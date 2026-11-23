import sqlite3
import json
import logging
import os
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

from db_conn import connect as _db_connect
from agents.regime_classifier import MarketRegimeClassifier
from data.data_integrations import AsyncHttpClient, YahooFinanceConnector

logger = logging.getLogger(__name__)


class LiveContextEnricher:
    """Enriquece contextos do comitê com dados estritamente causais (sem lookahead)."""

    def __init__(self, cache_db_path: str, micro_db_path: str, refresh_interval_s: float = 21600.0):
        self.cache_db_path = cache_db_path
        self.micro_db_path = micro_db_path
        self.refresh_interval_s = refresh_interval_s
        self._last_refresh_mono: float = 0.0
        self._enrich_error_count: int = 0
        # db_path="" evita que o classificador toque no market_cache.db padrão durante testes
        self._classifier = MarketRegimeClassifier(db_path="")

    def _connect_ro(self, path: str) -> Optional[sqlite3.Connection]:
        if not path or not os.path.exists(path):
            return None
        con = _db_connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True, timeout=5.0)
        con.execute("PRAGMA busy_timeout=5000")
        return con

    def _fetch_all(self, path: str, sql: str, params: Tuple[Any, ...] = ()) -> List[Tuple[Any, ...]]:
        con = None
        try:
            con = self._connect_ro(path)
            if con is None:
                return []
            return con.execute(sql, params).fetchall()
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.debug(f"Tabela opcional ausente em {path}: {e}")
            else:
                self._enrich_error_count += 1
                logger.warning(f"Erro ao consultar {path}: {e}")
            return []
        finally:
            if con is not None:
                con.close()

    def enrich(self, symbol: str, latest: Any, df_feat: Optional[pd.DataFrame], ts_ms: int) -> Dict[str, Any]:
        ctx = latest.to_dict() if hasattr(latest, "to_dict") else dict(latest or {})

        # 1. OHLCV-derived (causal)
        if "volume_sma" not in ctx or ctx.get("volume_sma") is None:
            v20 = ctx.get("vol_sma_20")
            try:
                if v20 is not None and np.isfinite(float(v20)) and float(v20) > 0:
                    ctx["volume_sma"] = float(v20)
            except (TypeError, ValueError):
                pass

        if df_feat is not None and not df_feat.empty:
            if ctx.get("volume_sma") is None and "volume" in df_feat.columns:
                vol = pd.to_numeric(df_feat["volume"], errors="coerce").dropna()
                if len(vol) > 0:
                    ctx["volume_sma"] = float(vol.iloc[-20:].mean())
            # ATR baselines
            if "atr_14" in df_feat.columns:
                atr = pd.to_numeric(df_feat["atr_14"], errors="coerce").dropna()
                if len(atr) >= 5:
                    median_hist = atr.iloc[-49:-1]
                    if len(median_hist) > 0:
                        ctx["atr_median"] = float(median_hist.median())
                    sma_hist = atr.iloc[-21:-1]
                    if len(sma_hist) > 0:
                        ctx["atr_sma"] = float(sma_hist.mean())
            # regime
            if "regime" not in ctx and len(df_feat) >= 50:
                try:
                    ctx["regime"] = self._classifier.classify_from_df(df_feat, symbol=symbol).regime
                except Exception as e:
                    logger.debug(f"Falha ao classificar regime: {e}")

        day_start_ms = (int(ts_ms) // 86_400_000) * 86_400_000
        try:
            rows = self._fetch_all(
                self.cache_db_path,
                "SELECT series_name, date_str, timestamp_ms, value FROM macro_daily_history WHERE timestamp_ms < ? ORDER BY date_str ASC, timestamp_ms ASC",
                (day_start_ms,),
            )
            series: Dict[str, List[Tuple[str, int, float]]] = {}
            for sn, ds, t, v in rows:
                try:
                    series.setdefault(sn, []).append((ds, int(t), float(v)))
                except (TypeError, ValueError):
                    continue

            # DXY
            dxy = series.get("DXY", [])
            if len(dxy) >= 6:
                latest_ts = dxy[-1][1]
                if day_start_ms - latest_ts <= 7 * 86_400_000:
                    vals = [x[2] for x in dxy]
                    if vals[-6] > 0:
                        ctx["dxy_ret_5d"] = (vals[-1] - vals[-6]) / vals[-6]
                    if len(vals) >= 20:
                        ctx["dxy_above_sma20"] = bool(vals[-1] > float(np.mean(vals[-20:])))

            # VIX
            vix = series.get("VIX", [])
            if vix and (day_start_ms - vix[-1][1] <= 7 * 86_400_000):
                ctx["vix_value"] = float(vix[-1][2])

            # SP500
            sp500 = series.get("SP500", [])
            if len(sp500) >= 6:
                latest_ts = sp500[-1][1]
                if day_start_ms - latest_ts <= 7 * 86_400_000:
                    vals = [x[2] for x in sp500]
                    if vals[-6] > 0:
                        ctx["sp500_ret_5d"] = (vals[-1] - vals[-6]) / vals[-6]

            # FEAR_AND_GREED
            fng = series.get("FEAR_AND_GREED", [])
            if fng and (day_start_ms - fng[-1][1] <= 3 * 86_400_000):
                latest_fng_ts = fng[-1][1]
                ctx["fear_and_greed"] = float(fng[-1][2])
                target_fng_7d = latest_fng_ts - 7 * 86_400_000
                prior_fng = [x for x in fng if x[1] <= target_fng_7d]
                if not prior_fng:
                    prior_fng = [
                        x for x in fng
                        if 6 * 86_400_000 <= (latest_fng_ts - x[1]) <= 8 * 86_400_000
                    ]
                if prior_fng:
                    best_prior = min(prior_fng, key=lambda x: abs((latest_fng_ts - x[1]) - 7 * 86_400_000)) if len(prior_fng) > 1 else prior_fng[-1]
                    ctx["fng_7d_chg"] = float(fng[-1][2] - best_prior[2])

            # STABLECOINS
            stbl = series.get("STABLECOINS_CIRCULATING_USD") or series.get("STABLECOIN_MCAP")
            if stbl:
                latest_row = stbl[-1]
                if day_start_ms - latest_row[1] <= 3 * 86_400_000 and len(stbl) >= 2:
                    latest_ts = latest_row[1]
                    target_ms = latest_ts - 30 * 86_400_000
                    prior = [x for x in stbl if x[1] <= target_ms]
                    if not prior:
                        prior = [x for x in stbl if 25 * 86_400_000 <= (latest_ts - x[1]) <= 35 * 86_400_000]
                    if prior:
                        ref_val = prior[-1][2]
                        if ref_val > 0:
                            ctx["stablecoin_growth_pct_30d"] = (stbl[-1][2] - ref_val) / ref_val

            # DEFI_TVL
            tvl = series.get("DEFI_TVL", [])
            if tvl:
                latest_row = tvl[-1]
                if day_start_ms - latest_row[1] <= 3 * 86_400_000:
                    if len(tvl) >= 2 and tvl[-2][2] > 0:
                        ctx["tvl_change_24h"] = (tvl[-1][2] - tvl[-2][2]) / tvl[-2][2]
                    target7 = latest_row[1] - 7 * 86_400_000
                    prior7 = [x for x in tvl if x[1] <= target7]
                    if not prior7:
                        prior7 = [x for x in tvl if 6 * 86_400_000 <= (latest_row[1] - x[1]) <= 8 * 86_400_000]
                    if prior7 and prior7[-1][2] > 0:
                        ctx["tvl_growth_pct_7d"] = (tvl[-1][2] - prior7[-1][2]) / prior7[-1][2]
        except Exception as e:
            logger.debug(f"Falha ao processar macro_daily_history: {e}")

        # 3. Macro Events Calendar
        ctx.setdefault("is_cpi_fomc_window", False)
        try:
            events = self._fetch_all(
                self.cache_db_path,
                "SELECT event_name, timestamp_ms FROM macro_events_calendar WHERE timestamp_ms BETWEEN ? AND ?",
                (ts_ms - 86_400_000, ts_ms + 86_400_000),
            )
            for event_name, event_ts_ms in events:
                effective_ts = int(event_ts_ms)
                if effective_ts % 86_400_000 == 0:
                    name = str(event_name).upper()
                    if "CPI" in name:
                        effective_ts += 13 * 3_600_000 + 30 * 60_000
                    elif "FOMC" in name:
                        effective_ts += 19 * 3_600_000
                if abs(int(ts_ms) - effective_ts) <= 30 * 60 * 1000:
                    ctx["is_cpi_fomc_window"] = True
                    break
        except Exception as e:
            logger.debug(f"Falha ao processar macro_events_calendar: {e}")

        # 4. Funding Rate & Z-Score
        rates: List[float] = []
        funding_source: Optional[str] = None
        try:
            def _extract_finite_rates(rows_in: List[Tuple[Any, Any]]) -> List[float]:
                out: List[float] = []
                for row in rows_in:
                    if row[1] is None:
                        continue
                    try:
                        val = float(row[1])
                        if np.isfinite(val):
                            out.append(val)
                    except (TypeError, ValueError):
                        continue
                return out

            funding_rows = self._fetch_all(
                self.cache_db_path,
                "SELECT timestamp, funding_rate FROM funding_rates WHERE symbol=? AND timestamp<=? AND funding_rate IS NOT NULL ORDER BY timestamp DESC LIMIT 90",
                (symbol, ts_ms),
            )
            is_cache_fresh = (
                bool(funding_rows)
                and (ts_ms - int(funding_rows[0][0]) <= 12 * 3_600_000)
                and funding_rows[0][1] is not None
            )
            if is_cache_fresh:
                rates = _extract_finite_rates(funding_rows)
                if rates:
                    funding_source = "cache"
            else:
                deriv_latest = self._fetch_all(
                    self.micro_db_path,
                    "SELECT ts_ms, funding_rate FROM derivs WHERE symbol=? AND ts_ms<=? AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 1",
                    (symbol, ts_ms),
                )
                if deriv_latest and (ts_ms - int(deriv_latest[0][0]) <= 12 * 3_600_000):
                    try:
                        fresh_rate = float(deriv_latest[0][1])
                    except (TypeError, ValueError):
                        fresh_rate = float("nan")
                    if np.isfinite(fresh_rate):
                        funding_source = "micro_derivs"
                        hist_rates_from_cache = _extract_finite_rates(funding_rows) if funding_rows else []
                        combined = [fresh_rate] + hist_rates_from_cache
                        if len(combined) >= 10:
                            rates = combined
                        else:
                            deriv_hist = self._fetch_all(
                                self.micro_db_path,
                                "SELECT ts_ms, funding_rate FROM derivs WHERE symbol=? AND ts_ms<=? AND funding_rate IS NOT NULL ORDER BY ts_ms DESC LIMIT 200",
                                (symbol, ts_ms),
                            )
                            seen_buckets = set()
                            bucket_rates: List[float] = []
                            for r_ts, r_val in deriv_hist:
                                try:
                                    rv = float(r_val)
                                    if not np.isfinite(rv):
                                        continue
                                    bucket = int(r_ts) // 3_600_000
                                    if bucket not in seen_buckets:
                                        seen_buckets.add(bucket)
                                        bucket_rates.append(rv)
                                except (TypeError, ValueError):
                                    continue
                            rates = bucket_rates if bucket_rates else [fresh_rate]

            if rates:
                ctx["funding_rate"] = float(rates[0])
                if funding_source is not None:
                    ctx["funding_source"] = funding_source
                if len(rates) >= 10:
                    std = float(np.std(rates, ddof=0))
                    if std > 1e-9:
                        ctx["funding_zscore"] = (rates[0] - float(np.mean(rates))) / std
                    else:
                        ctx["funding_zscore"] = 0.0
                else:
                    ctx["funding_zscore"] = 0.0
        except Exception as e:
            logger.debug(f"Falha ao processar funding_rates: {e}")

        # 5. Live Microstructure (schema-aware book_features fallback n10 -> n5 -> n2)
        try:
            book_cols_rows = self._fetch_all(self.micro_db_path, "PRAGMA table_info(book_features)")
            book_col_names = {r[1] for r in book_cols_rows} if book_cols_rows else set()
            want_cols = [c for c in ("ts_ms", "spread_bps", "bid_n10", "ask_n10", "bid_n5", "ask_n5", "bid_n2", "ask_n2") if c in book_col_names]
            if "ts_ms" in want_cols:
                sql_b = f"SELECT {', '.join(want_cols)} FROM book_features WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1"
                book = self._fetch_all(self.micro_db_path, sql_b, (symbol, ts_ms))
                if book:
                    row_map = dict(zip(want_cols, book[0]))
                    book_ts = row_map["ts_ms"]
                    if ts_ms - int(book_ts) <= 120_000:
                        try:
                            spread_bps = row_map.get("spread_bps")
                            if spread_bps is not None and np.isfinite(float(spread_bps)):
                                ctx["spread_pct"] = float(spread_bps) / 10_000.0
                                ctx["micro_spread_bps"] = float(spread_bps)
                            for bk, ak in (("bid_n10", "ask_n10"), ("bid_n5", "ask_n5"), ("bid_n2", "ask_n2")):
                                b_val, a_val = row_map.get(bk), row_map.get(ak)
                                if b_val is not None and a_val is not None:
                                    bid, ask = float(b_val), float(a_val)
                                    if np.isfinite(bid) and np.isfinite(ask) and (bid + ask) > 0:
                                        ctx["micro_book_imb_10"] = (bid - ask) / (bid + ask)
                                        break
                        except (TypeError, ValueError):
                            pass
        except Exception as e:
            logger.debug(f"Falha ao processar book_features: {e}")

        try:
            valid_taker = False
            tick_rows = self._fetch_all(
                self.micro_db_path,
                "SELECT minute_ms, buy_notional, sell_notional, saturated FROM tick_agg "
                "WHERE symbol=? AND minute_ms + 60000 <= ? AND minute_ms + 60000 > ? AND COALESCE(saturated,0)=0 ORDER BY minute_ms",
                (symbol, ts_ms, ts_ms - 5 * 60_000),
            )
            if tick_rows:
                minutes = {r[0] for r in tick_rows}
                max_min = max(r[0] for r in tick_rows)
                total_buy = sum(float(r[1] or 0) for r in tick_rows)
                total_sell = sum(float(r[2] or 0) for r in tick_rows)
                if len(minutes) >= 3 and (ts_ms - (int(max_min) + 60_000)) <= 180_000 and (total_buy + total_sell) > 0:
                    ctx["micro_taker_imb_5m"] = (total_buy - total_sell) / (total_buy + total_sell)
                    valid_taker = True
            if not valid_taker:
                bs_rows = self._fetch_all(
                    self.micro_db_path,
                    "SELECT ts_ms, v2, v3 FROM binance_series WHERE symbol=? AND kind='taker' AND ts_ms <= ? ORDER BY ts_ms DESC LIMIT 3",
                    (symbol, ts_ms),
                )
                if bs_rows:
                    deltas = [
                        int(bs_rows[i][0]) - int(bs_rows[i + 1][0])
                        for i in range(len(bs_rows) - 1)
                        if int(bs_rows[i][0]) - int(bs_rows[i + 1][0]) > 0
                    ]
                    if deltas:
                        period_ms = int(min(900_000, max(60_000, float(np.median(deltas)))))
                    else:
                        period_ms = 300_000
                    closed_rows = [r for r in bs_rows if int(r[0]) + period_ms <= ts_ms]
                    if closed_rows:
                        taker_ts, v2, v3 = closed_rows[0]
                        if ts_ms - (int(taker_ts) + period_ms) <= 720_000:
                            try:
                                sum_val = float(v2) + float(v3)
                                if sum_val > 0:
                                    ctx["micro_taker_imb_5m"] = (float(v2) - float(v3)) / sum_val
                            except (TypeError, ValueError):
                                pass
        except Exception as e:
            logger.debug(f"Falha ao processar tick_agg/binance_series: {e}")

        ctx["enrich_errors"] = int(self._enrich_error_count)
        return ctx

    async def maybe_refresh_external_cache(self, force: bool = False) -> Dict[str, int]:
        if not force and (time.monotonic() - self._last_refresh_mono) < self.refresh_interval_s:
            return {}
        self._last_refresh_mono = time.monotonic()

        http = AsyncHttpClient(timeout_seconds=15.0)
        yf = YahooFinanceConnector(http)
        counts: Dict[str, int] = {}
        try:
            results = await asyncio.gather(
                yf.get_history("DX-Y.NYB", range_str="3mo", interval="1d"),
                yf.get_history("^VIX", range_str="3mo", interval="1d"),
                yf.get_history("^GSPC", range_str="3mo", interval="1d"),
                http.get_json("https://api.alternative.me/fng/?limit=40"),
                http.get_json("https://stablecoins.llama.fi/stablecoincharts/all"),
                http.get_json("https://api.llama.fi/charts"),
                return_exceptions=True,
            )

            series_names = ["DXY", "VIX", "SP500", "FEAR_AND_GREED", "STABLECOINS_CIRCULATING_USD", "DEFI_TVL"]
            all_tuples: List[Tuple[str, str, int, float, str]] = []

            for idx, result in enumerate(results):
                name = series_names[idx]
                if isinstance(result, BaseException):
                    logger.warning(f"Fonte {name} falhou: {result}")
                    continue
                entries = []
                if idx in (0, 1, 2):
                    entries = self._parse_yf(result)
                elif idx == 3:
                    entries = self._parse_fng(result)
                elif idx == 4:
                    entries = self._parse_stablecoins(result)
                elif idx == 5:
                    entries = self._parse_defi_tvl(result)
                if entries:
                    counts[name] = len(entries)
                    for date_str, ts, val in entries:
                        all_tuples.append((name, date_str, ts, val, ""))

            if all_tuples:
                con = None
                try:
                    con = _db_connect(self.cache_db_path, timeout=15.0)
                    con.execute("PRAGMA journal_mode=WAL")
                    con.execute(
                        """
                        CREATE TABLE IF NOT EXISTS macro_daily_history (
                            series_name TEXT,
                            date_str TEXT,
                            timestamp_ms INTEGER,
                            value REAL,
                            extra_info TEXT,
                            PRIMARY KEY (series_name, date_str)
                        )
                        """
                    )
                    con.executemany(
                        "INSERT OR REPLACE INTO macro_daily_history VALUES (?,?,?,?,?)",
                        all_tuples,
                    )
                    con.commit()
                finally:
                    if con is not None:
                        con.close()
            else:
                # Se todas as fontes falharam, permite nova tentativa após 5 min (em vez de travar 6h)
                self._last_refresh_mono = max(0.0, time.monotonic() - max(0.0, self.refresh_interval_s - 300.0))
            return counts
        finally:
            await http.close()

    @staticmethod
    def _parse_yf(res: Any, utc_today: Optional[str] = None) -> List[Tuple[str, int, float]]:
        today_str = utc_today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        out: List[Tuple[str, int, float]] = []
        if isinstance(res, list):
            for item in res:
                if not isinstance(item, dict):
                    continue
                try:
                    val = item.get("close")
                    ts = item.get("timestamp")
                    ds = item.get("date")
                    if val is None or ts is None:
                        continue
                    fval = float(val)
                    its = int(ts)
                    if not np.isfinite(fval) or its <= 0:
                        continue
                    if not ds:
                        ds = datetime.fromtimestamp(its / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
                    date_str = str(ds)[:10]
                    if date_str >= today_str:
                        continue
                    out.append((date_str, its, fval))
                except (TypeError, ValueError):
                    continue
            return out[-90:]
        if isinstance(res, pd.DataFrame) and not res.empty:
            close_col = next((c for c in ("Close", "close", "Adj Close", "adjclose") if c in res.columns), None)
            if close_col is None:
                return []
            dates = res["Date"] if "Date" in res.columns else res.index
            for d, v in zip(dates, res[close_col]):
                try:
                    ts_dt = pd.Timestamp(d)
                    if pd.isna(v):
                        continue
                    date_str = ts_dt.strftime("%Y-%m-%d")
                    if date_str >= today_str:
                        continue
                    out.append((date_str, int(ts_dt.timestamp() * 1000), float(v)))
                except (TypeError, ValueError):
                    continue
            return out[-90:]
        return []

    @staticmethod
    def _parse_fng(resp: Any) -> List[Tuple[str, int, float]]:
        if not isinstance(resp, dict):
            return []
        data = resp.get("data", [])
        out = []
        for item in data:
            try:
                ts_sec = int(item.get("timestamp", 0))
                val = float(item.get("value"))
                ds = datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime("%Y-%m-%d")
                out.append((ds, ts_sec * 1000, val))
            except (TypeError, ValueError):
                continue
        return out[-40:]

    @staticmethod
    def _parse_stablecoins(resp: Any) -> List[Tuple[str, int, float]]:
        if isinstance(resp, list):
            data = resp
        elif isinstance(resp, dict):
            data = resp.get("data", [])
            if isinstance(data, dict):
                data = data.get("chart") or data.get("items") or data.get("list") or []
        else:
            return []
        out = []
        for item in data[-60:]:
            try:
                if not isinstance(item, dict):
                    continue
                ts = item.get("date") or item.get("timestamp")
                raw_val = item.get("totalCirculatingUSD")
                if isinstance(raw_val, dict):
                    val = raw_val.get("peggedUSD")
                else:
                    val = raw_val if raw_val is not None else (item.get("totalCirculating") or item.get("value"))
                if ts is None or val is None:
                    continue
                ts = int(ts)
                if ts < 10_000_000_000:
                    ts *= 1000
                ds = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
                out.append((ds, ts, float(val)))
            except (TypeError, ValueError):
                continue
        return out[-60:]

    @staticmethod
    def _parse_defi_tvl(resp: Any) -> List[Tuple[str, int, float]]:
        data = resp if isinstance(resp, list) else (resp.get("data", []) if isinstance(resp, dict) else [])
        if isinstance(data, dict):
            data = data.get("chart") or data.get("items") or data.get("list") or []
        out = []
        for item in data[-60:]:
            try:
                ts = item.get("date") or item.get("timestamp")
                val = item.get("totalLiquidityUSD") or item.get("tvl") or item.get("value")
                if ts is None or val is None:
                    continue
                ts = int(ts)
                if ts < 10_000_000_000:
                    ts *= 1000
                ds = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
                out.append((ds, ts, float(val)))
            except Exception:
                continue
        return out[-60:]
