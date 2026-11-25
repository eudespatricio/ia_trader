"""
agents/tape_reading_agent.py - Confirmação de fluxo (Tape Reading) para o gate de entrada (ADDENDUM E8).

Lê SOMENTE LEITURA (sqlite `mode=ro`) o banco datasets/microstructure.db alimentado por
scripts/microstructure_collector.py (tabelas book_features e tick_agg). Regras PRÉ-REGISTRADAS e fixas
(constantes abaixo; não otimizar). Sem lookahead: só linhas com ts <= now_ms.

Voto:
  BUY  : taker_imb_5m >= +TAKER_IMB_MIN  E book_imb_10 >= +BOOK_IMB_MIN  E microprice_dev >= 0
  SELL : espelhado (<= -TAKER_IMB_MIN, <= -BOOK_IMB_MIN, microprice_dev <= 0)
  HOLD : caso contrário, ou spread_bps > SPREAD_MAX_BPS (motivo 'spread_wide').
Status 'UNAVAILABLE' (fail-open no gate): símbolo fora da cobertura do coletor, DB ausente/ilegível,
dados ausentes, insuficientes ou velhos (book > BOOK_MAX_AGE_S, tick > TICK_MAX_AGE_S).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from db_conn import connect as _db_connect
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ia_trader.tape_reading")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MICROSTRUCTURE_DB = os.path.join(BASE, "datasets", "microstructure.db")

COVERED_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT"})   # = BASE SYMBOLS do coletor
TAKER_WINDOW_MIN = 5            # soma dos últimos 5 minutos de tick_agg
TAKER_MIN_MINUTES = 3           # mínimo de minutos distintos com dados na janela
TAKER_MIN_TRADES = 30           # mínimo de trades somados na janela
BOOK_SNAPSHOTS = 4              # média dos últimos 4 snapshots
TAKER_IMB_MIN = 0.15
BOOK_IMB_MIN = 0.10
MICROPRICE_DEV_MIN_BPS = 0.0
SPREAD_MAX_BPS = 5.0
BOOK_MAX_AGE_S = 120
TICK_MAX_AGE_S = 180

# Item 2 & 3: Score contínuo e absorção institucional
FLOW_SCORE_MIN = 0.12
ABSORPTION_TAKER_THRESHOLD = 0.30
ABSORPTION_BOOK_THRESHOLD = 0.15

LIVE_FALLBACK_TIMEOUT_S = 2.5
LIVE_FALLBACK_CACHE_TTL_MS = 10_000
LIVE_FALLBACK_ERR_COOLDOWN_MS = 15_000


class TapeReadingAgent:
    name = "TapeReadingAgent"

    def __init__(
        self,
        db_path: Optional[str] = None,
        use_continuous_flow: bool = True,
        covered_symbols: Optional[set] = None,
        fallback_cache_ttl_ms: int = LIVE_FALLBACK_CACHE_TTL_MS,
        fallback_err_cooldown_ms: int = LIVE_FALLBACK_ERR_COOLDOWN_MS,
    ):
        self.db_path = db_path or MICROSTRUCTURE_DB
        self.use_continuous_flow = use_continuous_flow
        self.covered_symbols = set(covered_symbols) if covered_symbols else set(COVERED_SYMBOLS)
        self._fallback_cache_ttl_ms = int(fallback_cache_ttl_ms)
        self._fallback_err_cooldown_ms = int(fallback_err_cooldown_ms)
        self._live_fallback_cache: Dict[str, Tuple[int, Dict[str, Any], bool]] = {}
        self._last_refresh_ms: int = 0
        self.last_result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    def _unavailable(self, symbol: str, reason: str, now_ms: int, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._finish({
            "agent_name": self.name, "symbol": symbol, "vote": "HOLD", "status": "UNAVAILABLE",
            "confidence": 0.0, "features": features or {}, "reason": reason, "timestamp": now_ms,
        })

    def _finish(self, res: Dict[str, Any]) -> Dict[str, Any]:
        self.last_result = res
        return res

    def _connect_ro(self) -> sqlite3.Connection:
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(self.db_path)
        con = _db_connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0)
        con.execute("PRAGMA busy_timeout=5000")
        return con

    def add_covered_symbol(self, sym: str) -> None:
        self.covered_symbols.add(str(sym).upper())

    def set_covered_symbols(self, symbols: Any) -> None:
        self.covered_symbols = {str(s).upper() for s in symbols}

    def refresh_covered_symbols_from_db(self, now_ms: Optional[int] = None) -> None:
        """Sincroniza símbolos monitorados a partir da tabela active_symbols gerenciada pelo coletor."""
        now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        self._last_refresh_ms = now_ms
        con = None
        try:
            con = self._connect_ro()
            rows = con.execute(
                "SELECT symbol FROM active_symbols WHERE updated_at >= ?",
                (now_ms - 3600_000,)
            ).fetchall()
            for r in rows:
                if r[0]:
                    self.covered_symbols.add(str(r[0]).upper())
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.debug(f"Tabela active_symbols ausente ao sincronizar: {e}")
            else:
                logger.warning(f"Falha ao sincronizar active_symbols: {e}")
        finally:
            if con is not None:
                con.close()

    def _evaluate_live_binance_fallback(self, sym: str, now_ms: int, fallback_reason: str) -> Dict[str, Any]:
        """Fallback on-demand REST (Binance Futures) para altcoins de Golden Pot ou majors com DB local degradado."""
        sym = str(sym).upper()
        cached = self._live_fallback_cache.get(sym)
        if cached is not None:
            c_ts, c_res, c_unavail = cached
            ttl = self._fallback_err_cooldown_ms if c_unavail else self._fallback_cache_ttl_ms
            if ttl > 0 and 0 <= (now_ms - c_ts) < ttl:
                return self._finish({**c_res, "features": dict(c_res.get("features") or {})})

        def _cache_and_return(res: Dict[str, Any]) -> Dict[str, Any]:
            if len(self._live_fallback_cache) >= 256 and sym not in self._live_fallback_cache:
                self._live_fallback_cache.clear()
            self._live_fallback_cache[sym] = (
                now_ms,
                {**res, "features": dict(res.get("features") or {})},
                res.get("status") != "OK",
            )
            return res

        try:
            q_depth = f"https://fapi.binance.com/fapi/v1/depth?symbol={urllib.parse.quote(sym)}&limit=100"
            req_d = urllib.request.Request(q_depth, headers={"User-Agent": "ia-trader-tape/1.0"})
            with urllib.request.urlopen(req_d, timeout=LIVE_FALLBACK_TIMEOUT_S) as r_d:
                depth = json.loads(r_d.read().decode())
            asks, bids = depth.get("asks") or [], depth.get("bids") or []
            if not asks or not bids:
                return _cache_and_return(self._unavailable(sym, fallback_reason, now_ms))
            ba, bb = float(asks[0][0]), float(bids[0][0])
            sa, sb = float(asks[0][1]), float(bids[0][1])
            mid = (ba + bb) / 2.0
            if mid <= 0:
                return _cache_and_return(self._unavailable(sym, fallback_reason, now_ms))
            spread = (ba - bb) / mid * 1e4
            micro = (ba * sb + bb * sa) / (sa + sb) if (sa + sb) > 0 else mid
            mdev = (micro - mid) / mid * 1e4
            bid_n10 = sum(float(p) * float(q) for p, q in bids if abs(float(p) - mid) / mid * 1e4 <= 10.0)
            ask_n10 = sum(float(p) * float(q) for p, q in asks if abs(float(p) - mid) / mid * 1e4 <= 10.0)
            if (bid_n10 + ask_n10) <= 0:
                bid_n10 = sum(float(p) * float(q) for p, q in bids[:10])
                ask_n10 = sum(float(p) * float(q) for p, q in asks[:10])
            if (bid_n10 + ask_n10) <= 0:
                return _cache_and_return(self._unavailable(sym, "book_incomplete", now_ms))
            book_imb = (bid_n10 - ask_n10) / (bid_n10 + ask_n10)

            q_trades = f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={urllib.parse.quote(sym)}&limit=500"
            req_t = urllib.request.Request(q_trades, headers={"User-Agent": "ia-trader-tape/1.0"})
            with urllib.request.urlopen(req_t, timeout=LIVE_FALLBACK_TIMEOUT_S) as r_t:
                trades = json.loads(r_t.read().decode())
            win_start = now_ms - TAKER_WINDOW_MIN * 60_000
            recent = [t for t in trades if int(t.get("T", 0)) >= win_start]
            if len(recent) < TAKER_MIN_TRADES:
                fallback_subset = trades[-max(TAKER_MIN_TRADES, 50):]
                # P0: Rejeitar ticks stale no fallback se os trades forem anteriores à janela
                if not fallback_subset or int(fallback_subset[-1].get("T", 0)) < win_start:
                    return _cache_and_return(self._unavailable(sym, "stale_tick_fallback", now_ms))
                recent = fallback_subset
            if not recent:
                return _cache_and_return(self._unavailable(sym, "no_tick_data", now_ms))
            buy = sum(float(t["p"]) * float(t["q"]) for t in recent if not t.get("m", False))
            sell = sum(float(t["p"]) * float(t["q"]) for t in recent if t.get("m", False))
            if (buy + sell) <= 0:
                return _cache_and_return(self._unavailable(sym, "tick_zero_volume", now_ms))
            taker = (buy - sell) / (buy + sell)
            # P0: Normalização de microprice por spread (evita saturação em 1 bp)
            mdev_norm = max(-1.0, min(1.0, mdev / max(float(spread), 1.0)))
            flow_score = 0.60 * taker + 0.30 * book_imb + 0.10 * mdev_norm
            feats: Dict[str, Any] = {
                "source": "binance_live_rest",
                "n_trades_5m": len(recent),
                "taker_imb_5m": taker,
                "book_imb_10": book_imb,
                "microprice_dev": mdev,
                "spread_bps": spread,
                "flow_score": round(flow_score, 4),
            }
            is_bullish_absorption = (taker <= -ABSORPTION_TAKER_THRESHOLD and book_imb >= ABSORPTION_BOOK_THRESHOLD)
            is_bearish_absorption = (taker >= ABSORPTION_TAKER_THRESHOLD and book_imb <= -ABSORPTION_BOOK_THRESHOLD)
            # Para altcoins em live fallback, toleramos spread até 15 bps (vs 5 bps nos majors BTC/ETH/SOL)
            max_spread = SPREAD_MAX_BPS if sym in COVERED_SYMBOLS else 15.0
            if spread > max_spread:
                vote, reason = "HOLD", "spread_wide"
            elif is_bullish_absorption:
                vote, reason = "BUY", "absorption_buy_confirmed"
            elif is_bearish_absorption:
                vote, reason = "SELL", "absorption_sell_confirmed"
            elif self.use_continuous_flow:
                if flow_score >= FLOW_SCORE_MIN:
                    vote, reason = "BUY", "flow_buy_confirmed"
                elif flow_score <= -FLOW_SCORE_MIN:
                    vote, reason = "SELL", "flow_sell_confirmed"
                else:
                    vote, reason = "HOLD", "no_flow_confluence"
            else:
                if taker >= TAKER_IMB_MIN and book_imb >= BOOK_IMB_MIN and mdev >= MICROPRICE_DEV_MIN_BPS:
                    vote, reason = "BUY", "flow_buy_confirmed"
                elif taker <= -TAKER_IMB_MIN and book_imb <= -BOOK_IMB_MIN and mdev <= -MICROPRICE_DEV_MIN_BPS:
                    vote, reason = "SELL", "flow_sell_confirmed"
                else:
                    vote, reason = "HOLD", "no_flow_confluence"
            conf = float(min(1.0, abs(flow_score) / (2.0 * FLOW_SCORE_MIN))) if vote != "HOLD" else 0.0
            return _cache_and_return(self._finish({
                "agent_name": self.name, "symbol": sym, "vote": vote, "status": "OK", "confidence": conf,
                "features": feats, "reason": reason, "timestamp": now_ms,
            }))
        except Exception as e:  # noqa: BLE001
            logger.warning("TapeReading live REST fallback failed for %s (%s): %s", sym, fallback_reason, e)
            return _cache_and_return(self._unavailable(sym, fallback_reason, now_ms))

    def _fetch_binance_series_taker(self, con: sqlite3.Connection, sym: str, now_ms: int) -> Optional[Tuple[float, float]]:
        """Retorna (taker_imb, age_s) para kind='taker' dos últimos 5m, ou None se indisponível/estragado."""
        try:
            exists = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='binance_series'"
            ).fetchone()
            if not exists:
                return None
            rows = con.execute(
                "SELECT ts_ms, v1, v2, v3 FROM binance_series "
                "WHERE symbol=? AND kind='taker' AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 4",
                (sym, now_ms),
            ).fetchall()
            if not rows:
                return None
            if len(rows) >= 2:
                deltas = sorted(
                    int(rows[i][0]) - int(rows[i + 1][0])
                    for i in range(len(rows) - 1)
                    if int(rows[i][0]) - int(rows[i + 1][0]) > 0
                )
                if deltas:
                    mid = len(deltas) // 2
                    med = deltas[mid] if len(deltas) % 2 == 1 else (deltas[mid - 1] + deltas[mid]) / 2.0
                    period_ms = int(min(900_000, max(60_000, med)))
                else:
                    period_ms = 300_000
                closed_rows = [r for r in rows if int(r[0]) + period_ms <= now_ms]
                if not closed_rows:
                    return None
                rows = closed_rows
                newest_ts = int(rows[0][0])
                age_s = (now_ms - (newest_ts + period_ms)) / 1000.0
                win_span = max(300_000, period_ms)
            else:
                newest_ts = int(rows[0][0])
                age_s = (now_ms - newest_ts) / 1000.0
                win_span = 300_000
            if age_s > 720.0:
                return None
            win_start = newest_ts - win_span
            buy = 0.0
            sell = 0.0
            for row in rows:
                ts = int(row[0])
                if ts < win_start:
                    continue
                buy += float(row[2] or 0.0)
                sell += float(row[3] or 0.0)
            if buy + sell <= 0.0:
                return None
            taker = (buy - sell) / (buy + sell)
            return (taker, age_s)
        except Exception as e:
            logger.debug(f"Falha ao ler binance_series taker para {sym}: {e}")
            return None

    def _fetch_liq_telemetry(self, con: sqlite3.Connection, sym: str, now_ms: int) -> Dict[str, float]:
        """Soma notional usd de liquidações dos últimos 5m por lado agressor (long/short)."""
        out: Dict[str, float] = {}
        try:
            exists = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='liquidations'"
            ).fetchone()
            if not exists:
                return out
            row = con.execute(
                "SELECT "
                "SUM(CASE WHEN side='SELL' THEN notional_usd ELSE 0 END), "
                "SUM(CASE WHEN side='BUY' THEN notional_usd ELSE 0 END) "
                "FROM liquidations "
                "WHERE symbol=? AND ts_ms<=? AND ts_ms>?",
                (sym, now_ms, now_ms - 300_000),
            ).fetchone()
            if row:
                if row[0] is not None:
                    out["liq_long_5m_usd"] = float(row[0])
                if row[1] is not None:
                    out["liq_short_5m_usd"] = float(row[1])
        except Exception as e:
            logger.debug(f"Falha ao ler liquidations para {sym}: {e}")
        return out

    def evaluate(
        self,
        symbol: str,
        direction: str = "",
        now_ms: Optional[int] = None,
        allow_live_fallback: bool = False,
    ) -> Dict[str, Any]:
        """Retorna dict AgentSignal-like. `direction` é só informativo (o voto não depende dele)."""
        is_live_call = (
            ((now_ms is None) or bool(allow_live_fallback))
            and (self.db_path == MICROSTRUCTURE_DB)
            and os.path.exists(self.db_path)
        )
        now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        sym = str(symbol).upper()
        if sym not in self.covered_symbols:
            if (now_ms - self._last_refresh_ms) > 30_000:
                self.refresh_covered_symbols_from_db(now_ms)
            if sym not in self.covered_symbols:
                if is_live_call:
                    return self._evaluate_live_binance_fallback(sym, now_ms, "symbol_not_covered")
                return self._unavailable(sym, "symbol_not_covered", now_ms)

        con: Optional[sqlite3.Connection] = None
        try:
            con = self._connect_ro()
        except Exception as e:  # noqa: BLE001
            return self._unavailable(sym, f"db_unavailable:{type(e).__name__}", now_ms)

        try:
            # --- Schema-aware book_features ---
            book_cols = [row[1] for row in con.execute("PRAGMA table_info(book_features)").fetchall()]
            has_book_fine = all(c in book_cols for c in ("bid_n5", "ask_n5", "bid_n2", "ask_n2"))
            if has_book_fine:
                books = con.execute(
                    "SELECT ts_ms, spread_bps, microprice_dev_bps, bid_n10, ask_n10, "
                    "bid_n5, ask_n5, bid_n2, ask_n2 "
                    "FROM book_features WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT ?",
                    (sym, now_ms, BOOK_SNAPSHOTS)).fetchall()
            else:
                books = con.execute(
                    "SELECT ts_ms, spread_bps, microprice_dev_bps, bid_n10, ask_n10 "
                    "FROM book_features WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT ?",
                    (sym, now_ms, BOOK_SNAPSHOTS)).fetchall()

            # --- Schema-aware tick_agg ---
            win_start = now_ms - TAKER_WINDOW_MIN * 60_000
            tick_cols = [row[1] for row in con.execute("PRAGMA table_info(tick_agg)").fetchall()]
            if "saturated" in tick_cols:
                ticks = con.execute(
                    "SELECT minute_ms, buy_notional, sell_notional, n_trades, "
                    "COALESCE(saturated, 0) AS saturated "
                    "FROM tick_agg WHERE symbol=? AND minute_ms<=? AND minute_ms>? "
                    "ORDER BY minute_ms DESC",
                    (sym, now_ms, win_start)).fetchall()
            else:
                ticks = con.execute(
                    "SELECT minute_ms, buy_notional, sell_notional, n_trades, 0 AS saturated "
                    "FROM tick_agg WHERE symbol=? AND minute_ms<=? AND minute_ms>? "
                    "ORDER BY minute_ms DESC",
                    (sym, now_ms, win_start)).fetchall()

            liq_feats = self._fetch_liq_telemetry(con, sym, now_ms)

            # --- Processamento ---
            if not books:
                if is_live_call:
                    con.close()
                    con = None
                    return self._evaluate_live_binance_fallback(sym, now_ms, "no_book_data")
                return self._unavailable(sym, "no_book_data", now_ms, liq_feats)

            book_age = (now_ms - int(books[0][0])) / 1000.0
            feats: Dict[str, Any] = {"book_age_s": round(book_age, 1)}
            feats.update(liq_feats)
            if book_age > BOOK_MAX_AGE_S:
                if is_live_call:
                    con.close()
                    con = None
                    return self._evaluate_live_binance_fallback(sym, now_ms, "book_stale")
                return self._unavailable(sym, "book_stale", now_ms, feats)

            sat_ratio = (sum(1 for r in ticks if int(r[4] or 0)) / len(ticks)) if ticks else 0.0
            tick_age = (now_ms - int(ticks[0][0])) / 1000.0 if ticks else None
            if tick_age is not None:
                feats["tick_age_s"] = round(tick_age, 1)

            n_min = len(ticks)
            n_trades = sum(int(r[3] or 0) for r in ticks)
            buy = sum(float(r[1] or 0.0) for r in ticks)
            sell = sum(float(r[2] or 0.0) for r in ticks)
            buy_plus_sell = buy + sell
            feats.update({"tick_minutes": n_min, "n_trades_5m": n_trades})

            tick_missing_or_bad = (
                (not ticks)
                or (tick_age is not None and tick_age > TICK_MAX_AGE_S)
                or (n_min < TAKER_MIN_MINUTES)
                or (n_trades < TAKER_MIN_TRADES)
                or (buy_plus_sell <= 0.0)
                or (sat_ratio > 0.50)
            )

            if tick_missing_or_bad:
                binance_taker = self._fetch_binance_series_taker(con, sym, now_ms)
                if binance_taker is not None:
                    taker_bs, bs_age = binance_taker
                    feats["taker_source"] = "binance_series_5m"
                    feats["taker_age_s"] = round(bs_age, 1)
                    feats["tick_saturated_ratio"] = round(sat_ratio, 2)
                    taker = taker_bs
                else:
                    if not ticks:
                        err_reason = "no_tick_data"
                    elif tick_age is not None and tick_age > TICK_MAX_AGE_S:
                        err_reason = "tick_stale"
                    elif n_min < TAKER_MIN_MINUTES or n_trades < TAKER_MIN_TRADES:
                        err_reason = "tick_insufficient"
                    elif buy_plus_sell <= 0.0:
                        err_reason = "tick_zero_volume"
                    else:
                        # sat_ratio > 0.50: trata como insuficiência de qualidade
                        err_reason = "tick_insufficient"
                    if is_live_call:
                        con.close()
                        con = None
                        return self._evaluate_live_binance_fallback(sym, now_ms, err_reason)
                    return self._unavailable(sym, err_reason, now_ms, feats)
            else:
                taker = (buy - sell) / buy_plus_sell
                feats["taker_source"] = "tick_agg"
                feats["tick_saturated_ratio"] = round(sat_ratio, 2)

            # Processamento de book imbalance com fallback de colunas
            imbs: List[float] = []
            for r in books:
                if has_book_fine:
                    b, a = r[3], r[4]  # bid_n10/ask_n10
                    if b is None or a is None or (float(b) + float(a)) <= 0.0:
                        b, a = r[5], r[6]  # fallback bid_n5/ask_n5
                    if b is None or a is None or (float(b) + float(a)) <= 0.0:
                        b, a = r[7], r[8]  # fallback bid_n2/ask_n2
                    if b is None or a is None or (float(b) + float(a)) <= 0.0:
                        continue
                    imbs.append((float(b) - float(a)) / (float(b) + float(a)))
                else:
                    b, a = r[3], r[4]
                    if b is None or a is None or (float(b) + float(a)) <= 0.0:
                        continue
                    imbs.append((float(b) - float(a)) / (float(b) + float(a)))

            devs = [float(r[2]) for r in books if r[2] is not None]
            spread = books[0][1]
            if not imbs or not devs or spread is None:
                return self._unavailable(sym, "book_incomplete", now_ms, feats)
            book_imb = sum(imbs) / len(imbs)
            mdev = sum(devs) / len(devs)
            spread = float(spread)

            # P0: Normalização de microprice por spread (evita saturação em 1 bp)
            mdev_norm = max(-1.0, min(1.0, mdev / max(spread, 1.0)))
            flow_score = 0.60 * taker + 0.30 * book_imb + 0.10 * mdev_norm
            feats.update({
                "taker_imb_5m": taker,
                "book_imb_10": book_imb,
                "microprice_dev": mdev,
                "spread_bps": spread,
                "flow_score": round(flow_score, 4),
            })

            is_bullish_absorption = (taker <= -ABSORPTION_TAKER_THRESHOLD and book_imb >= ABSORPTION_BOOK_THRESHOLD)
            is_bearish_absorption = (taker >= ABSORPTION_TAKER_THRESHOLD and book_imb <= -ABSORPTION_BOOK_THRESHOLD)

            if spread > SPREAD_MAX_BPS:
                vote, reason = "HOLD", "spread_wide"
            elif is_bullish_absorption:
                vote, reason = "BUY", "absorption_buy_confirmed"
            elif is_bearish_absorption:
                vote, reason = "SELL", "absorption_sell_confirmed"
            elif self.use_continuous_flow:
                if flow_score >= FLOW_SCORE_MIN:
                    vote, reason = "BUY", "flow_buy_confirmed"
                elif flow_score <= -FLOW_SCORE_MIN:
                    vote, reason = "SELL", "flow_sell_confirmed"
                else:
                    vote, reason = "HOLD", "no_flow_confluence"
            else:
                if taker >= TAKER_IMB_MIN and book_imb >= BOOK_IMB_MIN and mdev >= MICROPRICE_DEV_MIN_BPS:
                    vote, reason = "BUY", "flow_buy_confirmed"
                elif taker <= -TAKER_IMB_MIN and book_imb <= -BOOK_IMB_MIN and mdev <= -MICROPRICE_DEV_MIN_BPS:
                    vote, reason = "SELL", "flow_sell_confirmed"
                else:
                    vote, reason = "HOLD", "no_flow_confluence"

            conf = 0.0
            if vote != "HOLD":
                if "absorption" in reason:
                    conf = float(min(1.0, 0.5 * (abs(taker) / 0.50) + 0.5 * (abs(book_imb) / 0.30)))
                elif self.use_continuous_flow:
                    conf = float(min(1.0, abs(flow_score) / (2.0 * FLOW_SCORE_MIN)))
                else:
                    conf = float(min(1.0, 0.5 * min(1.0, abs(taker) / (2 * TAKER_IMB_MIN))
                                     + 0.5 * min(1.0, abs(book_imb) / (2 * BOOK_IMB_MIN))))
            return self._finish({
                "agent_name": self.name, "symbol": sym, "vote": vote, "status": "OK", "confidence": conf,
                "features": feats, "reason": reason, "timestamp": now_ms,
            })
        except Exception as e:  # noqa: BLE001
            return self._unavailable(sym, f"db_error:{type(e).__name__}", now_ms)
        finally:
            if con is not None:
                con.close()
