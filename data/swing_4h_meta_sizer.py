"""data/swing_4h_meta_sizer.py - Meta-Sizer causal SWING_4H (LogReg-L2 + Vol-Target + ATR-RP + RS 14d)."""
from __future__ import annotations
from contextlib import closing
import json, logging, math, os, sqlite3
from typing import Any, Dict, List, Optional, Tuple
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data.swing_4h_strategy import (
    SWING_4H_BAR_MS, SWING_4H_COOLDOWN_BARS, SWING_4H_MAX_DIST_SMA200, SWING_4H_MAX_HOLDING_BARS,
    SWING_4H_SL_ATR_MULT, SWING_4H_TRAIL_ATR_MULT, compute_swing_4h_indicators, resample_1h_to_closed_4h,
)
from db_conn import connect as _db_connect

logger = logging.getLogger(__name__)
META_FEATURE_COLS: Tuple[str, ...] = ("atr_pct", "dir_funding_z", "dir_fng", "dir_sp500_5d", "dir_btc_168h", "vix", "abs_dist200", "dir_stbl30", "dir_dxy_5d", "dir_breadth")
VALID_MODES: Tuple[str, ...] = ("shadow", "active")
CORE_SYMBOLS: Tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
_DAY_MS, _ANN_4H_SQRT, _RT_COST = 86_400_000, math.sqrt(6.0 * 365.0), 0.0016
_TELEMETRY_DDL = "CREATE TABLE IF NOT EXISTS swing_4h_meta_telemetry (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp INTEGER NOT NULL, symbol TEXT NOT NULL, side TEXT NOT NULL, position_id TEXT, mode TEXT NOT NULL, meta_prob REAL, meta_size_mult REAL NOT NULL, vol_target_mult REAL NOT NULL, atr_risk_parity_mult REAL NOT NULL, rs_14d_mult REAL NOT NULL, combined_shadow_mult REAL NOT NULL, applied_mult REAL NOT NULL, features_json TEXT)"


def _finite(val: Any, default: float = 0.0) -> float:
    try:
        fv = float(val)
        return fv if math.isfinite(fv) else default
    except (TypeError, ValueError):
        return default


class Swing4HMetaSizer:
    def __init__(self, cache_db_path: str = "datasets/market_cache.db", mode: str = "shadow", refit_interval_ms: int = 7 * 86_400_000, min_train_samples: int = 40, target_ann_vol: float = 0.25, target_risk_pct: float = 0.015) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}; expected one of {VALID_MODES}")
        self.cache_db_path, self.mode = cache_db_path, mode
        self.refit_interval_ms, self.min_train_samples = int(refit_interval_ms), int(min_train_samples)
        self.target_ann_vol, self.target_risk_pct = float(target_ann_vol), float(target_risk_pct)
        self._model, self._base_rate, self._train_n, self._last_fit_ms, self._last_fit_attempt_ms = None, 0.5, 0, None, None
        self._funding_map, self._macro_map = {}, {}
        self._ensure_telemetry_table()

    def _ctx_at(self, sym: str, close_ts: int) -> Dict[str, float]:
        c: Dict[str, float] = {}
        ft = self._funding_map.get(sym)
        if ft is not None and len(ft[0]):
            idx = int(np.searchsorted(ft[0], close_ts, side="right"))
            w = ft[1][max(0, idx - 30) : idx]
            if len(w) >= 5 and float(np.std(w)) > 1e-9:
                c["funding_zscore"] = float((w[-1] - np.mean(w)) / np.std(w))
        d0 = (close_ts // _DAY_MS) * _DAY_MS
        for k, col, lag in (("FEAR_AND_GREED", "fear_and_greed", 0), ("VIX", "vix_value", 0), ("SP500", "sp500_ret_5d", 5), ("DXY", "dxy_ret_5d", 5), ("STABLECOINS_CIRCULATING_USD", "stablecoin_growth_pct_30d", 30)):
            mt = self._macro_map.get(k) or (self._macro_map.get("STABLECOIN_MCAP") if k.startswith("STABLE") else None)
            if mt is not None and len(mt[0]):
                mi = int(np.searchsorted(mt[0], d0, side="left")) - 1
                if mi >= lag and mt[1][mi - lag] > 0:
                    c[col] = float(mt[1][mi]) if lag == 0 else float((mt[1][mi] - mt[1][mi - lag]) / mt[1][mi - lag])
        return c

    def _ensure_telemetry_table(self) -> bool:
        if not self.cache_db_path or not os.path.isdir(os.path.dirname(os.path.abspath(self.cache_db_path))):
            return False
        try:
            with closing(_db_connect(self.cache_db_path, timeout=5.0)) as con:
                with con:
                    con.execute(_TELEMETRY_DDL)
            return True
        except (sqlite3.Error, OSError) as exc:
            logger.debug(f"Telemetria swing_4h_meta_telemetry indisponível em {self.cache_db_path}: {exc}")
            return False

    @staticmethod
    def _closed_slice(df: Optional[pd.DataFrame], bar_ts_ms: int) -> pd.DataFrame:
        if df is None or df.empty or "timestamp" not in df.columns:
            return pd.DataFrame()
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        sub = df[ts <= int(bar_ts_ms) - SWING_4H_BAR_MS]
        if not sub.empty and ("atr_14" not in sub.columns or "sma_200" not in sub.columns):
            sub = compute_swing_4h_indicators(sub)
        return sub

    def extract_features(
        self, symbol: str, side: str, df_4h_map: Dict[str, pd.DataFrame], context: Dict[str, Any], bar_ts_ms: int,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        sgn = 1.0 if str(side).upper() in ("BUY", "LONG") else -1.0
        ctx = self._ctx_at(symbol, int(bar_ts_ms))
        if context:
            ctx.update(context)
        closed_map = {k: c for k, d in (df_4h_map or {}).items() if not (c := self._closed_slice(d, bar_ts_ms)).empty}
        sym_df = closed_map.get(symbol, pd.DataFrame())
        if not sym_df.empty:
            row = sym_df.iloc[-1]
            close, atr, sma200 = _finite(row.get("close"), 0.0), _finite(row.get("atr_14"), 0.0), _finite(row.get("sma_200"), 0.0)
            dist200 = _finite(row.get("dist_sma_200"), ((close - sma200) / sma200) if sma200 > 0 else 0.0)
            atr_pct = (atr / close) if (close > 0.0 and atr > 0.0) else _finite(ctx.get("atr_pct"), 0.02)
            rets = pd.to_numeric(sym_df["close"], errors="coerce").dropna().pct_change().dropna()
            ewma_std = float(rets.ewm(span=42, adjust=False).std().iloc[-1]) if len(rets) >= 2 else 0.0
            ewma_vol_ann = _finite(ewma_std * _ANN_4H_SQRT, self.target_ann_vol)
            if ewma_vol_ann <= 0.0:
                ewma_vol_ann = _finite(atr_pct * _ANN_4H_SQRT, self.target_ann_vol)
        else:
            atr_pct = _finite(ctx.get("atr_pct"), 0.02)
            dist200 = _finite(ctx.get("dist_sma_200", ctx.get("abs_dist200")), 0.0)
            ewma_vol_ann = _finite(ctx.get("ewma_vol_ann"), self.target_ann_vol)

        btc_df = closed_map.get("BTCUSDT", sym_df)
        if not btc_df.empty and len(btc_df) >= 2:
            bc = pd.to_numeric(btc_df["close"], errors="coerce").dropna()
            c_now, c_ref = float(bc.iloc[-1]), float(bc.iloc[-43 if len(bc) >= 43 else 0])
            btc_ret_42 = ((c_now / c_ref) - 1.0) if c_ref > 0.0 else 0.0
        else:
            btc_ret_42 = _finite(ctx.get("btc_ret_168h"), 0.0)

        above_cnt, valid_cnt, rs_scores = 0, 0, {}
        for s_key, s_df in closed_map.items():
            sl = s_df.iloc[-1]
            sc, ssma = _finite(sl.get("close"), 0.0), _finite(sl.get("sma_200"), 0.0)
            sdist = _finite(sl.get("dist_sma_200"), ((sc - ssma) / ssma) if ssma > 0 else float("nan"))
            if sc > 0.0 and (ssma > 0.0 or math.isfinite(sdist)):
                valid_cnt += 1
                if (sc > ssma and ssma > 0.0) or sdist > 0.0:
                    above_cnt += 1
            s_closes = pd.to_numeric(s_df["close"], errors="coerce").dropna()
            if len(s_closes) >= 2:
                c1, c0 = float(s_closes.iloc[-1]), float(s_closes.iloc[-85 if len(s_closes) >= 85 else 0])
                if c0 > 0.0:
                    rs_scores[s_key] = sgn * ((c1 / c0) - 1.0)

        frac_above = (above_cnt / valid_cnt) if valid_cnt > 0 else 0.5
        rs_14d_mult = 1.0
        if len(rs_scores) >= 3 and symbol in rs_scores and max(rs_scores.values()) > min(rs_scores.values()):
            ordered = [k for k, _ in sorted(rs_scores.items(), key=lambda kv: (kv[1], kv[0]))]
            rk = ordered.index(symbol)
            rs_14d_mult = 0.6 if rk == 0 else (1.4 if rk == len(ordered) - 1 else 1.0)

        fng, fz = _finite(ctx.get("fear_and_greed", ctx.get("fng")), 50.0), _finite(ctx.get("funding_zscore", ctx.get("funding_z")), 0.0)
        sp5, vix = _finite(ctx.get("sp500_ret_5d", ctx.get("sp500_5d")), 0.0), _finite(ctx.get("vix_value", ctx.get("vix")), 20.0)
        stbl, dxy5 = _finite(ctx.get("stablecoin_growth_pct_30d", ctx.get("stbl30")), 0.0), _finite(ctx.get("dxy_ret_5d", ctx.get("dxy_5d")), 0.0)
        features: Dict[str, float] = {
            "atr_pct": float(atr_pct), "dir_funding_z": _finite(ctx.get("dir_funding_z"), sgn * fz),
            "dir_fng": _finite(ctx.get("dir_fng"), sgn * ((fng - 50.0) / 50.0)), "dir_sp500_5d": _finite(ctx.get("dir_sp500_5d"), sgn * sp5),
            "dir_btc_168h": _finite(ctx.get("dir_btc_168h"), sgn * btc_ret_42), "vix": float(vix),
            "abs_dist200": float(abs(dist200)), "dir_stbl30": _finite(ctx.get("dir_stbl30"), sgn * stbl),
            "dir_dxy_5d": _finite(ctx.get("dir_dxy_5d"), sgn * dxy5), "dir_breadth": _finite(ctx.get("dir_breadth"), sgn * (frac_above - 0.5)),
        }
        risk_pct = float(SWING_4H_SL_ATR_MULT * atr_pct)
        vol_target_mult = float(np.clip((self.target_ann_vol / max(ewma_vol_ann, 0.15)) / 0.55, 0.4, 2.0))
        atr_rp_mult = float(np.clip((self.target_risk_pct / max(risk_pct, 0.01)) * 3.0, 0.4, 2.0))
        return features, {"ewma_vol_ann": float(ewma_vol_ann), "risk_pct": risk_pct, "vol_target_mult": vol_target_mult, "atr_risk_parity_mult": atr_rp_mult, "rs_14d_mult": float(rs_14d_mult)}

    def fit_expanding_from_db(self, as_of_ms: int) -> bool:
        cutoff = int(as_of_ms)
        self._last_fit_attempt_ms = cutoff
        if self._last_fit_ms is not None and cutoff < self._last_fit_ms:
            self._model, self._train_n, self._base_rate, self._last_fit_ms = None, 0, 0.5, None
        if not self.cache_db_path or not os.path.exists(self.cache_db_path):
            return False
        candles_4h, funding_map, macro_map = {}, {}, {}
        try:
            with closing(_db_connect(f"file:{os.path.abspath(self.cache_db_path)}?mode=ro", uri=True, timeout=5.0)) as con:
                with con:
                    q_c = "SELECT timestamp, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe='1h' AND timestamp < ? "
                    for sym in CORE_SYMBOLS:
                        try:
                            df1 = pd.read_sql(q_c + "AND exchange='binance' ORDER BY timestamp ASC", con, params=(sym, cutoff))
                        except (sqlite3.Error, pd.errors.DatabaseError):
                            df1 = pd.DataFrame()
                        if df1.empty:
                            df1 = pd.read_sql(q_c + "ORDER BY timestamp ASC", con, params=(sym, cutoff))
                        c4 = resample_1h_to_closed_4h(df1, now_ms=cutoff)
                        if not c4.empty:
                            candles_4h[sym] = compute_swing_4h_indicators(c4)
                    try:
                        fdf = pd.read_sql("SELECT symbol, timestamp, funding_rate FROM funding_rates WHERE timestamp < ? ORDER BY timestamp ASC", con, params=(cutoff,))
                        for sym, g in fdf.groupby("symbol"):
                            gd = g.drop_duplicates(subset=["timestamp"], keep="last")
                            funding_map[str(sym)] = (gd["timestamp"].to_numpy(np.int64), gd["funding_rate"].to_numpy(np.float64))
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(f"funding_rates ausente: {exc}")
                    try:
                        mdf = pd.read_sql("SELECT series_name, timestamp_ms, value FROM macro_daily_history WHERE timestamp_ms < ? ORDER BY timestamp_ms ASC", con, params=(cutoff,))
                        for sname, g in mdf.groupby("series_name"):
                            macro_map[str(sname)] = (g["timestamp_ms"].to_numpy(np.int64), g["value"].to_numpy(np.float64))
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(f"macro_daily_history ausente: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Falha ao ler cache_db em fit_expanding_from_db: {exc}")
            return False
        if not candles_4h:
            return False
        self._funding_map, self._macro_map = funding_map, macro_map

        samples: List[Tuple[int, List[float], float]] = []
        for sym, df in candles_4h.items():
            ts_a, o_a, h_a, l_a, c_a = df["timestamp"].to_numpy(np.int64), df["open"].to_numpy(float), df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
            atr_a, sma_a, h40_a, l40_a = df["atr_14"].to_numpy(float), df["sma_200"].to_numpy(float), df["h40"].to_numpy(float), df["l40"].to_numpy(float)
            h10_a, l10_a, dist_a = df["h10"].to_numpy(float), df["l10"].to_numpy(float), df["dist_sma_200"].to_numpy(float)
            cool_until, n_bars, ft = -1, len(df), funding_map.get(sym)
            for i in range(n_bars - 1):
                if i < cool_until or not (math.isfinite(sma_a[i]) and sma_a[i] > 0 and atr_a[i] > 0 and math.isfinite(h40_a[i]) and math.isfinite(l40_a[i])):
                    continue
                side = "BUY" if (c_a[i] > h40_a[i] and 0.0 <= dist_a[i] <= SWING_4H_MAX_DIST_SMA200) else ("SELL" if (c_a[i] < l40_a[i] and -SWING_4H_MAX_DIST_SMA200 <= dist_a[i] <= 0.0) else "")
                if not side or o_a[i + 1] <= 0.0 or not math.isfinite(o_a[i + 1]):
                    continue
                sgn, entry_px, entry_atr = (1.0 if side == "BUY" else -1.0), o_a[i + 1], atr_a[i]
                sl, trail_ref, exit_px, exit_j = entry_px - sgn * SWING_4H_SL_ATR_MULT * entry_atr, entry_px, None, -1
                for j in range(i + 1, min(n_bars, i + 1 + SWING_4H_MAX_HOLDING_BARS)):
                    if int(ts_a[j]) + SWING_4H_BAR_MS >= cutoff:
                        break
                    if (j - i) >= SWING_4H_MAX_HOLDING_BARS:
                        exit_px, exit_j = o_a[j], j
                        break
                    if (sgn > 0 and l_a[j] <= sl) or (sgn < 0 and h_a[j] >= sl):
                        exit_px, exit_j = (min(sl, o_a[j]) if sgn > 0 else max(sl, o_a[j])), j
                        break
                    trail_ref = max(trail_ref, h_a[j]) if sgn > 0 else min(trail_ref, l_a[j])
                    sl = max(sl, trail_ref - SWING_4H_TRAIL_ATR_MULT * entry_atr) if sgn > 0 else min(sl, trail_ref + SWING_4H_TRAIL_ATR_MULT * entry_atr)
                    if (sgn > 0 and math.isfinite(l10_a[j]) and c_a[j] < l10_a[j]) or (sgn < 0 and math.isfinite(h10_a[j]) and c_a[j] > h10_a[j]):
                        exit_px, exit_j = c_a[j], j
                        break
                if exit_px is not None and exit_j > i:
                    dec_ts = int(ts_a[i]) + SWING_4H_BAR_MS
                    f_dict, _ = self.extract_features(sym, side, candles_4h, self._ctx_at(sym, dec_ts), dec_ts)
                    fund_sum = 0.0
                    if ft is not None and len(ft[0]):
                        i0 = int(np.searchsorted(ft[0], int(ts_a[i + 1]), side="left"))
                        i1 = int(np.searchsorted(ft[0], int(ts_a[exit_j]) + SWING_4H_BAR_MS, side="right"))
                        if i1 > i0:
                            fund_sum = float(np.sum(ft[1][i0:i1]))
                    net_ret = sgn * ((exit_px - entry_px) / entry_px) - _RT_COST - sgn * fund_sum
                    samples.append((dec_ts, [float(f_dict[c]) for c in META_FEATURE_COLS], net_ret))
                    cool_until = exit_j + SWING_4H_COOLDOWN_BARS

        if len(samples) < self.min_train_samples:
            return False
        samples.sort(key=lambda x: x[0])
        X = np.nan_to_num(np.asarray([s[1] for s in samples], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        rets_arr = np.asarray([s[2] for s in samples], dtype=np.float64)
        y = (rets_arr > 0.0).astype(int)
        if len(np.unique(y)) < 2:
            y = (rets_arr >= float(np.median(rets_arr))).astype(int)
        if len(np.unique(y)) < 2:
            return False
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
        pipe.fit(X, y)
        self._model, self._train_n, self._base_rate, self._last_fit_ms = pipe, len(samples), float(np.mean(y)), cutoff
        return True

    def score_entry(self, symbol: str, side: str, df_4h_map: Dict[str, pd.DataFrame], context: Dict[str, Any], bar_ts_ms: int, mode_override: Optional[str] = None) -> Dict[str, Any]:
        ts_ms = int(bar_ts_ms)
        retry_ready = self._last_fit_attempt_ms is None or (ts_ms - self._last_fit_attempt_ms) >= SWING_4H_BAR_MS
        need_fit = (self._model is None and retry_ready) or (
            self._last_fit_ms is not None and (ts_ms < self._last_fit_ms or ((ts_ms - self._last_fit_ms) >= self.refit_interval_ms and retry_ready))
        )
        if need_fit:
            self.fit_expanding_from_db(ts_ms)
        features, diag = self.extract_features(symbol, side, df_4h_map, context, ts_ms)
        if self._model is not None:
            x_vec = np.nan_to_num(np.array([[features[c] for c in META_FEATURE_COLS]], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            meta_prob: Optional[float] = float(self._model.predict_proba(x_vec)[0, 1])
            meta_size_mult = float(np.clip(1.0 + (meta_prob - self._base_rate) * 3.0, 0.5, 1.5))
        else:
            meta_prob, meta_size_mult = None, 1.0
        vol_target_mult = float(diag["vol_target_mult"])
        combined_shadow_mult = float(np.clip(vol_target_mult * meta_size_mult, 0.4, 2.0))
        eff_mode = mode_override if mode_override in VALID_MODES else self.mode
        return {
            "mode": eff_mode, "meta_prob": meta_prob, "meta_size_mult": meta_size_mult, "vol_target_mult": vol_target_mult,
            "atr_risk_parity_mult": float(diag["atr_risk_parity_mult"]), "rs_14d_mult": float(diag["rs_14d_mult"]),
            "combined_shadow_mult": combined_shadow_mult, "applied_mult": combined_shadow_mult if eff_mode == "active" else 1.0,
            "train_n": int(self._train_n), "features": features,
        }

    def record_telemetry(self, timestamp: int, symbol: str, side: str, score_result: Dict[str, Any], position_id: Optional[str] = None) -> None:
        if not self._ensure_telemetry_table():
            return
        try:
            mp = score_result.get("meta_prob")
            vals = (
                int(timestamp), str(symbol), str(side), str(position_id) if position_id is not None else None,
                str(score_result.get("mode", self.mode)), float(mp) if mp is not None else None,
                float(score_result.get("meta_size_mult", 1.0)), float(score_result.get("vol_target_mult", 1.0)),
                float(score_result.get("atr_risk_parity_mult", 1.0)), float(score_result.get("rs_14d_mult", 1.0)),
                float(score_result.get("combined_shadow_mult", 1.0)), float(score_result.get("applied_mult", 1.0)),
                json.dumps(score_result.get("features") or {}, sort_keys=True),
            )
            with closing(_db_connect(self.cache_db_path, timeout=5.0)) as con:
                with con:
                    con.execute(
                        "INSERT INTO swing_4h_meta_telemetry (timestamp, symbol, side, position_id, mode, meta_prob, "
                        "meta_size_mult, vol_target_mult, atr_risk_parity_mult, rs_14d_mult, combined_shadow_mult, "
                        "applied_mult, features_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        vals,
                    )
        except (sqlite3.Error, OSError) as exc:
            logger.warning(f"Falha ao gravar swing_4h_meta_telemetry para {symbol}: {exc}")
