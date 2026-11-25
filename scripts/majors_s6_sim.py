#!/usr/bin/env python3
"""scripts/majors_s6_sim.py - Portfolio risk-cap simulator, zero-leakage fast parity helpers,
correlation/beta analytics, and GO-gate evaluator for S6.3 Majors Expansion (strict RO SQLite)."""
from __future__ import annotations

import math, os, sqlite3
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np, pandas as pd

from agents.technical_agent import _DefaultMLEnsemble
from data.feature_engineering import FeatureEngineer
from data.live_context_enricher import LiveContextEnricher
from smart_trade import STRATEGY_PROFILES
from statistical_rigor import bootstrap_performance_metrics, calculate_annualized_sharpe, deflated_sharpe_ratio

BASE_3: Tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
CANDIDATE_5: Tuple[str, ...] = ("XRPUSDT", "NEARUSDT", "DOGEUSDT", "LINKUSDT", "AAVEUSDT")
TIER1_8: Tuple[str, ...] = BASE_3 + CANDIDATE_5
DEFAULT_SYM_ORDER: Dict[str, int] = {
    "ETHUSDT": 0, "SOLUSDT": 1, "BTCUSDT": 2, "XRPUSDT": 3, "NEARUSDT": 4, "DOGEUSDT": 5, "LINKUSDT": 6, "AAVEUSDT": 7,
}
B15, B1H, B4H, B8H, DAY = 900_000, 3_600_000, 14_400_000, 28_800_000, 86_400_000
IS_START_1H = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
TRAIN_CUTOFF_1H = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
OOS_START_1H = int(datetime(2025, 1, 8, tzinfo=timezone.utc).timestamp() * 1000)
SPLIT_15M = int(datetime(2026, 7, 29, tzinfo=timezone.utc).timestamp() * 1000)
TRAIN_CUTOFF_15M, TRAIN_CUTOFF_1H_FOR_15M = SPLIT_15M - 10 * B15, SPLIT_15M - 7 * DAY
SWING_PARITY_OOS_MS = int(datetime(2025, 11, 20, tzinfo=timezone.utc).timestamp() * 1000)
TAKER_FEE, MAKER_FEE, TAKER_SLIP, MAKER_SLIP = 0.0006, 0.0002, 0.0002, 0.0001
BE_BUFFER_PCT, FUNDING_FALLBACK = 0.0004, 0.0001
NATIVE_NAN = frozenset({"dist_resistance_pct", "dist_support_pct"})
WORKERS, N_JOBS = 6, 6
SCALP_PROF, DIR_PROF = STRATEGY_PROFILES["SCALP"], STRATEGY_PROFILES["DIRECTIONAL"]


def ro_connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True, timeout=5.0)
    con.execute("PRAGMA query_only=ON"); con.execute("PRAGMA busy_timeout=5000")
    return con


def load_candles(con: sqlite3.Connection, sym: str, tf: str) -> pd.DataFrame:
    rows = con.execute(
        "SELECT timestamp, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe=? "
        "ORDER BY timestamp ASC, CASE WHEN exchange='binance' THEN 0 ELSE 1 END ASC", (sym, tf),
    ).fetchall()
    cols = ["timestamp", "open", "high", "low", "close", "volume"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols).drop_duplicates("timestamp", keep="first").sort_values("timestamp").reset_index(drop=True)
    return df.iloc[:-1].reset_index(drop=True) if len(df) > 1 else df


def load_funding_map(con: sqlite3.Connection, symbols: Sequence[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for sym in symbols:
        rows = con.execute("SELECT timestamp, funding_rate FROM funding_rates WHERE symbol=? AND funding_rate IS NOT NULL ORDER BY timestamp ASC", (sym,)).fetchall()
        dedup = {int(ts): float(r) for ts, r in rows if r is not None and math.isfinite(float(r))}
        ks = sorted(dedup)
        out[sym] = (np.array(ks, dtype=np.int64), np.array([dedup[k] for k in ks], dtype=float)) if ks else (np.empty(0, dtype=np.int64), np.empty(0, dtype=float))
    return out


def lookup_funding(ftuple: Optional[Tuple[np.ndarray, np.ndarray]], ts_ms: int) -> float:
    if ftuple is None or len(ftuple[0]) == 0:
        return FUNDING_FALLBACK
    ts_arr, rate_arr = ftuple
    idx = int(np.searchsorted(ts_arr, ts_ms, side="right")) - 1
    return float(rate_arr[idx]) if (idx >= 0 and (ts_ms - int(ts_arr[idx])) <= 29_400_000 and math.isfinite(float(rate_arr[idx]))) else FUNDING_FALLBACK


class SD:
    """Prepared symbol/timeframe feature container (bars with valid sma_200 > 0)."""


def build_sd(sym: str, raw: pd.DataFrame) -> SD:
    dff = FeatureEngineer.compute_all_features(raw, drop_na=False)
    kc_mid = dff["close"].astype(float).ewm(span=20, adjust=False).mean().to_numpy()
    sma = pd.to_numeric(dff["sma_200"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(sma) & (sma > 0.0)
    s = SD()
    s.sym, s.df, s.kc_mid = sym, dff.loc[mask].reset_index(drop=True), kc_mid[mask]
    f = s.df
    s.ts, s.o, s.h, s.l, s.c = f["timestamp"].to_numpy(dtype=np.int64), f["open"].to_numpy(float), f["high"].to_numpy(float), f["low"].to_numpy(float), f["close"].to_numpy(float)
    s.atr, s.n = pd.to_numeric(f["atr_14"], errors="coerce").to_numpy(dtype=float), len(f)
    return s


class FastCachedEnricher(LiveContextEnricher):
    """Read-only LiveContextEnricher preloading macro/events/funding + bounded micro RO queries (<= 1e-12 parity)."""

    def __init__(self, cache_db_path: str, micro_db_path: str):
        super().__init__(cache_db_path, micro_db_path)
        self._cache_abs = os.path.abspath(cache_db_path) if cache_db_path else ""
        self._micro_abs = os.path.abspath(micro_db_path) if micro_db_path else ""
        self._conns = {p: (ro_connect(p) if (p and os.path.exists(p)) else None) for p in (self._cache_abs, self._micro_abs)}
        self._macro_rows: List[Tuple[str, str, int, float]] = []
        self._macro_day_cache: Dict[int, List[Tuple[str, str, int, float]]] = {}
        self._event_ts, self._event_rows = np.empty(0, dtype=np.int64), []
        self._funding_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._table_info_cache: Dict[Tuple[str, str], List[Tuple[Any, ...]]] = {}
        self._micro_min_ts: Dict[str, int] = {}
        self._bs_taker_min_ts: Dict[str, int] = {}
        self._preload()

    def _connect_ro(self, path: str) -> Optional[sqlite3.Connection]:
        return ro_connect(path) if (path and os.path.exists(path)) else None

    def _preload(self) -> None:
        con = self._conns.get(self._cache_abs)
        if con is not None:
            try:
                self._macro_rows = [(str(r[0]), str(r[1]), int(r[2]), float(r[3])) for r in con.execute(
                    "SELECT series_name, date_str, timestamp_ms, value FROM macro_daily_history ORDER BY date_str ASC, timestamp_ms ASC").fetchall()
                    if r[2] is not None and r[3] is not None]
            except sqlite3.OperationalError:
                pass
            try:
                self._event_rows = [(str(r[0]), int(r[1])) for r in con.execute(
                    "SELECT event_name, timestamp_ms FROM macro_events_calendar ORDER BY timestamp_ms ASC").fetchall() if r[1] is not None]
                self._event_ts = np.array([r[1] for r in self._event_rows], dtype=np.int64)
            except sqlite3.OperationalError:
                pass
            try:
                by_s: Dict[str, List[Tuple[int, float]]] = {}
                for s, t, r in con.execute("SELECT symbol, timestamp, funding_rate FROM funding_rates WHERE funding_rate IS NOT NULL ORDER BY symbol ASC, timestamp ASC").fetchall():
                    by_s.setdefault(str(s), []).append((int(t), float(r)))
                self._funding_cache = {s: (np.array([p[0] for p in ps], dtype=np.int64), np.array([p[1] for p in ps], dtype=float)) for s, ps in by_s.items()}
            except sqlite3.OperationalError:
                pass
        mcon = self._conns.get(self._micro_abs)
        if mcon is not None:
            for tbl, col in (("book_features", "ts_ms"), ("derivs", "ts_ms"), ("tick_agg", "minute_ms"), ("binance_series", "ts_ms")):
                try:
                    row = mcon.execute(f"SELECT min({col}) FROM {tbl}").fetchone()
                    self._micro_min_ts[tbl] = int(row[0]) if (row and row[0] is not None) else (1 << 62)
                except sqlite3.OperationalError:
                    self._micro_min_ts[tbl] = 1 << 62
            try:
                self._bs_taker_min_ts = {str(s): int(t) for s, t in mcon.execute(
                    "SELECT symbol, min(ts_ms) FROM binance_series WHERE kind='taker' GROUP BY symbol").fetchall() if t is not None}
            except sqlite3.OperationalError:
                self._bs_taker_min_ts = {}

    def _fetch_all(self, path: str, sql: str, params: Tuple[Any, ...] = ()) -> List[Tuple[Any, ...]]:
        if not path:
            return []
        abs_p = os.path.abspath(path)
        if abs_p == self._cache_abs:
            if "FROM macro_daily_history WHERE timestamp_ms < ?" in sql and len(params) == 1:
                d_ms = int(params[0])
                return self._macro_day_cache.setdefault(d_ms, [r for r in self._macro_rows if r[2] < d_ms])
            if "FROM macro_events_calendar WHERE timestamp_ms BETWEEN ? AND ?" in sql and len(params) == 2:
                return self._event_rows[int(np.searchsorted(self._event_ts, int(params[0]), "left")):int(np.searchsorted(self._event_ts, int(params[1]), "right"))] if len(self._event_ts) else []
            if "FROM funding_rates WHERE symbol=? AND timestamp<=?" in sql and len(params) == 2:
                ent = self._funding_cache.get(str(params[0]))
                if ent is None:
                    return []
                ts_a, rt_a = ent; idx = int(np.searchsorted(ts_a, int(params[1]), "right"))
                return [(int(ts_a[k]), float(rt_a[k])) for k in range(idx - 1, max(0, idx - 90) - 1, -1)] if idx > 0 else []
        if sql.startswith("PRAGMA table_info("):
            key = (abs_p, sql)
            if key not in self._table_info_cache:
                con = self._conns.get(abs_p)
                self._table_info_cache[key] = con.execute(sql, params).fetchall() if con is not None else []
            return self._table_info_cache[key]
        if abs_p == self._micro_abs:
            if "FROM book_features WHERE symbol=? AND ts_ms<=?" in sql and len(params) == 2:
                sym, t = params[0], int(params[1])
                if t < self._micro_min_ts.get("book_features", 1 << 62): return []
                sql, params = sql.replace("WHERE symbol=? AND ts_ms<=?", "WHERE ts_ms>=? AND ts_ms<=? AND symbol=?"), (t - 120_000, t, sym)
            elif "FROM derivs WHERE symbol=? AND ts_ms<=?" in sql and len(params) == 2:
                sym, t = params[0], int(params[1])
                if t < self._micro_min_ts.get("derivs", 1 << 62): return []
                if "LIMIT 1" in sql:
                    sql, params = sql.replace("WHERE symbol=? AND ts_ms<=?", "WHERE ts_ms>=? AND ts_ms<=? AND symbol=?"), (t - 12 * B1H, t, sym)
            elif "FROM tick_agg WHERE symbol=? AND minute_ms + 60000 <= ? AND minute_ms + 60000 > ?" in sql and len(params) == 3:
                sym, hi, lo = params[0], int(params[1]) - 60_000, int(params[2]) - 60_000
                if hi < self._micro_min_ts.get("tick_agg", 1 << 62): return []
                sql, params = sql.replace("WHERE symbol=? AND minute_ms + 60000 <= ? AND minute_ms + 60000 > ?", "WHERE minute_ms <= ? AND minute_ms > ? AND symbol=?"), (hi, lo, sym)
            elif "FROM binance_series WHERE symbol=? AND kind='taker' AND ts_ms <= ?" in sql and len(params) == 2:
                sym, t = str(params[0]), int(params[1])
                if t < self._micro_min_ts.get("binance_series", 1 << 62) or t < self._bs_taker_min_ts.get(sym, 1 << 62): return []
        con = self._conns.get(abs_p)
        if con is None: return []
        try:
            return con.execute(sql, params).fetchall()
        except Exception as exc:
            if "no such table" not in str(exc).lower(): self._enrich_error_count += 1
            return []

    def close(self) -> None:
        for con in self._conns.values():
            if con is not None:
                try: con.close()
                except Exception: pass
        self._conns.clear()


class _PrecomputedEnsemble:
    """Vectorized per-symbol predict_proba cache serving exact rows to TechnicalAgent (<= 1e-12 parity)."""

    def __init__(self, fallback: _DefaultMLEnsemble, lookup: Dict[str, Dict[int, np.ndarray]]):
        self._fallback, self._lookup, self._active_symbol = fallback, lookup, None
        self.models_with_features, self.models, self.feature_names = fallback.models_with_features, fallback.models, fallback.feature_names
        self.classes_ = np.array([0, 1, 2])

    @classmethod
    def from_ensemble(cls, ens: _DefaultMLEnsemble, dfs_by_sym: Dict[str, pd.DataFrame]) -> "_PrecomputedEnsemble":
        lookup: Dict[str, Dict[int, np.ndarray]] = {}
        for sym, df in dfs_by_sym.items():
            if df is None or df.empty:
                lookup[sym] = {}; continue
            pr, ts = ens.predict_proba(df), df["timestamp"].to_numpy(dtype=np.int64)
            lookup[sym] = {int(ts[i]): pr[i : i + 1].copy() for i in range(len(ts))}
        return cls(ens, lookup)

    def set_active_symbol(self, sym: Optional[str]) -> None:
        self._active_symbol = sym

    def predict_proba(self, X: Any) -> np.ndarray:
        if isinstance(X, dict) and "timestamp" in X:
            sym = X.get("symbol") or self._active_symbol
            row = self._lookup.get(str(sym), {}).get(int(X["timestamp"])) if sym is not None else None
            if row is not None: return row
        return self._fallback.predict_proba(X)


def directional_outcome(sd: SD, i: int, d: int, funding_tuple: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Optional[Dict[str, Any]]:
    """Production DIRECTIONAL 1h trade ladder with stress costs + real 8h funding."""
    n = sd.n
    if n - 1 - i < 1 or not (sd.atr[i] > 0.0) or not (sd.c[i] > 0.0): return None
    sgn, atr = float(d), float(sd.atr[i])
    entry = sd.c[i] * (1.0 + sgn * TAKER_SLIP)
    stop, tp1, tp2 = entry - sgn * DIR_PROF.sl_atr_mult * atr, entry + sgn * DIR_PROF.tp1_atr_mult * atr, entry + sgn * DIR_PROF.tp2_atr_mult * atr
    f1, f2, be = DIR_PROF.tp1_fraction, DIR_PROF.tp2_fraction, entry * (1.0 + sgn * BE_BUFFER_PCT)
    rem, pnl, fpnl, tp1_h, tp2_h, best = 1.0, 0.0, 0.0, False, False, entry
    last = min(i + DIR_PROF.max_holding_bars, n - 1)
    exit_idx, reason, done = last, "TIMEOUT", False
    leg = lambda px, fr, fee, slp: fr * (sgn * (px * (1.0 - sgn * slp) - entry) / entry - fee)  # noqa: E731
    for j in range(i + 1, last + 1):
        if funding_tuple is not None and (j - i) % 8 == 0:
            fd = (-sgn) * rem * lookup_funding(funding_tuple, int(sd.ts[j]) + B1H); pnl += fd; fpnl += fd
        hi, lo = sd.h[j], sd.l[j]
        if (lo <= stop) if sgn > 0 else (hi >= stop):
            pnl += leg(stop, rem, TAKER_FEE, TAKER_SLIP); exit_idx, reason, done = j, ("SL" if not tp1_h else ("TRAIL" if tp2_h else "BE")), True; break
        if not tp1_h and ((hi >= tp1) if sgn > 0 else (lo <= tp1)):
            pnl += leg(tp1, f1, MAKER_FEE, MAKER_SLIP); rem -= f1; tp1_h, stop = True, be
        if tp1_h and not tp2_h and ((hi >= tp2) if sgn > 0 else (lo <= tp2)):
            pnl += leg(tp2, f2, MAKER_FEE, MAKER_SLIP); rem -= f2; tp2_h = True
        if tp2_h:
            best = max(best, hi) if sgn > 0 else min(best, lo)
            tr_px = best - sgn * DIR_PROF.trail_atr_mult * atr
            stop = max(stop, tr_px) if sgn > 0 else min(stop, tr_px)
    if not done: pnl += leg(sd.c[last], rem, TAKER_FEE, TAKER_SLIP)
    sl_d = DIR_PROF.sl_atr_mult * atr
    return {"exit_idx": exit_idx, "exit_ts": int(sd.ts[exit_idx]), "open_t": int(sd.ts[i]) + B1H, "close_t": int(sd.ts[exit_idx]) + B1H,
            "r": float((pnl - TAKER_FEE) * entry / sl_d), "funding_r": float(fpnl * entry / sl_d), "sl_pct": float(sl_d / entry), "reason": reason}


def scalp_outcome(sd: SD, i: int, d: int) -> Optional[Dict[str, Any]]:
    """Production SCALP 15m S0_PROD outcome (SL 1.4x / TP3 2.2x / timeout 30b, gap guard 30m)."""
    n = sd.n
    if n - 1 - i < 2 or not (sd.atr[i] > 0.0) or not (sd.c[i] > 0.0) or np.any(np.diff(sd.ts[i : min(i + 31, n)]) > 1_800_000): return None
    sgn, atr = float(d), float(sd.atr[i])
    entry, sl_d = sd.c[i] * (1.0 + sgn * TAKER_SLIP), SCALP_PROF.sl_atr_mult * atr
    sl_px, tp_px, last = entry - sgn * sl_d, entry + sgn * SCALP_PROF.tp3_atr_mult * atr, min(i + SCALP_PROF.max_holding_bars, n - 1)
    exit_idx, reason, ex_px, fee = last, "TIMEOUT", sd.c[last] * (1.0 - sgn * TAKER_SLIP), TAKER_FEE
    for j in range(i + 1, last + 1):
        if (sd.l[j] <= sl_px) if sgn > 0 else (sd.h[j] >= sl_px):
            exit_idx, reason, ex_px, fee = j, "SL", sl_px * (1.0 - sgn * TAKER_SLIP), TAKER_FEE; break
        if (sd.h[j] >= tp_px) if sgn > 0 else (sd.l[j] <= tp_px):
            exit_idx, reason, ex_px, fee = j, "TP3", tp_px * (1.0 - sgn * MAKER_SLIP), MAKER_FEE; break
    net = sgn * (ex_px - entry) / entry - TAKER_FEE - fee
    return {"exit_idx": exit_idx, "exit_ts": int(sd.ts[exit_idx]), "open_t": int(sd.ts[i]) + B15, "close_t": int(sd.ts[exit_idx]) + B15,
            "r": float(net * entry / sl_d), "sl_pct": float(sl_d / entry), "reason": reason}


def directional_cooldown_fn(cand: Dict[str, Any], out: Dict[str, Any], streak: int) -> int:
    return (2 if streak >= 2 else 1) * B1H


def scalp_cooldown_fn(cand: Dict[str, Any], out: Dict[str, Any], streak: int) -> int:
    return (8 if streak >= 2 else 4) * B15


def _dir_sign(c: Dict[str, Any]) -> int:
    raw = c.get("d", c.get("direction", 1))
    return (1 if raw.upper() in ("BUY", "LONG", "+1", "1") else -1) if isinstance(raw, str) else (1 if int(raw) > 0 else -1)


def compute_pairwise_corr_matrix(dfs_by_sym: Dict[str, pd.DataFrame], start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> Dict[str, Dict[str, Optional[float]]]:
    series = {}
    for sym, df in dfs_by_sym.items():
        if df is None or df.empty: continue
        sub = df[["timestamp", "close"]]
        if start_ms is not None: sub = sub[sub["timestamp"] >= start_ms]
        if end_ms is not None: sub = sub[sub["timestamp"] < end_ms]
        ret = sub.set_index("timestamp")["close"].astype(float).pct_change().dropna()
        if len(ret) >= 2: series[sym] = ret
    if not series: return {}
    corr = pd.DataFrame(series).corr()
    return {str(a): {str(b): (round(float(corr.loc[a, b]), 4) if math.isfinite(float(corr.loc[a, b])) else None) for b in corr.columns} for a in corr.index}


def compute_causal_beta_to_btc(dfs_by_sym: Dict[str, pd.DataFrame], cutoff_ms: int, btc_sym: str = "BTCUSDT", lookback_days: Optional[int] = 30) -> Dict[str, float]:
    if btc_sym not in dfs_by_sym or dfs_by_sym[btc_sym].empty: return {s: 1.0 for s in dfs_by_sym}
    lo_ms = (cutoff_ms - int(lookback_days) * DAY) if lookback_days is not None else None
    rets = lambda df: df[(df["timestamp"] < cutoff_ms) & ((df["timestamp"] >= lo_ms) if lo_ms is not None else True)].set_index("timestamp")["close"].astype(float).pct_change().dropna()  # noqa: E731
    btc_r, betas = rets(dfs_by_sym[btc_sym]), {}
    for sym, df in dfs_by_sym.items():
        if sym == btc_sym: betas[sym] = 1.0; continue
        al = pd.concat([rets(df), btc_r], axis=1, join="inner").dropna()
        if len(al) < 10: betas[sym] = 1.0; continue
        x, y = al.iloc[:, 0].to_numpy(float), al.iloc[:, 1].to_numpy(float)
        vb = float(np.var(y, ddof=1))
        beta = float(np.cov(x, y, ddof=1)[0, 1]) / vb if vb > 1e-12 else 1.0
        betas[sym] = round(beta, 4) if math.isfinite(beta) else 1.0
    return betas


def _pf_of(r: np.ndarray) -> float:
    gp, gl = float(r[r > 0].sum()), float(-r[r < 0].sum())
    return gp / gl if gl > 0.0 else (99.0 if gp > 0.0 else 0.0)


def compute_portfolio_metrics(
    trades: Sequence[Dict[str, Any]], t0: Optional[int] = None, t1: Optional[int] = None,
    initial_capital: float = 10_000.0, risk_pct: float = 0.01, var_trials: float = 1.0,
    n_trials: int = 1, bootstrap_iters: int = 2000,
) -> Dict[str, Any]:
    ordered = sorted(trades, key=lambda x: (int(x.get("close_t", x["exit_ts"])), int(x["ts"])))
    n = len(ordered)
    t0 = min((int(t["ts"]) for t in ordered), default=0) if t0 is None else t0
    t1 = max((int(t.get("close_t", t["exit_ts"])) for t in ordered), default=t0 + DAY) if t1 is None else t1
    days = max((t1 - t0) / DAY, 1.0)
    if n == 0:
        return {"n": 0, "trades_count": 0, "days": round(days, 2), "trades_per_day": 0.0, "wr_pct": 0.0, "pf": 0.0,
                "pf_ex_top1": 0.0, "sum_r": 0.0, "avg_r": 0.0, "max_dd_r": 0.0, "max_dd_pct": 0.0, "net_pnl_usd": 0.0,
                "net_pnl_pct": 0.0, "sharpe_net": 0.0, "calmar": 0.0, "dsr": 0.0, "dsr_passed_95pct": False, "bootstrap_ci95_avg_r": None}
    r_arr = np.array([float(t["r"]) for t in ordered], dtype=float)
    eq_r = np.cumsum(r_arr)
    max_dd_r = float(np.max(np.maximum.accumulate(np.r_[0.0, eq_r])[1:] - eq_r))
    eq, peak, max_dd_pct = float(initial_capital), float(initial_capital), 0.0
    d0, d1 = int(t0 // DAY), max(int(t0 // DAY), int(max(t1 - 1, t0) // DAY))
    daily_eq: Dict[int, float] = {}
    for t in ordered:
        eq = max(0.0, eq + eq * risk_pct * float(t["r"])); peak = max(peak, eq)
        if peak > 0.0: max_dd_pct = max(max_dd_pct, ((peak - eq) / peak) * 100.0)
        daily_eq[min(max(int(t.get("close_t", t["exit_ts"]) // DAY), d0), d1)] = eq
    eq_ser = pd.Series(daily_eq, dtype=float).reindex(np.arange(d0, d1 + 1, dtype=np.int64)).ffill().fillna(float(initial_capital)).to_numpy(float)
    prev_eq = np.insert(eq_ser[:-1], 0, float(initial_capital))
    d_rets = (eq_ser - prev_eq) / np.maximum(1.0, prev_eq)
    sharpe = float(calculate_annualized_sharpe(d_rets, 365.0))
    dsr_res = deflated_sharpe_ratio(sharpe, d_rets, max(1, int(n_trials)), max(float(var_trials), 1e-4), 365.0)
    net_usd = eq - float(initial_capital)
    ann_ret = (((eq / initial_capital) ** (365.0 / days) - 1.0) * 100.0) if eq > 0 else -100.0
    ci95 = [float(x) for x in bootstrap_performance_metrics(r_arr, n_iterations=bootstrap_iters, periods_per_year=1.0, seed=42)["ann_ret_ci"]] if (n >= 10 and bootstrap_iters > 0) else None
    return {
        "n": n, "trades_count": n, "days": round(days, 2), "trades_per_day": float(n / days), "wr_pct": float((r_arr > 0.0).mean() * 100.0),
        "pf": _pf_of(r_arr), "pf_ex_top1": _pf_of(np.delete(r_arr, int(np.argmax(r_arr)))) if n > 1 else 0.0,
        "sum_r": float(r_arr.sum()), "avg_r": float(r_arr.mean()), "max_dd_r": max_dd_r, "max_dd_pct": float(max_dd_pct),
        "net_pnl_usd": float(net_usd), "net_pnl_pct": float((net_usd / initial_capital) * 100.0) if initial_capital > 0 else 0.0,
        "sharpe_net": sharpe, "calmar": float((ann_ret / max_dd_pct) if max_dd_pct > 1e-9 else (99.0 if ann_ret > 0 else 0.0)),
        "dsr": float(dsr_res["dsr"]), "dsr_passed_95pct": bool(dsr_res["passed_95pct"]), "bootstrap_ci95_avg_r": ci95,
    }


def simulate_portfolio_with_caps(
    cands: Sequence[Dict[str, Any]], outcome_fn: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
    corr_cap: int = 2, max_net_risk_r: Optional[float] = None, beta_map: Optional[Dict[str, float]] = None,
    initial_capital: float = 10_000.0, risk_pct: float = 0.01, max_margin_util: float = 0.85,
    cooldown_fn: Optional[Callable[[Dict[str, Any], Dict[str, Any], int], int]] = None,
    sym_order: Optional[Dict[str, int]] = None, blocked_intervals: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    one_entry_per_bar: bool = False, t0: Optional[int] = None, t1: Optional[int] = None,
    var_trials: float = 1.0, n_trials: int = 1, bootstrap_iters: int = 2000,
) -> Dict[str, Any]:
    """Simulate multi-symbol portfolio enforcing corr_cap, net directional risk R cap (unit/beta), margin util & cooldown."""
    omap = sym_order if sym_order is not None else DEFAULT_SYM_ORDER
    free, streak, trades, last_ts = {}, {}, [], -1
    blocked = {"cooldown": 0, "corr_cap": 0, "net_risk_r": 0, "margin": 0, "dir_lock": 0, "bar_limit": 0}
    scands = sorted(cands, key=lambda x: (int(x["ts"]), omap.get(str(x.get("sym") or x.get("symbol") or ""), 999), str(x.get("sym") or x.get("symbol") or "")))
    for c in scands:
        sym, ts = str(c.get("sym") or c.get("symbol")), int(c["ts"])
        open_t, sgn = int(c.get("open_t", ts)), _dir_sign(c)
        if one_entry_per_bar and ts == last_ts: blocked["bar_limit"] += 1; continue
        if open_t < free.get(sym, -1): blocked["cooldown"] += 1; continue
        if blocked_intervals is not None and any(o <= open_t < e for o, e in blocked_intervals.get(sym, [])): blocked["dir_lock"] += 1; continue
        active = [t for t in trades if t["open_t"] <= open_t < t["close_t"]]
        if corr_cap is not None and sum(1 for t in active if t["d"] == sgn) >= int(corr_cap): blocked["corr_cap"] += 1; continue
        bw = float(beta_map.get(sym, 1.0)) if beta_map is not None else 1.0
        bw = bw if math.isfinite(bw) else 1.0
        if max_net_risk_r is not None and abs(sum(t["d"] * t["beta_w"] for t in active) + sgn * bw * 1.0) > float(max_net_risk_r) + 1e-12:
            blocked["net_risk_r"] += 1; continue
        out = outcome_fn(c)
        if out is None: continue
        m_frac = float(out["margin_frac"]) if out.get("margin_frac") is not None else (
            min(0.25, risk_pct / max(1e-4, float(out["sl_pct"]) * float(out.get("leverage", 2.0)))) if float(out.get("sl_pct", 0.0)) > 0.0 else min(0.25, risk_pct * 10.0))
        if sum(float(t["margin_frac"]) for t in active) + m_frac > float(max_margin_util) + 1e-12: blocked["margin"] += 1; continue
        ex_ts, cl_t, reason = int(out["exit_ts"]), int(out.get("close_t", out["exit_ts"])), str(out.get("reason", "EXIT"))
        streak[sym] = streak.get(sym, 0) + 1 if reason in ("SL", "STOP_LOSS") else 0
        free[sym], last_ts = cl_t + max(0, int(cooldown_fn(c, out, streak[sym])) if cooldown_fn is not None else 0), ts
        trades.append({"sym": sym, "ts": ts, "open_t": open_t, "exit_ts": ex_ts, "close_t": cl_t,
                       "d": sgn, "beta_w": bw, "r": float(out["r"]), "reason": reason, "margin_frac": m_frac})
    m = compute_portfolio_metrics(trades, t0=t0, t1=t1, initial_capital=initial_capital, risk_pct=risk_pct,
                                  var_trials=var_trials, n_trials=n_trials, bootstrap_iters=bootstrap_iters)
    return {**m, "trades": trades, "blocked_counts": blocked}


def evaluate_go_gate(
    candidate_oos: Dict[str, Any], baseline_oos: Dict[str, Any],
    marginal_by_sym: Dict[str, Dict[str, Any]], baseline_ml8_oos: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate S6.3 GO promotion gate & filter non-destructive marginal symbols."""
    b_sum, b_tpd, b_pf, b_dd = (float(baseline_oos.get(k, 0.0)) for k in ("sum_r", "trades_per_day", "pf", "max_dd_pct"))
    c_sum, c_tpd, c_pf, c_pf1, c_avg = (float(candidate_oos.get(k, 0.0)) for k in ("sum_r", "trades_per_day", "pf", "pf_ex_top1", "avg_r"))
    c_dd, ml8_ok = float(candidate_oos.get("max_dd_pct", 100.0)), True
    if baseline_ml8_oos is not None:
        m8_s, m8_pf, m8_dd = float(baseline_ml8_oos.get("sum_r", 0.0)), float(baseline_ml8_oos.get("pf", 0.0)), float(baseline_ml8_oos.get("max_dd_pct", 100.0))
        ml8_ok = (m8_s >= (b_sum * 0.90 if b_sum > 0.0 else b_sum)) and (m8_pf >= (b_pf * 0.95 if b_pf > 0.0 else 0.0)) and (m8_dd < 15.0) and (m8_dd <= b_dd + 2.0)
    promoted, rejected = [], []
    for sym, info in marginal_by_sym.items():
        avg_r = float(info.get("avg_r", info.get("standalone", {}).get("avg_r", -1.0)))
        d_s, d_sh, d_c = float(info.get("delta_sum_r", 0.0)), float(info.get("delta_sharpe", 0.0)), float(info.get("delta_calmar", 0.0))
        n_s = info.get("n", info.get("standalone", {}).get("n", None))
        (promoted if (((n_s is None) or (int(n_s) > 0)) and ((avg_r >= 0.0) or ((d_sh > 0.0 or d_c > 0.0) and d_s >= 0.0))) else rejected).append(sym)
    checks = {
        "n_ge_15": int(candidate_oos.get("n", 0)) >= 15,
        "sum_r_gt_baseline": c_sum > b_sum, "tpd_gt_baseline": c_tpd > b_tpd,
        "pf_ge_1_25": (c_pf >= 1.25) and (c_pf >= b_pf if b_pf < 1.25 else True), "pf_ex_top1_ge_1_10": c_pf1 >= 1.10,
        "avg_r_gt_0": c_avg > 0.0, "max_dd_lt_15": c_dd < 15.0,
        "dsr_passed_95pct": bool(candidate_oos.get("dsr_passed_95pct", float(candidate_oos.get("dsr", 0.0)) >= 0.95)),
        "ml8_non_degradation": ml8_ok, "has_promoted_symbols": (len(promoted) > 0) if marginal_by_sym else True,
    }
    passed = all(checks.values())
    return {"verdict": "GO" if passed else "NO-GO", "passed": passed, "checks": checks,
            "promoted_symbols": promoted, "rejected_symbols": rejected, "reasons": [k for k, ok in checks.items() if not ok]}


def sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict): return {str(k): sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [sanitize_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)): return float(obj) if math.isfinite(float(obj)) else None
    return int(obj) if isinstance(obj, np.integer) else (bool(obj) if isinstance(obj, np.bool_) else obj)
