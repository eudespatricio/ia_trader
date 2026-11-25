#!/usr/bin/env python3
"""Zero-leakage OOS ablation SCALP 15m (S0..S7) vs DIRECTIONAL 1h  ->  datasets/reports/s6_scalp_oos_ablation.json

Protocol: modelos 15m/1h treinados EM MEMORIA com purge/embargo antes de SPLIT_TS (nenhum .joblib de disco);
comite de 6 agentes (TradingCore.evaluate_market_deterministic, learning_tracker=None) avaliado apenas nas barras-gatilho;
DBs SQLite somente leitura; <= 5 workers de comite para nao sufocar o daemon de trading.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import sqlite3
import sys
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from agents.technical_agent import TechnicalAgent, _DefaultMLEnsemble, _disable_estimator_parallelism  # noqa: E402
from core import TradingCore  # noqa: E402
from data.feature_engineering import FeatureEngineer  # noqa: E402
from data.live_context_enricher import LiveContextEnricher  # noqa: E402
from model_trainer import ModelTrainer  # noqa: E402
from scripts.train_15m_scalp_models import select_stationary_features  # noqa: E402
from smart_trade import STRATEGY_PROFILES  # noqa: E402
from statistical_rigor import (bootstrap_performance_metrics, calculate_annualized_sharpe,  # noqa: E402
                               deflated_sharpe_ratio, get_prior_trials_count)

CACHE_DB = os.path.join(ROOT, "datasets", "market_cache.db")
MICRO_DB = os.path.join(ROOT, "datasets", "microstructure.db")
OUT_JSON = os.path.join(ROOT, "datasets", "reports", "s6_scalp_oos_ablation.json")
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SYM_ORDER = {"ETHUSDT": 0, "SOLUSDT": 1, "BTCUSDT": 2}   # ordem de avaliacao do daemon (1 entrada/barra)
B15, B1H, DAY = 900_000, 3_600_000, 86_400_000
SPLIT_TS = int(datetime(2026, 7, 29, tzinfo=timezone.utc).timestamp() * 1000)
DIAG_TS = int(datetime(2026, 11, 20, 13, tzinfo=timezone.utc).timestamp() * 1000)
NJ, WORKERS = 6, 5
TAKER_FEE, MAKER_FEE, TAKER_SLIP, MAKER_SLIP = 0.0006, 0.0002, 0.0002, 0.0001
CONF_MIN, N_VARIANTS = 0.70, 8
NATIVE_NAN = {"dist_resistance_pct", "dist_support_pct"}
SCALP, DIRP = STRATEGY_PROFILES["SCALP"], STRATEGY_PROFILES["DIRECTIONAL"]
G: Dict[str, Any] = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ dados / features
def ro_connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True, timeout=5.0)
    con.execute("PRAGMA query_only=ON")
    return con


def load_candles(con: sqlite3.Connection, sym: str, tf: str) -> pd.DataFrame:
    rows = con.execute(
        "SELECT timestamp, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe=? "
        "ORDER BY timestamp ASC, CASE WHEN exchange='binance' THEN 0 ELSE 1 END ASC", (sym, tf)).fetchall()
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp").reset_index(drop=True)
    return df.iloc[:-1].reset_index(drop=True)   # descarta a ultima vela possivelmente em formacao


class SD:
    """Dados preparados de um simbolo/timeframe (apenas barras com sma_200 valida)."""


def wilder_adx(raw: pd.DataFrame, n: int = 14) -> np.ndarray:
    """ADX(14) de Wilder, causal (FeatureEngineer nao expoe adx_14); usado apenas como feature do meta-labeler S7."""
    h, l, c = (raw[k].astype(float).to_numpy() for k in ("high", "low", "close"))
    pc = np.r_[c[0], c[:-1]]
    up, dn = np.diff(h, prepend=h[0]), -np.diff(l, prepend=l[0])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    sm = lambda a: pd.Series(a).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()  # noqa: E731
    pdi = 100 * sm(np.where((up > dn) & (up > 0), up, 0.0)) / np.maximum(sm(tr), 1e-12)
    mdi = 100 * sm(np.where((dn > up) & (dn > 0), dn, 0.0)) / np.maximum(sm(tr), 1e-12)
    return sm(100 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-12))


def build_sd(sym: str, raw: pd.DataFrame) -> SD:
    dff = FeatureEngineer.compute_all_features(raw, drop_na=False)
    kc_mid = dff["close"].astype(float).ewm(span=20, adjust=False).mean().to_numpy()
    sma = dff["sma_200"].astype(float).to_numpy()
    mask = np.isfinite(sma) & (sma > 0)
    s = SD()
    s.sym, s.df, s.kc_mid = sym, dff[mask].reset_index(drop=True), kc_mid[mask]
    s.adx = wilder_adx(raw)[mask]
    f = s.df
    s.ts = f["timestamp"].to_numpy(np.int64)
    for k in ("open", "high", "low", "close"):
        setattr(s, k[0] if k != "close" else "c", f[k].to_numpy(float))
    s.atr = f["atr_14"].astype(float).to_numpy()
    s.n = len(f)
    return s


def h1_dist_aligned(sd15: SD, raw1h: pd.DataFrame) -> np.ndarray:
    """(close-SMA200)/SMA200 da ultima vela 1h FECHADA em ts+15m (close_ts_1h = ts_1h+3.6M <= ts+900k)."""
    c = raw1h["close"].astype(float)
    dist = ((c - c.rolling(200, min_periods=200).mean()) / c.rolling(200, min_periods=200).mean()).to_numpy()
    idx = np.searchsorted(raw1h["timestamp"].to_numpy(np.int64) + B1H, sd15.ts + B15, side="right") - 1
    return np.where(idx >= 0, dist[np.maximum(idx, 0)], np.nan)


def fit_ensemble(sds: Dict[str, SD], cols: List[str], cutoff: int, tp: float, sl: float, native_nan: bool):
    prep = ModelTrainer(model_type="lgbm", horizon=5, multiple_tp=tp, multiple_sl=sl, n_jobs=NJ)
    xs, ys, tss = [], [], []
    for s in SYMS:
        sub = sds[s].df[sds[s].df["timestamp"] < cutoff].reset_index(drop=True)
        X, y = prep.prepare_dataset(sub, feature_columns=cols)
        xs.append(X.replace([np.inf, -np.inf], np.nan)); ys.append(y)
        tss.append(sub.loc[X.index, "timestamp"].to_numpy(np.int64))
    order = np.argsort(np.concatenate(tss), kind="mergesort")
    X = pd.concat(xs, ignore_index=True).iloc[order].reset_index(drop=True)
    y = pd.concat(ys, ignore_index=True).iloc[order].reset_index(drop=True)
    models = []
    for mt in ("lgbm", "rf"):
        Xf = X.copy()
        keep = [c for c in cols if c not in NATIVE_NAN] if (mt == "lgbm" and native_nan) else cols
        Xf[keep] = Xf[keep].fillna(0.0)
        tr = ModelTrainer(model_type=mt, horizon=5, multiple_tp=tp, multiple_sl=sl, n_jobs=NJ)
        tr.device = "cpu"
        est = tr._build_estimator(random_state=42)
        est.fit(Xf, y)
        models.append(_disable_estimator_parallelism(est))
    return models, int(len(X))


# ------------------------------------------------------------------ comite (paralelo, fork)
class CachedEnricher(LiveContextEnricher):
    """Mesmo enriquecimento causal; apenas memoiza a query diaria de macro_daily_history (params = dia UTC)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._macro_cache: Dict[Any, Any] = {}

    def _connect_ro(self, path):  # noqa: ANN001 - sempre read-only + query_only
        con = super()._connect_ro(path)
        if con is not None:
            con.execute("PRAGMA query_only=ON")
        return con

    def _fetch_all(self, path, sql, params=()):  # noqa: ANN001
        if "macro_daily_history" in sql:
            key = (path, params)
            if key not in self._macro_cache:
                self._macro_cache[key] = super()._fetch_all(path, sql, params)
            return self._macro_cache[key]
        return super()._fetch_all(path, sql, params)


def committee_task(task: Tuple[str, str, int]) -> Tuple[str, int, Optional[Tuple[str, float, float, bool]]]:
    sym, tf, i = task
    sd, bar = (G["sd15"][sym], B15) if tf == "15m" else (G["sd1h"][sym], B1H)
    try:
        sub = sd.df.iloc[max(0, i - 399): i + 1]
        ctx = G["enr"].enrich(sym, sd.df.iloc[i], sub, int(sd.ts[i]) + bar)
        sig = G["core"].evaluate_market_deterministic(
            context=ctx, symbol=sym, current_price=float(sd.c[i]), atr=float(sd.atr[i]), timestamp=int(sd.ts[i]))
        return (f"{sym}|{tf}", i, (sig.direction, float(sig.confidence), float(sig.consensus_score), bool(sig.is_vetoed)))
    except Exception as exc:  # noqa: BLE001 - nunca silencioso: devolve erro explicito contado no metadata
        return (f"{sym}|{tf}", i, ("ERR:" + type(exc).__name__ + ":" + str(exc)[:80], 0.0, 0.0, True))


def run_committee(tasks: List[Tuple[str, str, int]]) -> Dict[Tuple[str, str, int], Tuple[str, float, float, bool]]:
    out: Dict[Tuple[str, str, int], Any] = {}
    if not tasks:
        return out
    ctx = mp.get_context("fork")
    t0 = time.time()
    with ctx.Pool(WORKERS) as pool:
        for k, (key, i, res) in enumerate(pool.imap_unordered(committee_task, tasks, chunksize=32), 1):
            sym, tf = key.split("|")
            out[(sym, tf, i)] = res
            if k % 2000 == 0:
                log(f"  comite {k}/{len(tasks)} ({time.time() - t0:.0f}s)")
    return out


# ------------------------------------------------------------------ simulacao de trades
def scalp_outcome(sd: SD, i: int, d: int) -> Optional[Dict[str, Any]]:
    n = sd.n
    if n - 1 - i < 2 or not (sd.atr[i] > 0) or not (sd.c[i] > 0):
        return None
    if np.any(np.diff(sd.ts[i: min(i + 31, n)]) > 1_800_000):   # gap guard 30m
        return None
    sgn, atr = float(d), float(sd.atr[i])
    entry = sd.c[i] * (1 + sgn * TAKER_SLIP)
    sl_px, tp_px = entry - sgn * SCALP.sl_atr_mult * atr, entry + sgn * SCALP.tp3_atr_mult * atr
    last = min(i + SCALP.max_holding_bars, n - 1)
    exit_idx, reason, exit_px, fee = last, "TIMEOUT", sd.c[last] * (1 - sgn * TAKER_SLIP), TAKER_FEE
    for j in range(i + 1, last + 1):
        sl_hit = sd.l[j] <= sl_px if sgn > 0 else sd.h[j] >= sl_px
        tp_hit = sd.h[j] >= tp_px if sgn > 0 else sd.l[j] <= tp_px
        if sl_hit:   # tie-break conservador: SL antes de TP
            exit_idx, reason, exit_px, fee = j, "SL", sl_px * (1 - sgn * TAKER_SLIP), TAKER_FEE
            break
        if tp_hit:
            exit_idx, reason, exit_px, fee = j, "TP3", tp_px * (1 - sgn * MAKER_SLIP), MAKER_FEE
            break
    net = sgn * (exit_px - entry) / entry - TAKER_FEE - fee
    return {"exit_idx": exit_idx, "exit_ts": int(sd.ts[exit_idx]), "r": float(net * entry / (SCALP.sl_atr_mult * atr)), "reason": reason}


def directional_outcome(sd: SD, i: int, d: int) -> Optional[Dict[str, Any]]:
    n = sd.n
    if n - 1 - i < 1 or not (sd.atr[i] > 0) or not (sd.c[i] > 0):
        return None
    sgn, atr = float(d), float(sd.atr[i])
    entry = sd.c[i] * (1 + sgn * TAKER_SLIP)
    stop = entry - sgn * DIRP.sl_atr_mult * atr
    tp1, tp2 = entry + sgn * DIRP.tp1_atr_mult * atr, entry + sgn * DIRP.tp2_atr_mult * atr
    f1, f2 = DIRP.tp1_fraction, DIRP.tp2_fraction
    be = entry * (1 + sgn * 0.0004)
    rem, pnl, tp1_hit, tp2_hit, best = 1.0, 0.0, False, False, entry
    last = min(i + DIRP.max_holding_bars, n - 1)
    exit_idx, reason = last, "TIMEOUT"

    def leg(px: float, frac: float, fee: float, slip: float) -> float:
        return frac * (sgn * (px * (1 - sgn * slip) - entry) / entry - fee)

    done = False
    for j in range(i + 1, last + 1):
        hi, lo = sd.h[j], sd.l[j]
        if (lo <= stop) if sgn > 0 else (hi >= stop):   # stop/BE/trailing SEMPRE antes de TP na mesma barra
            pnl += leg(stop, rem, TAKER_FEE, TAKER_SLIP)
            exit_idx, reason, done = j, ("SL" if not tp1_hit else ("TRAIL" if tp2_hit else "BE")), True
            break
        if not tp1_hit and ((hi >= tp1) if sgn > 0 else (lo <= tp1)):
            pnl += leg(tp1, f1, MAKER_FEE, MAKER_SLIP); rem -= f1; tp1_hit, stop = True, be
        if tp1_hit and not tp2_hit and ((hi >= tp2) if sgn > 0 else (lo <= tp2)):
            pnl += leg(tp2, f2, MAKER_FEE, MAKER_SLIP); rem -= f2; tp2_hit = True
        if tp2_hit:   # trailing 1.2*ATR atualizado com a barra fechada; nunca afrouxa
            best = max(best, hi) if sgn > 0 else min(best, lo)
            trail = best - sgn * DIRP.trail_atr_mult * atr
            stop = max(stop, trail) if sgn > 0 else min(stop, trail)
    if not done:
        pnl += leg(sd.c[last], rem, TAKER_FEE, TAKER_SLIP)
    net = pnl - TAKER_FEE
    return {"exit_idx": exit_idx, "exit_ts": int(sd.ts[exit_idx]), "r": float(net * entry / (DIRP.sl_atr_mult * atr)), "reason": reason}


def run_scalp_portfolio(cands: List[Dict[str, Any]], dir_intervals: Optional[Dict[str, List[Tuple[int, int]]]]) -> List[Dict[str, Any]]:
    free: Dict[str, int] = {s: 0 for s in SYMS}
    streak: Dict[str, int] = {s: 0 for s in SYMS}
    trades, last_entry_ts = [], -1
    for c in sorted(cands, key=lambda x: (x["ts"], SYM_ORDER[x["sym"]])):
        if c["ts"] == last_entry_ts or c["ts"] < free[c["sym"]]:   # 1 entrada por barra / cooldown
            continue
        if dir_intervals is not None:
            t = c["ts"] + B15
            if any(o <= t < e for o, e in dir_intervals.get(c["sym"], [])):
                continue
        out = c.get("_out") or scalp_outcome(G["sd15"][c["sym"]], c["i"], c["d"])
        c["_out"] = out
        if out is None:
            continue
        streak[c["sym"]] = streak[c["sym"]] + 1 if out["reason"] == "SL" else 0
        cd = 8 if streak[c["sym"]] >= 2 else 4
        free[c["sym"]] = out["exit_ts"] + cd * B15
        last_entry_ts = c["ts"]
        trades.append({"sym": c["sym"], "ts": c["ts"], "exit_ts": out["exit_ts"], "r": out["r"], "reason": out["reason"], "c": c})
    return trades


def run_directional_portfolio(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    free: Dict[str, int] = {s: 0 for s in SYMS}
    streak: Dict[str, int] = {s: 0 for s in SYMS}
    trades: List[Dict[str, Any]] = []
    for c in sorted(cands, key=lambda x: (x["ts"], SYM_ORDER[x["sym"]])):
        dec = c["ts"] + B1H
        if dec < free[c["sym"]]:
            continue
        same = sum(1 for t in trades if t["d"] == c["d"] and t["open_t"] <= dec < t["close_t"])
        if same >= 2:   # MAX_SAME_DIR_MAJORS
            continue
        out = directional_outcome(G["sd1h"][c["sym"]], c["i"], c["d"])
        if out is None:
            continue
        streak[c["sym"]] = streak[c["sym"]] + 1 if out["reason"] == "SL" else 0
        free[c["sym"]] = out["exit_ts"] + (2 if streak[c["sym"]] >= 2 else 1) * B1H
        trades.append({"sym": c["sym"], "ts": c["ts"], "exit_ts": out["exit_ts"], "r": out["r"], "reason": out["reason"],
                       "d": c["d"], "open_t": dec, "close_t": out["exit_ts"] + B1H})
    return trades


# ------------------------------------------------------------------ metricas
def pf_of(r: np.ndarray) -> float:
    gp, gl = float(r[r > 0].sum()), float(-r[r < 0].sum())
    return gp / gl if gl > 0 else (99.0 if gp > 0 else 0.0)


def daily_series(trades: List[Dict[str, Any]], t0: int, t1: int) -> np.ndarray:
    d0, d1 = t0 // DAY, max(t0 // DAY, (t1 - 1) // DAY)
    arr = np.zeros(d1 - d0 + 1)
    for t in trades:
        arr[min(max(t["exit_ts"] // DAY - d0, 0), d1 - d0)] += t["r"]
    return arr


def metrics(trades: List[Dict[str, Any]], t0: int, t1: int, var_trials: float, n_trials: int) -> Dict[str, Any]:
    days = max((t1 - t0) / DAY, 1e-9)
    r = np.array([t["r"] for t in sorted(trades, key=lambda x: x["exit_ts"])], dtype=float)
    n = len(r)
    m: Dict[str, Any] = {"n": n, "trades_per_day": n / days}
    if n == 0:
        m.update(wr_pct=0.0, pf=0.0, pf_ex_top1=0.0, sum_r=0.0, avg_r=0.0, max_dd_r=0.0, sharpe_per_trade=0.0,
                 dsr=0.0, bootstrap_ci95_avg_r=None)
        return m
    eq = np.cumsum(r)
    sd_r = float(np.std(r, ddof=1)) if n > 1 else 0.0
    daily = daily_series(trades, t0, t1)
    dsr = deflated_sharpe_ratio(calculate_annualized_sharpe(daily), daily, n_trials, var_trials, 365.0)
    ci = None
    if n >= 10:   # periods_per_year=1 => ann_ret_ci = IC95 bootstrap iid da media de R por trade
        b = bootstrap_performance_metrics(r, n_iterations=2000, periods_per_year=1.0, seed=42)
        ci = [float(b["ann_ret_ci"][0]), float(b["ann_ret_ci"][1])]
    m.update(wr_pct=float((r > 0).mean() * 100), pf=pf_of(r), pf_ex_top1=pf_of(np.delete(r, int(np.argmax(r)))) if n > 1 else 0.0,
             sum_r=float(r.sum()), avg_r=float(r.mean()), max_dd_r=float(np.max(np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq)),
             sharpe_per_trade=float(r.mean() / sd_r) if sd_r > 1e-12 else 0.0, dsr=float(dsr["dsr"]),
             dsr_passed_95pct=bool(dsr["passed_95pct"]), bootstrap_ci95_avg_r=ci)
    return m


def verdict(oos: Dict[str, Any], is_m: Dict[str, Any], s0_tpd: float) -> str:
    ok = (oos["n"] >= 15 and oos["pf"] >= 1.25 and oos["pf_ex_top1"] >= 1.10 and oos["avg_r"] > 0
          and is_m["pf"] >= 1.00 and oos["trades_per_day"] > s0_tpd)
    return "GO" if ok else "NO-GO"


def sanitize(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ------------------------------------------------------------------ gatilhos
def trigger_dirs(s: SD, h1: np.ndarray, ml: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
    c, o, atr = s.c, s.o, s.atr
    ok = np.isfinite(atr) & (atr > 0) & (c > 0) & np.isfinite(h1)
    bb_hi, bb_lo = s.df["bb_high"].astype(float).to_numpy(), s.df["bb_low"].astype(float).to_numpy()
    kc_hi, kc_lo = s.kc_mid + 1.5 * atr, s.kc_mid - 1.5 * atr
    sqz = (bb_hi < kc_hi) & (bb_lo > kc_lo)
    sh = lambda a, k: np.r_[np.zeros(k, dtype=a.dtype), a[:-k]]  # noqa: E731 - a[i-k]
    sqz1, sqz2 = sh(sqz, 1), sh(sqz, 1) & sh(sqz, 2)
    vz = s.df["volume_zscore_20"].astype(float).to_numpy()
    natr = atr / np.where(c > 0, c, np.nan) * 10_000.0
    buy = (c > o) & (c > kc_hi) & (h1 >= 0)
    sell = (c < o) & (c < kc_lo) & (h1 <= 0)
    brk = np.where(buy, 1, np.where(sell, -1, 0))
    hi_s, lo_s = pd.Series(s.h), pd.Series(s.l)
    ph, pl = hi_s.rolling(20).max().shift(1).to_numpy(), lo_s.rolling(20).min().shift(1).to_numpy()
    rng = s.h - s.l
    cpos = np.where(rng > 0, (c - s.l) / np.where(rng > 0, rng, 1), 0.5)
    mom_ok = (vz >= 0.8) & (rng / np.where(atr > 0, atr, np.nan) >= 1.0)
    mom = np.where(mom_ok & (c > o) & (cpos >= 0.70) & (c > ph) & (h1 >= 0), 1,
                   np.where(mom_ok & (c < o) & (cpos <= 0.30) & (c < pl) & (h1 <= 0), -1, 0))
    idx = np.arange(s.n)
    base = ok & (idx >= 3)
    sq = lambda m, v, nt: np.where(base & m & (vz >= v) & (natr >= nt), brk, 0)  # noqa: E731
    p_buy, p_sell = ml
    raw = {"S1": sq(sqz2, 1.0, 35.0), "S4": sq(sqz1, 1.0, 35.0), "S5": sq(sqz2, 0.5, 25.0), "S6": np.where(base, mom, 0)}
    p_dir = lambda d: np.where(d > 0, p_buy, p_sell)  # noqa: E731
    p_opp = lambda d: np.where(d > 0, p_sell, p_buy)  # noqa: E731
    s1 = raw["S1"]
    return {"raw": raw, "natr": natr, "vz": vz,
            "ml": {"S0": (p_dir(s1) >= 0.45) & (p_dir(s1) > p_opp(s1)), "S2": p_dir(s1) > p_opp(s1),
                   "S3": (p_dir(s1) >= 0.38) & (p_dir(s1) > p_opp(s1))}}


VARIANT_BASE = {"S0_PROD": ("S1", "S0"), "S1_NO_ML15": ("S1", None), "S2_ML15_DIR_ONLY": ("S1", "S2"),
                "S3_ML15_THR38": ("S1", "S3"), "S4_SQZ1B_NO_ML15": ("S4", None),
                "S5_SQZ2B_RELAX_VOL_NATR": ("S5", None), "S6_MOM20_H1_NO_ML15": ("S6", None)}


def cand_features(c: Dict[str, Any]) -> List[float]:
    s, i = G["sd15"][c["sym"]], c["i"]
    f = s.df
    return [c["natr"], float(f["volume_zscore_20"].iloc[i]), float(f["rsi_14"].iloc[i]), float(s.adx[i]),
            abs(float(G["h1"][c["sym"]][i])), c["conf"], abs(c["score"]), c["p_dir"]]


def main() -> int:
    t_start = time.time()
    con = ro_connect(CACHE_DB)
    raw15 = {s: load_candles(con, s, "15m") for s in SYMS}
    raw1h = {s: load_candles(con, s, "1h") for s in SYMS}
    con.close()
    sd15 = {s: build_sd(s, raw15[s]) for s in SYMS}
    sd1h = {s: build_sd(s, raw1h[s]) for s in SYMS}
    G.update(sd15=sd15, sd1h=sd1h, h1={s: h1_dist_aligned(sd15[s], raw1h[s]) for s in SYMS})
    end_ts = max(int(sd15[s].ts[-1]) for s in SYMS) + B15
    is_start = min(int(sd15[s].ts[0]) for s in SYMS)
    log(f"15m bars: { {s: sd15[s].n for s in SYMS} } | 1h bars: { {s: sd1h[s].n for s in SYMS} } | IS_START={is_start} END={end_ts}")

    cols15 = sorted(set.intersection(*[set(select_stationary_features(sd15[s].df)) for s in SYMS]))
    cols1h = sorted(set.intersection(*[set(select_stationary_features(sd1h[s].df)) for s in SYMS]))
    log("treinando ensemble 15m purgado (cutoff = SPLIT - 10 barras)...")
    m15, n15 = fit_ensemble(sd15, cols15, SPLIT_TS - 10 * B15, 1.2, 1.2, native_nan=False)
    log(f"15m train_n={n15}; treinando ensemble 1h purgado (embargo 7d)...")
    m1h, n1h = fit_ensemble(sd1h, cols1h, SPLIT_TS - 7 * DAY, 1.5, 1.0, native_nan=True)
    log(f"1h train_n={n1h}")
    ens_1h = _DefaultMLEnsemble(models=m1h, feature_names=cols1h)
    core = TradingCore(min_consensus_threshold=0.30, technical_agent=TechnicalAgent(model=ens_1h))
    core.learning_tracker = None
    G.update(core=core, enr=CachedEnricher(CACHE_DB, MICRO_DB))

    ml_probs, trig = {}, {}
    for s in SYMS:
        X = sd15[s].df.reindex(columns=cols15).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        pr = [np.asarray(m.predict_proba(X), dtype=float) for m in m15]
        ml_probs[s] = (np.mean([p[:, 1] for p in pr], axis=0), np.mean([p[:, 2] for p in pr], axis=0))
        trig[s] = trigger_dirs(sd15[s], G["h1"][s], ml_probs[s])

    # --- tarefas de comite: gatilhos 15m (uniao S4|S5|S6; S1/S0/S2/S3 sao subconjuntos) + todas as barras 1h do periodo
    tasks: List[Tuple[str, str, int]] = []
    for s in SYMS:
        need = (trig[s]["raw"]["S4"] != 0) | (trig[s]["raw"]["S5"] != 0) | (trig[s]["raw"]["S6"] != 0)
        tasks += [(s, "15m", int(i)) for i in np.nonzero(need)[0]]
    n_scalp_tasks = len(tasks)
    for s in SYMS:
        sd = sd1h[s]
        ok = (sd.ts >= is_start) & (sd.ts + B1H <= end_ts) & np.isfinite(sd.atr) & (sd.atr > 0)
        tasks += [(s, "1h", int(i)) for i in np.nonzero(ok)[0]]
    if os.environ.get("SMOKE"):
        tasks = tasks[:60] + tasks[n_scalp_tasks:n_scalp_tasks + 60]
    log(f"tarefas de comite: scalp15m={n_scalp_tasks} dir1h={len(tasks) - n_scalp_tasks}")
    cache_p = os.environ.get("S6_COMM_CACHE")   # apenas dev: memoiza avaliacoes do comite (mesmos modelos deterministicos)
    if cache_p and os.path.exists(cache_p):
        import pickle
        comm = pickle.load(open(cache_p, "rb"))
    else:
        comm = run_committee(tasks)
        if cache_p:
            import pickle
            pickle.dump(comm, open(cache_p, "wb"))
    n_err = sum(1 for v in comm.values() if v[0].startswith("ERR"))
    log(f"comite concluido ({time.time() - t_start:.0f}s total) erros={n_err}")
    if n_err:
        errs = sorted({v[0] for v in comm.values() if v[0].startswith("ERR")})[:3]
        raise RuntimeError(f"{n_err} avaliacoes de comite com erro: {errs}")

    # --- DIRECTIONAL 1h
    dcands = []
    for (s, tf, i), (dr, conf, sc, veto) in comm.items():
        if tf == "1h" and dr in ("BUY", "SELL") and not veto and conf >= CONF_MIN:
            dcands.append({"sym": s, "i": i, "ts": int(sd1h[s].ts[i]), "d": 1 if dr == "BUY" else -1})
    dtr = run_directional_portfolio(dcands)
    dir_iv: Dict[str, List[Tuple[int, int]]] = {s: [] for s in SYMS}
    for t in dtr:
        dir_iv[t["sym"]].append((t["open_t"], t["close_t"]))
    log(f"DIRECTIONAL: candidatos={len(dcands)} trades={len(dtr)}")

    # --- candidatos por variante
    def build_cands(base: str, ml_key: Optional[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
        out, fun = [], {w: {"raw_triggers": 0, "post_ml": 0, "post_committee": 0} for w in ("is", "oos", "diag")}
        for s in SYMS:
            t = trig[s]; sd = sd15[s]
            raw_d = t["raw"][base]
            mlm = t["ml"][ml_key] if ml_key else np.ones(sd.n, dtype=bool)
            for i in np.nonzero(raw_d != 0)[0]:
                d, ts = int(raw_d[i]), int(sd.ts[i])
                ws = ["is" if ts < SPLIT_TS else "oos"] + (["diag"] if ts >= DIAG_TS else [])
                for w in ws:
                    fun[w]["raw_triggers"] += 1
                    fun[w]["post_ml"] += int(mlm[i])
                if not mlm[i]:
                    continue
                dr, conf, sc, veto = comm[(s, "15m", int(i))]
                if veto or dr != ("BUY" if d > 0 else "SELL") or conf < CONF_MIN:
                    continue
                for w in ws:
                    fun[w]["post_committee"] += 1
                out.append({"sym": s, "i": int(i), "ts": ts, "d": d, "conf": conf, "score": sc, "natr": float(t["natr"][i]),
                            "p_dir": float(ml_probs[s][0][i] if d > 0 else ml_probs[s][1][i])})
        return out, fun

    cands: Dict[str, List[Dict[str, Any]]] = {}
    funnels: Dict[str, Any] = {}
    for name, (base, mlk) in VARIANT_BASE.items():
        cands[name], funnels[name] = build_cands(base, mlk)
    sim: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for name in VARIANT_BASE:
        sim[name] = {"standalone": run_scalp_portfolio(cands[name], None), "dirlock": run_scalp_portfolio(cands[name], dir_iv)}

    # --- S7 meta-labeler (treino estrito em IS: trades S1 com saida < SPLIT)
    def meta_variant(src: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        tr = [t for t in sim[src]["standalone"] if t["exit_ts"] < SPLIT_TS]
        info: Dict[str, Any] = {"source": src, "train_n": len(tr)}
        if len(tr) < 10 or len({t["r"] > 0 for t in tr}) < 2:
            info["status"] = "INSUFFICIENT_IS_TRADES_OR_SINGLE_CLASS"
            return [], info
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        Xtr = np.array([cand_features(t["c"]) for t in tr])
        ytr = np.array([1 if t["r"] > 0 else 0 for t in tr])
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000))
        pipe.fit(Xtr, ytr)
        allc = cands[src]
        p = pipe.predict_proba(np.array([cand_features(c) for c in allc]))[:, 1] if allc else np.array([])
        info.update(status="OK", train_win_rate=float(ytr.mean()), n_candidates=len(allc), n_pass=int((p >= 0.5).sum()))
        return [c for c, pw in zip(allc, p) if pw >= 0.5], info

    meta_info: Dict[str, Any] = {}
    s7_src = [("S7_META_LABEL_SQZ", "S1_NO_ML15")]
    if len([t for t in sim["S1_NO_ML15"]["standalone"] if t["exit_ts"] < SPLIT_TS]) < 40:
        s7_src.append(("S7b_META_LABEL_SQZ1B", "S4_SQZ1B_NO_ML15"))
    for name, src in s7_src:
        cands[name], meta_info[name] = meta_variant(src)
        funnels[name] = funnels[src]
        sim[name] = {"standalone": run_scalp_portfolio(cands[name], None), "dirlock": run_scalp_portfolio(cands[name], dir_iv)}

    # --- metricas (variancia entre trials = var dos Sharpes anualizados diarios OOS das 8 variantes)
    prior = get_prior_trials_count()
    n_trials = prior + N_VARIANTS
    oos = lambda tr: [t for t in tr if t["ts"] >= SPLIT_TS]  # noqa: E731
    sr_list = [calculate_annualized_sharpe(daily_series(oos(sim[n]["standalone"]), SPLIT_TS, end_ts), 365.0)
               for n in list(VARIANT_BASE) + ["S7_META_LABEL_SQZ"]]
    var_trials = max(float(np.var(sr_list)), 1e-4)
    M = lambda tr, a, b: metrics([t for t in tr if a <= t["ts"] < b], a, b, var_trials, n_trials)  # noqa: E731
    dir_out = {"is": M(dtr, is_start, SPLIT_TS), "oos": M(dtr, SPLIT_TS, end_ts)}
    variants: Dict[str, Any] = {}
    for name in sim:
        st, dl = sim[name]["standalone"], sim[name]["dirlock"]
        variants[name] = {"is_standalone": M(st, is_start, SPLIT_TS), "oos_standalone": M(st, SPLIT_TS, end_ts),
                          "oos_with_dir_lock": M(dl, SPLIT_TS, end_ts), "diagnostic_4_5d": M(st, DIAG_TS, end_ts),
                          "funnel": funnels[name], "n_candidates_post_committee": len(cands[name])}
        if name in meta_info:
            variants[name]["meta_labeler"] = meta_info[name]
    s0_tpd = variants["S0_PROD"]["oos_standalone"]["trades_per_day"]
    s0_tpd_dl = variants["S0_PROD"]["oos_with_dir_lock"]["trades_per_day"]
    for name, v in variants.items():
        v["verdict"] = verdict(v["oos_standalone"], v["is_standalone"], s0_tpd)
        v["verdict_with_dir_lock"] = verdict(v["oos_with_dir_lock"], v["is_standalone"], s0_tpd_dl)
        v["verdict_parity"] = v["verdict"] == v["verdict_with_dir_lock"]

    meta = {"split_ts": SPLIT_TS, "split_utc": "2026-07-29T00:00:00Z", "diagnostic_ts": DIAG_TS, "diagnostic_utc": "2026-11-20T13:00:00Z",
            "is_start_ts": is_start, "end_ts": end_ts, "oos_days": (end_ts - SPLIT_TS) / DAY, "diag_days": (end_ts - DIAG_TS) / DAY,
            "symbols": SYMS, "bars_15m_valid": {s: sd15[s].n for s in SYMS}, "bars_1h_valid": {s: sd1h[s].n for s in SYMS},
            "bars_15m_oos": {s: int((sd15[s].ts >= SPLIT_TS).sum()) for s in SYMS}, "prior_trials": prior, "n_trials_dsr": n_trials,
            "dsr_variance_of_trials": var_trials, "purged_15m_train_n": n15, "purged_1h_train_n": n1h,
            "committee_evals": len(comm), "committee_errors": n_err, "directional_candidates": len(dcands),
            "runtime_s": time.time() - t_start,
            "notes": ["Modelos treinados em memoria (sem .joblib). IS e in-sample para os modelos (so OOS e limpo).",
                      "adx_14 (meta-labeler S7) = ADX Wilder causal calculado no script (FeatureEngineer nao o expoe).",
                      "Comite = TradingCore.evaluate_market_deterministic (6 agentes), learning_tracker=None, enrich causal ts+bar.",
                      "Candles: prioridade binance, bitget preenche historico < 2026-10-04 (costura de volume possivel).",
                      "Janelas por ts de entrada; trades_per_day = n / dias da janela; DSR sobre serie diaria de R (365/ano).",
                      "DIRECTIONAL 1h avaliado de IS_START (1a barra 15m valida) ate o fim; trava SCALP = DIRECTIONAL aberto no simbolo em ts+15m.",
                      "S0 e NO-GO por construcao (tpd > proprio tpd e falso)."]}
    result = sanitize({"metadata": meta, "directional_1h": dir_out, "variants": variants})
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON + ".tmp", "w") as fh:
        json.dump(result, fh, indent=2, allow_nan=False)
    os.replace(OUT_JSON + ".tmp", OUT_JSON)
    log(f"JSON escrito: {OUT_JSON} | runtime {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
