#!/usr/bin/env python3
"""scripts/eval_majors_expansion_s6_oos.py - Zero-Leakage OOS Majors Expansion Evaluator (S6.3).
Strictly read-only SQLite (`?mode=ro` + `PRAGMA query_only=ON`); zero `.joblib` disk loads."""
from __future__ import annotations

import argparse, json, math, multiprocessing as mp, os, sys, time, warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

from agents.technical_agent import TechnicalAgent, _DefaultMLEnsemble, _disable_estimator_parallelism  # noqa: E402
from core import TradingCore  # noqa: E402
from data.live_context_enricher import LiveContextEnricher  # noqa: E402
from data.swing_4h_strategy import (SWING_4H_BAR_MS, SWING_4H_COOLDOWN_BARS, SWING_4H_MAX_HOLDING_BARS,  # noqa: E402
    SWING_4H_SL_ATR_MULT, SWING_4H_TRAIL_ATR_MULT, compute_swing_4h_indicators, evaluate_swing_4h_entry, resample_1h_to_closed_4h)
from leverage_policy import decide_leverage  # noqa: E402
from model_trainer import ModelTrainer  # noqa: E402
from scripts.majors_s6_sim import (  # noqa: E402
    B15, B1H, BASE_3, CANDIDATE_5, DAY, IS_START_1H, N_JOBS, NATIVE_NAN, OOS_START_1H, SPLIT_15M, SWING_PARITY_OOS_MS,
    TAKER_FEE, TAKER_SLIP, TIER1_8, TRAIN_CUTOFF_15M, TRAIN_CUTOFF_1H, TRAIN_CUTOFF_1H_FOR_15M, WORKERS,
    FastCachedEnricher, SD, _PrecomputedEnsemble, build_sd, compute_causal_beta_to_btc, compute_pairwise_corr_matrix,
    directional_cooldown_fn, directional_outcome, evaluate_go_gate, load_candles, load_funding_map, lookup_funding,
    ro_connect, sanitize_json, scalp_cooldown_fn, scalp_outcome, simulate_portfolio_with_caps)
from scripts.train_15m_scalp_models import select_stationary_features  # noqa: E402
from statistical_rigor import calculate_annualized_sharpe, deflated_sharpe_ratio, get_prior_trials_count, log_trial  # noqa: E402
CACHE_DB, MICRO_DB = os.path.join(ROOT, "datasets", "market_cache.db"), os.path.join(ROOT, "datasets", "microstructure.db")
OUT_JSON, CONF_MIN = os.path.join(ROOT, "datasets", "reports", "s6_majors_expansion_oos.json"), 0.70
RISK_GRID = ((None, "unit"), (2.0, "unit"), (3.0, "unit"), (4.0, "unit"), (2.0, "beta"), (3.0, "beta"), (4.0, "beta"))
G: Dict[str, Any] = {}
def log(msg: str) -> None: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit_purged_ensemble(sds: Dict[str, SD], syms: Sequence[str], cols: List[str], cutoff_ms: int, tp: float, sl: float, native_nan: bool, start_ms: Optional[int] = None) -> Tuple[List[Any], int]:
    """Train in-memory LGBM + RF on ts < cutoff_ms; Triple-Barrier per symbol before mergesort concat."""
    prep = ModelTrainer(model_type="lgbm", horizon=5, multiple_tp=tp, multiple_sl=sl, n_jobs=N_JOBS)
    xs, ys, tss = [], [], []
    for s in syms:
        df = sds[s].df
        sub = df.loc[(df["timestamp"] < cutoff_ms) & ((df["timestamp"] >= start_ms) if start_ms is not None else True)].reset_index(drop=True)
        if len(sub) < 30: continue
        X, y = prep.prepare_dataset(sub, feature_columns=cols)
        xs.append(X.replace([np.inf, -np.inf], np.nan)); ys.append(y); tss.append(sub.loc[X.index, "timestamp"].to_numpy(np.int64))
    if not xs: raise RuntimeError(f"Empty training set before cutoff={cutoff_ms}")
    order = np.argsort(np.concatenate(tss), kind="mergesort")
    Xa, ya = pd.concat(xs, ignore_index=True).iloc[order].reset_index(drop=True), pd.concat(ys, ignore_index=True).iloc[order].reset_index(drop=True)
    models = []
    for mt in ("lgbm", "rf"):
        Xf = Xa.copy()
        fc = [c for c in cols if c not in NATIVE_NAN] if (mt == "lgbm" and native_nan) else cols
        Xf[fc] = Xf[fc].fillna(0.0)
        tr = ModelTrainer(model_type=mt, horizon=5, multiple_tp=tp, multiple_sl=sl, n_jobs=N_JOBS); tr.device = "cpu"
        est = tr._build_estimator(random_state=42); est.fit(Xf, ya)
        models.append(_disable_estimator_parallelism(est))
    return models, int(len(Xa))


def verify_parity(cache_db: str, micro_db: str, sd1h: Dict[str, SD], ens_raw: _DefaultMLEnsemble, ens_pre: _PrecomputedEnsemble, syms: Sequence[str]) -> Dict[str, Any]:
    enr_f, enr_r, ed, pdiff, n = FastCachedEnricher(cache_db, micro_db), LiveContextEnricher(cache_db, micro_db), 0.0, 0.0, 0
    try:
        for sym in list(syms)[:5]:
            sd = sd1h[sym]
            for i in sorted({sd.n - 1} | {min(sd.n - 1, max(55, int(sd.n * q))) for q in (0.65, 0.985, 0.995, 0.999)}):
                sub = sd.df.iloc[max(0, i - 199) : i + 1]
                cf, cr = enr_f.enrich(sym, sd.df.iloc[i], sub, int(sd.ts[i]) + B1H), enr_r.enrich(sym, sd.df.iloc[i], sub, int(sd.ts[i]) + B1H)
                assert set(cf) == set(cr), f"Key mismatch {sym}@{i}"
                for k, v2 in cr.items():
                    v1 = cf[k]
                    if isinstance(v1, (int, float, np.number)) and isinstance(v2, (int, float, np.number)) and not isinstance(v1, bool):
                        f1, f2 = float(v1), float(v2); assert math.isfinite(f1) == math.isfinite(f2), f"NaN mismatch {sym}.{k}"
                        if math.isfinite(f1): ed = max(ed, abs(f1 - f2))
                    else: assert v1 == v2, f"Val mismatch {sym}.{k}: {v1} vs {v2}"
                ens_pre.set_active_symbol(sym); pdiff = max(pdiff, float(np.max(np.abs(ens_pre.predict_proba(cf) - ens_raw.predict_proba(cf))))); n += 1
    finally: enr_f.close()
    assert ed <= 1e-12 and pdiff <= 1e-12, f"Parity violation: enr={ed:.3e}, ens={pdiff:.3e}"
    return {"n_samples": n, "enricher_max_diff": ed, "ensemble_max_diff": pdiff, "passed": True}


def _comm_init() -> None: G["enr"] = FastCachedEnricher(G["cache_db"], G["micro_db"])

def _comm_worker(task: Tuple[str, str, int]) -> Tuple[str, int, Any, Any]:
    tf, sym, i = task
    sd, bar = (G["sd1h"][sym], B1H) if tf == "1h" else (G["sd15"][sym], B15)
    e3, e8, c3, c8 = (G["e1_b3"], G["e1_p8"], G["c1_b3"], G["c1_p8"]) if tf == "1h" else (G["e15_b3"], G["e15_p8"], G["c15_b3"], G["c15_p8"])
    try:
        ctx = G["enr"].enrich(sym, sd.df.iloc[i], sd.df.iloc[max(0, i - 199) : i + 1], int(sd.ts[i]) + bar)
        e3.set_active_symbol(sym); s3 = c3.evaluate_market_deterministic(ctx, sym, float(sd.c[i]), float(sd.atr[i]), int(sd.ts[i]))
        e8.set_active_symbol(sym); s8 = c8.evaluate_market_deterministic(ctx, sym, float(sd.c[i]), float(sd.atr[i]), int(sd.ts[i]))
        return sym, i, (s3.direction, float(s3.confidence), bool(s3.is_vetoed)), (s8.direction, float(s8.confidence), bool(s8.is_vetoed))
    except Exception as exc:  # noqa: BLE001
        err = (f"ERR:{type(exc).__name__}:{str(exc)[:80]}", 0.0, True); return sym, i, err, err

def run_pool(tf: str, tasks: List[Tuple[str, int]], label: str) -> Dict[str, Dict[Tuple[str, int], Tuple[str, float, bool]]]:
    res: Dict[str, Dict[Tuple[str, int], Tuple[str, float, bool]]] = {"base3": {}, "pooled8": {}}
    if not tasks: return res
    t0 = time.time()
    with mp.get_context("fork").Pool(WORKERS, initializer=_comm_init) as pool:
        for k, (sym, i, r3, r8) in enumerate(pool.imap_unordered(_comm_worker, [(tf, s, idx) for s, idx in tasks], chunksize=64), 1):
            res["base3"][(sym, i)], res["pooled8"][(sym, i)] = r3, r8
            if k % 25000 == 0: log(f"  {label} {k}/{len(tasks)} ({time.time() - t0:.1f}s)")
    errs = [v[0] for sc in res.values() for v in sc.values() if str(v[0]).startswith("ERR:")]
    if errs: raise RuntimeError(f"{label} committee errors ({len(errs)}): {errs[:3]}")
    return res

def h1_dist_aligned(sd15: SD, raw1h: pd.DataFrame) -> np.ndarray:
    c = raw1h["close"].astype(float); sma = c.rolling(200, min_periods=200).mean()
    dist = ((c - sma) / sma).to_numpy(float)
    idx = np.searchsorted(raw1h["timestamp"].to_numpy(np.int64) + B1H, sd15.ts + B15, side="right") - 1
    return np.where(idx >= 0, dist[np.maximum(idx, 0)], np.nan)


def scalp_s0_trigger_dirs(s: SD, h1: np.ndarray, p_buy: np.ndarray, p_sell: np.ndarray) -> np.ndarray:
    c, o, atr = s.c, s.o, s.atr
    ok = np.isfinite(atr) & (atr > 0.0) & (c > 0.0) & np.isfinite(h1) & (np.arange(s.n) >= 3)
    kc_hi, kc_lo = s.kc_mid + 1.5 * atr, s.kc_mid - 1.5 * atr
    sqz = (s.df["bb_high"].astype(float).to_numpy() < kc_hi) & (s.df["bb_low"].astype(float).to_numpy() > kc_lo)
    sqz2 = np.r_[False, sqz[:-1]] & np.r_[False, False, sqz[:-2]]
    vz, natr = pd.to_numeric(s.df["volume_zscore_20"], errors="coerce").to_numpy(float), atr / np.where(c > 0.0, c, np.nan) * 10_000.0
    brk = np.where((c > o) & (c > kc_hi) & (h1 >= 0.0), 1, np.where((c < o) & (c < kc_lo) & (h1 <= 0.0), -1, 0))
    rd = np.where(ok & sqz2 & (vz >= 1.0) & (natr >= 35.0), brk, 0)
    pd_v, po_v = np.where(rd > 0, p_buy, p_sell), np.where(rd > 0, p_sell, p_buy)
    return np.where((rd != 0) & (pd_v >= 0.45) & (pd_v > po_v), rd, 0)


def prepare_swing_4h_cache(raw1h_map: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    for sym, df1h in raw1h_map.items():
        df4 = compute_swing_4h_indicators(resample_1h_to_closed_4h(df1h))
        if df4.empty: cache[sym] = {"n": 0, "ts": np.empty(0, np.int64)}; continue
        entries = {}
        for i in range(len(df4)):
            trig = evaluate_swing_4h_entry(df4.iloc[i : i + 1])
            if trig is not None:
                d_s, _, m = trig; av, cv = float(m["atr_14"]), float(m["close"])
                entries[i] = (d_s, av, min(2.0, decide_leverage(0.75, atr_pct=av / cv, regime_alignment=True, timeframe="4h", max_leverage=2.0)))
        cache[sym] = {"n": len(df4), "ts": df4["timestamp"].to_numpy(np.int64), "o": df4["open"].to_numpy(float), "h": df4["high"].to_numpy(float),
                      "l": df4["low"].to_numpy(float), "c": df4["close"].to_numpy(float), "l10": df4["l10"].to_numpy(float), "h10": df4["h10"].to_numpy(float), "entries": entries}
    return cache


def simulate_swing_4h(
    sw_cache: Dict[str, Dict[str, Any]], funding_map: Dict[str, Tuple[np.ndarray, np.ndarray]], symbols: Sequence[str],
    start_ms: int, end_ms: Optional[int] = None, corr_cap: Optional[int] = None, max_net_risk_r: Optional[float] = None,
    beta_map: Optional[Dict[str, float]] = None, n_trials: int = 1, var_trials: float = 1.0,
) -> Dict[str, Any]:
    rows_by_sym, all_ts = {}, set()
    for sym in symbols:
        imap = {int(t): i for i, t in enumerate(sw_cache.get(sym, {"ts": []})["ts"]) if int(t) >= start_ms and (end_ms is None or int(t) < end_ms)}
        rows_by_sym[sym] = imap; all_ts.update(imap)
    timeline = sorted(all_ts)
    if not timeline:
        return {k: (0 if k in ("n", "trades_count") else (False if k == "dsr_passed_95pct" else (None if k == "bootstrap_ci95_avg_r" else 0.0)))
                for k in ("n", "trades_count", "days", "trades_per_day", "wr_pct", "pf", "pf_ex_top1", "sum_r", "avg_r", "max_dd_r", "max_dd_pct", "net_pnl_usd", "net_pnl_pct", "sharpe_net", "calmar", "dsr", "dsr_passed_95pct", "bootstrap_ci95_avg_r")}
    cap, rpct, m_util, cpct, bal, pos, pend, cd, tr_r, deq = 10_000.0, 0.01, 0.85, 0.25, 10_000.0, {}, {}, {}, [], {}

    def _close(sym: str, raw_px: float, cts: int) -> None:
        nonlocal bal
        p = pos.pop(sym); sgn = 1.0 if p["s"] == "LONG" else -1.0
        ex = raw_px * (1.0 - sgn * TAKER_SLIP); ef = p["q"] * ex * TAKER_FEE; gr = sgn * (ex - p["e"]) * p["q"]
        bal += p["f"] + gr - ef; tr_r.append(float((gr - p["ef"] - ef + p["f"]) / max(1e-6, p["ru"])))
        cd[sym] = (cts // SWING_4H_BAR_MS) + SWING_4H_COOLDOWN_BARS

    for ts in timeline:
        bid = ts // SWING_4H_BAR_MS
        for sym in symbols:
            if sym not in pend or sym in pos or ts not in rows_by_sym[sym]: continue
            d_s, atr, lev = pend.pop(sym); opx = float(sw_cache[sym]["o"][rows_by_sym[sym][ts]])
            if not (opx > 0.0 and math.isfinite(opx)): continue
            side, sgn = ("LONG", 1.0) if d_s == "BUY" else ("SHORT", -1.0)
            if corr_cap is not None and sum(1 for p in pos.values() if p["s"] == side) >= int(corr_cap): continue
            bw = float(beta_map.get(sym, 1.0)) if beta_map is not None else 1.0
            if max_net_risk_r is not None and abs(sum((1.0 if p["s"] == "LONG" else -1.0) * p["bw"] for p in pos.values()) + sgn * bw) > float(max_net_risk_r) + 1e-12: continue
            epx = opx * (1.0 + sgn * TAKER_SLIP); sl = epx - sgn * SWING_4H_SL_ATR_MULT * atr; mu = sum(p["m"] for p in pos.values())
            mrg = min((bal * rpct) / max(1e-4, (abs(epx - sl) / epx) * lev), max(0.0, bal * m_util - mu) * 0.95, bal * cpct)
            ntn = mrg * lev; ef = ntn * TAKER_FEE
            if mrg < 10.0 or (mu + mrg + ef) > bal * m_util: continue
            bal -= ef
            pos[sym] = {"s": side, "e": epx, "q": ntn / epx, "m": mrg, "n": ntn, "sl": sl, "a": atr, "tr": epx, "ef": ef, "f": 0.0, "b": 0, "ru": max(1.0, abs(epx - sl) * (ntn / epx)), "bw": bw}
        for sym in symbols:
            idx = rows_by_sym[sym].get(ts)
            if idx is None: continue
            sc = sw_cache[sym]; o, h, l, c = float(sc["o"][idx]), float(sc["h"][idx]), float(sc["l"][idx]), float(sc["c"][idx])
            if sym in pos:
                p = pos[sym]; p["b"] += 1
                if p["b"] % 2 == 0: p["f"] += (-1.0 if p["s"] == "LONG" else 1.0) * p["n"] * lookup_funding(funding_map.get(sym), ts + SWING_4H_BAR_MS)
                if p["b"] >= SWING_4H_MAX_HOLDING_BARS: _close(sym, o, ts)
                elif (p["s"] == "LONG" and l <= p["sl"]) or (p["s"] == "SHORT" and h >= p["sl"]): _close(sym, min(p["sl"], o) if p["s"] == "LONG" else max(p["sl"], o), ts)
                elif p["s"] == "LONG":
                    p["tr"] = max(p["tr"], h); p["sl"] = max(p["sl"], p["tr"] - SWING_4H_TRAIL_ATR_MULT * p["a"])
                    if idx >= 11 and math.isfinite(sc["l10"][idx]) and c < sc["l10"][idx]: _close(sym, c, ts + SWING_4H_BAR_MS)
                else:
                    p["tr"] = min(p["tr"], l); p["sl"] = min(p["sl"], p["tr"] + SWING_4H_TRAIL_ATR_MULT * p["a"])
                    if idx >= 11 and math.isfinite(sc["h10"][idx]) and c > sc["h10"][idx]: _close(sym, c, ts + SWING_4H_BAR_MS)
            if sym not in pos and bid >= cd.get(sym, -1) and idx in sc["entries"]: pend[sym] = sc["entries"][idx]
        deq[(ts + SWING_4H_BAR_MS) // DAY] = bal + sum(((1.0 if p["s"] == "LONG" else -1.0) * (float(sw_cache[s]["c"][rows_by_sym[s][ts]]) - p["e"]) * p["q"]) + p["f"] for s, p in pos.items() if ts in rows_by_sym[s])

    dk = sorted(deq); eq_v = pd.Series([deq[d] for d in dk], index=dk).reindex(np.arange(dk[0], dk[-1] + 1, dtype=np.int64)).ffill().fillna(cap).to_numpy(float)
    prev_e = np.insert(eq_v[:-1], 0, cap); d_rets = (eq_v - prev_e) / np.maximum(1.0, prev_e)
    peaks = np.maximum.accumulate(np.insert(eq_v, 0, cap))[1:]
    max_dd = float(np.max((peaks - eq_v) / np.maximum(1.0, peaks)) * 100.0) if len(eq_v) else 0.0
    days = max(1.0, (timeline[-1] - timeline[0] + SWING_4H_BAR_MS) / float(DAY))
    ra, n = np.array(tr_r, float), len(tr_r)
    pf_fn = lambda a: (float(a[a > 0].sum()) / float(-a[a < 0].sum())) if float(-a[a < 0].sum()) > 0 else (99.0 if float(a[a > 0].sum()) > 0 else 0.0)  # noqa: E731
    sh = float(calculate_annualized_sharpe(d_rets, 365.0)); dsr = deflated_sharpe_ratio(sh, d_rets, max(1, n_trials), max(var_trials, 1e-4), 365.0)
    net_usd = float(eq_v[-1] - cap) if len(eq_v) else 0.0
    ann = (((eq_v[-1] / cap) ** (365.0 / days) - 1.0) * 100.0) if (len(eq_v) and eq_v[-1] > 0) else -100.0
    return {"n": n, "trades_count": n, "days": round(days, 2), "trades_per_day": round(n / days, 4), "wr_pct": round(float((ra > 0).mean() * 100.0), 2) if n else 0.0,
            "pf": round(pf_fn(ra), 4) if n else 0.0, "pf_ex_top1": round(pf_fn(np.delete(ra, int(np.argmax(ra)))), 4) if n > 1 else 0.0,
            "sum_r": round(float(ra.sum()), 4) if n else 0.0, "avg_r": round(float(ra.mean()), 4) if n else 0.0,
            "max_dd_r": round(float(np.max(np.maximum.accumulate(np.r_[0.0, np.cumsum(ra)])[1:] - np.cumsum(ra))), 4) if n else 0.0,
            "max_dd_pct": round(max_dd, 2), "net_pnl_usd": round(net_usd, 2), "net_pnl_pct": round((net_usd / cap) * 100.0, 2),
            "sharpe_net": round(sh, 4), "calmar": round((ann / max_dd) if max_dd > 1e-9 else (99.0 if ann > 0 else 0.0), 4),
            "dsr": round(float(dsr["dsr"]), 4), "dsr_passed_95pct": bool(dsr["passed_95pct"]), "bootstrap_ci95_avg_r": None}


def eval_strategy_matrix(
    cands_sc: Dict[str, List[Dict[str, Any]]], out_fn: Callable, cd_fn: Callable, beta_map: Dict[str, float],
    t0: int, t1: int, n_prior: int, locks: Optional[Dict[str, Dict[str, List[Tuple[int, int]]]]] = None, one_bar: bool = False,
) -> Dict[str, Any]:
    nt, strip = n_prior + 42, lambda d: {k: v for k, v in d.items() if k != "trades"}  # noqa: E731
    run = lambda sc, syms, cc, nr, ub, lk=None, nb=2000, vt=1.0: simulate_portfolio_with_caps(  # noqa: E731
        [c for c in cands_sc[sc] if c["sym"] in set(syms)], out_fn, corr_cap=cc, max_net_risk_r=nr,
        beta_map=(beta_map if ub else None), cooldown_fn=cd_fn, blocked_intervals=(locks.get(lk) if (locks and lk) else None),
        one_entry_per_bar=one_bar, t0=t0, t1=t1, var_trials=vt, n_trials=nt, bootstrap_iters=nb)
    grid_keys = [(sc, cc, nr, wm, f"{sc}_cc{cc}_nr{'none' if nr is None else int(nr)}_{wm}") for sc in ("base3", "pooled8") for cc in (2, 3, 4) for nr, wm in RISK_GRID]
    m1_p1 = {k: strip(run(sc, TIER1_8, cc, nr, wm == "beta", f"{sc}_cc{cc}", 0, 1.0)) for sc, cc, nr, wm, k in grid_keys}
    vt = max(float(np.var([v["sharpe_net"] for v in m1_p1.values()])), 1e-4)
    m0, m0_ml8 = run("base3", BASE_3, 2, None, False, "base3_cc2", 2000, vt), run("pooled8", BASE_3, 2, None, False, "pooled8_cc2", 2000, vt)
    m1_grid = {k: strip(run(sc, TIER1_8, cc, nr, wm == "beta", f"{sc}_cc{cc}", 0, vt)) for sc, cc, nr, wm, k in grid_keys}
    bk = max(m1_grid, key=lambda k: (m1_grid[k]["sum_r"], m1_grid[k]["sharpe_net"]))
    psc, pcc, pnr, pwm = bk.split("_")
    m1_grid[bk] = best_m1 = strip(run(psc, TIER1_8, int(pcc[2:]), None if pnr == "nrnone" else float(pnr[2:]), pwm == "beta", f"{psc}_{pcc}", 2000, vt))
    m2: Dict[str, Any] = {}
    for sym in TIER1_8:
        st3, st8 = strip(run("base3", [sym], 2, None, False, "base3_cc2", 0, vt)), strip(run("pooled8", [sym], 2, None, False, "pooled8_cc2", 0, vt))
        ent: Dict[str, Any] = {"standalone_base3": st3, "standalone_pooled8": st8}
        if sym in CANDIDATE_5:
            l3, l8 = strip(run("base3", list(BASE_3) + [sym], 2, None, False, "base3_cc2", 0, vt)), strip(run("pooled8", list(BASE_3) + [sym], 2, None, False, "pooled8_cc2", 0, vt))
            ent.update(leave_one_in_base3=l3, leave_one_in_pooled8=l8, n=st3["n"], avg_r=st3["avg_r"],
                       delta_sum_r=round(l3["sum_r"] - m0["sum_r"], 4), delta_sharpe=round(l3["sharpe_net"] - m0["sharpe_net"], 4), delta_calmar=round(l3["calmar"] - m0["calmar"], 4))
        m2[sym] = ent
    marg = {s: m2[s] for s in CANDIDATE_5}
    gate = evaluate_go_gate(best_m1, m0, marg, m0_ml8 if bk.startswith("pooled8") else None)
    prom = gate["promoted_symbols"]
    m3: Dict[str, Any] = {"status": "NO_STRICT_SUBSET", "exploratory_only": True, "eligible_for_promotion": False, "promoted_symbols": prom}
    if 0 < len(prom) < len(CANDIDATE_5):
        ss = list(BASE_3) + list(prom)
        m3_b3, m3_p8 = strip(run("base3", ss, 2, 3.0, True, "base3_cc2", 2000, vt)), strip(run("pooled8", ss, 2, 3.0, True, "pooled8_cc2", 2000, vt))
        bsc, bm3 = ("pooled8", m3_p8) if m3_p8["sum_r"] > m3_b3["sum_r"] else ("base3", m3_b3)
        m3 = {"status": "EXPLORATORY_REQUIRES_FORWARD_VALIDATION", "exploratory_only": True, "eligible_for_promotion": False, "symbols": ss,
              "promoted_symbols": prom, "base3_cc2_nr3_beta": m3_b3, "pooled8_cc2_nr3_beta": m3_p8, "best_scope": bsc,
              "exploratory_gate": evaluate_go_gate(bm3, m0, marg, m0_ml8 if bsc == "pooled8" else None)}
    return {"M0_BASE_3": strip(m0), "M0_BASE_3_ML8": strip(m0_ml8), "M1_TIER1_8": {"best_key": bk, "best": best_m1, "var_trials": round(vt, 6), "go_gate": gate, "grid": m1_grid},
            "M2_PER_SYMBOL": m2, "M3_SUBSET_CANDIDATE": m3}


def eval_g8_gp_fast(cache_db: str, micro_db: str, ens_1h: _DefaultMLEnsemble, budget_s: float = 28.0) -> Dict[str, Any]:
    """Optional fast G8_ATR_EXP_CAP_3X (ATR_exp <= 3.0x on GOLDEN_POT OOS) bounded to < 30s."""
    t0 = time.time(); from scripts import eval_golden_pot_s6_oos as gp
    con = ro_connect(cache_db)
    try:
        rows = con.execute("SELECT symbol, exchange, timestamp, open, high, low, close, volume FROM candles WHERE timeframe='5m' AND timestamp >= ? ORDER BY symbol ASC, timestamp ASC", (gp.SPLIT_TS_GP - 300 * gp.GP_BAR_MS,)).fetchall()
    finally: con.close()
    if not rows or (time.time() - t0) > budget_s * 0.5: return {"status": "SKIPPED_EMPTY_OR_BUDGET", "elapsed_s": round(time.time() - t0, 2)}
    df = pd.DataFrame(rows, columns=["symbol", "exchange", "timestamp", "open", "high", "low", "close", "volume"])
    df["_p"] = np.where(df["exchange"] == "binance", 0, 1); df = df.sort_values(["symbol", "timestamp", "_p"]).drop_duplicates(["symbol", "timestamp"], keep="first")
    segs, sid = [], 0
    for sym, grp in df.groupby("symbol", sort=True):
        su = str(sym).strip().upper()
        if su in gp.MAJORS or su in gp.STABLECOIN_EXCLUSIONS or gp.GoldenPotAgent.extract_base_asset(su) in gp.ALL_TRADFI_BASE_EXCLUSIONS or len(grp) <= gp.MIN_SEGMENT_BARS: continue
        g = grp.iloc[:-1][["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        brks = [0] + (np.where(np.diff(g["timestamp"].to_numpy(np.int64)) > gp.GAP_THRESHOLD_5M_MS)[0] + 1).tolist() + [len(g)]
        for a, b in zip(brks[:-1], brks[1:]):
            if b - a >= gp.MIN_SEGMENT_BARS: segs.append((sid, su, g.iloc[a:b].copy().reset_index(drop=True))); sid += 1
    cands_oos, seg_dfs = [], {}
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for res in pool.imap_unordered(gp._process_segment_worker, segs, chunksize=16):
            if res is not None:
                ocs = [c for c in res[3] if c["split"] == "OOS" and c["std_dir"] is not None]
                if ocs: seg_dfs[res[0]] = res[2]; cands_oos.extend(ocs)
            if (time.time() - t0) > budget_s: return {"status": "SKIPPED_BUDGET_30S", "elapsed_s": round(time.time() - t0, 2)}
    enr, core = FastCachedEnricher(cache_db, micro_db), TradingCore(min_consensus_threshold=0.30, technical_agent=TechnicalAgent(model=ens_1h))
    core.learning_tracker, g0_s, g8_s, gp_p = None, [], [], gp.STRATEGY_PROFILES["GOLDEN_POT"]
    try:
        for c in cands_oos:
            if (time.time() - t0) > budget_s: return {"status": "SKIPPED_BUDGET_30S", "elapsed_s": round(time.time() - t0, 2)}
            d, df_s, i = c["std_dir"], seg_dfs[c["seg_id"]], c["idx"]
            if gp.is_climax_blocked(d, c["rsi"], c["cum_move_atr"], "prod"): continue
            sig = core.evaluate_market_deterministic(enr.enrich(c["sym"], df_s.iloc[i], df_s.iloc[max(0, i - 60) : i + 1], int(c["ts"]) + gp.GP_BAR_MS), c["sym"], float(c["close"]), float(c["atr"]), int(c["ts"]))
            if sig.is_vetoed or sig.direction != d or sig.confidence < CONF_MIN: continue
            rec = {"sym": c["sym"], "ts": c["ts"], "direction": d, "rank_score": c["rank_score"], **gp.simulate_single_ladder_trade(df_s, i, d, float(c["sl_mult"]), float(gp_p.tp1_atr_mult), float(gp_p.tp2_atr_mult), float(gp_p.trail_atr_mult), gp.GP_MAX_HOLDING_BARS, gp.GP_BAR_MS)}
            g0_s.append(rec)
            if float(c["atr_exp"]) <= 3.0: g8_s.append(rec)
    finally: enr.close()
    od = max(1.0, (max((c["ts"] for c in cands_oos), default=gp.SPLIT_TS_GP + DAY) - gp.SPLIT_TS_GP) / DAY)
    g0_tr, _ = gp.simulate_gp_portfolio_split(g0_s, od, 12, 6, 3); g8_tr, _ = gp.simulate_gp_portfolio_split(g8_s, od, 12, 6, 3)
    return {"status": "OK", "elapsed_s": round(time.time() - t0, 2), "g0_oos_n": len(g0_tr), "g0_oos_sum_r": round(sum(t["r_mult"] for t in g0_tr), 4), "g8_oos_n": len(g8_tr), "g8_oos_sum_r": round(sum(t["r_mult"] for t in g8_tr), 4), "g8_oos_avg_r": round(float(np.mean([t["r_mult"] for t in g8_tr])), 4) if g8_tr else 0.0}


def eval_swing_windows(sw_cache: Dict[str, Any], funding_map: Dict[str, Any], beta_30d: Dict[str, float], oos_1h: int, prior: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for wn, ws in (("oos_primary_2025_01_08", oos_1h), ("oos_parity_2025_11_20", SWING_PARITY_OOS_MS)):
        specs = [(cc, nr, wm, f"cc{cc}_nr{'none' if nr is None else int(nr)}_{wm}") for cc in (2, 3, 4) for nr, wm in RISK_GRID]
        sg0 = {k: simulate_swing_4h(sw_cache, funding_map, TIER1_8, ws, None, cc, nr, beta_30d if wm == "beta" else None, prior + 21, 1.0) for cc, nr, wm, k in specs}
        vt = max(float(np.var([v["sharpe_net"] for v in sg0.values()])), 1e-4)
        sm0 = simulate_swing_4h(sw_cache, funding_map, BASE_3, ws, None, None, None, None, prior + 21, vt)
        sg = {k: simulate_swing_4h(sw_cache, funding_map, TIER1_8, ws, None, cc, nr, beta_30d if wm == "beta" else None, prior + 21, vt) for cc, nr, wm, k in specs}
        sm2 = {}
        for s in TIER1_8:
            st = simulate_swing_4h(sw_cache, funding_map, [s], ws, None, None, None, None, prior + 21, vt); en: Dict[str, Any] = {"standalone": st}
            if s in CANDIDATE_5:
                loi = simulate_swing_4h(sw_cache, funding_map, list(BASE_3) + [s], ws, None, 2, None, None, prior + 21, vt)
                en.update(leave_one_in=loi, n=st["n"], avg_r=st["avg_r"], delta_sum_r=round(loi["sum_r"] - sm0["sum_r"], 4), delta_sharpe=round(loi["sharpe_net"] - sm0["sharpe_net"], 4), delta_calmar=round(loi["calmar"] - sm0["calmar"], 4))
            sm2[s] = en
        bk = max(sg, key=lambda k: (sg[k]["sum_r"], sg[k]["sharpe_net"])); gt = evaluate_go_gate(sg[bk], sm0, {s: sm2[s] for s in CANDIDATE_5})
        sm3: Dict[str, Any] = {"status": "NO_STRICT_SUBSET", "exploratory_only": True, "eligible_for_promotion": False, "promoted_symbols": gt["promoted_symbols"]}
        if 0 < len(gt["promoted_symbols"]) < len(CANDIDATE_5):
            ss = list(BASE_3) + list(gt["promoted_symbols"]); subm = simulate_swing_4h(sw_cache, funding_map, ss, ws, None, 2, 3.0, beta_30d, prior + 21, vt)
            sm3 = {"status": "EXPLORATORY_REQUIRES_FORWARD_VALIDATION", "exploratory_only": True, "eligible_for_promotion": False, "symbols": ss, "cc2_nr3_beta": subm, "exploratory_gate": evaluate_go_gate(subm, sm0, {s: sm2[s] for s in CANDIDATE_5})}
        out[wn] = {"M0_BASE_3": sm0, "M1_TIER1_8": {"best_key": bk, "best": sg[bk], "var_trials": round(vt, 6), "go_gate": gt, "grid": sg}, "M2_PER_SYMBOL": sm2, "M3_SUBSET_CANDIDATE": sm3}
    return out


def _synth_clone(src: pd.DataFrame, seed: int) -> pd.DataFrame:
    df, sc = src.copy(), np.clip(1.0 + np.cumsum(np.random.default_rng(seed).normal(0.0, 0.0015, len(src))), 0.4, 2.5)
    for col in ("open", "high", "low", "close"): df[col] = df[col].astype(float) * sc
    df["high"], df["low"] = np.maximum(df["high"], np.maximum(df["open"], df["close"])), np.minimum(df["low"], np.minimum(df["open"], df["close"]))
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="S6.3 Zero-Leakage OOS Majors Expansion Evaluator")
    for flag, kw in (("--smoke", {"action": "store_true"}), ("--log-trial", {"action": "store_true"}), ("--skip-g8-gp", {"action": "store_true"}),
                     ("--out", {"default": OUT_JSON}), ("--cache-db", {"default": CACHE_DB}), ("--micro-db", {"default": MICRO_DB})):
        ap.add_argument(flag, **kw)
    args = ap.parse_args(); t0 = time.time()

    con = ro_connect(args.cache_db)
    try: raw1h, raw15, funding_map = {s: load_candles(con, s, "1h") for s in TIER1_8}, {s: load_candles(con, s, "15m") for s in TIER1_8}, load_funding_map(con, TIER1_8)
    finally: con.close()
    synth: List[str] = []
    for idx, s in enumerate(TIER1_8):
        if len(raw1h[s]) < 500 or len(raw15[s]) < 500:
            if not args.smoke: raise RuntimeError(f"Insufficient history for {s}: 1h={len(raw1h[s])}, 15m={len(raw15[s])}")
            src = BASE_3[idx % len(BASE_3)]
            raw1h[s] = _synth_clone(raw1h[src], 100 + idx) if len(raw1h[s]) < 500 else raw1h[s]
            raw15[s] = _synth_clone(raw15[src], 200 + idx) if len(raw15[s]) < 500 else raw15[s]
            funding_map[s] = funding_map[src] if not len(funding_map[s][0]) else funding_map[s]; synth.append(s)
    if args.smoke: raw1h, raw15 = {s: df.iloc[-6000:].reset_index(drop=True) for s, df in raw1h.items()}, {s: df.iloc[-4000:].reset_index(drop=True) for s, df in raw15.items()}
    log(f"Building features (smoke={args.smoke}, synth={synth})...")
    sd1h, sd15 = {s: build_sd(s, raw1h[s]) for s in TIER1_8}, {s: build_sd(s, raw15[s]) for s in TIER1_8}
    end_1h, end_15m = max(int(sd1h[s].ts[-1]) for s in TIER1_8) + B1H, max(int(sd15[s].ts[-1]) for s in TIER1_8) + B15
    cut_1h = TRAIN_CUTOFF_1H if not args.smoke else int(sd1h["BTCUSDT"].ts[int(sd1h["BTCUSDT"].n * 0.7)])
    oos_1h = OOS_START_1H if not args.smoke else (cut_1h + 7 * DAY)
    cut_15m = TRAIN_CUTOFF_15M if not args.smoke else int(sd15["BTCUSDT"].ts[int(sd15["BTCUSDT"].n * 0.7)])
    split_15m, cut_1h_15m = (SPLIT_15M, TRAIN_CUTOFF_1H_FOR_15M) if not args.smoke else (cut_15m + 10 * B15, cut_15m + 10 * B15 - 7 * DAY)
    cols1h, cols15 = sorted(set.intersection(*[set(select_stationary_features(sd1h[s].df)) for s in TIER1_8])), sorted(set.intersection(*[set(select_stationary_features(sd15[s].df)) for s in TIER1_8]))
    s1h = IS_START_1H if not args.smoke else None
    ens_specs = (("1h_b3", sd1h, BASE_3, cols1h, cut_1h, 1.5, 1.0, True, s1h), ("1h_p8", sd1h, TIER1_8, cols1h, cut_1h, 1.5, 1.0, True, s1h),
                 ("15m_b3", sd15, BASE_3, cols15, cut_15m, 1.2, 1.2, False, None), ("15m_p8", sd15, TIER1_8, cols15, cut_15m, 1.2, 1.2, False, None),
                 ("15c_b3", sd1h, BASE_3, cols1h, cut_1h_15m, 1.5, 1.0, True, None), ("15c_p8", sd1h, TIER1_8, cols1h, cut_1h_15m, 1.5, 1.0, True, None))
    fitted = {k: fit_purged_ensemble(*a) for k, *a in ens_specs}
    m1h_b3, m1h_p8, m15_b3, m15_p8, m15c_b3, m15c_p8 = (fitted[k][0] for k, *_ in ens_specs)
    train_n = {k: fitted[k][1] for k, *_ in ens_specs}
    e1_b3_r, e1_p8_r, e15_b3_r, e15_p8_r = _DefaultMLEnsemble(m1h_b3, cols1h), _DefaultMLEnsemble(m1h_p8, cols1h), _DefaultMLEnsemble(m15c_b3, cols1h), _DefaultMLEnsemble(m15c_p8, cols1h)
    d1o, d15o = {s: sd1h[s].df[sd1h[s].df["timestamp"] >= oos_1h] for s in TIER1_8}, {s: sd15[s].df[sd15[s].df["timestamp"] >= split_15m] for s in TIER1_8}
    e1_b3, e1_p8, e15_b3, e15_p8 = (_PrecomputedEnsemble.from_ensemble(e, d) for e, d in ((e1_b3_r, d1o), (e1_p8_r, d1o), (e15_b3_r, d15o), (e15_p8_r, d15o)))
    parity = verify_parity(args.cache_db, args.micro_db, sd1h, e1_b3_r, e1_b3, list(BASE_3) + ["XRPUSDT", "NEARUSDT"])
    log(f"Parity OK: enricher={parity['enricher_max_diff']:.2e}, ensemble={parity['ensemble_max_diff']:.2e} (n={parity['n_samples']})")
    mk_c = lambda e: (lambda c: (setattr(c, "learning_tracker", None) or c))(TradingCore(min_consensus_threshold=0.30, technical_agent=TechnicalAgent(model=e)))  # noqa: E731
    G.update(cache_db=args.cache_db, micro_db=args.micro_db, sd1h=sd1h, sd15=sd15, e1_b3=e1_b3, e1_p8=e1_p8, e15_b3=e15_b3, e15_p8=e15_p8,
             c1_b3=mk_c(e1_b3), c1_p8=mk_c(e1_p8), c15_b3=mk_c(e15_b3), c15_p8=mk_c(e15_p8))
    t1h = [(s, int(i)) for s in TIER1_8 for i in (np.nonzero((sd1h[s].ts >= oos_1h) & np.isfinite(sd1h[s].atr) & (sd1h[s].atr > 0.0))[0][-200:] if args.smoke else np.nonzero((sd1h[s].ts >= oos_1h) & np.isfinite(sd1h[s].atr) & (sd1h[s].atr > 0.0))[0])]
    log(f"Evaluating DIRECTIONAL 1h ({len(t1h)} bars)..."); comm_1h = run_pool("1h", t1h, "dir1h")
    trig15, t15_set = {"base3": {}, "pooled8": {}}, set()
    for s in TIER1_8:
        h1d, X15 = h1_dist_aligned(sd15[s], raw1h[s]), sd15[s].df.reindex(columns=cols15).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        for sc, mods in (("base3", m15_b3), ("pooled8", m15_p8)):
            pr = [np.asarray(m.predict_proba(X15), float) for m in mods]
            td = scalp_s0_trigger_dirs(sd15[s], h1d, np.mean([p[:, 1] for p in pr], 0), np.mean([p[:, 2] for p in pr], 0))
            trig15[sc][s] = td; idxs = np.nonzero((td != 0) & (sd15[s].ts >= split_15m))[0]
            t15_set.update((s, int(i)) for i in (idxs[-40:] if args.smoke else idxs))
    log(f"Evaluating SCALP 15m ({len(t15_set)} triggers)..."); comm_15m = run_pool("15m", sorted(t15_set), "scalp15m")
    dfs_all = {s: sd1h[s].df for s in TIER1_8}
    corr_is, corr_oos = compute_pairwise_corr_matrix(dfs_all, end_ms=cut_1h), compute_pairwise_corr_matrix(dfs_all, start_ms=oos_1h, end_ms=end_1h)
    beta_is, beta_30d = compute_causal_beta_to_btc(dfs_all, cut_1h, lookback_days=None), compute_causal_beta_to_btc(dfs_all, oos_1h, lookback_days=30)
    dir_cands = {sc: [{"sym": s, "i": i, "ts": int(sd1h[s].ts[i]), "open_t": int(sd1h[s].ts[i]) + B1H, "d": 1 if dr == "BUY" else -1}
                      for (s, i), (dr, conf, veto) in comm_1h[sc].items() if not veto and dr in ("BUY", "SELL") and conf >= CONF_MIN] for sc in ("base3", "pooled8")}
    d_c: Dict[Tuple[str, int, int], Any] = {}
    d_out = lambda c: d_c.setdefault((c["sym"], c["i"], c["d"]), directional_outcome(sd1h[c["sym"]], c["i"], c["d"], funding_map.get(c["sym"])))  # noqa: E731
    prior, dir_t0 = get_prior_trials_count(), (min((c["ts"] for c in dir_cands["base3"]), default=oos_1h) if args.smoke else oos_1h)
    dir_res = eval_strategy_matrix(dir_cands, d_out, directional_cooldown_fn, beta_30d, dir_t0, end_1h, prior)
    dir_locks = {}
    for sc in ("base3", "pooled8"):
        for cc in (2, 3, 4):
            iv: Dict[str, List[Tuple[int, int]]] = {s: [] for s in TIER1_8}
            for tr in simulate_portfolio_with_caps(dir_cands[sc], d_out, corr_cap=cc, cooldown_fn=directional_cooldown_fn, t0=dir_t0, t1=end_1h, bootstrap_iters=0)["trades"]:
                iv[tr["sym"]].append((tr["open_t"], tr["close_t"]))
            dir_locks[f"{sc}_cc{cc}"] = iv
    scalp_cands = {sc: [{"sym": s, "i": i, "ts": int(sd15[s].ts[i]), "open_t": int(sd15[s].ts[i]) + B15, "d": int(trig15[sc][s][i])}
                        for (s, i), (dr, conf, veto) in comm_15m[sc].items()
                        if int(trig15[sc][s][i]) != 0 and not veto and dr == ("BUY" if int(trig15[sc][s][i]) > 0 else "SELL") and conf >= CONF_MIN] for sc in ("base3", "pooled8")}
    s_c: Dict[Tuple[str, int, int], Any] = {}
    s_out = lambda c: s_c.setdefault((c["sym"], c["i"], c["d"]), scalp_outcome(sd15[c["sym"]], c["i"], c["d"]))  # noqa: E731
    scalp_st, scalp_dl = eval_strategy_matrix(scalp_cands, s_out, scalp_cooldown_fn, beta_30d, split_15m, end_15m, prior, None, True), eval_strategy_matrix(scalp_cands, s_out, scalp_cooldown_fn, beta_30d, split_15m, end_15m, prior, dir_locks, True)
    log("Evaluating SWING_4H 4h (primary >=2025-01-08 & parity >=2025-11-20)...")
    swing_out = eval_swing_windows(prepare_swing_4h_cache(raw1h), funding_map, beta_30d, oos_1h, prior)
    g8 = {"status": "SKIPPED_SMOKE_OR_FLAG"} if (args.smoke or args.skip_g8_gp) else eval_g8_gp_fast(args.cache_db, args.micro_db, e15_b3_r, 28.0)
    report = sanitize_json({
        "metadata": {"smoke": bool(args.smoke), "synthetic_smoke_symbols": synth, "base3": list(BASE_3), "candidate5": list(CANDIDATE_5), "tier1_8": list(TIER1_8),
                     "train_cutoff_1h": cut_1h, "oos_start_1h": oos_1h, "end_1h": end_1h, "train_cutoff_15m": cut_15m, "split_15m": split_15m,
                     "train_cutoff_1h_for_15m": cut_1h_15m, "end_15m": end_15m, "common_features": {"1h": len(cols1h), "15m": len(cols15)},
                     "train_n": train_n, "parity_checks": parity, "prior_trials": prior, "workers": WORKERS, "runtime_s": round(time.time() - t0, 2)},
        "correlation_and_beta": {"corr_1h_is": corr_is, "corr_1h_oos": corr_oos, "beta_is_to_btc": beta_is, "beta_30d_to_btc": beta_30d},
        "directional_1h": dir_res, "scalp_15m_standalone": scalp_st, "scalp_15m_dirlock": scalp_dl, "swing_4h": swing_out, "g8_atr_exp_cap_3x": g8,
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".tmp", "w", encoding="utf-8") as fh: json.dump(report, fh, indent=2, allow_nan=False)
    os.replace(args.out + ".tmp", args.out); log(f"Report written: {args.out} ({report['metadata']['runtime_s']}s)")
    if args.log_trial:
        swp = swing_out["oos_primary_2025_01_08"]
        lbl = lambda g: "CANDIDATE" if g["passed"] else "FAIL_REFUTED"  # noqa: E731
        rows_t = [
            ("S6_3_DIR1H_M0_BASE3", "DIRECTIONAL_1H_M0", "BASE_3", "1h", dir_res["M0_BASE_3"], "2025-01-08..OOS", "BASELINE"),
            ("S6_3_DIR1H_M1_BEST", f"DIRECTIONAL_1H_{dir_res['M1_TIER1_8']['best_key']}", "TIER1_8", "1h", dir_res["M1_TIER1_8"]["best"], "2025-01-08..OOS", lbl(dir_res["M1_TIER1_8"]["go_gate"])),
            ("S6_3_SCALP15M_M0_BASE3", "SCALP_15M_M0", "BASE_3", "15m", scalp_dl["M0_BASE_3"], "2026-07-29..OOS", "BASELINE"),
            ("S6_3_SCALP15M_M1_BEST", f"SCALP_15M_{scalp_dl['M1_TIER1_8']['best_key']}", "TIER1_8", "15m", scalp_dl["M1_TIER1_8"]["best"], "2026-07-29..OOS", lbl(scalp_dl["M1_TIER1_8"]["go_gate"])),
            ("S6_3_SWING4H_M0_BASE3", "SWING_4H_M0", "BASE_3", "4h", swp["M0_BASE_3"], "2025-01-08..OOS", "BASELINE"),
            ("S6_3_SWING4H_M1_BEST", f"SWING_4H_{swp['M1_TIER1_8']['best_key']}", "TIER1_8", "4h", swp["M1_TIER1_8"]["best"], "2025-01-08..OOS", lbl(swp["M1_TIER1_8"]["go_gate"])),
        ]
        for exp, st_n, sym_s, tf, m, per, st_l in rows_t:
            tid = log_trial(phase="S6_3_OOS", experiment=exp, strategy_name=st_n, symbol=sym_s, timeframe=tf, params={"smoke": args.smoke}, sample_period=per, is_oos=True,
                            trades_count=int(m["n"]), net_pnl=float(m["net_pnl_usd"]), net_pnl_pct=float(m["net_pnl_pct"]), sharpe_ratio=float(m["sharpe_net"]),
                            max_drawdown_pct=float(m["max_dd_pct"]), win_rate_pct=float(m["wr_pct"]), status_label=st_l,
                            notes=f"PF={m['pf']:.3f};sumR={m['sum_r']:.3f};tpd={m['trades_per_day']:.4f};DSR={m['dsr']:.4f}")
            log(f"Logged {exp}: {tid} ({st_l})")
        if g8.get("status") == "OK":
            g8_l = "CANDIDATE" if (g8.get("g8_oos_avg_r", 0.0) > 0.0 and g8.get("g8_oos_sum_r", 0.0) > g8.get("g0_oos_sum_r", 0.0)) else "FAIL_REFUTED"
            tid = log_trial(phase="S6_3_OOS", experiment="S6_3_GP_G8_ATR_EXP_3X", strategy_name="GOLDEN_POT_G8_ATR_EXP_3X", symbol="ALTCOINS", timeframe="5m",
                            params={"atr_exp_max": 3.0}, sample_period="2026-11-05..OOS", is_oos=True, trades_count=int(g8["g8_oos_n"]), net_pnl=0.0, net_pnl_pct=0.0,
                            sharpe_ratio=0.0, max_drawdown_pct=0.0, win_rate_pct=0.0, status_label=g8_l, notes=f"g0_sumR={g8['g0_oos_sum_r']};g8_sumR={g8['g8_oos_sum_r']};g8_avgR={g8['g8_oos_avg_r']}")
            log(f"Logged S6_3_GP_G8_ATR_EXP_3X: {tid} ({g8_l})")
    return 0


if __name__ == "__main__": raise SystemExit(main())
