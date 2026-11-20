#!/usr/bin/env python3
"""Wave P2-B Zero-Leakage Benchmark (`v1` vs `v2`) & Conditional ML Promotion Gate."""

from __future__ import annotations
import argparse, json, math, os, shutil, sqlite3, sys, tempfile, time
from datetime import datetime, timezone
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple
import joblib, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.technical_agent import _disable_estimator_parallelism  # noqa: E402
from data.feature_engineering import FeatureEngineer  # noqa: E402
from data.ml_features_v2 import TIMEFRAME_MS, V2_FEATURE_COLUMNS, _NEUTRAL_DEFAULTS, build_v2_features_batch, is_v2_feature_bundle  # noqa: E402
from db_conn import connect as _db_connect  # noqa: E402
from model_trainer import ModelTrainer, _atomic_joblib_dump, create_triple_barrier_labels  # noqa: E402
from smart_trade import MAKER_SLIPPAGE_PCT, MARKET_SLIPPAGE_PCT  # noqa: E402
from statistical_rigor import bootstrap_performance_metrics, calculate_annualized_sharpe, deflated_sharpe_ratio, get_prior_trials_count, log_trial, probability_of_backtest_overfitting  # noqa: E402

MAJORS: Tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
DAY_MS, HOUR_MS, M15_MS = 86_400_000, 3_600_000, 900_000
TAKER_FEE, MAKER_FEE, RISK_PER_TRADE = 0.0006, 0.0002, 0.01
HOLDOUT_1H_START_MS = int(pd.Timestamp("2025-11-20", tz="UTC").timestamp() * 1000)
MICRO_ABLATION_START_MS = int(pd.Timestamp("2026-09-15", tz="UTC").timestamp() * 1000)
MICRO_EXTRA_COLS: List[str] = ["ms_oi_chg_1h", "ms_ls_global_z", "ms_taker_imb_15m"]

V1_FEATURE_COLUMNS: List[str] = [
    "macd_line", "macd_signal", "macd_hist", "macd_diff", "dist_sma_20", "dist_sma_50", "dist_sma_200", "dist_ema_9", "dist_ema_21",
    "rsi_14", "stoch_k", "stoch_d", "rsi_overbought", "rsi_oversold", "atr_pct", "natr_14", "bb_width", "bb_pband",
    "vol_sma_ratio", "volume_zscore_20", "dist_vwap_pct", "ob_spread", "ob_imbalance", "returns", "log_returns", "volatility_20", "hl_spread",
    "macro_cpi", "macro_fedfunds", "macro_vix", "macro_dxy", "macro_sp500", "macro_regime_risk_off", "funding_rate", "funding_zscore",
    "liquidity_sweep_bull", "liquidity_sweep_bear", "dist_resistance_pct", "dist_support_pct",
    "inside_bullish_fvg", "inside_bearish_fvg", "inside_bullish_ob", "inside_bearish_ob", "at_bullish_poi", "at_bearish_poi",
]

CandidateSpec = NamedTuple("CandidateSpec", [
    ("name", str), ("version", str), ("timeframe", str), ("train_symbols", Tuple[str, ...]),
    ("horizon", int), ("multiple_tp", float), ("multiple_sl", float), ("decision_threshold", float),
])

CANDIDATES_1H: Tuple[CandidateSpec, ...] = (
    CandidateSpec("BEFORE_V1_1H", "v1", "1h", ("BTCUSDT",), 5, 1.2, 1.2, 0.50),
    CandidateSpec("AFTER_V2_1H_H5", "v2", "1h", MAJORS, 5, 1.2, 1.2, 0.50),
    CandidateSpec("AFTER_V2_1H_H18", "v2", "1h", MAJORS, 18, 2.0, 1.2, 0.50),
    CandidateSpec("AFTER_V2_1H_H24", "v2", "1h", MAJORS, 24, 2.2, 1.4, 0.50),
)
CANDIDATES_15M: Tuple[CandidateSpec, ...] = (
    CandidateSpec("BEFORE_V1_15M", "v1", "15m", MAJORS, 5, 1.2, 1.2, 0.45),
    CandidateSpec("AFTER_V2_15M_H5", "v2", "15m", MAJORS, 5, 1.2, 1.2, 0.45),
    CandidateSpec("AFTER_V2_15M_H16", "v2", "15m", MAJORS, 16, 2.0, 1.4, 0.45),
    CandidateSpec("AFTER_V2_15M_H30", "v2", "15m", MAJORS, 30, 2.2, 1.4, 0.45),
)

def _default_db_path(filename: str) -> str:
    local, fallback = os.path.join(ROOT, "datasets", filename), os.path.join("/home/eudespatricio/ia_trader/datasets", filename)
    return local if os.path.exists(local) or not os.path.exists(fallback) else fallback

def _connect_ro(db_path: str) -> sqlite3.Connection:
    ap = os.path.abspath(db_path)
    if not os.path.exists(ap):
        raise FileNotFoundError(f"SQLite DB not found: {ap}")
    con = _db_connect(f"file:{ap}?mode=ro", uri=True, timeout=20.0)
    con.execute("PRAGMA busy_timeout=5000")
    return con

def _ts_iso(ts_ms: int) -> str:
    return pd.Timestamp(int(ts_ms), unit="ms", tz="UTC").isoformat()

def _strip_warmup(df_feat: pd.DataFrame) -> pd.DataFrame:
    if "sma_200" not in df_feat.columns or df_feat.empty:
        return df_feat.reset_index(drop=True)
    sma = pd.to_numeric(df_feat["sma_200"], errors="coerce").to_numpy("float64")
    valid = np.isfinite(sma) & (sma > 0.0)
    return df_feat.loc[valid].reset_index(drop=True) if np.any(valid) else df_feat.reset_index(drop=True)

def load_candles_ro(cache_db: str, symbol: str, timeframe: str) -> pd.DataFrame:
    tf_ms, now_ms = TIMEFRAME_MS[timeframe], int(time.time() * 1000)
    sql = (
        "SELECT exchange, timestamp, open, high, low, close, volume FROM candles WHERE exchange='binance' AND symbol=? AND timeframe='1h' ORDER BY timestamp ASC"
        if timeframe == "1h" else
        "SELECT exchange, timestamp, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe=? ORDER BY timestamp ASC, CASE WHEN LOWER(exchange)='binance' THEN 0 ELSE 1 END ASC"
    )
    con = _connect_ro(cache_db)
    try:
        rows = con.execute(sql, (symbol,) if timeframe == "1h" else (symbol, timeframe)).fetchall()
    finally:
        con.close()
    cols = ["exchange", "timestamp", "open", "high", "low", "close", "volume"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols).drop_duplicates("timestamp", keep="first").sort_values("timestamp").reset_index(drop=True)
    return df.iloc[:-1].reset_index(drop=True) if len(df) > 1 and (int(df["timestamp"].iloc[-1]) + tf_ms) > now_ms else df

def load_suite_features(cache_db: str, timeframe: str, symbols: Sequence[str] = MAJORS, df_1h_btc: Optional[pd.DataFrame] = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    v1_map, v2_map = {}, {}
    btc_1h = df_1h_btc if df_1h_btc is not None else load_candles_ro(cache_db, "BTCUSDT", "1h")
    for sym in symbols:
        raw = btc_1h.copy() if (timeframe == "1h" and sym == "BTCUSDT" and not btc_1h.empty) else load_candles_ro(cache_db, sym, timeframe)
        if raw.empty:
            raise RuntimeError(f"No {timeframe} candles found for {sym} in {cache_db}")
        v1_map[sym] = _strip_warmup(FeatureEngineer.compute_all_features(raw.drop(columns=["exchange"], errors="ignore"), drop_na=False))
        v2_map[sym] = _strip_warmup(build_v2_features_batch(raw, sym, timeframe=timeframe, cache_db_path=cache_db, df_1h_btc=btc_1h))
    return v1_map, v2_map

def _predict_proba_3(est: Any, X: pd.DataFrame) -> np.ndarray:
    raw, classes = np.asarray(est.predict_proba(X), dtype=np.float64), getattr(est, "classes_", np.array([0, 1, 2]))
    if raw.shape[1] == 3 and np.array_equal(classes, [0, 1, 2]):
        return raw
    out = np.zeros((len(X), 3), dtype=np.float64)
    for col_i, cls_val in enumerate(classes):
        if int(cls_val) in (0, 1, 2) and col_i < raw.shape[1]:
            out[:, int(cls_val)] = raw[:, col_i]
    return out

def _sanitize_feature_frame(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Reindex feature columns and fill non-finite values with explicit domain neutral defaults."""
    out = df.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan)
    fill_map = {c: _NEUTRAL_DEFAULTS.get(c, 0.0) for c in feature_cols}
    return out.fillna(value=fill_map)


def _build_pooled_labeled(
    feats_map: Dict[str, pd.DataFrame], symbols: Sequence[str], feature_cols: List[str],
    horizon: int, multiple_tp: float, multiple_sl: float, cutoff_ms: Optional[int] = None, embargo_ms: int = 0,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    x_parts, y_parts, ts_parts = [], [], []
    for sym in symbols:
        df_sym = feats_map[sym]
        sub = df_sym[df_sym["timestamp"] < int(cutoff_ms)].reset_index(drop=True) if cutoff_ms is not None else df_sym.reset_index(drop=True)
        if len(sub) <= horizon + 5:
            continue
        tgt = create_triple_barrier_labels(df=sub, horizon=horizon, multiple_tp=multiple_tp, multiple_sl=multiple_sl)
        valid = tgt.notna().to_numpy()
        if cutoff_ms is not None:
            valid = valid & (sub["timestamp"].to_numpy(dtype=np.int64) < int(cutoff_ms - embargo_ms))
        if not np.any(valid):
            continue
        x_parts.append(_sanitize_feature_frame(sub.loc[valid], feature_cols))
        y_parts.append(tgt.loc[valid].astype(int))
        ts_parts.append(sub.loc[valid, "timestamp"].to_numpy(dtype=np.int64))
    if not x_parts:
        raise RuntimeError("Insufficient samples after purge/embargo.")
    X_pool, y_pool, ts_all = pd.concat(x_parts, ignore_index=True), pd.concat(y_parts, ignore_index=True), np.concatenate(ts_parts)
    order = np.argsort(ts_all, kind="mergesort")
    return X_pool.iloc[order].reset_index(drop=True), y_pool.iloc[order].reset_index(drop=True), ts_all[order]

def _compute_feature_health(feature_cols: List[str], est_lgbm: Any, est_rf: Any) -> Dict[str, Any]:
    imp_l = np.asarray(getattr(est_lgbm, "feature_importances_", np.zeros(len(feature_cols))), dtype=np.float64)
    imp_r = np.asarray(getattr(est_rf, "feature_importances_", np.zeros(len(feature_cols))), dtype=np.float64)
    z_l, z_r = imp_l <= 0.0, imp_r <= 0.0
    pct_l = (imp_l / imp_l.sum() * 100.0) if imp_l.sum() > 0 else np.zeros_like(imp_l)
    pct_r = (imp_r / imp_r.sum() * 100.0) if imp_r.sum() > 0 else np.zeros_like(imp_r)
    avg_pct = 0.5 * (pct_l + pct_r)
    top_10 = [{"feature": feature_cols[int(i)], "avg_pct": round(float(avg_pct[int(i)]), 4), "lgbm_pct": round(float(pct_l[int(i)]), 4), "rf_pct": round(float(pct_r[int(i)]), 4)} for i in np.argsort(-avg_pct)[:10]]
    return {"n_features": len(feature_cols), "zero_imp_lgbm": int(np.sum(z_l)), "zero_imp_rf": int(np.sum(z_r)), "zero_imp_both": int(np.sum(z_l & z_r)), "zero_imp_either": int(np.sum(z_l | z_r)), "top_10_features": top_10}

def _r_stats(r_list: Sequence[float]) -> Dict[str, Any]:
    if not r_list:
        return {"n_trades": 0, "win_rate": 0.0, "win_rate_pct": 0.0, "profit_factor": 0.0, "expectancy_r": 0.0, "total_net_r": 0.0}
    arr = np.asarray(r_list, dtype=np.float64)
    gp, gl, wr = float(arr[arr > 0].sum()), float(np.abs(arr[arr < 0]).sum()), float(np.mean(arr > 0))
    pf = gp / gl if gl > 1e-12 else (99.0 if gp > 0 else 0.0)
    return {"n_trades": len(r_list), "win_rate": round(wr, 4), "win_rate_pct": round(wr * 100.0, 2), "profit_factor": round(pf, 4), "expectancy_r": round(float(np.mean(arr)), 4), "total_net_r": round(float(np.sum(arr)), 4)}

def simulate_oos_candidate(
    feats_map: Dict[str, pd.DataFrame], feature_cols: List[str], est_lgbm: Any, est_rf: Any,
    spec: CandidateSpec, holdout_start_ms: int, holdout_end_ms: int, eval_symbols: Sequence[str] = MAJORS,
) -> Tuple[Dict[str, Any], np.ndarray]:
    cooldown_bars = 2 if spec.timeframe == "1h" else 4
    H, tp_m, sl_m, thresh = spec.horizon, spec.multiple_tp, spec.multiple_sl, spec.decision_threshold
    start_day_ms = (int(holdout_start_ms) // DAY_MS) * DAY_MS
    n_days = int((max(start_day_ms, (int(holdout_end_ms) // DAY_MS) * DAY_MS) - start_day_ms) // DAY_MS) + 1
    all_events, all_r, per_symbol = [], [], {}

    for sym in eval_symbols:
        df_oos = feats_map[sym][feats_map[sym]["timestamp"] >= int(holdout_start_ms)].reset_index(drop=True)
        n = len(df_oos)
        if n <= H + 1:
            per_symbol[sym] = _r_stats([])
            continue
        X_oos = _sanitize_feature_frame(df_oos, feature_cols)
        proba = 0.5 * (_predict_proba_3(est_lgbm, X_oos) + _predict_proba_3(est_rf, X_oos))
        p_buy, p_sell, ts = proba[:, 1], proba[:, 2], df_oos["timestamp"].to_numpy(dtype=np.int64)
        highs, lows = pd.to_numeric(df_oos["high"], errors="coerce").to_numpy("float64"), pd.to_numeric(df_oos["low"], errors="coerce").to_numpy("float64")
        closes, atrs = pd.to_numeric(df_oos["close"], errors="coerce").to_numpy("float64"), pd.to_numeric(df_oos["atr_14"], errors="coerce").to_numpy("float64")

        sym_r, next_free_idx = [], 0
        for i in range(0, n - H):
            if i < next_free_idx:
                continue
            sgn = 1.0 if (p_buy[i] >= thresh and p_buy[i] > p_sell[i]) else (-1.0 if (p_sell[i] >= thresh and p_sell[i] > p_buy[i]) else 0.0)
            c0, atr0 = closes[i], atrs[i]
            if sgn == 0.0 or not (math.isfinite(c0) and c0 > 0.0 and math.isfinite(atr0) and atr0 > 0.0):
                continue
            entry_px = c0 * (1.0 + sgn * MARKET_SLIPPAGE_PCT)
            sl_px, tp_px = entry_px - sgn * sl_m * atr0, entry_px + sgn * tp_m * atr0
            if sl_px <= 0.0 or tp_px <= 0.0:
                continue
            risk_pct = max(abs(entry_px - sl_px) / entry_px, 1e-4)
            exit_idx, exit_px, exit_fee = i + H, closes[i + H] * (1.0 - sgn * MARKET_SLIPPAGE_PCT), TAKER_FEE
            for j in range(i + 1, i + H + 1):
                if (lows[j] <= sl_px) if sgn > 0.0 else (highs[j] >= sl_px):
                    exit_px, exit_fee, exit_idx = sl_px * (1.0 - sgn * MARKET_SLIPPAGE_PCT), TAKER_FEE, j
                    break
                if (highs[j] >= tp_px) if sgn > 0.0 else (lows[j] <= tp_px):
                    exit_px, exit_fee, exit_idx = tp_px * (1.0 - sgn * MAKER_SLIPPAGE_PCT), MAKER_FEE, j
                    break
            r_mult = float((sgn * (exit_px - entry_px) / entry_px - TAKER_FEE - exit_fee) / risk_pct)
            sym_r.append(r_mult)
            all_r.append(r_mult)
            all_events.append((int(ts[exit_idx]), r_mult))
            next_free_idx = exit_idx + cooldown_bars
        per_symbol[sym] = _r_stats(sym_r)

    daily_mult = np.ones(n_days, dtype=np.float64)
    for exit_ts, r_mult in sorted(all_events, key=lambda x: x[0]):
        daily_mult[min(max(int((exit_ts - start_day_ms) // DAY_MS), 0), n_days - 1)] *= 1.0 + max(RISK_PER_TRADE * r_mult, -0.95)
    daily_rets = daily_mult - 1.0
    eq = np.concatenate([[1.0], np.cumprod(1.0 + daily_rets)])
    peaks = np.maximum.accumulate(eq)
    max_dd, sharpe = float(np.min((eq - peaks) / np.maximum(peaks, 1e-12))), calculate_annualized_sharpe(daily_rets, periods_per_year=365.0)
    boot = bootstrap_performance_metrics(daily_rets, n_iterations=1000, periods_per_year=365.0, seed=42, block_size=7)
    oos = {
        **_r_stats(all_r), "total_return_pct": round(float((eq[-1] - 1.0) * 100.0), 4), "n_calendar_days": n_days,
        "active_days": int(np.sum(np.abs(daily_rets) > 1e-12)), "sharpe": round(sharpe, 4), "max_dd": round(max_dd, 4),
        "bootstrap_ci95": {k: [round(float(v[0]), 4), round(float(v[1]), 4)] for k, v in boot.items()}, "per_symbol": per_symbol,
    }
    return oos, daily_rets

def _evaluate_candidate(
    spec: CandidateSpec, feats_map: Dict[str, pd.DataFrame], feature_cols: List[str],
    split_ms: int, holdout_end_ms: int, embargo_ms: int, models_dir: str,
) -> Dict[str, Any]:
    X_dev, y_dev, ts_dev = _build_pooled_labeled(feats_map, spec.train_symbols, feature_cols, spec.horizon, spec.multiple_tp, spec.multiple_sl, cutoff_ms=split_ms, embargo_ms=embargo_ms)
    wf_summary, fitted, drop_wf = {}, {}, {"test_indices", "y_true", "y_pred", "y_proba"}
    tmp_dir = tempfile.gettempdir()
    for mtype in ("lgbm", "rf"):
        tr = ModelTrainer(model_type=mtype, horizon=spec.horizon, multiple_tp=spec.multiple_tp, multiple_sl=spec.multiple_sl, confidence_threshold=spec.decision_threshold, n_jobs=8, models_dir=tmp_dir)
        wf = tr.train_walk_forward(X_dev, y_dev, n_splits=4, timestamps=ts_dev, embargo_ms=embargo_ms)
        wf_summary[mtype] = {k: v for k, v in wf.items() if k not in drop_wf}
        est = _disable_estimator_parallelism(tr._build_estimator(random_state=42).fit(X_dev, y_dev))
        fitted[mtype] = est
    health = _compute_feature_health(feature_cols, fitted["lgbm"], fitted["rf"])
    oos, daily_rets = simulate_oos_candidate(feats_map, feature_cols, fitted["lgbm"], fitted["rf"], spec, split_ms, holdout_end_ms)
    return {
        "name": spec.name, "version": spec.version, "timeframe": spec.timeframe, "train_symbols": list(spec.train_symbols),
        "horizon": spec.horizon, "multiple_tp": spec.multiple_tp, "multiple_sl": spec.multiple_sl, "decision_threshold": spec.decision_threshold,
        "embargo_ms": int(embargo_ms), "dev_samples": int(len(X_dev)), "n_features": health["n_features"],
        "zero_imp_lgbm": health["zero_imp_lgbm"], "zero_imp_rf": health["zero_imp_rf"], "zero_imp_both": health["zero_imp_both"],
        "feature_health": health, "walk_forward_dev": wf_summary, "oos": oos, "_daily_rets": daily_rets,
    }

def classify_suite_gate(candidates: List[Dict[str, Any]], suite_pbo: float) -> Dict[str, Any]:
    v1 = next(c for c in candidates if c["version"] == "v1")
    v2_cands = [c for c in candidates if c["version"] == "v2"]
    eligible = [c for c in v2_cands if c["oos"]["n_trades"] >= 20]
    key_fn = (lambda c: (c["oos"]["profit_factor"], c["oos"]["expectancy_r"], c["oos"]["sharpe"])) if eligible else (lambda c: (c["oos"]["n_trades"], c["oos"]["profit_factor"], c["oos"]["expectancy_r"]))
    cand = sorted(eligible or v2_cands, key=key_fn, reverse=True)[0]
    gate_min_trades = bool(eligible)
    gate_zero_imp = bool(cand["zero_imp_lgbm"] == 0 and cand["zero_imp_rf"] == 0)
    gate_beats_v1 = bool(cand["oos"]["profit_factor"] > v1["oos"]["profit_factor"] and cand["oos"]["expectancy_r"] > v1["oos"]["expectancy_r"] and cand["oos"]["expectancy_r"] > 0.0)
    gate_pbo, gate_dsr = bool(suite_pbo < 0.20), bool(cand["dsr"]["dsr"] >= 0.95)
    decision = ("PASS_ALL" if gate_dsr else "PARTIAL_DSR_BLOCK") if (gate_min_trades and gate_zero_imp and gate_beats_v1 and gate_pbo) else "FAIL_REFUTED"
    return {"best_v2_candidate": cand["name"], "gate_min_trades_20": gate_min_trades, "gate_zero_imp": gate_zero_imp, "gate_beats_v1": gate_beats_v1, "gate_pbo": gate_pbo, "gate_dsr": gate_dsr, "decision": decision}

def _align_series_le(src_ts: np.ndarray, vals: np.ndarray, close_ts: np.ndarray, max_age_ms: int = 2 * HOUR_MS) -> np.ndarray:
    if len(src_ts) == 0:
        return np.zeros(len(close_ts), dtype=np.float64)
    idx = np.searchsorted(src_ts, close_ts, side="right") - 1
    sidx = np.maximum(idx, 0)
    return np.where((idx >= 0) & ((close_ts - src_ts[sidx]) <= max_age_ms), vals[sidx], 0.0)

def load_microstructure_features_15m(v2_15m_map: Dict[str, pd.DataFrame], micro_db: str, start_ms: int = MICRO_ABLATION_START_MS) -> Optional[Dict[str, pd.DataFrame]]:
    if not micro_db or not os.path.exists(micro_db):
        return None
    con = _connect_ro(micro_db)
    try:
        if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='binance_series'").fetchone():
            return None
        rows = con.execute("SELECT ts_ms, symbol, kind, v1, v2, v3 FROM binance_series WHERE symbol IN ('BTCUSDT','ETHUSDT','SOLUSDT') AND kind IN ('oi','ls_global','taker') ORDER BY ts_ms ASC").fetchall()
    finally:
        con.close()
    if not rows:
        return None

    mdf = pd.DataFrame(rows, columns=["ts_ms", "symbol", "kind", "v1", "v2", "v3"])
    mdf["ts_ms"] = pd.to_numeric(mdf["ts_ms"], errors="coerce").astype("int64")
    eff_start_ms = int(start_ms) if max(int(df["timestamp"].max()) for df in v2_15m_map.values()) >= int(start_ms) else int(mdf["ts_ms"].min())
    enriched: Dict[str, pd.DataFrame] = {}

    for sym, base_df in v2_15m_map.items():
        sub = base_df[base_df["timestamp"] >= eff_start_ms].copy().reset_index(drop=True)
        if sub.empty:
            return None
        open_ts = sub["timestamp"].to_numpy(dtype=np.int64)
        close_ts, s_df = open_ts + M15_MS, mdf[mdf["symbol"] == sym]
        for col in MICRO_EXTRA_COLS:
            sub[col] = 0.0

        oi = s_df[s_df["kind"] == "oi"].drop_duplicates("ts_ms", keep="last")
        if not oi.empty:
            oi_ts, oi_v = oi["ts_ms"].to_numpy(dtype=np.int64), pd.to_numeric(oi["v1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            p_idx = np.searchsorted(oi_ts, oi_ts - HOUR_MS, side="right") - 1
            ps = np.maximum(p_idx, 0)
            oi_chg = np.where((p_idx >= 0) & ((oi_ts - oi_ts[ps]) <= 2 * HOUR_MS) & (oi_v[ps] > 0), (oi_v - oi_v[ps]) / (oi_v[ps] + 1e-8), 0.0)
            sub["ms_oi_chg_1h"] = _align_series_le(oi_ts, oi_chg, close_ts)

        ls = s_df[s_df["kind"] == "ls_global"].drop_duplicates("ts_ms", keep="last")
        if not ls.empty:
            ls_ts, ls_s = ls["ts_ms"].to_numpy(dtype=np.int64), pd.to_numeric(ls["v1"], errors="coerce").reset_index(drop=True)
            m, sd = ls_s.rolling(288, min_periods=12).mean(), ls_s.rolling(288, min_periods=12).std()
            sub["ms_ls_global_z"] = _align_series_le(ls_ts, np.asarray(np.where(m.notna() & (sd > 1e-8), (ls_s - m) / sd, 0.0), dtype=np.float64), close_ts)

        tk = s_df[s_df["kind"] == "taker"].drop_duplicates("ts_ms", keep="last")
        if not tk.empty:
            end_ts = tk["ts_ms"].to_numpy(dtype=np.int64) + 300_000
            cs_b = np.concatenate([[0.0], np.cumsum(np.maximum(pd.to_numeric(tk["v2"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0))])
            cs_s = np.concatenate([[0.0], np.cumsum(np.maximum(pd.to_numeric(tk["v3"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64), 0.0))])
            hi, lo = np.searchsorted(end_ts, close_ts, side="right"), np.searchsorted(end_ts, open_ts + 300_000, side="left")
            bs, ss = cs_b[hi] - cs_b[lo], cs_s[hi] - cs_s[lo]
            denom_tk = bs + ss
            sub["ms_taker_imb_15m"] = np.divide(bs - ss, denom_tk, out=np.zeros_like(denom_tk), where=denom_tk > 1e-8)

        for col in MICRO_EXTRA_COLS:
            arr = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=np.float64)
            sub[col] = np.where(np.isfinite(arr), arr, 0.0)
        enriched[sym] = sub
    return enriched

def run_micro_ablation_15m(v2_15m_map: Dict[str, pd.DataFrame], micro_db: str, models_dir: str) -> Dict[str, Any]:
    enriched = load_microstructure_features_15m(v2_15m_map, micro_db)
    if not enriched:
        return {"status": "SKIPPED_NO_MICRO_DATA", "informational_only": True}
    min_ts, max_ts = min(int(df["timestamp"].min()) for df in enriched.values()), max(int(df["timestamp"].max()) for df in enriched.values())
    split_ms, embargo_ms = min_ts + int((max_ts - min_ts) * 0.67), (16 + 10) * M15_MS
    s0 = CandidateSpec("ABLATION_15M_H16_NO_MICRO", "v2", "15m", MAJORS, 16, 2.0, 1.4, 0.45)
    s1 = CandidateSpec("ABLATION_15M_H16_WITH_MICRO", "v2_micro", "15m", MAJORS, 16, 2.0, 1.4, 0.45)
    try:
        r0 = _evaluate_candidate(s0, enriched, list(V2_FEATURE_COLUMNS), split_ms, max_ts, embargo_ms, models_dir)
        r1 = _evaluate_candidate(s1, enriched, [*V2_FEATURE_COLUMNS, *MICRO_EXTRA_COLS], split_ms, max_ts, embargo_ms, models_dir)
    except Exception as exc:
        return {"status": f"SKIPPED_INSUFFICIENT_WINDOW: {exc}", "informational_only": True}
    b_oos, m_oos = r0["oos"], r1["oos"]
    delta = {
        "n_trades_delta": int(m_oos["n_trades"] - b_oos["n_trades"]), "profit_factor_delta": round(float(m_oos["profit_factor"] - b_oos["profit_factor"]), 4),
        "expectancy_r_delta": round(float(m_oos["expectancy_r"] - b_oos["expectancy_r"]), 4), "sharpe_delta": round(float(m_oos["sharpe"] - b_oos["sharpe"]), 4),
        "win_rate_pct_delta": round(float(m_oos["win_rate_pct"] - b_oos["win_rate_pct"]), 2),
    }
    return {
        "status": "OK", "informational_only": True, "window_start_utc": _ts_iso(min_ts), "split_utc": _ts_iso(split_ms), "window_end_utc": _ts_iso(max_ts),
        "without_micro": {"oos": b_oos, "feature_health": r0["feature_health"]}, "with_micro": {"oos": m_oos, "feature_health": r1["feature_health"]}, "oos_delta": delta,
    }

def promote_suite_models(suite_name: str, suite_res: Dict[str, Any], v2_feats_map: Dict[str, pd.DataFrame], models_dir: str) -> Dict[str, Any]:
    best_name = suite_res["gate"]["best_v2_candidate"]
    best_cand = next(c for c in suite_res["candidates"] if c["name"] == best_name)
    X_full, y_full, ts_full = _build_pooled_labeled(v2_feats_map, MAJORS, list(V2_FEATURE_COLUMNS), best_cand["horizon"], best_cand["multiple_tp"], best_cand["multiple_sl"])
    archive_dir = os.path.join(models_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    promoted_files, backed_up_files = [], []

    for mtype in ("lgbm", "rf"):
        save_name = f"{mtype}_model_v1.joblib" if suite_name == "1h" else f"{mtype}_15m_v1.joblib"
        target_path = os.path.join(models_dir, save_name)
        if os.path.exists(target_path):
            bak_path = os.path.join(archive_dir, f"{save_name[:-7]}_{ts_tag}.joblib")
            shutil.copy2(target_path, bak_path)
            backed_up_files.append(bak_path)
        trainer = ModelTrainer(model_type=mtype, horizon=best_cand["horizon"], multiple_tp=best_cand["multiple_tp"], multiple_sl=best_cand["multiple_sl"], confidence_threshold=best_cand["decision_threshold"], n_jobs=8, models_dir=models_dir)
        extra_meta = {"schema_version": "2.0.0", "version": "2.0.0", "feature_pipeline": "ml_features_v2", "timeframe": suite_name, "symbols": list(MAJORS), "candidate_name": best_cand["name"], "pbo": suite_res["pbo"]["pbo"], "dsr": best_cand["dsr"], "oos_metrics": best_cand["oos"]}
        payload = trainer.train_full_model(X_full, y_full, save_name=save_name, timestamps=ts_full, embargo_ms=best_cand["embargo_ms"], extra_metadata=extra_meta)
        _disable_estimator_parallelism(payload["model"])
        _atomic_joblib_dump(payload, target_path, compress=3)
        loaded = joblib.load(target_path)
        proba = _predict_proba_3(loaded["model"], X_full.iloc[: min(8, len(X_full))])
        if not is_v2_feature_bundle(loaded) or not np.all(np.isfinite(proba)):
            raise RuntimeError(f"Promoted artifact validation failed: {target_path}")
        promoted_files.append(target_path)
    return {"promoted": True, "candidate": best_name, "files": promoted_files, "backups": backed_up_files}

def _json_clean(obj: Any) -> Any:
    if isinstance(obj, dict): return {str(k): _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_json_clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)): return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, (np.floating, float)): return round(float(obj), 6) if math.isfinite(float(obj)) else 0.0
    return obj

def _print_summary(report: Dict[str, Any]) -> None:
    print("\n=== WAVE P2-B ZERO-LEAKAGE ML v1 vs v2 BENCHMARK ===", flush=True)
    for sname in ("1h", "15m"):
        s, g = report["suites"][sname], report["suites"][sname]["gate"]
        print(f"\n[{sname.upper()} SUITE] Holdout: {s['holdout_start_utc'][:10]} -> {s['holdout_end_utc'][:10]} | PBO={s['pbo']['pbo']:.3f} | Best v2={g['best_v2_candidate']} | Decision={g['decision']}", flush=True)
        print(f"  {'Candidate':<18} {'Ver':<4} {'N':>5} {'WR%':>6} {'PF':>7} {'E[R]':>8} {'NetR':>8} {'Sharpe':>7} {'DSR':>6} {'ZImp(L/R)':>10}", flush=True)
        for c in s["candidates"]:
            o = c["oos"]
            print(f"  {c['name']:<18} {c['version']:<4} {o['n_trades']:>5d} {o['win_rate_pct']:>6.2f} {o['profit_factor']:>7.3f} {o['expectancy_r']:>+8.4f} {o['total_net_r']:>+8.2f} {o['sharpe']:>7.3f} {c['dsr']['dsr']:>6.3f} {c['zero_imp_lgbm']:>4d}/{c['zero_imp_rf']:<4d}", flush=True)
    ab = report["microstructure_ablation_15m"]
    if ab.get("status") == "OK":
        d = ab["oos_delta"]
        print(f"\n[15M 66D MICROSTRUCTURE ABLATION (INFO ONLY)] {ab['window_start_utc'][:10]} -> {ab['window_end_utc'][:10]} | dPF={d['profit_factor_delta']:+.4f} dE[R]={d['expectancy_r_delta']:+.4f} dSharpe={d['sharpe_delta']:+.4f}", flush=True)

def run_benchmark(
    cache_db: str, micro_db: str, models_dir: str, report_path: str,
    log_trials: bool = False, promote_1h: bool = False, promote_15m: bool = False,
    dry: bool = False, holdout_1h_start_ms: int = HOLDOUT_1H_START_MS,
) -> Dict[str, Any]:
    n_prior_trials = get_prior_trials_count()
    n_trials_total = n_prior_trials + len(CANDIDATES_1H) + len(CANDIDATES_15M)
    btc_1h_raw = load_candles_ro(cache_db, "BTCUSDT", "1h")
    v1_1h, v2_1h = load_suite_features(cache_db, "1h", MAJORS, df_1h_btc=btc_1h_raw)
    v1_15m, v2_15m = load_suite_features(cache_db, "15m", MAJORS, df_1h_btc=btc_1h_raw)

    max_1h_ts = max(int(df["timestamp"].max()) for df in v2_1h.values())
    min_15m_ts, max_15m_ts = min(int(df["timestamp"].min()) for df in v2_15m.values()), max(int(df["timestamp"].max()) for df in v2_15m.values())
    split_15m_ms = min_15m_ts + int((max_15m_ts - min_15m_ts) * 0.67)

    cands_1h = [_evaluate_candidate(s, v1_1h if s.version == "v1" else v2_1h, V1_FEATURE_COLUMNS if s.version == "v1" else list(V2_FEATURE_COLUMNS), holdout_1h_start_ms, max_1h_ts, (s.horizon + 5) * HOUR_MS, models_dir) for s in CANDIDATES_1H]
    cands_15m = [_evaluate_candidate(s, v1_15m if s.version == "v1" else v2_15m, V1_FEATURE_COLUMNS if s.version == "v1" else list(V2_FEATURE_COLUMNS), split_15m_ms, max_15m_ts, (s.horizon + 10) * M15_MS, models_dir) for s in CANDIDATES_15M]

    all_cands = cands_1h + cands_15m
    sharpes = [float(c["oos"]["sharpe"]) for c in all_cands]
    var_trials = max(float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.25, 0.25)
    for c in all_cands:
        c["dsr"] = deflated_sharpe_ratio(c["oos"]["sharpe"], c["_daily_rets"], n_trials=n_trials_total, variance_of_trials=var_trials, periods_per_year=365.0)

    suites_out: Dict[str, Any] = {}
    for sname, cands, s_start, s_end in (("1h", cands_1h, holdout_1h_start_ms, max_1h_ts), ("15m", cands_15m, split_15m_ms, max_15m_ts)):
        pbo_res = probability_of_backtest_overfitting(np.column_stack([c["_daily_rets"] for c in cands]), n_splits=10, periods_per_year=365.0)
        pbo_res["logits"] = [round(float(x), 4) for x in pbo_res.get("logits", [])]
        for c in cands:
            c.pop("_daily_rets", None)
        gate = classify_suite_gate(cands, float(pbo_res["pbo"]))
        suites_out[sname] = {"timeframe": sname, "holdout_start_ms": int(s_start), "holdout_start_utc": _ts_iso(s_start), "holdout_end_ms": int(s_end), "holdout_end_utc": _ts_iso(s_end), "pbo": pbo_res, "gate": gate, "decision": gate["decision"], "candidates": cands, "promotion": {"promoted": False, "requested": bool(promote_1h if sname == "1h" else promote_15m)}}

    if log_trials and not dry:
        for sname, sdata in suites_out.items():
            for c in sdata["candidates"]:
                o = c["oos"]
                log_trial(phase="Wave_P2B", experiment=f"ML_V2_BENCH_{sname.upper()}", strategy_name=c["name"], symbol="BTC/ETH/SOL" if len(c["train_symbols"]) > 1 else "BTCUSDT", timeframe=sname, params={"version": c["version"], "H": c["horizon"], "tp": c["multiple_tp"], "sl": c["multiple_sl"]}, sample_period=f"{sdata['holdout_start_utc'][:10]}..{sdata['holdout_end_utc'][:10]}", is_oos=True, trades_count=o["n_trades"], net_pnl=o["total_net_r"], net_pnl_pct=o["total_return_pct"], sharpe_ratio=o["sharpe"], max_drawdown_pct=o["max_dd"] * 100.0, win_rate_pct=o["win_rate_pct"], status_label=sdata["decision"], notes=f"PF={o['profit_factor']};PBO={sdata['pbo']['pbo']:.3f};DSR={c['dsr']['dsr']:.3f}")

    if not dry and promote_1h and suites_out["1h"]["decision"] == "PASS_ALL":
        suites_out["1h"]["promotion"] = promote_suite_models("1h", suites_out["1h"], v2_1h, models_dir)
    if not dry and promote_15m and suites_out["15m"]["decision"] == "PASS_ALL":
        suites_out["15m"]["promotion"] = promote_suite_models("15m", suites_out["15m"], v2_15m, models_dir)

    report = _json_clean({"generated_at_utc": datetime.now(timezone.utc).isoformat(), "prior_trials_count": n_prior_trials, "total_trials_dsr": n_trials_total, "variance_of_trials_used": var_trials, "suites": suites_out, "microstructure_ablation_15m": run_micro_ablation_15m(v2_15m, micro_db, models_dir)})
    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, allow_nan=False)
    _print_summary(report)
    return report

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wave P2-B Zero-Leakage ML v1 vs v2 Benchmark & Promotion Gate")
    p.add_argument("--cache-db", default=_default_db_path("market_cache.db"))
    p.add_argument("--micro-db", default=_default_db_path("microstructure.db"))
    p.add_argument("--models-dir", default=os.path.join(ROOT, "models"))
    p.add_argument("--report-path", default=os.path.join(ROOT, "datasets", "reports", "p2_ml_v2_benchmark.json"))
    p.add_argument("--log-trials", action="store_true")
    p.add_argument("--promote-1h", action="store_true")
    p.add_argument("--promote-15m", action="store_true")
    p.add_argument("--dry", action="store_true", help="Force dry benchmark mode (never promote or log trials)")
    return p.parse_args(argv)

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_benchmark(args.cache_db, args.micro_db, args.models_dir, args.report_path, args.log_trials, args.promote_1h, args.promote_15m, args.dry)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
