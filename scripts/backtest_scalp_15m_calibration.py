"""scripts/backtest_scalp_15m_calibration.py - Purged Walk-Forward IS->OOS Calibration para SCALP 15m.

Zero vazamento temporal / zero seleção no conjunto de teste:
1. Divide o histórico estritamente no tempo: IS (`< split_ts`, 67%) e OOS holdout (`>= split_ts`, 33%).
2. Dentro da janela IS, treina LGBM+RF (`ModelTrainer._build_estimator`) no sub-bloco [0..45%] (-embargo)
   e ranqueia a grade de thresholds exclusivamente na validação interna IS [45%..67%].
3. Treina o ensemble IS completo (`< split_ts - embargo`) e avalia UMA vez no holdout OOS (`>= split_ts`).
4. Reporta também a janela pós-treino (`ts >= created_at`) dos artefatos `.joblib` carregados em disco.
"""
from __future__ import annotations

import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.technical_agent import _disable_estimator_parallelism
from data.feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from scripts.train_15m_scalp_models import select_stationary_features
from smart_trade import MAKER_SLIPPAGE_PCT, MARKET_SLIPPAGE_PCT, STRATEGY_PROFILES

CACHE_DB = os.path.join(ROOT, "datasets", "market_cache.db")
MODELS_DIR = os.path.join(ROOT, "models")
MAJORS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SCALP_BAR_MS = 900_000
H1_BAR_MS = 3_600_000
MAKER_FEE = 0.0002
TAKER_FEE = 0.0006
COOLDOWN_BARS = 4
EMBARGO_BARS = 10


@dataclass
class PreparedSymbolData:
    symbol: str
    df: pd.DataFrame
    p_buy: np.ndarray
    p_sell: np.ndarray
    natr_bps: np.ndarray
    vol_z: np.ndarray
    in_sqz: np.ndarray
    kc_hi: np.ndarray
    kc_lo: np.ndarray
    h1_dist: np.ndarray


def _load_candles(con: sqlite3.Connection, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    rows = con.execute(
        """
        SELECT timestamp, open, high, low, close, volume
        FROM candles
        WHERE symbol = ? AND timeframe = ?
        ORDER BY timestamp DESC, CASE WHEN exchange = 'binance' THEN 0 ELSE 1 END ASC
        """,
        (symbol, timeframe),
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp").reset_index(drop=True)
    if len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)
    return df


def _load_ml15_bundles() -> List[Dict]:
    bundles = []
    for fname in ("lgbm_15m_v1.joblib", "rf_15m_v1.joblib"):
        p = os.path.join(MODELS_DIR, fname)
        if os.path.exists(p):
            b = joblib.load(p)
            if isinstance(b, dict) and b.get("model") is not None and b.get("feature_names"):
                _disable_estimator_parallelism(b["model"])
                bundles.append(b)
    return bundles


def _fit_purged_bundles(feats_by_sym: Dict[str, pd.DataFrame], feature_cols: List[str], cutoff_ts: int) -> List[Dict]:
    embargo_ts = int(cutoff_ts) - EMBARGO_BARS * SCALP_BAR_MS
    prep = ModelTrainer(model_type="lgbm", horizon=5, multiple_tp=1.2, multiple_sl=1.2, n_jobs=8)
    x_parts, y_parts, ts_parts = [], [], []
    for sym in MAJORS:
        dff = feats_by_sym[sym]
        sub = dff[dff["timestamp"] < embargo_ts].reset_index(drop=True)
        if len(sub) < 100:
            continue
        Xs, ys = prep.prepare_dataset(sub, feature_columns=feature_cols)
        Xs = Xs.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        x_parts.append(Xs)
        y_parts.append(ys)
        ts_parts.append(sub.loc[Xs.index, "timestamp"].to_numpy(dtype=np.int64))
    X_pool = pd.concat(x_parts, axis=0, ignore_index=True)
    y_pool = pd.concat(y_parts, axis=0, ignore_index=True)
    order = np.argsort(np.concatenate(ts_parts), kind="mergesort")
    X_tr, y_tr = X_pool.iloc[order].reset_index(drop=True), y_pool.iloc[order].reset_index(drop=True)

    bundles = []
    for mtype in ("lgbm", "rf"):
        tr = ModelTrainer(model_type=mtype, horizon=5, multiple_tp=1.2, multiple_sl=1.2, n_jobs=8)
        tr.device = "cpu"
        est = tr._build_estimator(random_state=42)
        est.fit(X_tr, y_tr)
        _disable_estimator_parallelism(est)
        bundles.append({"model": est, "feature_names": feature_cols})
    return bundles


def _build_prepared(symbol: str, df_feat: pd.DataFrame, df1h: pd.DataFrame, bundles: List[Dict]) -> PreparedSymbolData:
    h1_close = df1h["close"].astype(float)
    h1_sma200 = h1_close.rolling(200, min_periods=200).mean()
    h1_dist_arr = ((h1_close - h1_sma200) / h1_sma200).to_numpy(dtype=float)
    h1_close_ts = df1h["timestamp"].to_numpy(dtype=np.int64) + H1_BAR_MS
    target_close_ts = df_feat["timestamp"].to_numpy(dtype=np.int64) + SCALP_BAR_MS
    idx_h1 = np.searchsorted(h1_close_ts, target_close_ts, side="right") - 1
    h1_dist = np.where(idx_h1 >= 0, h1_dist_arr[np.maximum(idx_h1, 0)], np.nan)

    p_buys, p_sells = [], []
    for b in bundles:
        fnames = list(b["feature_names"])
        X = df_feat.reindex(columns=fnames, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        proba = np.asarray(b["model"].predict_proba(X), dtype=float)
        p_buys.append(proba[:, 1])
        p_sells.append(proba[:, 2])
    p_buy = np.mean(p_buys, axis=0) if p_buys else np.zeros(len(df_feat))
    p_sell = np.mean(p_sells, axis=0) if p_sells else np.zeros(len(df_feat))

    c = df_feat["close"].astype(float).to_numpy()
    atr = df_feat["atr_14"].astype(float).to_numpy()
    bb_hi = df_feat["bb_high"].astype(float).to_numpy()
    bb_lo = df_feat["bb_low"].astype(float).to_numpy()
    if "kc_mid_20" in df_feat.columns:
        kc_mid = df_feat["kc_mid_20"].astype(float).to_numpy()
    else:
        kc_mid = df_feat["close"].astype(float).ewm(span=20, adjust=False).mean().to_numpy()
    kc_hi = kc_mid + 1.5 * atr
    kc_lo = kc_mid - 1.5 * atr
    in_sqz = (bb_hi < kc_hi) & (bb_lo > kc_lo)
    natr_bps = (atr / np.where(c > 0, c, np.nan)) * 10_000.0
    vol_z = df_feat["volume_zscore_20"].astype(float).to_numpy()

    return PreparedSymbolData(
        symbol=symbol, df=df_feat, p_buy=p_buy, p_sell=p_sell,
        natr_bps=natr_bps, vol_z=vol_z, in_sqz=in_sqz,
        kc_hi=kc_hi, kc_lo=kc_lo, h1_dist=h1_dist,
    )


def simulate_symbol(
    data: PreparedSymbolData,
    natr_min: float,
    vol_z_min: float,
    ml_min: float,
    ts_start: int = 0,
    ts_end: int = 10**18,
) -> List[float]:
    prof = STRATEGY_PROFILES["SCALP"]
    df = data.df
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    atrs = df["atr_14"].to_numpy(dtype=float)

    n = len(df)
    pnls_r: List[float] = []
    next_free_idx = 3

    for i in range(3, n - 2):
        if ts[i] < ts_start or ts[i] >= ts_end or i < next_free_idx:
            continue
        if not (data.in_sqz[i - 1] and data.in_sqz[i - 2]):
            continue
        if data.natr_bps[i] < natr_min or data.vol_z[i] < vol_z_min or not np.isfinite(data.h1_dist[i]):
            continue

        direction: Optional[str] = None
        if closes[i] > opens[i] and closes[i] > data.kc_hi[i] and data.h1_dist[i] >= 0.0:
            if data.p_buy[i] >= ml_min and data.p_buy[i] > data.p_sell[i]:
                direction = "BUY"
        elif closes[i] < opens[i] and closes[i] < data.kc_lo[i] and data.h1_dist[i] <= 0.0:
            if data.p_sell[i] >= ml_min and data.p_sell[i] > data.p_buy[i]:
                direction = "SELL"

        if direction is None or atrs[i] <= 0.0 or closes[i] <= 0.0:
            continue

        sgn = 1.0 if direction == "BUY" else -1.0
        entry_px = closes[i] * (1.0 + sgn * MAKER_SLIPPAGE_PCT)
        sl_px = entry_px - sgn * prof.sl_atr_mult * atrs[i]
        tp_px = entry_px + sgn * prof.tp3_atr_mult * atrs[i]
        sl_dist = abs(entry_px - sl_px)

        exit_px = closes[min(i + prof.max_holding_bars, n - 1)]
        exit_reason = "TIMEOUT"
        exit_idx = min(i + prof.max_holding_bars, n - 1)

        for j in range(i + 1, min(i + prof.max_holding_bars + 1, n)):
            if direction == "BUY":
                if lows[j] <= sl_px:
                    exit_px, exit_reason, exit_idx = sl_px * (1.0 - MARKET_SLIPPAGE_PCT), "SL", j
                    break
                if highs[j] >= tp_px:
                    exit_px, exit_reason, exit_idx = tp_px * (1.0 - MAKER_SLIPPAGE_PCT), "TP3", j
                    break
            else:
                if highs[j] >= sl_px:
                    exit_px, exit_reason, exit_idx = sl_px * (1.0 + MARKET_SLIPPAGE_PCT), "SL", j
                    break
                if lows[j] <= tp_px:
                    exit_px, exit_reason, exit_idx = tp_px * (1.0 + MAKER_SLIPPAGE_PCT), "TP3", j
                    break

        exit_fee = MAKER_FEE if exit_reason == "TP3" else TAKER_FEE
        gross_ret = sgn * (exit_px - entry_px) / entry_px
        net_ret = gross_ret - MAKER_FEE - exit_fee
        pnls_r.append(float(net_ret / (sl_dist / entry_px)))
        next_free_idx = exit_idx + COOLDOWN_BARS

    return pnls_r


def _metrics(pnls_r: List[float]) -> Dict[str, float]:
    n = len(pnls_r)
    if n == 0:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "sum_r": 0.0}
    arr = np.asarray(pnls_r, dtype=float)
    wins, losses = arr[arr > 0], arr[arr < 0]
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
    return {"n": n, "wr": float(len(wins) / n * 100.0), "pf": pf, "sum_r": float(arr.sum())}


def run_zero_leakage_calibration() -> None:
    con = sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True, timeout=15.0)
    feats_by_sym, h1_by_sym, cols_sets = {}, {}, []
    try:
        for sym in MAJORS:
            df15 = _load_candles(con, sym, "15m", limit=20_000)
            df1h = _load_candles(con, sym, "1h", limit=5_500)
            dff = FeatureEngineer.compute_all_features(df15, drop_na=False)
            dff["kc_mid_20"] = dff["close"].astype(float).ewm(span=20, adjust=False).mean()
            dff = dff[np.isfinite(dff["sma_200"].astype(float)) & (dff["sma_200"].astype(float) > 0.0)].reset_index(drop=True)
            feats_by_sym[sym] = dff
            h1_by_sym[sym] = df1h
            cols_sets.append(set(select_stationary_features(dff)))
    finally:
        con.close()

    feature_cols = [c for c in sorted(set.intersection(*cols_sets)) if c != "kc_mid_20"]
    min_ts = min(int(df["timestamp"].min()) for df in feats_by_sym.values())
    max_ts = max(int(df["timestamp"].max()) for df in feats_by_sym.values())
    is_tr_end = min_ts + int((max_ts - min_ts) * 0.45)
    split_ts = min_ts + int((max_ts - min_ts) * 0.67)

    # 1. Rank STRICTLY on Purged IS validation [0.45 .. 0.67] (zero test-set selection!)
    is_bundles = _fit_purged_bundles(feats_by_sym, feature_cols, is_tr_end)
    prep_is_val = [_build_prepared(s, feats_by_sym[s], h1_by_sym[s], is_bundles) for s in MAJORS]

    # 2. Evaluate ONCE on untouched OOS [0.67 .. 1.00] using full-IS model (< split_ts - embargo)
    oos_bundles = _fit_purged_bundles(feats_by_sym, feature_cols, split_ts)
    prep_oos = [_build_prepared(s, feats_by_sym[s], h1_by_sym[s], oos_bundles) for s in MAJORS]

    is_rows = []
    for natr in (35.0, 28.0, 25.0, 20.0):
        for vz in (1.0, 0.8, 0.6):
            for ml in (0.45, 0.42, 0.40, 0.38):
                r_is = []
                for d in prep_is_val:
                    r_is.extend(simulate_symbol(d, natr, vz, ml, is_tr_end, split_ts))
                m_is = _metrics(r_is)
                is_rows.append({"natr": natr, "vz": vz, "ml": ml, **m_is})

    # Select winner STRICTLY on IS validation metrics
    ranked_is = sorted(is_rows, key=lambda r: (r["n"] >= 8, r["pf"], r["sum_r"]), reverse=True)
    winner = ranked_is[0]
    w_oos = _metrics([x for d in prep_oos for x in simulate_symbol(d, winner["natr"], winner["vz"], winner["ml"], split_ts, 10**18)])
    base_oos = _metrics([x for d in prep_oos for x in simulate_symbol(d, 35.0, 1.0, 0.45, split_ts, 10**18)])

    print("=== IS-SELECTED (Purged [0.45..0.67], ZERO OOS peeking) ===")
    for r in ranked_is[:5]:
        print(f"  natr>={r['natr']:4.0f} vz>={r['vz']:.1f} ml>={r['ml']:.2f} | IS_VAL(n={r['n']:2d}, WR={r['wr']:5.1f}%, PF={r['pf']:5.2f}, R={r['sum_r']:+5.1f})")
    print(f"WINNER ON IS-VAL -> natr>={winner['natr']:.0f}, vz>={winner['vz']:.1f}, ml>={winner['ml']:.2f}")
    print(f"UNTOUCHED PURGED OOS (winner): n={w_oos['n']}, WR={w_oos['wr']:.1f}%, PF={w_oos['pf']:.2f}, R={w_oos['sum_r']:+.1f}")
    print(f"UNTOUCHED PURGED OOS (base 35/1.0/0.45): n={base_oos['n']}, WR={base_oos['wr']:.1f}%, PF={base_oos['pf']:.2f}, R={base_oos['sum_r']:+.1f}")

    # 3. Also report strict post-created_at unseen slice for production .joblib models
    prod_bundles = _load_ml15_bundles()
    if prod_bundles:
        created_iso = prod_bundles[0].get("created_at")
        post_cut = int(datetime.fromisoformat(str(created_iso)).timestamp() * 1000) if created_iso else split_ts
        prep_prod = [_build_prepared(s, feats_by_sym[s], h1_by_sym[s], prod_bundles) for s in MAJORS]
        for label, natr, vz, ml in (("PROD_BASE", 35.0, 1.0, 0.45), ("P1B_CAND", winner["natr"], winner["vz"], winner["ml"]), ("SQUEEZE_20_08_45", 20.0, 0.8, 0.45)):
            m_post = _metrics([x for d in prep_prod for x in simulate_symbol(d, natr, vz, ml, post_cut, 10**18)])
            print(f"[PROD .joblib POST-{created_iso}] {label} ({natr:.0f}/{vz:.1f}/{ml:.2f}): n={m_post['n']}, WR={m_post['wr']:.1f}%, PF={m_post['pf']:.2f}, R={m_post['sum_r']:+.1f}")


if __name__ == "__main__":
    run_zero_leakage_calibration()
