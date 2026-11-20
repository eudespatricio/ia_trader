"""scripts/backtest_scalp_15m_calibration.py - OOS A/B Backtest para calibração do SCALP 15m (Squeeze BB/KC + ML 15m).

Lê exclusivamente datasets/market_cache.db (mode=ro), calcula features causais 15m e H1 SMA200,
executa ensemble LGBM+RF 15m com n_jobs=1 e avalia a grade de thresholds sob split temporal estrito (67% IS / 33% OOS).
"""
from __future__ import annotations

import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.technical_agent import _disable_estimator_parallelism
from data.feature_engineering import FeatureEngineer
from smart_trade import MAKER_SLIPPAGE_PCT, MARKET_SLIPPAGE_PCT, STRATEGY_PROFILES

CACHE_DB = os.path.join(ROOT, "datasets", "market_cache.db")
MODELS_DIR = os.path.join(ROOT, "models")
MAJORS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
SCALP_BAR_MS = 900_000
H1_BAR_MS = 3_600_000
MAKER_FEE = 0.0002
TAKER_FEE = 0.0006
COOLDOWN_BARS = 4


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


def prepare_symbol(con: sqlite3.Connection, symbol: str, bundles: List[Dict], limit_15m: int = 20_000) -> Optional[PreparedSymbolData]:
    df15 = _load_candles(con, symbol, "15m", limit=limit_15m)
    df1h = _load_candles(con, symbol, "1h", limit=limit_15m // 4 + 300)
    if len(df15) < 300 or len(df1h) < 250:
        return None

    df_feat = FeatureEngineer.compute_all_features(df15, drop_na=False)
    valid_sma = np.isfinite(df_feat["sma_200"].astype(float)) & (df_feat["sma_200"].astype(float) > 0.0)
    df_feat = df_feat[valid_sma].reset_index(drop=True)
    if len(df_feat) < 100:
        return None

    # Causal H1 SMA200 distance alignment: decision happens at bar close (ts15 + 900_000)
    h1_close = df1h["close"].astype(float)
    h1_sma200 = h1_close.rolling(200, min_periods=200).mean()
    h1_dist_arr = ((h1_close - h1_sma200) / h1_sma200).to_numpy(dtype=float)
    h1_close_ts = df1h["timestamp"].to_numpy(dtype=np.int64) + H1_BAR_MS
    target_close_ts = df_feat["timestamp"].to_numpy(dtype=np.int64) + SCALP_BAR_MS
    idx_h1 = np.searchsorted(h1_close_ts, target_close_ts, side="right") - 1
    h1_dist = np.where(idx_h1 >= 0, h1_dist_arr[np.maximum(idx_h1, 0)], np.nan)

    # Batch predict ML15 ensemble
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
    ml_margin: float = 0.0,
    min_prior_sqz: int = 2,
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

    for i in range(max(3, 2), n - 2):
        if ts[i] < ts_start or ts[i] >= ts_end:
            continue
        if i < next_free_idx:
            continue

        if min_prior_sqz == 2:
            if not (data.in_sqz[i - 1] and data.in_sqz[i - 2]):
                continue
        else:
            if not data.in_sqz[i - 1]:
                continue

        if data.natr_bps[i] < natr_min or data.vol_z[i] < vol_z_min or not np.isfinite(data.h1_dist[i]):
            continue

        direction: Optional[str] = None
        if closes[i] > opens[i] and closes[i] > data.kc_hi[i] and data.h1_dist[i] >= 0.0:
            if data.p_buy[i] >= ml_min and (data.p_buy[i] - data.p_sell[i]) > ml_margin and data.p_buy[i] > data.p_sell[i]:
                direction = "BUY"
        elif closes[i] < opens[i] and closes[i] < data.kc_lo[i] and data.h1_dist[i] <= 0.0:
            if data.p_sell[i] >= ml_min and (data.p_sell[i] - data.p_buy[i]) > ml_margin and data.p_sell[i] > data.p_buy[i]:
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
        r_multiple = net_ret / (sl_dist / entry_px)
        pnls_r.append(float(r_multiple))
        next_free_idx = exit_idx + COOLDOWN_BARS

    return pnls_r


def _metrics(pnls_r: List[float]) -> Dict[str, float]:
    n = len(pnls_r)
    if n == 0:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "sum_r": 0.0}
    arr = np.asarray(pnls_r, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float(abs(losses.sum())) if len(losses) else 0.0
    pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
    return {"n": n, "wr": float(len(wins) / n * 100.0), "pf": pf, "sum_r": float(arr.sum())}


def run_calibration_grid() -> List[Dict]:
    bundles = _load_ml15_bundles()
    con = sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True, timeout=15.0)
    try:
        prepared = [d for s in MAJORS if (d := prepare_symbol(con, s, bundles)) is not None]
    finally:
        con.close()

    min_ts = min(int(d.df["timestamp"].min()) for d in prepared)
    max_ts = max(int(d.df["timestamp"].max()) for d in prepared)
    split_ts = min_ts + int((max_ts - min_ts) * 0.67)

    results = []
    for min_sqz in (2, 1):
        for natr_min in (35.0, 28.0, 25.0, 20.0):
            for vol_z_min in (1.0, 0.8, 0.6):
                for ml_min in (0.45, 0.40, 0.38, 0.35):
                    for ml_margin in (0.0, 0.05):
                        is_r, oos_r = [], []
                        for d in prepared:
                            is_r.extend(simulate_symbol(d, natr_min, vol_z_min, ml_min, ml_margin, min_sqz, 0, split_ts))
                            oos_r.extend(simulate_symbol(d, natr_min, vol_z_min, ml_min, ml_margin, min_sqz, split_ts, 10**18))
                        mi, mo = _metrics(is_r), _metrics(oos_r)
                        results.append({
                            "sqz": min_sqz, "natr": natr_min, "vz": vol_z_min, "ml": ml_min, "margin": ml_margin,
                            "is_n": mi["n"], "is_wr": mi["wr"], "is_pf": mi["pf"], "is_r": mi["sum_r"],
                            "oos_n": mo["n"], "oos_wr": mo["wr"], "oos_pf": mo["pf"], "oos_r": mo["sum_r"],
                            "tot_n": mi["n"] + mo["n"],
                        })
    return results


if __name__ == "__main__":
    rows = run_calibration_grid()
    passing = [r for r in rows if r["oos_pf"] >= 1.25 and r["oos_n"] >= 8 and r["tot_n"] >= 15]
    print(f"Total configs tested: {len(rows)} | Passing gate (OOS PF>=1.25, OOS n>=8, total>=15): {len(passing)}")
    ranked = sorted(rows, key=lambda x: (x["oos_n"] >= 8, x["oos_pf"], x["oos_r"]), reverse=True)
    for r in ranked[:15]:
        print(
            f"sqz={r['sqz']} natr>={r['natr']:4.1f} vz>={r['vz']:.1f} ml>={r['ml']:.2f} d>{r['margin']:.2f} | "
            f"IS(n={r['is_n']:3d}, WR={r['is_wr']:4.1f}%, PF={r['is_pf']:4.2f}, R={r['is_r']:+5.1f}) | "
            f"OOS(n={r['oos_n']:3d}, WR={r['oos_wr']:4.1f}%, PF={r['oos_pf']:4.2f}, R={r['oos_r']:+5.1f})"
        )
