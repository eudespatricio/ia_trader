"""
data/swing_4h_strategy.py - Motor puro e compartilhado da estratégia SWING_4H (Donchian40 / ExitCh10).

Usado tanto pelo daemon de paper trading quanto pelo harness de referência offline
(`scripts/eval_swing_4h_reference.py`) para garantir paridade exata de reamostragem,
indicadores, entradas e saídas sem lookahead.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

SWING_4H_BAR_MS: int = 14_400_000
SWING_4H_DONCHIAN_ENTRY: int = 40
SWING_4H_DONCHIAN_EXIT: int = 10
SWING_4H_SMA_PERIOD: int = 200
SWING_4H_ATR_PERIOD: int = 14
SWING_4H_MAX_DIST_SMA200: float = 0.20
SWING_4H_SL_ATR_MULT: float = 2.5
SWING_4H_TRAIL_ATR_MULT: float = 3.5
SWING_4H_MAX_HOLDING_BARS: int = 120
SWING_4H_COOLDOWN_BARS: int = 2

_OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def resample_1h_to_closed_4h(
    df_1h: Optional[pd.DataFrame],
    now_ms: Optional[int] = None,
) -> pd.DataFrame:
    """Reamostra velas 1h em barras 4h estritamente fechadas (4 velas/bucket e fechamento <= now_ms)."""
    if df_1h is None or df_1h.empty:
        return pd.DataFrame(columns=_OHLCV_COLS)

    work = df_1h[_OHLCV_COLS].copy()
    work["timestamp"] = pd.to_numeric(work["timestamp"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["timestamp", "open", "high", "low", "close"])
    if work.empty:
        return pd.DataFrame(columns=_OHLCV_COLS)

    work["timestamp"] = work["timestamp"].astype(np.int64)
    work = work.sort_values("timestamp", ascending=True).drop_duplicates(subset=["timestamp"], keep="first")
    work["bucket"] = (work["timestamp"] // SWING_4H_BAR_MS) * SWING_4H_BAR_MS

    agg = (
        work.groupby("bucket", sort=True, as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            cnt=("close", "count"),
        )
        .rename(columns={"bucket": "timestamp"})
    )
    agg = agg[agg["cnt"] >= 4].drop(columns=["cnt"])
    if now_ms is not None:
        agg = agg[agg["timestamp"] + SWING_4H_BAR_MS <= int(now_ms)]
    if agg.empty:
        return pd.DataFrame(columns=_OHLCV_COLS)

    agg["timestamp"] = agg["timestamp"].astype(np.int64)
    return agg.reset_index(drop=True)


def compute_swing_4h_indicators(df_4h: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Calcula Wilder ATR14, SMA200, canais Donchian shift(1) (40 e 10) e dist_sma_200."""
    if df_4h is None or df_4h.empty:
        return pd.DataFrame()

    df = df_4h.copy().reset_index(drop=True)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["atr_14"] = tr.ewm(alpha=1.0 / SWING_4H_ATR_PERIOD, adjust=False).mean()
    df["sma_200"] = close.rolling(SWING_4H_SMA_PERIOD, min_periods=SWING_4H_SMA_PERIOD).mean()
    df["h40"] = high.shift(1).rolling(SWING_4H_DONCHIAN_ENTRY, min_periods=SWING_4H_DONCHIAN_ENTRY).max()
    df["l40"] = low.shift(1).rolling(SWING_4H_DONCHIAN_ENTRY, min_periods=SWING_4H_DONCHIAN_ENTRY).min()
    df["h10"] = high.shift(1).rolling(SWING_4H_DONCHIAN_EXIT, min_periods=SWING_4H_DONCHIAN_EXIT).max()
    df["l10"] = low.shift(1).rolling(SWING_4H_DONCHIAN_EXIT, min_periods=SWING_4H_DONCHIAN_EXIT).min()
    df["dist_sma_200"] = (close - df["sma_200"]) / df["sma_200"]
    return df


def evaluate_swing_4h_entry(
    df_4h: Optional[pd.DataFrame],
    max_dist_sma200: float = SWING_4H_MAX_DIST_SMA200,
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Avalia a última barra 4h fechada para rompimento Donchian40 alinhado à SMA200."""
    if df_4h is None or df_4h.empty:
        return None

    row = df_4h.iloc[-1]
    vals = {}
    for k in ("close", "atr_14", "sma_200", "h40", "l40", "h10", "l10", "dist_sma_200"):
        v = row.get(k)
        if v is None:
            return None
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(fv):
            return None
        vals[k] = fv

    close = vals["close"]
    atr = vals["atr_14"]
    sma_200 = vals["sma_200"]
    dist = vals["dist_sma_200"]
    if close <= 0.0 or atr <= 0.0 or sma_200 <= 0.0:
        return None

    if close > vals["h40"] and 0.0 <= dist <= max_dist_sma200:
        rationale = (
            f"Swing 4h Donchian40 BUY: close {close:.2f} > h40 {vals['h40']:.2f} | "
            f"dist_sma200 {dist:+.2%} | ATR14 {atr:.2f}"
        )
        return "BUY", rationale, vals

    if close < vals["l40"] and -max_dist_sma200 <= dist <= 0.0:
        rationale = (
            f"Swing 4h Donchian40 SELL: close {close:.2f} < l40 {vals['l40']:.2f} | "
            f"dist_sma200 {dist:+.2%} | ATR14 {atr:.2f}"
        )
        return "SELL", rationale, vals

    return None


def should_exit_swing_4h_channel(
    df_4h: Optional[pd.DataFrame],
    side: str,
) -> Tuple[bool, str]:
    """Verifica saída pelo canal Donchian10 oposto na última barra 4h fechada."""
    if df_4h is None or df_4h.empty:
        return False, ""

    row = df_4h.iloc[-1]
    try:
        close = float(row.get("close"))
    except (TypeError, ValueError):
        return False, ""
    if not np.isfinite(close):
        return False, ""

    if side == "LONG":
        try:
            l10 = float(row.get("l10"))
        except (TypeError, ValueError):
            return False, ""
        if np.isfinite(l10) and close < l10:
            return True, f"ExitCh10 LONG: close {close:.2f} < l10 {l10:.2f}"
    elif side == "SHORT":
        try:
            h10 = float(row.get("h10"))
        except (TypeError, ValueError):
            return False, ""
        if np.isfinite(h10) and close > h10:
            return True, f"ExitCh10 SHORT: close {close:.2f} > h10 {h10:.2f}"

    return False, ""
