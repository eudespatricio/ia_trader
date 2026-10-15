"""Features de padrões de price action (puras e CAUSAIS: o valor em t usa apenas barras <= t).

Colunas (prefixo ``pat_``):
  pat_pin_bull / pat_pin_bear      pin bar (corpo <= 1/3 do range, pavio >= 2x corpo, fechamento no terço oposto ao pavio)
  pat_engulf_bull / pat_engulf_bear engolfo (corpo atual envolve o corpo da barra anterior, cores opostas)
  pat_inside                        inside bar (high < high[t-1] e low > low[t-1])
  pat_div_bull / pat_div_bear       divergência REGULAR de RSI(14): pivot confirmado em t (pivot em t-n, n barras de atraso)
                                    com preço em novo extremo vs. pivot anterior (janela <= 30 barras) e RSI não confirmando
  pat_dist_swing_high / _low        distância % (close) ao último swing high/low CONFIRMADO (market_structure, atraso n)
  pat_near_high / pat_near_low      barra a <= 0.5 ATR da zona (high vs swing high / low vs swing low)
  pat_bull_at_sr / pat_bear_at_sr   padrão de alta (pin/engolfo/div) perto do swing low / de baixa perto do swing high
"""
from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

PATTERN_COLUMNS: List[str] = [
    "pat_pin_bull", "pat_pin_bear", "pat_engulf_bull", "pat_engulf_bear", "pat_inside",
    "pat_div_bull", "pat_div_bear", "pat_dist_swing_high", "pat_dist_swing_low",
    "pat_near_high", "pat_near_low", "pat_bull_at_sr", "pat_bear_at_sr",
]


def _confirmed_pivots(series: np.ndarray, n: int, kind: str) -> np.ndarray:
    """Máscara booleana: True em t se o pivot em t-n ficou confirmado em t (usa só barras <= t).
    kind='low': series[t-n] <= mínimo das n barras anteriores e < mínimo das n posteriores (até t)."""
    s = pd.Series(series)
    if kind == "low":
        left = s.shift(1).rolling(n).min().shift(n)          # barras j-n..j-1 (j=t-n)
        right = s.rolling(n).min()                            # barras j+1..j+n = t-n+1..t
        center = s.shift(n)
        m = (center <= left) & (center < right)
    else:
        left = s.shift(1).rolling(n).max().shift(n)
        right = s.rolling(n).max()
        center = s.shift(n)
        m = (center >= left) & (center > right)
    return m.fillna(False).to_numpy(dtype=bool)


def _divergence(price: np.ndarray, rsi: np.ndarray, piv: np.ndarray, n: int, max_bars: int, bull: bool) -> np.ndarray:
    """Divergência regular avaliada nos eventos de confirmação (t); pivot em j=t-n, anterior em k (j-k <= max_bars)."""
    out = np.zeros(len(price), dtype=bool)
    prev_j = -1
    for t in np.flatnonzero(piv):
        j = t - n
        if prev_j >= 0 and (j - prev_j) <= max_bars and np.isfinite(rsi[j]) and np.isfinite(rsi[prev_j]):
            if bull:
                out[t] = price[j] < price[prev_j] and rsi[j] > rsi[prev_j]
            else:
                out[t] = price[j] > price[prev_j] and rsi[j] < rsi[prev_j]
        prev_j = j
    return out


def compute_pattern_features(df: pd.DataFrame, pivot_n: int = 3, div_window: int = 30, rsi_period: int = 14,
                             atr_period: int = 14, sr_atr: float = 0.5) -> pd.DataFrame:
    """Retorna DataFrame (mesmo índice de ``df``) com as colunas de PATTERN_COLUMNS. Requer open/high/low/close."""
    from data.market_structure import MarketStructureDetector

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    idx = df.index
    rng = h - lo
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - lo
    ok = rng > 0
    small = ok & (body <= rng / 3.0)
    pin_bull = small & (lower >= 2.0 * body) & (c >= lo + 2.0 * rng / 3.0)
    pin_bear = small & (upper >= 2.0 * body) & (c <= lo + rng / 3.0)

    o1, c1, h1, l1 = (np.r_[np.nan, x[:-1]] for x in (o, c, h, lo))
    with np.errstate(invalid="ignore"):
        eng_bull = (c1 < o1) & (c > o) & (o <= c1) & (c >= o1)
        eng_bear = (c1 > o1) & (c < o) & (o >= c1) & (c <= o1)
        inside = (h < h1) & (lo > l1)

    rsi = RSIIndicator(close=pd.Series(c, index=idx), window=rsi_period).rsi().to_numpy(float)
    piv_l = _confirmed_pivots(lo, pivot_n, "low")
    piv_h = _confirmed_pivots(h, pivot_n, "high")
    div_bull = _divergence(lo, rsi, piv_l, pivot_n, div_window, True)
    div_bear = _divergence(h, rsi, piv_h, pivot_n, div_window, False)

    # S/R: último swing high/low confirmado (market_structure: topo em i só é conhecido em i+n)
    sh, sl = MarketStructureDetector(swing_window=pivot_n).identify_swings(df[["high", "low"]].reset_index(drop=True))
    sh_last = sh.ffill().to_numpy(float)
    sl_last = sl.ffill().to_numpy(float)
    if len(c) > atr_period:
        atr = AverageTrueRange(high=pd.Series(h, index=idx), low=pd.Series(lo, index=idx), close=pd.Series(c, index=idx),
                               window=atr_period).average_true_range().to_numpy(float).copy()
        atr[:atr_period - 1] = np.nan                         # aquecimento (ta devolve 0)
    else:
        atr = np.full(len(c), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        dist_h = (sh_last - c) / c
        dist_l = (c - sl_last) / c
        near_h = np.abs(h - sh_last) <= sr_atr * atr
        near_l = np.abs(lo - sl_last) <= sr_atr * atr
    near_h = near_h & np.isfinite(sh_last) & np.isfinite(atr)
    near_l = near_l & np.isfinite(sl_last) & np.isfinite(atr)
    bull_any = pin_bull | eng_bull | div_bull
    bear_any = pin_bear | eng_bear | div_bear

    f = lambda a: np.asarray(a, dtype=float)  # noqa: E731
    return pd.DataFrame({
        "pat_pin_bull": f(pin_bull), "pat_pin_bear": f(pin_bear),
        "pat_engulf_bull": f(eng_bull), "pat_engulf_bear": f(eng_bear), "pat_inside": f(inside),
        "pat_div_bull": f(div_bull), "pat_div_bear": f(div_bear),
        "pat_dist_swing_high": dist_h, "pat_dist_swing_low": dist_l,
        "pat_near_high": f(near_h), "pat_near_low": f(near_l),
        "pat_bull_at_sr": f(bull_any & near_l), "pat_bear_at_sr": f(bear_any & near_h),
    }, index=idx)[PATTERN_COLUMNS]
