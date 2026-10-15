"""
leverage_policy.py - Política de alavancagem decidida pelo Smart Trader (ADDENDUM E5).

A alavancagem NÃO é configurável pelo usuário. Função pura, determinística e sem I/O.
Alavancar amplifica ganhos e perdas na mesma proporção: não há evidência de que melhore
o resultado esperado em % do notional - apenas escala a variância (e o risco de liquidação).
"""
from __future__ import annotations

from typing import Optional

MAX_LEVERAGE = 5.0               # teto duro (também reforçado no engine)
MIN_LEVERAGE = 1.0
LEVERAGE_STEP = 1.0              # passo da política (engine aceita passos de 0.5)
ATR_PCT_EXTREME = 0.03           # ATR/preço >= 3% por vela = volatilidade extrema -> -1 degrau

# (confiança mínima inclusiva, alavancagem base) em ordem decrescente
CONFIDENCE_LADDER = (
    (0.90, 5.0),
    (0.80, 4.0),
    (0.70, 3.0),
    (0.60, 2.0),
)


def clamp_leverage(lev: float, max_leverage: float = MAX_LEVERAGE) -> float:
    """Limita a [1, max] (<= MAX_LEVERAGE) em passos de 0.5 (arredonda para baixo)."""
    try:
        lev = float(lev)
    except (TypeError, ValueError):
        return MIN_LEVERAGE
    if lev != lev:  # NaN
        return MIN_LEVERAGE
    cap = min(float(max_leverage), MAX_LEVERAGE)
    lev = max(MIN_LEVERAGE, min(lev, cap))
    return max(MIN_LEVERAGE, float(int(lev * 2.0)) / 2.0)


def decide_leverage(
    confidence: float,
    atr_pct: float = 0.0,
    regime_alignment: Optional[bool] = None,
    *,
    is_vetoed: bool = False,
    low_history_or_liquidity: bool = False,
    max_leverage: float = MAX_LEVERAGE,
) -> float:
    """
    Decide a alavancagem (1..MAX_LEVERAGE) a partir da confiança REAL do comitê.

    - veto, símbolo sem histórico/baixa liquidez -> 1x (sem alavancar);
    - confiança <0.60 -> 1x; 0.60-0.70 -> 2x; 0.70-0.80 -> 3x; 0.80-0.90 -> 4x; >=0.90 -> 5x;
    - -1 degrau se ATR% >= ATR_PCT_EXTREME (volatilidade extrema);
    - -1 degrau se a entrada é contra a tendência (regime_alignment is False; sma_200);
    - resultado sempre em [1, 5].
    """
    if is_vetoed or low_history_or_liquidity:
        return MIN_LEVERAGE
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        return MIN_LEVERAGE
    if conf != conf:
        return MIN_LEVERAGE

    lev = MIN_LEVERAGE
    for threshold, base in CONFIDENCE_LADDER:
        if conf >= threshold:
            lev = base
            break

    try:
        atr = float(atr_pct)
    except (TypeError, ValueError):
        atr = 0.0
    if atr == atr and atr >= ATR_PCT_EXTREME:
        lev -= LEVERAGE_STEP
    if regime_alignment is False:
        lev -= LEVERAGE_STEP
    return clamp_leverage(lev, max_leverage)
