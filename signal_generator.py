"""
Gerador de Sinais Executáveis (Signal Generator) da Fase 4.
Transforma o resultado do Juiz de Consenso em uma ordem parametrizada e segura:
1. Níveis de Stop Loss (SL) e Take Profit (TP) proporcionais ao ATR.
2. Dimensionamento de posição via risco fixo (1% institucional).
3. Racional textual completo sintetizando a deliberação dos agentes.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from consensus_judge import ConsensusDecision


@dataclass
class TradeSignal:
    """Sinal operacional final pronto para roteamento ou backtest."""
    signal_id: str
    timestamp: int
    symbol: str
    direction: str              # 'BUY', 'SELL', 'HOLD'
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    confidence: float
    consensus_score: float
    position_size: float
    risk_dollars: float
    rationale: str
    is_vetoed: bool = False
    contributing_agents: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SignalGenerator:
    """
    Construtor determinístico de sinais operacionais com gestão de risco ATR.
    """

    def __init__(
        self,
        capital: float = 10000.0,
        risk_per_trade: float = 0.01,   # 1% de risco no Stop Loss
        tp_atr_multiple: float = 1.5,
        sl_atr_multiple: float = 1.0,
        leverage: float = 1.0,
    ):
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.tp_atr_multiple = tp_atr_multiple
        self.sl_atr_multiple = sl_atr_multiple
        self.leverage = leverage

    def generate(
        self,
        decision: ConsensusDecision,
        symbol: str,
        current_price: float,
        atr: float,
        timestamp: int,
        capital_override: Optional[float] = None,
        structure_context: Optional[Dict[str, Any]] = None,
    ) -> TradeSignal:
        """
        Gera a estrutura formal do sinal operacional a partir da deliberação do comitê.
        Suporta parametrização assimétrica de SL/TP com confluência de POI e Liquidity Sweeps.
        """
        capital = capital_override if capital_override is not None else self.capital
        direction = decision.direction
        p0 = float(current_price)
        atr_raw = float(atr)
        atr_val = max(1e-6, atr_raw) if math.isfinite(atr_raw) else 1e-6
        # Preço/ATR inválidos (0, NaN, negativo) tornariam SL/TP e sizing sem sentido
        # (ZeroDivisionError em max_size ou stop colado na entrada): não opera.
        invalid_inputs = (not math.isfinite(p0)) or p0 <= 0.0 or (not math.isfinite(atr_raw)) or atr_raw <= 0.0

        if decision.is_vetoed or direction == "HOLD" or invalid_inputs:
            return TradeSignal(
                signal_id=str(uuid.uuid4())[:8],
                timestamp=timestamp,
                symbol=symbol,
                direction="HOLD",
                entry_price=p0,
                stop_loss=0.0,
                take_profit=0.0,
                atr=atr_val,
                confidence=decision.confidence,
                consensus_score=decision.aggregated_score,
                position_size=0.0,
                risk_dollars=0.0,
                rationale=decision.consensus_summary,
                is_vetoed=decision.is_vetoed,
                contributing_agents=[s.to_dict() for s in decision.agent_signals],
            )

        # Múltiplos de SL/TP com adaptação assimétrica institucional (Smart Money Concepts)
        eff_tp_mult = self.tp_atr_multiple
        eff_sl_mult = self.sl_atr_multiple
        is_asymmetric_poi = False

        if structure_context:
            sweep_bull = bool(structure_context.get("liquidity_sweep_bull", 0))
            sweep_bear = bool(structure_context.get("liquidity_sweep_bear", 0))
            at_bull_poi = bool(structure_context.get("at_bullish_poi", 0))
            at_bear_poi = bool(structure_context.get("at_bearish_poi", 0))

            if direction == "BUY" and (sweep_bull or at_bull_poi):
                eff_sl_mult = 0.7 if sweep_bull else 0.85
                eff_tp_mult = 2.1 if sweep_bull else 1.8
                is_asymmetric_poi = True
            elif direction == "SELL" and (sweep_bear or at_bear_poi):
                eff_sl_mult = 0.7 if sweep_bear else 0.85
                eff_tp_mult = 2.1 if sweep_bear else 1.8
                is_asymmetric_poi = True

        # Cálculo de Barreiras ATR
        if direction == "BUY":
            take_profit = p0 + (eff_tp_mult * atr_val)
            stop_loss = p0 - (eff_sl_mult * atr_val)
        else:  # SELL (SHORT)
            take_profit = p0 - (eff_tp_mult * atr_val)
            stop_loss = p0 + (eff_sl_mult * atr_val)

        sl_distance = abs(p0 - stop_loss)

        # Dimensionamento por Risco Fixo (1% do capital arriscado no SL)
        dollar_risk = capital * self.risk_per_trade
        desired_size = dollar_risk / max(1e-6, sl_distance)
        max_size = (capital * self.leverage) / p0
        size = min(desired_size, max_size)
        # Risco efetivo: se o teto de alavancagem limitou o tamanho, o risco real é menor que o nominal
        effective_risk = size * sl_distance

        # Construção da justificativa textual unificada
        agent_reasons = [
            f"[{s.agent_name} ({s.direction}, {s.confidence:.0%})]: {s.rationale}"
            for s in decision.agent_signals
            if s.direction == direction
        ]
        full_rationale = (
            f"{decision.consensus_summary} | TP={take_profit:.2f} (+{eff_tp_mult}x ATR), "
            f"SL={stop_loss:.2f} (-{eff_sl_mult}x ATR). "
            f"Evidências convergentes: {' // '.join(agent_reasons)}"
        )

        return TradeSignal(
            signal_id=str(uuid.uuid4())[:8],
            timestamp=timestamp,
            symbol=symbol,
            direction=direction,
            entry_price=p0,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr_val,
            confidence=decision.confidence,
            consensus_score=decision.aggregated_score,
            position_size=size,
            risk_dollars=effective_risk,
            rationale=full_rationale,
            is_vetoed=False,
            contributing_agents=[s.to_dict() for s in decision.agent_signals],
        )
