"""
Agente On-Chain da Fase 4.
Avalia a liquidez estrutural da blockchain utilizando dados do DefiLlama:
- Variação do suprimento circulante de Stablecoins (USDT, USDC, DAI): Medida de novos influxos de capital fiduciário.
- TVL (Total Value Locked): Utilização de capital e demanda por garantias em DeFi.
- Rigor Anti-Lookahead: Atualizações diárias indexadas por D-1.
"""

from __future__ import annotations

from typing import Any, Dict
from agents.base_agent import BaseAgent, AgentSignal


class OnChainAgent(BaseAgent):
    """
    Agente especialista em fluxos de liquidez e métricas on-chain.
    """

    def __init__(
        self,
        name: str = "OnChainAgent",
        weight: float = 1.0,
    ):
        super().__init__(name=name, weight=weight, is_backtestable=True)

    def evaluate(self, context: Dict[str, Any]) -> AgentSignal:
        """
        Avalia o contexto on-chain:
        - `stablecoin_net_inflow_7d`: Influxo líquido de stablecoins nos últimos 7 dias ($)
        - `stablecoin_growth_pct_30d`: Crescimento percentual do suprimento em 30 dias
        - `tvl_growth_pct_7d`: Crescimento do TVL em 7 dias
        """
        stbl_growth = context.get("stablecoin_growth_pct_30d")
        tvl_growth = context.get("tvl_growth_pct_7d")
        if tvl_growth is None:
            tvl_growth = context.get("tvl_change_24h")

        if stbl_growth is None and tvl_growth is not None:
            stbl_growth = float(tvl_growth) * 0.75

        direction = "HOLD"
        confidence = 0.50
        rationale = "Métricas on-chain em regime de estabilidade."
        metrics = {
            "stablecoin_growth_pct_30d": stbl_growth,
            "tvl_growth_pct_7d": tvl_growth,
        }

        score = 0.0

        if stbl_growth is not None:
            stbl_val = float(stbl_growth)
            if stbl_val > 0.005:  # Expansão de stablecoins > 0.5% -> Entrada de capital
                score += 1.2
            elif stbl_val < -0.005:  # Contração de stablecoins > 0.5% -> Saída para fiat
                score -= 1.2

        if tvl_growth is not None:
            tvl_val = float(tvl_growth)
            if tvl_val > 0.01:
                score += 0.8
            elif tvl_val < -0.01:
                score -= 0.8

        if score >= 0.8:
            direction = "BUY"
            confidence = min(0.85, 0.58 + (score * 0.10))
            rationale = f"Neurônio On-Chain Ativo: Expansão de capital e liquidez positiva (Score={score:+.1f})."
        elif score <= -0.8:
            direction = "SELL"
            confidence = min(0.85, 0.58 + (abs(score) * 0.10))
            rationale = f"Neurônio On-Chain Ativo: Retração de liquidez e fluxo defensivo (Score={score:+.1f})."
        else:
            confidence = 0.55
            rationale = "Liquidez on-chain sem tendência direcional expressiva. Neurônio em vigília."

        return AgentSignal(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            weight=self.weight,
            rationale=rationale,
            metrics=metrics,
            timestamp=context.get("timestamp"),
        )
