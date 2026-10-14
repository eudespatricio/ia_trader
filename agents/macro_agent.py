"""
Agente Macroeconômico da Fase 4.
Avalia o regime econômico global utilizando dados do Yahoo Finance e FRED:
- DXY (Índice Dólar): Correlação inversa com criptoativos.
- VIX (Índice de Volatilidade / Pânico em Ações).
- S&P 500: Apetite a risco institucional (Risk-On / Risk-Off).
- Rigor Anti-Lookahead: Estritamente com defasagem de 1 dia (fechamento de D-1 para decisões em D).
"""

from __future__ import annotations

from typing import Any, Dict
from agents.base_agent import BaseAgent, AgentSignal


class MacroAgent(BaseAgent):
    """
    Agente especialista em liquidez macroeconômica e política monetária.
    """

    def __init__(
        self,
        name: str = "MacroAgent",
        weight: float = 1.0,
    ):
        super().__init__(name=name, weight=weight, is_backtestable=True)

    def evaluate(self, context: Dict[str, Any]) -> AgentSignal:
        """
        Avalia o contexto macroeconômico recebido:
        - `dxy_ret_5d`: Retorno de 5 dias do DXY
        - `dxy_above_sma20`: Booleano se DXY > média de 20 dias
        - `vix_value`: Valor atual do VIX
        - `sp500_ret_5d`: Retorno de 5 dias do S&P 500
        """
        dxy_ret = context.get("dxy_ret_5d")
        dxy_above_sma = context.get("dxy_above_sma20")
        vix = context.get("vix_value")
        sp500_ret = context.get("sp500_ret_5d")

        direction = "HOLD"
        confidence = 0.50
        rationale = "Ambiente macroeconômico em equilíbrio neutro."
        metrics = {
            "dxy_ret_5d": dxy_ret,
            "vix": vix,
            "sp500_ret_5d": sp500_ret,
        }

        score = 0.0  # Positivo = favorável a cripto (BUY), Negativo = desfavorável (SELL)

        # 1. Análise DXY (Correlação Inversa)
        if dxy_ret is not None:
            if float(dxy_ret) < -0.005:  # Dólar enfraquecendo -> Risk-On
                score += 1.0
            elif float(dxy_ret) > 0.005:  # Dólar fortalecendo -> Liquidez drenada
                score -= 1.0

        if dxy_above_sma is not None:
            if not dxy_above_sma:
                score += 0.5
            else:
                score -= 0.5

        # 2. Análise VIX (Pânico nos Mercados Globais)
        if vix is not None:
            vix_val = float(vix)
            if vix_val >= 28.0:  # Pânico em equities -> Fuga de risco
                score -= 1.5
            elif vix_val <= 18.0:  # Calmaria e liquidez abundante -> Risk-On
                score += 0.5

        # 3. Análise S&P 500 (Apetite Institucional a Risco)
        if sp500_ret is not None:
            if float(sp500_ret) > 0.01:
                score += 1.0
            elif float(sp500_ret) < -0.01:
                score -= 1.0

        # Conclusão do Neurônio Macro Ativo
        if score >= 0.8:
            direction = "BUY"
            confidence = min(0.85, 0.58 + (score * 0.10))
            rationale = f"Neurônio Macro Ativo: Liquidez global favorável (DXY em retração, apetite institucional Risk-On, score={score:+.1f})."
        elif score <= -0.8:
            direction = "SELL"
            confidence = min(0.85, 0.58 + (abs(score) * 0.10))
            rationale = f"Neurônio Macro Ativo: Ambiente macro restritivo (Dólar em ascensão ou cautela em ativos de risco, score={score:+.1f})."
        else:
            confidence = 0.55
            rationale = f"Equilíbrio monetário global (score={score:+.1f}). Neurônio macro em prontidão."

        return AgentSignal(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            weight=self.weight,
            rationale=rationale,
            metrics=metrics,
            timestamp=context.get("timestamp"),
        )
