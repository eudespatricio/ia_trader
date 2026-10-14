"""
Agente de Sentimento de Mercado da Fase 4.
Avalia sentimento de mercado através de duas fontes:
1. Backtest Histórico: Índice Fear & Greed (alternative.me) com histórico diário gratuito desde 2018,
   alinhado estritamente com lag de 1 dia (sem lookahead bias).
2. Modo Ao Vivo: CryptoPanic sentiment headlines + Fear & Greed em tempo real.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from agents.base_agent import BaseAgent, AgentSignal


class SentimentAgent(BaseAgent):
    """
    Agente especialista em sentimento de multidão e psicologia de mercado.
    Opera com lógica contrária e momentum de sentimento.
    """

    def __init__(
        self,
        name: str = "SentimentAgent",
        weight: float = 1.0,
        extreme_fear_threshold: float = 25.0,
        extreme_greed_threshold: float = 75.0,
    ):
        super().__init__(name=name, weight=weight, is_backtestable=True)
        self.extreme_fear_threshold = extreme_fear_threshold
        self.extreme_greed_threshold = extreme_greed_threshold

    def evaluate(self, context: Dict[str, Any]) -> AgentSignal:
        """
        Avalia o contexto de sentimento:
        - `fear_and_greed`: valor numérico de 0 a 100 (ontem / D-1 sem lookahead)
        - `news_score`: opcional (média ponderada de sentimento CryptoPanic em live)
        """
        fng = context.get("fear_and_greed")
        news_score = context.get("news_score")

        direction = "HOLD"
        confidence = 0.50
        rationale = "Sentimento de mercado neutro ou moderado."
        metrics = {}

        if fng is not None:
            fng_val = float(fng)
            metrics["fear_and_greed"] = fng_val

            # Lógica de sentimento contrária institucional
            if fng_val <= self.extreme_fear_threshold:
                direction = "BUY"
                # Quanto mais extremo o medo, maior a convicção contrária
                confidence = min(0.85, 0.60 + ((self.extreme_fear_threshold - fng_val) / self.extreme_fear_threshold) * 0.25)
                rationale = f"Medo Extremo no mercado (F&G={fng_val:.0f}). Assimetria compradora contrária favorável."
            elif fng_val >= self.extreme_greed_threshold:
                direction = "SELL"
                confidence = min(0.85, 0.60 + ((fng_val - self.extreme_greed_threshold) / (100.0 - self.extreme_greed_threshold)) * 0.25)
                rationale = f"Ganância Extrema no mercado (F&G={fng_val:.0f}). Risco elevado de liquidação de topos."
            elif fng_val <= 42.0:
                direction = "BUY"
                confidence = min(0.75, 0.58 + ((42.0 - fng_val) / 42.0) * 0.15)
                rationale = f"Neurônio de Sentimento Ativo: Pessimismo moderado (F&G={fng_val:.0f}). Assimetria tática favorável para acúmulo."
            elif fng_val >= 58.0:
                direction = "SELL"
                confidence = min(0.75, 0.58 + ((fng_val - 58.0) / 42.0) * 0.15)
                rationale = f"Neurônio de Sentimento Ativo: Otimismo aquecido (F&G={fng_val:.0f}). Prudência contra exaustão de compradores."
            else:
                direction = "HOLD"
                confidence = 0.55
                rationale = f"Equilíbrio psicológico no mercado (F&G={fng_val:.0f}). Neurônio de sentimento em vigília."

        # Modulação ao vivo via notícias (se presente)
        if news_score is not None:
            metrics["news_score"] = float(news_score)
            if float(news_score) >= 0.5 and direction == "BUY":
                confidence = min(0.95, confidence + 0.10)
                rationale += " Confirmado por notícias positivas recentes (CryptoPanic)."
            elif float(news_score) <= -0.5 and direction == "SELL":
                confidence = min(0.95, confidence + 0.10)
                rationale += " Confirmado por notícias negativas recentes (CryptoPanic)."

        return AgentSignal(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            weight=self.weight,
            rationale=rationale,
            metrics=metrics,
            timestamp=context.get("timestamp"),
        )
