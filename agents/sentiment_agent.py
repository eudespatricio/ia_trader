"""
Agente de Sentimento de Mercado da Fase 4.
Avalia sentimento de mercado através de duas fontes:
1. Backtest Histórico: Índice Fear & Greed (alternative.me) com histórico diário gratuito desde 2018,
   alinhado estritamente com lag de 1 dia (sem lookahead bias).
2. Modo Ao Vivo: CryptoPanic sentiment headlines + Fear & Greed em tempo real.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional
from agents.base_agent import BaseAgent, AgentSignal


def _to_finite_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return None
    return fv if math.isfinite(fv) else None


class SentimentAgent(BaseAgent):
    """
    Agente especialista em sentimento de multidão e psicologia de mercado.
    Suporta modos: 'trend_momentum' (padrão), 'neutral_telemetry' e 'legacy_contrarian'.
    """

    SUPPORTED_MODES = ("trend_momentum", "neutral_telemetry", "legacy_contrarian")

    def __init__(
        self,
        name: str = "SentimentAgent",
        weight: float = 1.0,
        extreme_fear_threshold: float = 25.0,
        extreme_greed_threshold: float = 75.0,
        mode: str = "trend_momentum",
    ):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Modo inválido para SentimentAgent: {mode!r}. Esperado: {self.SUPPORTED_MODES}")
        super().__init__(name=name, weight=weight, is_backtestable=True)
        self.extreme_fear_threshold = extreme_fear_threshold
        self.extreme_greed_threshold = extreme_greed_threshold
        self.mode = mode

    def evaluate(self, context: Dict[str, Any]) -> AgentSignal:
        """
        Avalia o contexto de sentimento:
        - `fear_and_greed`: valor numérico de 0 a 100 (ontem / D-1 sem lookahead)
        - `fng_7d_chg`: variação de 7 dias do Fear & Greed (causal)
        - `news_score`: opcional (média ponderada de sentimento CryptoPanic em live)
        """
        fng_val = _to_finite_float(context.get("fear_and_greed"))
        fng_7d_val = _to_finite_float(context.get("fng_7d_chg"))
        news_val = _to_finite_float(context.get("news_score"))

        direction = "HOLD"
        confidence = 0.50
        rationale = "Sentimento de mercado neutro ou moderado."
        metrics: Dict[str, Any] = {"sentiment_mode": self.mode}

        if fng_val is not None:
            metrics["fear_and_greed"] = fng_val
        if fng_7d_val is not None:
            metrics["fng_7d_chg"] = fng_7d_val

        if self.mode == "neutral_telemetry":
            direction = "HOLD"
            confidence = 0.50
            rationale = "Sentimento em modo telemetria neutra."
        elif self.mode == "trend_momentum":
            if fng_val is not None and fng_7d_val is not None:
                if fng_val >= 55.0 and fng_7d_val >= 5.0:
                    direction = "BUY"
                    confidence = min(0.80, 0.58 + min(abs(fng_7d_val), 30.0) / 30.0 * 0.22)
                    rationale = f"Momentum de sentimento altista (F&G={fng_val:.0f}, 7d={fng_7d_val:+.1f})."
                elif fng_val <= 45.0 and fng_7d_val <= -5.0:
                    direction = "SELL"
                    confidence = min(0.80, 0.58 + min(abs(fng_7d_val), 30.0) / 30.0 * 0.22)
                    rationale = f"Momentum de sentimento baixista (F&G={fng_val:.0f}, 7d={fng_7d_val:+.1f})."
                else:
                    direction = "HOLD"
                    confidence = 0.55
                    rationale = f"Momentum de sentimento neutro (F&G={fng_val:.0f}, 7d={fng_7d_val:+.1f})."
            elif fng_val is not None:
                direction = "HOLD"
                confidence = 0.55
                rationale = f"Momentum de sentimento indisponível sem fng_7d_chg (F&G={fng_val:.0f})."
        elif self.mode == "legacy_contrarian" and fng_val is not None:
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

        # Modulação ao vivo via notícias (se presente) em todos os modos
        if news_val is not None:
            metrics["news_score"] = news_val
            if news_val >= 0.5 and direction == "BUY":
                confidence = min(0.95, confidence + 0.10)
                rationale += " Confirmado por notícias positivas recentes (CryptoPanic)."
            elif news_val <= -0.5 and direction == "SELL":
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
