"""
Módulo de Juiz de Consenso (Consensus Judge) da Fase 4.
Agrega os votos ponderados dos agentes especialistas em uma decisão unificada.
Garante:
1. Precedência absoluta de VETO do RiskSentinelAgent.
2. Ponderação matemática configurada a priori ou otimizada estritamente no treino.
3. Timeout estrito de 2.5s para árbitro LLM com fallback determinístico local imediato.
4. Isolamento estrito: LLM desativada por padrão e 100% ausente no backtesting.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from agents.base_agent import AgentSignal

logger = logging.getLogger("ia_trader.consensus")


@dataclass
class ConsensusDecision:
    """Decisão soberana consolidada do Comitê de Agentes."""
    direction: str                     # 'BUY', 'SELL', 'HOLD'
    confidence: float                  # [0.0, 1.0]
    is_vetoed: bool = False
    veto_rationale: Optional[str] = None
    aggregated_score: float = 0.0      # [-1.0, +1.0]
    agent_signals: List[AgentSignal] = field(default_factory=list)
    consensus_summary: str = ""
    llm_arbitrated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["agent_signals"] = [s.to_dict() for s in self.agent_signals]
        return res


class ConsensusJudge:
    """
    Árbitro e sintetizador de consenso entre agentes independentes.
    """

    def __init__(
        self,
        min_consensus_threshold: float = 0.30,
        default_weights: Optional[Dict[str, float]] = None,
        llm_timeout_seconds: float = 2.5,
    ):
        self.min_consensus_threshold = min_consensus_threshold
        self.default_weights = default_weights or {
            "RiskSentinelAgent": 1.0,
            "TechnicalAgent": 1.0,
            "OrderFlowAgent": 0.8,
            "SentimentAgent": 0.6,
            "MacroAgent": 0.6,
            "OnChainAgent": 0.5,
        }
        self.llm_timeout_seconds = llm_timeout_seconds

    def judge_deterministic(
        self,
        signals: List[AgentSignal],
        weights_override: Optional[Dict[str, float]] = None,
    ) -> ConsensusDecision:
        """
        Julgamento matemático rápido e determinístico.
        Executado no backtest e como fallback do árbitro LLM.
        """
        # 1. Checagem prioritária de VETO
        for sig in signals:
            if sig.is_veto:
                return ConsensusDecision(
                    direction="HOLD",
                    confidence=1.0,
                    is_vetoed=True,
                    veto_rationale=sig.rationale,
                    aggregated_score=0.0,
                    agent_signals=signals,
                    consensus_summary=f"OPERAÇÃO VETADA PELO SENTINELA: {sig.rationale}",
                    llm_arbitrated=False,
                )

        # 2. Ponderação dos Votos
        weights = weights_override or self.default_weights
        weighted_score = 0.0
        total_weight = 0.0

        for sig in signals:
            w = float(weights.get(sig.agent_name, sig.weight))
            total_weight += w

            if sig.direction == "BUY":
                weighted_score += (w * sig.confidence)
            elif sig.direction == "SELL":
                weighted_score -= (w * sig.confidence)
            elif sig.direction == "HOLD":
                # HOLD contribui para neutralidade (aproxima score de 0)
                pass

        norm_score = (weighted_score / total_weight) if total_weight > 0 else 0.0

        # 3. Classificação Direcional e Convicção da Maioria
        if norm_score >= self.min_consensus_threshold:
            direction = "BUY"
            buy_weights = [float(weights.get(sig.agent_name, sig.weight)) for sig in signals if sig.direction == "BUY"]
            buy_conf_w = [float(weights.get(sig.agent_name, sig.weight)) * sig.confidence for sig in signals if sig.direction == "BUY"]
            avg_maj_conf = sum(buy_conf_w) / sum(buy_weights) if sum(buy_weights) > 0 else abs(norm_score)

            active_weights = sum(float(weights.get(sig.agent_name, sig.weight)) for sig in signals if sig.direction in ("BUY", "SELL"))
            buy_total_w = sum(buy_weights)
            agreement_ratio = buy_total_w / active_weights if active_weights > 0 else 1.0

            confidence = min(0.99, max(0.51, avg_maj_conf * (0.80 + 0.20 * agreement_ratio)))
            dissidents = [sig.agent_name.replace("Agent", "") for sig in signals if sig.direction == "SELL"]
            dissent_txt = f" | {len(dissidents)} divergente(s): {', '.join(dissidents)}" if dissidents else " | Unânime"
            summary = f"Consenso COMPRADOR aprovado (Score Líquido: {norm_score:+.2f} >= Limiar +{self.min_consensus_threshold:.2f}{dissent_txt})."
        elif norm_score <= -self.min_consensus_threshold:
            direction = "SELL"
            sell_weights = [float(weights.get(sig.agent_name, sig.weight)) for sig in signals if sig.direction == "SELL"]
            sell_conf_w = [float(weights.get(sig.agent_name, sig.weight)) * sig.confidence for sig in signals if sig.direction == "SELL"]
            avg_maj_conf = sum(sell_conf_w) / sum(sell_weights) if sum(sell_weights) > 0 else abs(norm_score)

            active_weights = sum(float(weights.get(sig.agent_name, sig.weight)) for sig in signals if sig.direction in ("BUY", "SELL"))
            sell_total_w = sum(sell_weights)
            agreement_ratio = sell_total_w / active_weights if active_weights > 0 else 1.0

            confidence = min(0.99, max(0.51, avg_maj_conf * (0.80 + 0.20 * agreement_ratio)))
            dissidents = [sig.agent_name.replace("Agent", "") for sig in signals if sig.direction == "BUY"]
            dissent_txt = f" | {len(dissidents)} divergente(s): {', '.join(dissidents)}" if dissidents else " | Unânime"
            summary = f"Consenso VENDEDOR aprovado (Score Líquido: {norm_score:+.2f} <= Limiar {-self.min_consensus_threshold:.2f}{dissent_txt})."
        else:
            direction = "HOLD"
            confidence = 0.50
            summary = f"Sem consenso mínimo unificado (Score Líquido: {norm_score:+.2f} entre limiares +/-{self.min_consensus_threshold:.2f})."

        return ConsensusDecision(
            direction=direction,
            confidence=confidence,
            is_vetoed=False,
            aggregated_score=norm_score,
            agent_signals=signals,
            consensus_summary=summary,
            llm_arbitrated=False,
        )

    async def judge_async(
        self,
        signals: List[AgentSignal],
        weights_override: Optional[Dict[str, float]] = None,
        llm_assistant: Optional[Any] = None,
        use_llm: bool = False,
    ) -> ConsensusDecision:
        """
        Orquestra o julgamento com suporte opcional a árbitro LLM em caso de discórdia acentuada.
        Se `use_llm=False` ou ocorrer timeout (>2.5s), utiliza estritamente o julgamento determinístico.
        """
        base_decision = self.judge_deterministic(signals, weights_override)

        # Se já foi vetado ou LLM não foi solicitada/instanciada, encerra imediatamente
        if base_decision.is_vetoed or not use_llm or llm_assistant is None:
            return base_decision

        # Checa se há discórdia (ex: votos conflitantes BUY e SELL entre agentes com alta convicção)
        directions = {s.direction for s in signals if s.direction in ["BUY", "SELL"] and s.confidence >= 0.60}
        has_discordance = len(directions) > 1

        if not has_discordance:
            return base_decision

        # Tentativa de arbitragem LLM com timeout rigoroso de 2.5s
        try:
            prompt = (
                f"Você é o Juiz de Consenso do Comitê IA Trader. Analise os sinais conflitantes e emita uma decisão final.\n"
                f"Sinais dos Agentes: {[s.to_dict() for s in signals]}\n"
                f"Responda apenas em JSON com: direction ('BUY', 'SELL', 'HOLD'), confidence (0.0 a 1.0) e rationale."
            )
            # Executa com timeout rigoroso
            llm_resp = await asyncio.wait_for(
                llm_assistant.analyze_async(prompt),
                timeout=self.llm_timeout_seconds,
            )
            if llm_resp and isinstance(llm_resp, dict) and "direction" in llm_resp:
                direction = llm_resp["direction"].upper()
                confidence = float(llm_resp.get("confidence", 0.60))
                rationale = llm_resp.get("rationale", "Decisão arbitrada por LLM com sucesso.")
                return ConsensusDecision(
                    direction=direction,
                    confidence=confidence,
                    is_vetoed=False,
                    aggregated_score=base_decision.aggregated_score,
                    agent_signals=signals,
                    consensus_summary=f"ÁRBITRO LLM: {rationale}",
                    llm_arbitrated=True,
                )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout de {self.llm_timeout_seconds}s no árbitro LLM. Ativando fallback determinístico.")
        except Exception as e:
            logger.warning(f"Falha na arbitragem LLM ({e}). Ativando fallback determinístico.")

        return base_decision
