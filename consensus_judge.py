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
        llm_timeout_seconds: float = 1.0,
        allow_contextual_abstain: bool = True,
        contextual_agents: Optional[set] = None,
        min_directional_agents: int = 2,
        agent_calibration: Optional[Dict[str, Dict[str, float]]] = None,
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
        self.allow_contextual_abstain = allow_contextual_abstain
        self.contextual_agents = contextual_agents or {"SentimentAgent", "MacroAgent", "OnChainAgent"}
        self.min_directional_agents = min_directional_agents
        self.agent_calibration = dict(agent_calibration) if agent_calibration else {}

    def set_agent_calibration(self, calib: Dict[str, Dict[str, float]]) -> None:
        """Define ou atualiza mapa de calibração empírica dos agentes."""
        self.agent_calibration = dict(calib)

    def _apply_confidence_calibration(
        self,
        signals: List[AgentSignal],
        regime: Optional[str] = None,
        agent_calibration: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[AgentSignal]:
        """
        Nível 1 & Nível 2 do Protocolo S2 (docs/S2_CALIBRACAO_AGENTES_BACKTEST.md):
        Dampening de confiança para agentes com sobreconfiança severa em qualquer regime explícito,
        incluindo SIDEWAYS_RANGE e demais regimes (ex.: TREND_FOLLOWING, BREAKOUT).
        """
        calib_map = agent_calibration if agent_calibration is not None else self.agent_calibration
        if not signals:
            return signals

        calibrated_signals = []

        for sig in signals:
            if sig.is_veto:
                calibrated_signals.append(sig)
                continue

            conf = float(sig.confidence)
            should_dampen = False
            dampen_reason = ""
            calibrated_conf = conf

            # Busca informações de calibração: primeiro no mapa, depois nas métricas do próprio sinal
            c_info = calib_map.get(sig.agent_name) if calib_map else None
            if c_info is None and getattr(sig, "metrics", None):
                c_info = sig.metrics

            if c_info:
                # Guarda de amostragem: não aplicar dampening se houver poucas amostras
                n_samples = c_info.get("total_samples")
                if n_samples is not None and int(n_samples) < 5:
                    calibrated_signals.append(sig)
                    continue

                acc = float(c_info.get("accuracy", 1.0))
                ece = float(c_info.get("ece", 0.0))

                if regime == "SIDEWAYS_RANGE":
                    # Patch S4 (v1.0.6-s4): Dampening em SIDEWAYS_RANGE só penaliza sobreconfiança
                    # improdutiva (ece > 0.15 com acurácia < 55%) ou baixa assertividade (< 45%).
                    # Agentes com acurácia empírica vencedora (>= 55%, ex.: 64.3% na S3) não são
                    # travados em 0.55 (o que causava deadlock abaixo dos mínimos de 0.65/0.70/0.75).
                    if ece > 0.15 and acc < 0.55:
                        should_dampen = True
                        dampen_reason = f"ECE {ece:.2f} > 0.15 (acc={acc:.1%}) em SIDEWAYS_RANGE"
                        calibrated_conf = min(0.55, conf)
                    elif acc < 0.45:
                        should_dampen = True
                        dampen_reason = f"Acurácia {acc:.1%} < 45% em SIDEWAYS_RANGE"
                        calibrated_conf = min(0.55, conf)
                elif regime:
                    # Demais regimes explícitos: penalizar sobreconfiança severa (ECE > 0.20 e acurácia < 50%)
                    # ou acurácia muito baixa (< 40%).
                    if (ece > 0.20 and acc < 0.50) or (acc < 0.40):
                        should_dampen = True
                        target_cap = 0.55 if acc < 0.35 else min(0.58, conf * max(0.65, acc + 0.25))
                        calibrated_conf = min(conf, target_cap)
                        dampen_reason = f"ECE {ece:.2f} e acc {acc:.1%} em {regime}"

            if should_dampen:
                from dataclasses import replace
                new_sig = replace(sig, confidence=calibrated_conf)
                new_metrics = dict(sig.metrics) if getattr(sig, "metrics", None) else {}
                new_metrics["raw_confidence"] = conf
                new_metrics["calibrated_dampened"] = True
                new_metrics["dampen_reason"] = dampen_reason
                new_sig.metrics = new_metrics
                calibrated_signals.append(new_sig)
            else:
                calibrated_signals.append(sig)

        return calibrated_signals

    def judge_deterministic(
        self,
        signals: List[AgentSignal],
        weights_override: Optional[Dict[str, float]] = None,
        regime: Optional[str] = None,
        agent_calibration: Optional[Dict[str, Dict[str, float]]] = None,
        **kwargs: Any,
    ) -> ConsensusDecision:
        """
        Julgamento matemático rápido e determinístico com calibração de confiança.
        Executado no backtest e como fallback do árbitro LLM.
        """
        signals = self._apply_confidence_calibration(signals, regime=regime, agent_calibration=agent_calibration)

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
        # Agentes sinalizados com weight<=0 (ex.: fallback por falha em core.evaluate_market_async)
        # não devem diluir o denominador nem receber peso default do dicionário.
        all_signals = signals
        signals = [s for s in signals if s.weight > 0.0 or s.is_veto]
        weighted_score = 0.0
        total_weight = 0.0

        for sig in signals:
            # Abstenção não-punitiva: agentes sem sinal ou contextuais neutros não diluem o denominador
            if sig.direction == "ABSTAIN":
                continue
            if self.allow_contextual_abstain and sig.agent_name in self.contextual_agents and sig.direction == "HOLD":
                continue

            w = float(weights.get(sig.agent_name, sig.weight))
            total_weight += w

            if sig.direction == "BUY":
                weighted_score += (w * sig.confidence)
            elif sig.direction == "SELL":
                weighted_score -= (w * sig.confidence)
            elif sig.direction == "HOLD":
                # HOLD de agente primário ativo contribui para neutralidade (aproxima score de 0)
                pass

        norm_score = (weighted_score / total_weight) if total_weight > 0 else 0.0

        # 3. Classificação Direcional com Quórum Mínimo Obrigatório
        buy_signals = [sig for sig in signals if sig.direction == "BUY"]
        sell_signals = [sig for sig in signals if sig.direction == "SELL"]

        if norm_score >= self.min_consensus_threshold:
            if len(buy_signals) < self.min_directional_agents:
                direction = "HOLD"
                confidence = 0.50
                summary = (
                    f"Quórum direcional insuficiente: {len(buy_signals)} agente(s) comprador(es) "
                    f"< mínimo exigido ({self.min_directional_agents}). Operação bloqueada em HOLD."
                )
            else:
                direction = "BUY"
                buy_weights = [float(weights.get(sig.agent_name, sig.weight)) for sig in buy_signals]
                buy_conf_w = [float(weights.get(sig.agent_name, sig.weight)) * sig.confidence for sig in buy_signals]
                avg_maj_conf = sum(buy_conf_w) / sum(buy_weights) if sum(buy_weights) > 0 else abs(norm_score)

                active_weights = sum(float(weights.get(sig.agent_name, sig.weight)) for sig in signals if sig.direction in ("BUY", "SELL"))
                buy_total_w = sum(buy_weights)
                agreement_ratio = buy_total_w / active_weights if active_weights > 0 else 1.0

                confidence = min(0.99, max(0.51, avg_maj_conf * (0.80 + 0.20 * agreement_ratio)))
                dissidents = [sig.agent_name.replace("Agent", "") for sig in signals if sig.direction == "SELL"]
                dissent_txt = f" | {len(dissidents)} divergente(s): {', '.join(dissidents)}" if dissidents else " | Unânime"
                summary = f"Consenso COMPRADOR aprovado (Score Líquido: {norm_score:+.2f} >= Limiar +{self.min_consensus_threshold:.2f}{dissent_txt})."
        elif norm_score <= -self.min_consensus_threshold:
            if len(sell_signals) < self.min_directional_agents:
                direction = "HOLD"
                confidence = 0.50
                summary = (
                    f"Quórum direcional insuficiente: {len(sell_signals)} agente(s) vendedor(es) "
                    f"< mínimo exigido ({self.min_directional_agents}). Operação bloqueada em HOLD."
                )
            else:
                direction = "SELL"
                sell_weights = [float(weights.get(sig.agent_name, sig.weight)) for sig in sell_signals]
                sell_conf_w = [float(weights.get(sig.agent_name, sig.weight)) * sig.confidence for sig in sell_signals]
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
            agent_signals=all_signals,
            consensus_summary=summary,
            llm_arbitrated=False,
        )

    async def judge_async(
        self,
        signals: List[AgentSignal],
        weights_override: Optional[Dict[str, float]] = None,
        llm_assistant: Optional[Any] = None,
        use_llm: bool = False,
        regime: Optional[str] = None,
        agent_calibration: Optional[Dict[str, Dict[str, float]]] = None,
        **kwargs: Any,
    ) -> ConsensusDecision:
        """
        Orquestra o julgamento com suporte opcional a árbitro LLM em caso de discórdia acentuada.
        Se `use_llm=False` ou ocorrer timeout (>4.5s), utiliza estritamente o julgamento determinístico.
        """
        base_decision = self.judge_deterministic(
            signals,
            weights_override=weights_override,
            regime=regime,
            agent_calibration=agent_calibration,
            **kwargs,
        )

        # Se já foi vetado ou LLM não foi solicitada/instanciada, encerra imediatamente
        if base_decision.is_vetoed or not use_llm or llm_assistant is None:
            return base_decision

        # Checa se há discórdia (ex: votos conflitantes BUY e SELL entre agentes com alta convicção)
        # ou se o consenso é limítrofe com setup técnico ativo
        directions = {s.direction for s in signals if s.direction in ["BUY", "SELL"] and s.confidence >= 0.60}
        has_discordance = len(directions) > 1
        is_borderline = (
            base_decision.direction == "HOLD"
            and abs(base_decision.aggregated_score) >= 0.20
            and any(s.direction in ["BUY", "SELL"] and s.confidence >= 0.65 for s in signals)
        )

        if not has_discordance and not is_borderline:
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
                direction = str(llm_resp["direction"]).upper()
                if direction not in ("BUY", "SELL", "HOLD"):
                    raise ValueError(f"direção inválida do LLM: {direction!r}")

                # P0: O LLM atua estritamente como auditor de risco/sanidade.
                # Não possui autoridade para criar BUY ou SELL a partir de HOLD determinístico.
                if base_decision.direction == "HOLD" and direction in ("BUY", "SELL"):
                    logger.warning(
                        "Árbitro LLM sugeriu '%s', mas o comitê determinístico não atingiu quórum quantitativo. "
                        "Mantendo HOLD de segurança.", direction
                    )
                    return base_decision

                confidence = float(llm_resp.get("confidence", 0.60))
                confidence = 0.0 if confidence != confidence else max(0.0, min(1.0, confidence))
                rationale = llm_resp.get("rationale", "Decisão auditada por LLM com sucesso.")
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
