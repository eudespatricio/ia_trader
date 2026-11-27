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
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

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
        veto_only_agents: Optional[set] = None,
        max_contextual_weight_ratio: Optional[float] = 0.35,
        require_technical_confirmation: bool = True,
        block_primary_opposition: bool = True,
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
        self.contextual_agents = (
            contextual_agents if contextual_agents is not None else {"SentimentAgent", "MacroAgent", "OnChainAgent"}
        )
        self.min_directional_agents = min_directional_agents
        self.agent_calibration = dict(agent_calibration) if agent_calibration else {}
        # Veto-only sentinels emit HOLD when clear; counting that HOLD in total_weight dilutes directional consensus.
        self.veto_only_agents = veto_only_agents if veto_only_agents is not None else {"RiskSentinelAgent"}
        # Cap active contextual agents (slow D-1 Macro/OnChain/Sentiment) relative to active primary agents so
        # persistent macro/on-chain regimes cannot mathematically block counter-trend setups.
        self.max_contextual_weight_ratio = self._validate_contextual_ratio(max_contextual_weight_ratio)
        self.require_technical_confirmation = bool(require_technical_confirmation)
        self.block_primary_opposition = bool(block_primary_opposition)

    @staticmethod
    def _validate_contextual_ratio(ratio: Optional[float]) -> Optional[float]:
        if ratio is None:
            return None
        val = float(ratio)
        if not math.isfinite(val) or val < 0.0:
            raise ValueError(f"max_contextual_weight_ratio inválido: {ratio!r} (esperado None ou float finito >= 0.0)")
        return val

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

        weighted_entries: List[Tuple[AgentSignal, float]] = []
        for sig in signals:
            # Abstenção não-punitiva: agentes sem sinal, contextuais neutros ou sentinelas sem veto não diluem o denominador
            if sig.direction == "ABSTAIN":
                continue
            if self.allow_contextual_abstain and sig.agent_name in self.contextual_agents and sig.direction == "HOLD":
                continue
            if sig.agent_name in self.veto_only_agents and sig.direction == "HOLD":
                continue

            w = float(weights.get(sig.agent_name, sig.weight))
            if not math.isfinite(w) or w <= 0.0:
                continue
            weighted_entries.append((sig, w))

        # Limita proporcionalmente o peso efetivo dos agentes contextuais frente aos primários ativos
        ratio = self._validate_contextual_ratio(self.max_contextual_weight_ratio)
        if ratio is not None:
            primary_weight = sum(
                w
                for s, w in weighted_entries
                if s.agent_name not in self.contextual_agents and s.agent_name not in self.veto_only_agents
            )
            contextual_weight = sum(
                w for s, w in weighted_entries if s.agent_name in self.contextual_agents
            )
            max_ctx_weight = ratio * primary_weight
            if primary_weight > 0.0 and contextual_weight > max_ctx_weight:
                scale = (max_ctx_weight / contextual_weight) if contextual_weight > 0.0 else 0.0
                weighted_entries = [
                    (s, w * scale if s.agent_name in self.contextual_agents else w)
                    for s, w in weighted_entries
                ]
                # Remove entradas cujo peso efetivo caiu para zero para não inflar quórum ou dissidentes
                weighted_entries = [(s, w) for s, w in weighted_entries if w > 0.0]

        total_weight = sum(w for _, w in weighted_entries)
        weighted_score = 0.0
        for sig, w in weighted_entries:
            if sig.direction == "BUY":
                weighted_score += w * sig.confidence
            elif sig.direction == "SELL":
                weighted_score -= w * sig.confidence
            elif sig.direction == "HOLD":
                # HOLD de agente primário ativo contribui para neutralidade (aproxima score de 0)
                pass

        norm_score = (weighted_score / total_weight) if total_weight > 0.0 else 0.0

        # 3. Classificação Direcional com Quórum Mínimo e Confirmação Primária
        buy_entries = [(s, w) for s, w in weighted_entries if s.direction == "BUY"]
        sell_entries = [(s, w) for s, w in weighted_entries if s.direction == "SELL"]
        tech_sig = next((s for s, _ in weighted_entries if s.agent_name == "TechnicalAgent"), None)
        flow_sig = next((s for s, _ in weighted_entries if s.agent_name == "OrderFlowAgent"), None)

        if norm_score >= self.min_consensus_threshold:
            direction, confidence, summary = self._resolve_directional_side(
                target_dir="BUY",
                opposite_dir="SELL",
                maj_entries=buy_entries,
                weighted_entries=weighted_entries,
                norm_score=norm_score,
                tech_sig=tech_sig,
                flow_sig=flow_sig,
            )
        elif norm_score <= -self.min_consensus_threshold:
            direction, confidence, summary = self._resolve_directional_side(
                target_dir="SELL",
                opposite_dir="BUY",
                maj_entries=sell_entries,
                weighted_entries=weighted_entries,
                norm_score=norm_score,
                tech_sig=tech_sig,
                flow_sig=flow_sig,
            )
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

    def _resolve_directional_side(
        self,
        *,
        target_dir: str,
        opposite_dir: str,
        maj_entries: List[Tuple[AgentSignal, float]],
        weighted_entries: List[Tuple[AgentSignal, float]],
        norm_score: float,
        tech_sig: Optional[AgentSignal],
        flow_sig: Optional[AgentSignal],
    ) -> Tuple[str, float, str]:
        """Valida quórum, confirmação técnica, ausência de oposição primária e calcula confiança."""
        noun = "comprador(es)" if target_dir == "BUY" else "vendedor(es)"
        if len(maj_entries) < self.min_directional_agents:
            return (
                "HOLD",
                0.50,
                (
                    f"Quórum direcional insuficiente: {len(maj_entries)} agente(s) {noun} "
                    f"< mínimo exigido ({self.min_directional_agents}). Operação bloqueada em HOLD."
                ),
            )
        # Impede que fluxo + contextuais disparem entrada sem o sinal técnico confirmar a direção.
        if self.require_technical_confirmation and tech_sig is not None and tech_sig.direction != target_dir:
            return (
                "HOLD",
                0.50,
                f"Confirmação técnica ausente para {target_dir} (TechnicalAgent={tech_sig.direction}). Operação bloqueada em HOLD.",
            )
        if (
            self.block_primary_opposition
            and tech_sig is not None
            and flow_sig is not None
            and tech_sig.direction in ("BUY", "SELL")
            and flow_sig.direction in ("BUY", "SELL")
            and tech_sig.direction != flow_sig.direction
        ):
            return (
                "HOLD",
                0.50,
                f"Conflito entre agentes primários (TechnicalAgent={tech_sig.direction} vs OrderFlowAgent={flow_sig.direction}). Operação bloqueada em HOLD.",
            )

        maj_total_w = sum(w for _, w in maj_entries)
        maj_conf_w = sum(w * s.confidence for s, w in maj_entries)
        avg_maj_conf = maj_conf_w / maj_total_w if maj_total_w > 0.0 else abs(norm_score)

        active_weights = sum(w for s, w in weighted_entries if s.direction in ("BUY", "SELL"))
        agreement_ratio = maj_total_w / active_weights if active_weights > 0.0 else 1.0

        confidence = min(0.99, max(0.51, avg_maj_conf * (0.80 + 0.20 * agreement_ratio)))
        dissidents = [s.agent_name.replace("Agent", "") for s, _ in weighted_entries if s.direction == opposite_dir]
        dissent_txt = f" | {len(dissidents)} divergente(s): {', '.join(dissidents)}" if dissidents else " | Unânime"
        if target_dir == "BUY":
            summary = f"Consenso COMPRADOR aprovado (Score Líquido: {norm_score:+.2f} >= Limiar +{self.min_consensus_threshold:.2f}{dissent_txt})."
        else:
            summary = f"Consenso VENDEDOR aprovado (Score Líquido: {norm_score:+.2f} <= Limiar {-self.min_consensus_threshold:.2f}{dissent_txt})."
        return target_dir, confidence, summary

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
                # Não possui autoridade para criar direção a partir de HOLD nem inverter BUY <-> SELL.
                if direction in ("BUY", "SELL") and direction != base_decision.direction:
                    logger.warning(
                        "Árbitro LLM sugeriu '%s', divergindo da decisão determinística '%s'. "
                        "Mantendo decisão determinística de segurança.",
                        direction,
                        base_decision.direction,
                    )
                    return base_decision

                if base_decision.direction == "HOLD" and direction == "HOLD":
                    return base_decision

                confidence = float(llm_resp.get("confidence", 0.60))
                confidence = 0.0 if confidence != confidence else max(0.0, min(1.0, confidence))
                rationale = llm_resp.get("rationale", "Decisão auditada por LLM com sucesso.")
                return ConsensusDecision(
                    direction=direction,
                    confidence=confidence,
                    is_vetoed=False,
                    aggregated_score=base_decision.aggregated_score,
                    agent_signals=base_decision.agent_signals,
                    consensus_summary=f"ÁRBITRO LLM: {rationale}",
                    llm_arbitrated=True,
                )
        except asyncio.TimeoutError:
            logger.warning(f"Timeout de {self.llm_timeout_seconds}s no árbitro LLM. Ativando fallback determinístico.")
        except Exception as e:
            logger.warning(f"Falha na arbitragem LLM ({e}). Ativando fallback determinístico.")

        return base_decision
