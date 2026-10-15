"""
Núcleo Orquestrador Central (Core Orchestrator) da Fase 4.
Coordena a execução concorrente dos agentes especialistas, a deliberação do Juiz de Consenso
e a geração do sinal operacional parametrizado.

Princípios de Segurança e Governança:
1. Modo SOMENTE SIMULAÇÃO ativo e estrito: Nenhuma ordem real ou chamada autenticada de trading.
2. Prioridade Absoluta do Sentinela de Risco: Veto verificado antes de qualquer arbitragem.
3. Concorrência Assíncrona via asyncio: Agentes executados em paralelo sem gargalos de I/O.
4. Fallback Determinístico: Sem dependência obrigatória de LLM ou conectividade externa não-determinística.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent, AgentSignal
from agents.technical_agent import TechnicalAgent
from agents.orderflow_agent import OrderFlowAgent
from agents.sentiment_agent import SentimentAgent
from agents.macro_agent import MacroAgent
from agents.onchain_agent import OnChainAgent
from agents.risk_sentinel_agent import RiskSentinelAgent
from consensus_judge import ConsensusJudge, ConsensusDecision
from signal_generator import SignalGenerator, TradeSignal
from chatgpt_assistant import ChatGPTAssistant

logger = logging.getLogger("ia_trader.core")


class TradingCore:
    """
    Orquestrador central do Comitê Multi-Agente do IA Trader.
    """

    def __init__(
        self,
        capital: float = 10000.0,
        risk_per_trade: float = 0.01,
        min_consensus_threshold: float = 0.30,
        simulation_mode: bool = True,
        technical_agent: Optional[TechnicalAgent] = None,
        llm_assistant: Optional[ChatGPTAssistant] = None,
    ):
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.simulation_mode = simulation_mode

        # Inicialização do Sentinela de Risco (com poder de veto)
        self.risk_sentinel = RiskSentinelAgent()

        # Inicialização dos Agentes Especialistas
        self.technical_agent = technical_agent or TechnicalAgent()
        self.orderflow_agent = OrderFlowAgent()
        self.sentiment_agent = SentimentAgent()
        self.macro_agent = MacroAgent()
        self.onchain_agent = OnChainAgent()

        # Comitê completo
        self.agents: List[BaseAgent] = [
            self.risk_sentinel,
            self.technical_agent,
            self.orderflow_agent,
            self.sentiment_agent,
            self.macro_agent,
            self.onchain_agent,
        ]

        # Juiz de Consenso e Gerador de Sinais
        self.consensus_judge = ConsensusJudge(min_consensus_threshold=min_consensus_threshold)
        self.signal_generator = SignalGenerator(capital=capital, risk_per_trade=risk_per_trade)
        self.llm_assistant = llm_assistant or ChatGPTAssistant(enabled=False)

    def evaluate_market_deterministic(
        self,
        context: Dict[str, Any],
        symbol: str = "BTCUSDT",
        current_price: float = 0.0,
        atr: float = 0.0,
        timestamp: int = 0,
        weights_override: Optional[Dict[str, float]] = None,
    ) -> TradeSignal:
        """
        Avaliação determinística síncrona de alta performance para Backtesting Institucional.
        """
        # 1. Checagem prioritária do Sentinela de Risco
        sentinel_signal = self.risk_sentinel.evaluate(context)
        if sentinel_signal.is_veto:
            decision = ConsensusDecision(
                direction="HOLD",
                confidence=1.0,
                is_vetoed=True,
                veto_rationale=sentinel_signal.rationale,
                aggregated_score=0.0,
                agent_signals=[sentinel_signal],
                consensus_summary=f"OPERAÇÃO VETADA PELO SENTINELA: {sentinel_signal.rationale}",
            )
            return self.signal_generator.generate(
                decision=decision,
                symbol=symbol,
                current_price=current_price,
                atr=atr,
                timestamp=timestamp,
            )

        # 2. Avaliação dos demais agentes
        signals: List[AgentSignal] = [sentinel_signal]
        for agent in self.agents:
            if agent.name != self.risk_sentinel.name and agent.is_backtestable:
                signals.append(agent.evaluate(context))

        # 3. Deliberação do Juiz de Consenso
        decision = self.consensus_judge.judge_deterministic(signals, weights_override)

        # 4. Geração do Sinal Operacional
        return self.signal_generator.generate(
            decision=decision,
            symbol=symbol,
            current_price=current_price,
            atr=atr,
            timestamp=timestamp,
        )

    async def evaluate_market_async(
        self,
        context: Dict[str, Any],
        symbol: str = "BTCUSDT",
        current_price: float = 0.0,
        atr: float = 0.0,
        timestamp: int = 0,
        use_llm: bool = False,
    ) -> TradeSignal:
        """
        Avaliação assíncrona concorrente para Live Trading / Daemon de simulação.
        Executa agentes em paralelo via asyncio.gather mantendo a prioridade do veto.
        """
        # 1. Executa o Sentinela primeiro
        sentinel_signal = await self.risk_sentinel.evaluate_async(context)
        if sentinel_signal.is_veto:
            decision = ConsensusDecision(
                direction="HOLD",
                confidence=1.0,
                is_vetoed=True,
                veto_rationale=sentinel_signal.rationale,
                aggregated_score=0.0,
                agent_signals=[sentinel_signal],
                consensus_summary=f"OPERAÇÃO VETADA PELO SENTINELA: {sentinel_signal.rationale}",
            )
            return self.signal_generator.generate(
                decision=decision,
                symbol=symbol,
                current_price=current_price,
                atr=atr,
                timestamp=timestamp,
            )

        # 2. Executa os demais agentes em paralelo
        other_agents = [a for a in self.agents if a.name != self.risk_sentinel.name]
        tasks = [agent.evaluate_async(context) for agent in other_agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals: List[AgentSignal] = [sentinel_signal]
        for res, agent in zip(results, other_agents):
            if isinstance(res, Exception):
                logger.error(f"Erro no agente {agent.name}: {res}. Fallback neutro emitido.")
                signals.append(AgentSignal(
                    agent_name=agent.name,
                    direction="HOLD",
                    confidence=0.5,
                    weight=0.0,
                    rationale=f"Falha na execução: {res}",
                ))
            elif isinstance(res, AgentSignal):
                signals.append(res)

        # 3. Deliberação do Juiz de Consenso (com timeout LLM opcional de 2.5s)
        decision = await self.consensus_judge.judge_async(
            signals=signals,
            llm_assistant=self.llm_assistant,
            use_llm=use_llm,
        )

        # 4. Geração do Sinal Operacional
        return self.signal_generator.generate(
            decision=decision,
            symbol=symbol,
            current_price=current_price,
            atr=atr,
            timestamp=timestamp,
        )

    def evaluate_multi_strategy(
        self,
        symbol: str,
        current_price: float,
        context: Dict[str, Any],
        annualized_funding_rate: float = 0.0,
        timestamp: int = 0,
    ) -> Dict[str, Any]:
        """
        Avaliação Multiestratégia Institucional Integrada:
        1. Estratégia S1: Funding Carry Delta-Neutro (Spot Long + Perp Short).
        2. Estratégia S2: Direcional Estrutural via Comitê com features curadas e filtro de regime.
        3. Alocação Híbrida: Balanço dinâmico entre rendimento de funding e alfa direcional.
        """
        # 1. Avalia o sinal direcional do comitê com os filtros institucionais
        directional_signal = self.evaluate_market_deterministic(
            context=context,
            symbol=symbol,
            current_price=current_price,
            atr=float(context.get("atr_14", current_price * 0.02)),
            timestamp=timestamp,
        )

        # 2. Avalia a oportunidade de Funding Carry Delta-Neutro
        funding_ann = float(annualized_funding_rate or context.get("funding_rate_annualized", 0.0))
        funding_z = float(context.get("funding_zscore", 0.0))

        # Critérios homologados na Fase R:
        # Entra em carry se funding anualizado >= 8% a.a. e Z-score de funding >= -1.0
        carry_viable = funding_ann >= 0.08 and funding_z >= -1.0
        carry_exit_warning = funding_ann < 0.02 or funding_z < -2.0

        # Regime macro dominante (SMA 200)
        dist200 = float(context.get("dist_sma_200", 0.0) or 0.0)
        if dist200 != dist200:  # NaN >= 0 é False -> classificaria BEAR espuriamente
            dist200 = 0.0
        macro_bull = dist200 >= 0.0
        regime_str = "BULL_TREND" if macro_bull else "BEAR_TREND"

        carry_rec = (
            "DELTA_NEUTRAL_CARRY_ACTIVE" if carry_viable else
            ("CLOSE_CARRY" if carry_exit_warning else "HOLD_CARRY_NEUTRAL")
        )

        # Alocação teórica de capital (Paper Trading)
        carry_w = 0.60 if carry_viable else 0.0
        dir_w = 0.40 if directional_signal.direction in ["BUY", "SELL"] else 0.0
        cash_w = max(0.0, 1.0 - carry_w - dir_w)

        return {
            "symbol": symbol,
            "timestamp": timestamp,
            "current_price": current_price,
            "regime": regime_str,
            "directional_signal": directional_signal.to_dict(),
            "carry_opportunity": {
                "annualized_funding_pct": round(funding_ann * 100.0, 2),
                "funding_zscore": round(funding_z, 2),
                "is_carry_viable": carry_viable,
                "carry_exit_warning": carry_exit_warning,
                "strategy_recommendation": carry_rec,
            },
            "suggested_hybrid_allocation": {
                "carry_allocation_pct": round(carry_w * 100.0, 1),
                "directional_allocation_pct": round(dir_w * 100.0, 1),
                "cash_reserve_pct": round(cash_w * 100.0, 1),
            },
        }

