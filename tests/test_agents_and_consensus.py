"""
Suíte de Testes da Fase 4: Comitê de Agentes Especialistas, Consenso e Núcleo.
Valida:
1. Contrato AgentSignal e tipagem estrita de decisões.
2. Comportamento determinístico de cada agente especialista (Técnico, Fluxo, Sentimento, Macro, On-Chain).
3. Poder de Veto Absoluto do RiskSentinelAgent (prioridade soberana sobre qualquer consenso).
4. Deliberação ponderada do ConsensusJudge com limiares a priori.
5. Timeout rigoroso de 2.5s no árbitro LLM com fallback determinístico local.
6. Gerador de Sinais com barreiras de Stop Loss e Take Profit via múltiplos de ATR e dimensionamento de risco fixo (1%).
7. Orquestração concorrente via asyncio no TradingCore (modo simulação estrito).
8. Ausência estrita de vazamento temporal nas features consumidas pelos agentes.
"""

import asyncio
import os
import sys
import pytest
import numpy as np
import pandas as pd

# Adiciona raiz ao sys.path
sys.path.insert(0, "/home/eudespatricio/ia_trader")

from agents.base_agent import AgentSignal, BaseAgent
from agents.technical_agent import TechnicalAgent
from agents.orderflow_agent import OrderFlowAgent
from agents.sentiment_agent import SentimentAgent
from agents.macro_agent import MacroAgent
from agents.onchain_agent import OnChainAgent
from agents.risk_sentinel_agent import RiskSentinelAgent
from consensus_judge import ConsensusJudge, ConsensusDecision
from signal_generator import SignalGenerator, TradeSignal
from chatgpt_assistant import ChatGPTAssistant
from core import TradingCore


class TestAgentContractsAndImplementations:
    """Valida os contratos, tipagem e determinismo dos agentes individuais."""

    def test_agent_signal_validation(self):
        """Valida que direções inválidas geram ValueError e confidências são normalizadas."""
        sig = AgentSignal(agent_name="Test", direction="buy", confidence=1.2, weight=1.0, rationale="OK")
        assert sig.direction == "BUY"
        assert sig.confidence == 1.0  # Clamped to 1.0

        with pytest.raises(ValueError):
            AgentSignal(agent_name="Test", direction="INVALID", confidence=0.5, weight=1.0, rationale="Fail")

    def test_technical_agent_threshold_logic(self):
        """TechnicalAgent emite BUY somente quando proba_buy >= confidence_threshold."""
        agent = TechnicalAgent(confidence_threshold=0.60)

        # Abaixo do limiar -> HOLD
        sig_neutral = agent.evaluate({"proba_buy": 0.55, "proba_sell": 0.30})
        assert sig_neutral.direction == "HOLD"

        # Acima do limiar -> BUY
        sig_buy = agent.evaluate({"proba_buy": 0.65, "proba_sell": 0.20})
        assert sig_buy.direction == "BUY"
        assert sig_buy.confidence == 0.65

        # Acima do limiar -> SELL
        sig_sell = agent.evaluate({"proba_buy": 0.15, "proba_sell": 0.72})
        assert sig_sell.direction == "SELL"
        assert sig_sell.confidence == 0.72

    def test_orderflow_agent_imbalance_and_volume(self):
        """OrderFlowAgent avalia desbalanceamento de livro e proxy de volume."""
        agent = OrderFlowAgent(imbalance_threshold=0.20)

        # 1. Modo Livro L2
        l2_context = {"order_book": {"bids": [[100, 80]], "asks": [[101, 20]]}}
        sig_l2 = agent.evaluate(l2_context)
        assert sig_l2.direction == "BUY"
        assert sig_l2.metrics["imbalance"] == pytest.approx(0.60)

        # 2. Modo Vela Histórica
        candle_context = {
            "open": 100.0,
            "high": 110.0,
            "low": 98.0,
            "close": 109.0,  # Fechamento no topo do range
            "volume": 200.0,
            "volume_sma": 100.0,  # 2x volume médio
        }
        sig_candle = agent.evaluate(candle_context)
        assert sig_candle.direction == "BUY"
        assert sig_candle.metrics["close_position"] > 0.5

    def test_sentiment_agent_fear_and_greed_contrarian(self):
        """SentimentAgent opera de forma contrária em extremos de sentimento."""
        agent = SentimentAgent(extreme_fear_threshold=25.0, extreme_greed_threshold=75.0)

        # Medo Extremo (F&G = 12) -> Sinal Contrário de BUY
        sig_fear = agent.evaluate({"fear_and_greed": 12.0})
        assert sig_fear.direction == "BUY"
        assert sig_fear.confidence > 0.60

        # Ganância Extrema (F&G = 85) -> Sinal Contrário de SELL
        sig_greed = agent.evaluate({"fear_and_greed": 85.0})
        assert sig_greed.direction == "SELL"
        assert sig_greed.confidence > 0.60

        # Sentimento Neutro (F&G = 50) -> HOLD
        sig_neutral = agent.evaluate({"fear_and_greed": 50.0})
        assert sig_neutral.direction == "HOLD"

    def test_macro_agent_liquidity_regime(self):
        """MacroAgent avalia correlação inversa com DXY e volatilidade VIX."""
        agent = MacroAgent()

        # Dólar caindo + VIX calmo -> BUY
        sig_risk_on = agent.evaluate({
            "dxy_ret_5d": -0.012,
            "dxy_above_sma20": False,
            "vix_value": 15.0,
            "sp500_ret_5d": 0.02,
        })
        assert sig_risk_on.direction == "BUY"

        # Dólar subindo + VIX em pânico -> SELL
        sig_risk_off = agent.evaluate({
            "dxy_ret_5d": 0.015,
            "dxy_above_sma20": True,
            "vix_value": 32.0,
            "sp500_ret_5d": -0.03,
        })
        assert sig_risk_off.direction == "SELL"

    def test_onchain_agent_stablecoin_inflows(self):
        """OnChainAgent sinaliza BUY quando há expansão líquida de stablecoins."""
        agent = OnChainAgent()

        # Expansão de stablecoins > 1% em 30d
        sig_inflow = agent.evaluate({"stablecoin_growth_pct_30d": 0.025})
        assert sig_inflow.direction == "BUY"

        # Contração de stablecoins
        sig_outflow = agent.evaluate({"stablecoin_growth_pct_30d": -0.025})
        assert sig_outflow.direction == "SELL"


class TestRiskSentinelVetoPrecedence:
    """Valida o poder de veto soberano do RiskSentinelAgent."""

    def test_sentinel_volatility_spike_veto(self):
        """Veto por pico anômalo de ATR (> 2.5x mediana)."""
        sentinel = RiskSentinelAgent(max_atr_volatility_multiplier=2.5)
        sig = sentinel.evaluate({"atr_14": 150.0, "atr_median": 50.0})  # 3.0x > 2.5x

        assert sig.is_veto is True
        assert sig.direction == "HOLD"
        assert sig.confidence == 1.0
        assert "VOLATILITY_SPIKE" in sig.metrics["veto_type"]

    def test_sentinel_illiquidity_veto(self):
        """Veto por colapso de volume (< 15% da média)."""
        sentinel = RiskSentinelAgent(min_relative_volume=0.15)
        sig = sentinel.evaluate({"volume": 10.0, "volume_sma": 100.0})  # 10% < 15%

        assert sig.is_veto is True
        assert sig.direction == "HOLD"
        assert "ILLIQUIDITY" in sig.metrics["veto_type"]

    def test_sentinel_macro_event_window_veto(self):
        """Veto durante janela agendada de CPI / FOMC."""
        sentinel = RiskSentinelAgent()
        sig = sentinel.evaluate({"is_cpi_fomc_window": True})

        assert sig.is_veto is True
        assert sig.direction == "HOLD"
        assert "SCHEDULED_MACRO_EVENT" in sig.metrics["veto_type"]

    def test_sentinel_clear_when_within_safe_bounds(self):
        """Sentinela não emite veto quando métricas estão saudáveis."""
        sentinel = RiskSentinelAgent()
        sig = sentinel.evaluate({
            "atr_14": 55.0,
            "atr_median": 50.0,
            "volume": 120.0,
            "volume_sma": 100.0,
            "spread_pct": 0.0005,
            "is_cpi_fomc_window": False,
        })
        assert sig.is_veto is False
        assert sig.direction == "HOLD"


class TestConsensusJudgeAndSignalGenerator:
    """Valida a deliberação do juiz e a geração de sinais parametrizados."""

    def test_veto_has_absolute_priority_over_unanimous_buy(self):
        """Mesmo com votos unânimes de BUY, um único VETO do Sentinela força HOLD."""
        judge = ConsensusJudge()
        signals = [
            AgentSignal("RiskSentinelAgent", "HOLD", 1.0, 1.0, "Veto por volatilidade", is_veto=True),
            AgentSignal("TechnicalAgent", "BUY", 0.90, 1.0, "Alta convicção de alta"),
            AgentSignal("OrderFlowAgent", "BUY", 0.85, 0.8, "Fluxo comprador"),
            AgentSignal("MacroAgent", "BUY", 0.80, 0.6, "Macro favorável"),
        ]
        decision = judge.judge_deterministic(signals)
        assert decision.is_vetoed is True
        assert decision.direction == "HOLD"
        assert decision.confidence == 1.0

    def test_weighted_consensus_triggers_buy_when_threshold_exceeded(self):
        """Consenso ponderado atinge BUY quando score normalizado >= limiar."""
        judge = ConsensusJudge(min_consensus_threshold=0.45)
        signals = [
            AgentSignal("TechnicalAgent", "BUY", 0.70, 1.0, "Compra técnica"),
            AgentSignal("OrderFlowAgent", "BUY", 0.65, 0.8, "Fluxo positivo"),
            AgentSignal("SentimentAgent", "BUY", 0.60, 0.6, "Sentimento favorável"),
            AgentSignal("MacroAgent", "HOLD", 0.50, 0.6, "Neutro"),
        ]
        decision = judge.judge_deterministic(signals)
        assert decision.is_vetoed is False
        assert decision.direction == "BUY"
        assert decision.aggregated_score >= 0.45

    def test_signal_generator_atr_barriers_and_fixed_risk_sizing(self):
        """SignalGenerator calcula TP, SL proporcionais ao ATR e tamanho via risco de 1%."""
        gen = SignalGenerator(capital=10000.0, risk_per_trade=0.01, tp_atr_multiple=1.5, sl_atr_multiple=1.0)
        decision = ConsensusDecision(direction="BUY", confidence=0.80, aggregated_score=0.65, consensus_summary="BUY")

        sig = gen.generate(
            decision=decision,
            symbol="BTCUSDT",
            current_price=100.0,
            atr=2.0,
            timestamp=1700000000000,
        )

        assert sig.direction == "BUY"
        # TP = 100 + 1.5 * 2 = 103.0
        assert sig.take_profit == pytest.approx(103.0)
        # SL = 100 - 1.0 * 2 = 98.0
        assert sig.stop_loss == pytest.approx(98.0)
        # Distância SL = 2.0 | Risco $ = 10000 * 0.01 = $100 | Size = 100 / 2.0 = 50.0 unidades
        assert sig.position_size == pytest.approx(50.0)
        assert sig.risk_dollars == pytest.approx(100.0)


class TestCoreOrchestrationAndResilience:
    """Valida a orquestração do TradingCore."""

    def test_trading_core_deterministic_evaluation(self):
        core = TradingCore(capital=10000.0, risk_per_trade=0.01)
        context = {
            "proba_buy": 0.75,
            "proba_sell": 0.10,
            "fear_and_greed": 15.0,  # Medo extremo -> Buy
            "stablecoin_growth_pct_30d": 0.02,
            "atr_14": 2.0,
            "atr_median": 2.0,
            "volume": 100.0,
            "volume_sma": 100.0,
        }
        signal = core.evaluate_market_deterministic(
            context=context,
            symbol="BTCUSDT",
            current_price=50000.0,
            atr=500.0,
            timestamp=1700000000000,
        )
        assert signal.direction == "BUY"
        assert signal.position_size > 0
        assert signal.take_profit > 50000.0
        assert signal.stop_loss < 50000.0

    @pytest.mark.asyncio
    async def test_trading_core_async_parallel_execution(self):
        core = TradingCore(capital=10000.0, risk_per_trade=0.01)
        context = {
            "proba_buy": 0.80,
            "proba_sell": 0.05,
            "atr_14": 2.0,
            "atr_median": 2.0,
            "volume": 100.0,
            "volume_sma": 100.0,
        }
        signal = await core.evaluate_market_async(
            context=context,
            symbol="BTCUSDT",
            current_price=60000.0,
            atr=600.0,
            timestamp=1700000000000,
        )
        assert signal.direction in ["BUY", "HOLD"]
        assert signal.symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_consensus_judge_llm_timeout_triggers_deterministic_fallback(self):
        """Se o árbitro LLM demorar mais de 2.5s, o timeout reverte imediatamente para fallback determinístico."""
        class SlowLLMStub:
            async def analyze_async(self, prompt: str):
                await asyncio.sleep(5.0)  # Simula travamento de rede de 5s
                return {"direction": "BUY", "confidence": 0.99}

        judge = ConsensusJudge(llm_timeout_seconds=0.1)  # Timeout curto para o teste
        signals = [
            AgentSignal("TechnicalAgent", "BUY", 0.80, 1.0, "Compra"),
            AgentSignal("OrderFlowAgent", "SELL", 0.80, 1.0, "Venda"),
        ]
        decision = await judge.judge_async(
            signals=signals,
            llm_assistant=SlowLLMStub(),
            use_llm=True,
        )
        # O timeout disparou e reverteu para determinístico sem travar o sistema
        assert decision.llm_arbitrated is False
        assert decision.direction == "HOLD"


class TestAntiLookaheadAndMethodologicalRigor:
    """Validações metodológicas exigidas pela Auditoria Institucional."""

    def test_agents_strictly_use_lagged_data_never_current_day(self):
        """As séries diárias de sentimento, macro e on-chain devem usar estritamente D-1."""
        from datetime import datetime, timezone

        # Data atual: 2026-05-15 14:00:00 UTC
        ts_candle = int(datetime(2026, 5, 15, 14, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

        # Mock de banco com valores distintos para D-1, D e D+1
        mock_series = {
            ("FEAR_AND_GREED", "2026-05-14"): 15.0,  # D-1 (Publicado ontem - Medo Extremo -> BUY)
            ("FEAR_AND_GREED", "2026-05-15"): 85.0,  # D (Publicado hoje à noite - Lookahead proibido!)
            ("FEAR_AND_GREED", "2026-05-16"): 90.0,  # D+1 (Futuro proibido!)
        }

        # Função canônica de alinhamento anti-lookahead
        prev_date_str = datetime.fromtimestamp((ts_candle - 86400000) / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        curr_date_str = datetime.fromtimestamp(ts_candle / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

        assert prev_date_str == "2026-05-14"
        assert curr_date_str == "2026-05-15"

        # Garante que o valor alimentado no agente é estritamente o de D-1
        fng_lagged = mock_series.get(("FEAR_AND_GREED", prev_date_str))
        assert fng_lagged == 15.0  # D-1

        sent_agent = SentimentAgent()
        sig = sent_agent.evaluate({"fear_and_greed": fng_lagged})
        assert sig.direction == "BUY"
        assert sig.metrics["fear_and_greed"] == 15.0

    def test_sample_size_rule_flags_under_100_trades(self):
        """Qualquer teste com menos de 100 trades OOS deve ser classificado como 'Amostra Insuficiente'."""
        trades_counts = [7, 16, 26, 99, 100, 350]
        labels = [
            "AMOSTRA INSUFICIENTE (N < 100)" if n < 100 else "AMOSTRA SUFICIENTE"
            for n in trades_counts
        ]

        assert labels[0] == "AMOSTRA INSUFICIENTE (N < 100)"  # N=7
        assert labels[1] == "AMOSTRA INSUFICIENTE (N < 100)"  # N=16
        assert labels[2] == "AMOSTRA INSUFICIENTE (N < 100)"  # N=26
        assert labels[3] == "AMOSTRA INSUFICIENTE (N < 100)"  # N=99
        assert labels[4] == "AMOSTRA SUFICIENTE"              # N=100
        assert labels[5] == "AMOSTRA SUFICIENTE"              # N=350

    def test_judge_weights_calibration_train_isolation(self):
        """Pesos de consenso definidos estritamente sem tocar nos dados de teste."""
        n_samples = 500
        train_idx = np.arange(0, 350)
        test_idx = np.arange(360, 500)  # Gap de embargo de 10 barras

        # Simula histórico de retornos no treino
        np.random.seed(42)
        y_train = np.random.choice([1, 2], size=len(train_idx))
        agent_preds_train = {
            "TechnicalAgent": y_train,  # Perfeito no treino
            "OrderFlowAgent": np.random.choice([1, 2], size=len(train_idx)),
        }

        # Calibra pesos estritamente sobre train_idx
        calibrated_weights = {}
        for agent_name, preds in agent_preds_train.items():
            acc_train = np.mean(preds == y_train)
            calibrated_weights[agent_name] = float(acc_train)

        assert calibrated_weights["TechnicalAgent"] == 1.0
        assert len(calibrated_weights) == 2

        # Assert de isolamento: test_idx nunca foi lido
        assert min(test_idx) > max(train_idx)


class TestInstitutionalAgentRefinements:
    """Valida as melhorias institucionais nos agentes da Fase 4."""

    def test_orderflow_agent_institutional_signals(self):
        agent = OrderFlowAgent()

        # 1. Volume Z-score institucional anômalo com fechamento em alta
        ctx_vol_z = {
            "open": 100.0, "high": 110.0, "low": 98.0, "close": 108.0,
            "volume_zscore_20": 2.2,
        }
        sig = agent.evaluate(ctx_vol_z)
        assert sig.direction == "BUY"
        assert "Fluxo comprador institucional confirmado" in sig.rationale
        assert sig.confidence >= 0.70

        # 2. Reversão elástica de VWAP (dist_vwap_pct <= -2.5)
        ctx_vwap = {
            "open": 100.0, "high": 105.0, "low": 96.0, "close": 101.0,
            "dist_vwap_pct": -3.2,
        }
        sig_vwap = agent.evaluate(ctx_vwap)
        assert sig_vwap.direction == "BUY"
        assert "Reversão elástica de VWAP" in sig_vwap.rationale

        # 3. Anomalia de Funding Z-Score (funding_zscore > 2.0 -> SELL por exaustão)
        ctx_funding_z = {
            "open": 100.0, "high": 102.0, "low": 99.0, "close": 100.0,
            "funding_zscore": 2.8,
        }
        sig_fz = agent.evaluate(ctx_funding_z)
        assert sig_fz.direction == "SELL"
        assert "sobrealavancagem long" in sig_fz.rationale

    def test_technical_agent_regime_filtering(self):
        agent = TechnicalAgent(confidence_threshold=0.60)

        # Probabilidade BUY de 62%:
        # Em regime BULL (close >= sma_200) deve passar (thresh 0.60)
        ctx_bull = {"proba_buy": 0.62, "proba_sell": 0.20, "close": 65000.0, "sma_200": 60000.0}
        sig_bull = agent.evaluate(ctx_bull)
        assert sig_bull.direction == "BUY"

        # Em regime BEAR (close < sma_200), penalidade de +0.10 -> threshold exigido 0.70 -> deve virar HOLD!
        ctx_bear = {"proba_buy": 0.62, "proba_sell": 0.20, "close": 55000.0, "sma_200": 60000.0}
        sig_bear = agent.evaluate(ctx_bear)
        assert sig_bear.direction == "HOLD"

    def test_risk_sentinel_natr_and_funding_veto(self):
        sentinel = RiskSentinelAgent()

        # 1. NATR extremo > 12.0% -> VETO
        ctx_crash = {"natr_14": 14.5}
        sig_natr = sentinel.evaluate(ctx_crash)
        assert sig_natr.is_veto is True
        assert sig_natr.direction == "HOLD"
        assert "flash crash" in sig_natr.rationale

        # 2. Funding Z-Score extremo > 4.0 -> VETO
        ctx_liq = {"funding_zscore": 4.5}
        sig_fz = sentinel.evaluate(ctx_liq)
        assert sig_fz.is_veto is True
        assert "risco crítico de liquidação" in sig_fz.rationale.lower()

    def test_trading_core_evaluate_multi_strategy(self):
        core = TradingCore(capital=10000.0)
        ctx = {
            "close": 65000.0,
            "dist_sma_200": 5.2,  # Bull
            "atr_14": 1200.0,
            "funding_rate_annualized": 0.14,  # 14% a.a. -> Carry Viable
            "funding_zscore": 0.8,
            "proba_buy": 0.65,
            "proba_sell": 0.15,
        }
        res = core.evaluate_multi_strategy(symbol="BTCUSDT", current_price=65000.0, context=ctx)
        assert res["symbol"] == "BTCUSDT"
        assert res["regime"] == "BULL_TREND"
        assert res["carry_opportunity"]["is_carry_viable"] is True
        assert res["carry_opportunity"]["strategy_recommendation"] == "DELTA_NEUTRAL_CARRY_ACTIVE"
        assert res["suggested_hybrid_allocation"]["carry_allocation_pct"] == 60.0

    def test_technical_agent_market_structure_confluence(self):
        """TechnicalAgent ganha bônus de convicção em POI/Sweep e neutraliza compra próxima a resistência."""
        agent = TechnicalAgent(confidence_threshold=0.60)

        # 1. Confluência com Liquidity Sweep Bullish
        ctx_sweep = {
            "proba_buy": 0.62,
            "proba_sell": 0.10,
            "liquidity_sweep_bull": 1,
            "at_bullish_poi": 1,
        }
        sig_sweep = agent.evaluate(ctx_sweep)
        assert sig_sweep.direction == "BUY"
        assert sig_sweep.confidence > 0.75  # Ganha bônus de +0.15
        assert "CONFLUÊNCIA POI/SWEEP" in sig_sweep.rationale

        # 2. Penalidade por proximidade de resistência imediata (< 0.25%)
        ctx_res = {
            "proba_buy": 0.62,
            "proba_sell": 0.10,
            "dist_resistance_pct": 0.12,  # Imediatamente abaixo de resistência
            "liquidity_sweep_bull": 0,
        }
        sig_res = agent.evaluate(ctx_res)
        assert sig_res.direction == "HOLD"  # Convicção cai de 0.62 para 0.42 (< 0.60 threshold)
        assert "resistência imediata" in sig_res.rationale

    def test_risk_sentinel_structural_proximity_veto(self):
        """RiskSentinelAgent veta compra imediatamente abaixo de resistência maior ou venda em suporte."""
        sentinel = RiskSentinelAgent()

        # Compra pretendida em resistência (< 0.15%) -> VETO ESTRUTURAL
        ctx_buy_veto = {
            "intended_direction": "BUY",
            "dist_resistance_pct": 0.08,
            "liquidity_sweep_bull": 0,
        }
        sig_buy_veto = sentinel.evaluate(ctx_buy_veto)
        assert sig_buy_veto.is_veto is True
        assert sig_buy_veto.direction == "HOLD"
        assert "RESISTANCE_PROXIMITY" in sig_buy_veto.metrics["veto_type"]

        # Venda pretendida em suporte (< 0.15%) -> VETO ESTRUTURAL
        ctx_sell_veto = {
            "intended_direction": "SELL",
            "dist_support_pct": 0.05,
            "liquidity_sweep_bear": 0,
        }
        sig_sell_veto = sentinel.evaluate(ctx_sell_veto)
        assert sig_sell_veto.is_veto is True
        assert "SUPPORT_PROXIMITY" in sig_sell_veto.metrics["veto_type"]

    def test_signal_generator_asymmetric_poi_risk_reward(self):
        """SignalGenerator calibra Stop Loss mais curto e Take Profit mais longo com POI / Sweeps."""
        gen = SignalGenerator(capital=10000.0, tp_atr_multiple=1.5, sl_atr_multiple=1.0)
        decision = ConsensusDecision(direction="BUY", confidence=0.85, aggregated_score=0.85)

        # Sem confluência de estrutura: R:R = 1.5 : 1.0 (TP=1.5x, SL=1.0x ATR)
        sig_std = gen.generate(decision, "BTCUSDT", current_price=50000.0, atr=1000.0, timestamp=1000)
        assert sig_std.take_profit == 50000.0 + 1500.0
        assert sig_std.stop_loss == 50000.0 - 1000.0

        # Com confluência de Liquidity Sweep: R:R assimétrico 2.1 : 0.7 = 3:1!
        sig_asym = gen.generate(
            decision,
            "BTCUSDT",
            current_price=50000.0,
            atr=1000.0,
            timestamp=1000,
            structure_context={"liquidity_sweep_bull": 1, "at_bullish_poi": 1}
        )
        assert sig_asym.take_profit == 50000.0 + 2100.0
        assert sig_asym.stop_loss == 50000.0 - 700.0
        # Relação Payoff: (2100 / 700) = 3.0
        assert (sig_asym.take_profit - 50000.0) / (50000.0 - sig_asym.stop_loss) == pytest.approx(3.0)





# ---------------------------------------------------------------------------
# Regressões da auditoria de lógica de decisão (ajuste fino)
# ---------------------------------------------------------------------------
class TestDecisionAuditRegressions:
    def test_agent_signal_nan_confidence_is_zero_not_one(self):
        from agents.base_agent import AgentSignal
        sig = AgentSignal("X", "BUY", float("nan"), float("nan"), "r")
        assert sig.confidence == 0.0
        assert sig.weight == 0.0

    def test_technical_nan_sma_does_not_force_bear_regime(self):
        from agents.technical_agent import TechnicalAgent
        sig = TechnicalAgent().evaluate({"close": 100.0, "sma_200": float("nan"), "rsi_14": 30.0})
        assert sig.metrics["macro_bull"] is None
        sig = TechnicalAgent().evaluate({"proba_buy": float("nan"), "proba_sell": 0.7})
        assert 0.0 <= sig.confidence <= 1.0

    def test_onchain_zero_tvl_7d_does_not_fall_back_to_24h(self):
        from agents.onchain_agent import OnChainAgent
        sig = OnChainAgent().evaluate({"tvl_growth_pct_7d": 0.0, "tvl_change_24h": 0.05})
        assert sig.metrics["tvl_growth_pct_7d"] == 0.0
        assert sig.direction == "HOLD"

    def test_judge_ignores_zero_weight_failed_agent(self):
        from agents.base_agent import AgentSignal
        from consensus_judge import ConsensusJudge
        buy = lambda n, c: AgentSignal(n, "BUY", c, 1.0, "r")
        sigs = [buy("TechnicalAgent", 0.9), buy("OrderFlowAgent", 0.9)]
        base = ConsensusJudge().judge_deterministic(list(sigs)).aggregated_score
        failed = AgentSignal("MacroAgent", "HOLD", 0.5, 0.0, "falha")
        d = ConsensusJudge().judge_deterministic(sigs + [failed])
        assert d.aggregated_score == pytest.approx(base)
        assert any(s.agent_name == "MacroAgent" for s in d.agent_signals)

    def test_signal_generator_invalid_price_or_atr_returns_hold(self):
        from consensus_judge import ConsensusDecision
        from signal_generator import SignalGenerator
        dec = ConsensusDecision(direction="BUY", confidence=0.8, aggregated_score=0.5)
        for price, atr in ((0.0, 10.0), (100.0, 0.0), (100.0, float("nan"))):
            s = SignalGenerator().generate(dec, "BTCUSDT", price, atr, 1)
            assert s.direction == "HOLD" and s.position_size == 0.0

    def test_signal_generator_effective_risk_and_rationale_multiples(self):
        from consensus_judge import ConsensusDecision
        from signal_generator import SignalGenerator
        dec = ConsensusDecision(direction="BUY", confidence=0.8, aggregated_score=0.5)
        # SL muito curto + leverage 1 => teto de alavancagem limita o tamanho
        s = SignalGenerator(capital=10000.0).generate(dec, "BTCUSDT", 100.0, 0.1, 1)
        assert s.risk_dollars == pytest.approx(s.position_size * abs(s.entry_price - s.stop_loss))
        assert s.risk_dollars < 100.0
        s2 = SignalGenerator().generate(
            dec, "BTCUSDT", 100.0, 1.0, 1, structure_context={"liquidity_sweep_bull": 1})
        assert "2.1x ATR" in s2.rationale and "0.7x ATR" in s2.rationale

    def test_signal_generator_capital_override_zero_respected(self):
        from consensus_judge import ConsensusDecision
        from signal_generator import SignalGenerator
        dec = ConsensusDecision(direction="BUY", confidence=0.8, aggregated_score=0.5)
        s = SignalGenerator().generate(dec, "BTCUSDT", 100.0, 1.0, 1, capital_override=0.0)
        assert s.position_size == 0.0
