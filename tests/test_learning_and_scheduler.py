"""
Suíte de Testes de Aprendizado Adaptativo, Memória e Re-treinamento Contínuo.
Valida:
1. LearningTracker: registro de decisões episódicas, atualização de desfechos de trade e recalibração dinâmica de pesos por regime.
2. ModelEvolutionManager: protocolo Champion-Challenger e critérios de gatilho para retreinamento.
3. TaskScheduler: ciclo assíncrono de tarefas e manutenção de cache.
"""

import asyncio
import os
import sys
import time
import sqlite3
import pytest

sys.path.insert(0, "/home/eudespatricio/ia_trader")

from learning_tracker import LearningTracker, ModelEvolutionManager
from scheduler import TaskScheduler


@pytest.fixture
def temp_db(tmp_path):
    """Cria banco SQLite temporário isolado."""
    return str(tmp_path / "test_learning.db")


class TestLearningTracker:
    """Valida a memória de decisões e o feedback loop adaptativo da IA."""

    def test_record_decision_and_update_outcome(self, temp_db):
        tracker = LearningTracker(db_path=temp_db)

        # 1. Registra deliberação de comitê
        signals = [
            {"agent_name": "TechnicalAgent", "direction": "BUY", "confidence": 0.80},
            {"agent_name": "OrderFlowAgent", "direction": "BUY", "confidence": 0.75},
            {"agent_name": "MacroAgent", "direction": "SELL", "confidence": 0.60},
        ]
        tracker.record_decision(
            decision_id="dec_001",
            symbol="BTCUSDT",
            regime="BULL_TREND",
            consensus_direction="BUY",
            consensus_confidence=0.78,
            agent_signals=signals,
            position_id="pos_123"
        )

        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT * FROM agent_learning_memory WHERE decision_id = 'dec_001'").fetchone()
        conn.close()

        assert row is not None
        assert row[3] == "BULL_TREND"
        assert row[7] == "PENDING"

        # 2. Atualiza desfecho positivo (WIN de +$50.0)
        tracker.update_trade_outcome(position_id="pos_123", pnl=50.0)

        conn = sqlite3.connect(temp_db)
        row_updated = conn.execute("SELECT outcome, realized_pnl FROM agent_learning_memory WHERE position_id = 'pos_123'").fetchone()
        conn.close()

        assert row_updated[0] == "WIN"
        assert row_updated[1] == 50.0

    def test_adaptive_weight_recalibration_by_regime(self, temp_db):
        tracker = LearningTracker(db_path=temp_db)

        # Simula 6 decisões em SIDEWAYS_RANGE onde OrderFlow acerta e Macro erra
        for i in range(6):
            signals = [
                {"agent_name": "OrderFlowAgent", "direction": "BUY", "confidence": 0.85},
                {"agent_name": "MacroAgent", "direction": "SELL", "confidence": 0.70},
                {"agent_name": "TechnicalAgent", "direction": "BUY", "confidence": 0.65},
            ]
            pos_id = f"pos_sim_{i}"
            tracker.record_decision(
                decision_id=f"dec_{i}",
                symbol="BTCUSDT",
                regime="SIDEWAYS_RANGE",
                consensus_direction="BUY",
                consensus_confidence=0.75,
                agent_signals=signals,
                position_id=pos_id
            )
            # Todos foram WIN (confirmando o acerto do OrderFlow e erro do Macro)
            tracker.update_trade_outcome(position_id=pos_id, pnl=25.0)

        weights = tracker.get_calibrated_weights_for_regime("SIDEWAYS_RANGE")

        # OrderFlow acertou 6/6 (100%) -> seu peso deve ter aumentado acima do base (0.8 -> ~1.4)
        assert weights["OrderFlowAgent"] > tracker.DEFAULT_BASE_WEIGHTS["OrderFlowAgent"]
        # Macro errou 6/6 (0%) -> seu peso deve ter sido penalizado para baixo do base (0.7 -> ~0.35)
        assert weights["MacroAgent"] < tracker.DEFAULT_BASE_WEIGHTS["MacroAgent"]
        # RiskSentinel sempre mantém prioridade de veto
        assert weights["RiskSentinelAgent"] == 1.5


class TestModelEvolutionManager:
    """Valida os critérios do protocolo Champion-Challenger para retreinamento."""

    def test_should_trigger_retraining_rules(self, temp_db):
        manager = ModelEvolutionManager(db_path=temp_db, retrain_cadence_candles=1000)

        now_ms = int(time.time() * 1000)
        recent_ts = now_ms - (86400 * 1000 * 2)   # 2 dias atrás
        old_ts = now_ms - (86400 * 1000 * 16)      # 16 dias atrás

        # Menos de 1000 velas e recente -> Não retreina
        assert manager.should_trigger_retraining(recent_ts, now_ms, new_candles_count=500) is False

        # Mais de 1000 novas velas -> Retreina imediatamente
        assert manager.should_trigger_retraining(recent_ts, now_ms, new_candles_count=1200) is True

        # Mais de 14 dias sem retreino -> Retreina mesmo com poucas velas
        assert manager.should_trigger_retraining(old_ts, now_ms, new_candles_count=200) is True

    def test_champion_challenger_promotion_logic(self, temp_db):
        manager = ModelEvolutionManager(db_path=temp_db, min_improvement_pct=0.5)

        # 1. Challenger com melhora de 2.0% em AUC -> APROVADO
        promoted, reason = manager.evaluate_challenger(champion_metric=0.5500, challenger_metric=0.5610, metric_name="roc_auc")
        assert promoted is True
        assert "PROMOÇÃO APROVADA" in reason

        # 2. Challenger estagnado ou pior -> REJEITADO
        promoted_worse, reason_worse = manager.evaluate_challenger(champion_metric=0.5500, challenger_metric=0.5480, metric_name="roc_auc")
        assert promoted_worse is False
        assert "PROMOÇÃO REJEITADA" in reason_worse


@pytest.mark.asyncio
async def test_scheduler_lifecycle(temp_db):
    """Valida inicialização e cancelamento seguro do TaskScheduler."""
    scheduler = TaskScheduler(db_path=temp_db)
    await scheduler.start()
    assert scheduler.is_running is True
    assert len(scheduler._tasks) == 3

    # Pausa breve para validar execução não-bloqueante
    await asyncio.sleep(0.05)

    await scheduler.stop()
    assert scheduler.is_running is False
    assert len(scheduler._tasks) == 0
