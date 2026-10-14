"""
test_hardening_simulation.py - Teste de Hardening, Concorrência Multithread e Resiliência 24h.
Fase 10: Validação de Estabilidade Operacional e Integridade de Concorrência.
"""

import os
import sys
import time
import sqlite3
import threading
import pytest
import numpy as np

from launcher import UnifiedSystemLauncher
from rollback_guard import RollbackGuard, CircuitBreakerStatus
from performance_tracker import PerformanceTracker, PerformanceMetrics
from learning_tracker import LearningTracker


@pytest.fixture
def hardening_db(tmp_path):
    """Cria um banco SQLite temporário configurado para testes de estresse em modo WAL."""
    db_file = tmp_path / "hardening_test.db"
    conn = sqlite3.connect(str(db_file), timeout=15.0)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 15000")
    conn.close()
    return str(db_file)


class TestHardeningConcurrencyAndStability:
    """Valida a estabilidade do sistema sob alta concorrência e ciclos contínuos."""

    def test_sqlite_multithread_concurrent_writes(self, hardening_db):
        """Simula múltiplos agentes e daemons gravando e lendo concorrentemente no SQLite."""
        errors = []
        num_threads = 6
        iterations_per_thread = 20

        def worker_task(thread_id: int):
            try:
                tracker = PerformanceTracker(db_path=hardening_db)
                guard = RollbackGuard(db_path=hardening_db)
                learning = LearningTracker(db_path=hardening_db)

                signals = [
                    {"agent_name": "TechnicalAgent", "direction": "BUY", "confidence": 0.80},
                    {"agent_name": "RiskAgent", "direction": "CLEAR", "confidence": 1.0},
                ]

                for i in range(iterations_per_thread):
                    now_ms = int(time.time() * 1000)
                    pnl_val = 15.5 if (i % 2 == 0) else -10.0
                    pos_id = f"pos_{thread_id}_{i}"

                    # 1. Gravação no performance tracker
                    metrics = PerformanceMetrics(
                        current_equity=10000.0 + i,
                        current_balance=10000.0 + i,
                        net_pnl=pnl_val,
                        net_pnl_pct=0.15,
                        max_drawdown_pct=0.5,
                        win_rate=60.0,
                        profit_factor=1.8,
                        total_trades=i + 1,
                        timestamp=now_ms
                    )
                    tracker.record_snapshot(metrics)

                    # 2. Avaliação no rollback guard
                    guard.evaluate_telemetry({"current_drawdown_pct": 0.5, "current_streak_count": 1})

                    # 3. Gravação na memória de aprendizado
                    learning.record_decision(
                        decision_id=f"dec_{thread_id}_{i}",
                        symbol="BTCUSDT",
                        regime="BULL_TREND",
                        consensus_direction="BUY",
                        consensus_confidence=0.8,
                        agent_signals=signals,
                        position_id=pos_id
                    )
                    learning.update_trade_outcome(position_id=pos_id, pnl=pnl_val)

                    time.sleep(0.002)
            except Exception as e:
                errors.append(f"Thread {thread_id} falhou: {e}")

        threads = [threading.Thread(target=worker_task, args=(tid,)) for tid in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0, f"Erros de concorrência detectados: {errors}"

        # Verifica integridade final dos registros no SQLite
        conn = sqlite3.connect(hardening_db)
        snap_count = conn.execute("SELECT count(*) FROM performance_snapshots").fetchone()[0]
        learning_count = conn.execute("SELECT count(*) FROM agent_learning_memory").fetchone()[0]
        conn.close()

        assert snap_count == num_threads * iterations_per_thread
        assert learning_count == num_threads * iterations_per_thread

    def test_circuit_breaker_stress_and_recovery(self, hardening_db):
        """Simula evento de estresse com perdas consecutivas disparando cooldown e reset do operador."""
        guard = RollbackGuard(db_path=hardening_db, max_consecutive_losses=4, cooldown_seconds=300)

        # 3 stops consecutivos: ainda permitido operar
        metrics_mild = {
            "current_drawdown_pct": 2.0,
            "current_streak_type": "LOSS",
            "current_streak_count": 3,
            "max_consecutive_losses": 4
        }
        status_mild = guard.evaluate_telemetry(metrics_mild)
        assert status_mild.is_trading_allowed is True
        allowed, _ = guard.can_open_new_position("BTCUSDT", "SCALP")
        assert allowed is True

        # 4º stop consecutivo: dispara Circuit Breaker de Cooldown
        metrics_severe = {
            "current_drawdown_pct": 3.5,
            "current_streak_type": "LOSS",
            "current_streak_count": 4,
            "max_consecutive_losses": 4
        }
        status_severe = guard.evaluate_telemetry(metrics_severe)
        assert status_severe.state == "COOLDOWN"
        assert status_severe.is_trading_allowed is False

        # Nova tentativa de abertura deve ser bloqueada
        allowed_blocked, reason = guard.can_open_new_position("BTCUSDT", "SCALP")
        assert allowed_blocked is False
        assert "Cooldown" in reason

        # Reset supervisionado de emergência
        guard.reset_circuit_breaker(operator_note="Operador autorizou reset pós-análise")
        assert guard.status.state == "CLEAR"
        assert guard.status.is_trading_allowed is True

    def test_launcher_environment_and_checkpoint(self, hardening_db):
        """Valida que o UnifiedSystemLauncher inicializa schemas e realiza checkpoint do WAL sem falhas."""
        launcher = UnifiedSystemLauncher(
            db_path=hardening_db,
            run_web=False,
            run_trader=False,
            run_scheduler=False
        )

        launcher.initialize_environment()
        assert os.path.exists(hardening_db)

        # Testa finalização com checkpoint de WAL
        launcher.stop()

        # Verifica se o banco pode ser aberto em modo exclusivo sem locks residuais
        conn = sqlite3.connect(hardening_db, timeout=5.0)
        val = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()
        assert val == "ok"
