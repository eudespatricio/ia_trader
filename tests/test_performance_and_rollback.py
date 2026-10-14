"""
Suíte de Testes da Fase 7: Telemetria Financeira, Rastreio de Performance e Guardião de Rollback.
Testa:
1. PerformanceTracker: PnL, Drawdown, Sharpe, Sortino, Calmar, Streaks e decomposição por estratégia.
2. RollbackGuard: Circuit breakers de Drawdown, Perdas Consecutivas (Cooldown), Auto-Rollback por Degradação e Parada de Emergência.
3. Integração PaperTradingEngine: Bloqueio de novas posições sob trava ativa e atualização pós-fechamento.
4. Endpoints Flask de Telemetria e Circuit Breaker.
"""

import os
import sys
import time
import sqlite3
import pytest
import numpy as np

sys.path.insert(0, "/home/eudespatricio/ia_trader")

from performance_tracker import PerformanceTracker, PerformanceMetrics
from rollback_guard import RollbackGuard, CircuitBreakerStatus
from smart_trade import PaperTradingEngine, PaperPosition, PaperTrade
from flask_interface import app


@pytest.fixture
def temp_db(tmp_path):
    """Cria um banco SQLite temporário isolado para testes da Fase 7."""
    db_file = str(tmp_path / "test_telemetry.db")
    return db_file


class TestPerformanceTracker:
    """Valida o cálculo exato e determinístico de métricas financeiras da Fase 7."""

    def test_empty_database_initial_metrics(self, temp_db):
        tracker = PerformanceTracker(db_path=temp_db, initial_capital=10000.0)
        m = tracker.compute_metrics_from_db()
        assert m.initial_capital == 10000.0
        assert m.current_equity == 10000.0
        assert m.net_pnl == 0.0
        assert m.win_rate == 0.0
        assert m.total_trades == 0
        assert m.profit_factor == 0.0
        assert m.max_drawdown_pct == 0.0

    def test_metrics_computation_with_trades_and_equity(self, temp_db):
        tracker = PerformanceTracker(db_path=temp_db, initial_capital=10000.0)

        # Popula o banco com histórico simulado
        conn = sqlite3.connect(temp_db)
        now = int(time.time() * 1000)

        # 1. paper_portfolio
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_portfolio (
                timestamp INTEGER, equity REAL, balance REAL, margin_used REAL,
                unrealized_pnl REAL, realized_pnl_cum REAL, total_fees_paid REAL, open_positions_count INTEGER
            )
        """)
        # Simula equity: 10000 -> 10500 (pico) -> 10100 (drawdown) -> 10800
        conn.execute("INSERT INTO paper_portfolio VALUES (?, 10000.0, 10000.0, 0, 0, 0, 0, 0)", (now - 4000,))
        conn.execute("INSERT INTO paper_portfolio VALUES (?, 10500.0, 10500.0, 0, 0, 500, 5, 1)", (now - 3000,))
        conn.execute("INSERT INTO paper_portfolio VALUES (?, 10100.0, 10100.0, 0, 0, 100, 10, 1)", (now - 2000,))
        conn.execute("INSERT INTO paper_portfolio VALUES (?, 10800.0, 10800.0, 0, 0, 800, 15, 0)", (now - 1000,))

        # 2. paper_trades (3 trades: +500, -400, +700)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                trade_id TEXT, position_id TEXT, symbol TEXT, action TEXT,
                price REAL, quantity REAL, fee REAL, pnl REAL, timestamp INTEGER, rationale TEXT
            )
        """)
        conn.execute("INSERT INTO paper_trades VALUES ('t1', 'p1', 'BTCUSDT', 'TAKE_PROFIT', 50000, 0.1, 2.0, 500.0, ?, 'SCALP win')", (now - 3000,))
        conn.execute("INSERT INTO paper_trades VALUES ('t2', 'p2', 'ETHUSDT', 'STOP_LOSS', 3000, 1.0, 2.0, -400.0, ?, 'SCALP loss')", (now - 2000,))
        conn.execute("INSERT INTO paper_trades VALUES ('t3', 'p3', 'SOLUSDT', 'TP2_FULL', 150, 5.0, 2.0, 700.0, ?, 'GOLDEN_POT win')", (now - 1000,))
        conn.commit()
        conn.close()

        m = tracker.compute_metrics_from_db()

        assert m.current_equity == 10800.0
        assert m.net_pnl == 800.0
        assert m.net_pnl_pct == pytest.approx(8.0)
        assert m.total_trades == 3
        assert m.winning_trades == 2
        assert m.losing_trades == 1
        assert m.win_rate == pytest.approx(66.67, rel=1e-2)

        # Gross profit = 1200, Gross loss = 400 -> PF = 3.0
        assert m.profit_factor == pytest.approx(3.0)
        assert m.avg_win == 600.0
        assert m.avg_loss == 400.0
        assert m.payoff_ratio == pytest.approx(1.5)

        # Drawdown de 10500 para 10100 = 400 dólares -> ~3.81%
        assert m.peak_equity == 10800.0
        assert m.max_drawdown_dollars == pytest.approx(400.0)
        assert m.max_drawdown_pct == pytest.approx((400.0 / 10500.0) * 100.0, rel=1e-2)

        # Breakdown de estratégias
        assert "SCALP" in m.strategy_metrics
        assert m.strategy_metrics["SCALP"]["trades"] == 2
        assert m.strategy_metrics["SCALP"]["wins"] == 1
        assert "GOLDEN_POT" in m.strategy_metrics
        assert m.strategy_metrics["GOLDEN_POT"]["trades"] == 1
        assert m.strategy_metrics["GOLDEN_POT"]["wins"] == 1

    def test_record_snapshot_persistence(self, temp_db):
        tracker = PerformanceTracker(db_path=temp_db, initial_capital=10000.0)
        metrics = PerformanceMetrics(
            current_equity=10500.0,
            current_balance=10500.0,
            net_pnl=500.0,
            net_pnl_pct=5.0,
            max_drawdown_pct=1.5,
            win_rate=60.0,
            profit_factor=2.0,
            total_trades=5,
            timestamp=int(time.time() * 1000)
        )
        tracker.record_snapshot(metrics)

        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT * FROM performance_snapshots ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()

        assert row is not None
        assert row[2] == 10500.0  # equity
        assert row[4] == 500.0    # net_pnl


class TestRollbackGuard:
    """Valida o funcionamento dos Circuit Breakers e Rollback Automático."""

    def test_circuit_breaker_normal_state(self, temp_db):
        guard = RollbackGuard(db_path=temp_db)
        allowed, reason = guard.can_open_new_position("BTCUSDT", "SCALP")
        assert allowed is True
        assert guard.status.state == "CLEAR"

    def test_circuit_breaker_drawdown_limit_trips(self, temp_db):
        guard = RollbackGuard(db_path=temp_db, max_total_drawdown_pct=10.0)
        metrics = {"current_drawdown_pct": 10.5, "max_drawdown_pct": 10.5}

        status = guard.evaluate_telemetry(metrics)
        assert status.state == "TRIPPED"
        assert status.is_trading_allowed is False
        assert "CIRCUIT BREAKER DISPARADO" in status.reason

        # Nova tentativa de abertura é sumariamente rejeitada
        allowed, reason = guard.can_open_new_position("BTCUSDT", "SCALP")
        assert allowed is False
        assert "TRIPPED" in reason or "bloqueado" in reason.lower()

    def test_circuit_breaker_consecutive_losses_triggers_cooldown(self, temp_db):
        guard = RollbackGuard(db_path=temp_db, max_consecutive_losses=4, cooldown_seconds=3600)
        metrics = {
            "current_drawdown_pct": 3.0,
            "current_streak_type": "LOSS",
            "current_streak_count": 4,
            "max_consecutive_losses": 4
        }

        status = guard.evaluate_telemetry(metrics)
        assert status.state == "COOLDOWN"
        assert status.is_trading_allowed is False
        assert status.cooldown_until > int(time.time() * 1000)

        # Checagem de posição bloqueada em cooldown
        allowed, reason = guard.can_open_new_position("ETHUSDT", "SCALP")
        assert allowed is False
        assert "Cooldown" in reason

    def test_degradation_auto_rollback(self, temp_db):
        guard = RollbackGuard(
            db_path=temp_db,
            min_trades_for_degradation=10,
            min_win_rate_threshold=35.0,
            auto_rollback_enabled=True
        )
        # Amostra com 15 trades e win rate degradado de 20%
        metrics = {
            "total_trades": 15,
            "win_rate": 20.0,
            "profit_factor": 0.45,
            "current_drawdown_pct": 4.0
        }

        status = guard.evaluate_telemetry(metrics)
        assert status.is_rollback_active is True
        assert "ROLLBACK_TO_DEFENSIVE_MODE" in status.actions_taken

    def test_emergency_stop_and_manual_reset(self, temp_db):
        guard = RollbackGuard(db_path=temp_db)

        # 1. Parada de emergência manual
        guard.trigger_emergency_stop(reason="Teste de segurança do operador")
        assert guard.status.state == "EMERGENCY_HALT"
        assert guard.status.is_trading_allowed is False

        allowed, reason = guard.can_open_new_position("BTCUSDT", "SCALP")
        assert allowed is False

        # 2. Reset manual supervisionado
        guard.reset_circuit_breaker(operator_note="Auditoria concluída e autorizada")
        assert guard.status.state == "CLEAR"
        assert guard.status.is_trading_allowed is True

        allowed_after, _ = guard.can_open_new_position("BTCUSDT", "SCALP")
        assert allowed_after is True


class TestEngineIntegrationAndFlaskAPI:
    """Valida a integração entre PaperTradingEngine, Circuit Breakers e Endpoints Flask."""

    def test_engine_rejects_trade_when_circuit_breaker_tripped(self, temp_db):
        engine = PaperTradingEngine(initial_capital=10000.0, db_path=temp_db)
        assert engine.rollback_guard is not None

        # Dispara manualmente o circuit breaker
        engine.rollback_guard.trigger_emergency_stop("Simulação de colapso de mercado")

        # Testa verificação de cotas
        now = int(time.time() * 1000)
        can_scalp, reason_scalp = engine.can_execute_scalp(now)
        assert can_scalp is False
        assert "Circuit Breaker" in reason_scalp

        can_golden, reason_golden = engine.can_execute_golden_pot(now)
        assert can_golden is False
        assert "Circuit Breaker" in reason_golden

    def test_flask_telemetry_endpoints(self, temp_db):
        # Configura cliente Flask de teste
        client = app.test_client()

        # 1. GET /api/telemetry/performance
        res_perf = client.get("/api/telemetry/performance")
        assert res_perf.status_code == 200
        data_perf = res_perf.get_json()
        assert "current_equity" in data_perf
        assert "win_rate" in data_perf

        # 2. GET /api/telemetry/circuit-breaker
        res_cb = client.get("/api/telemetry/circuit-breaker")
        assert res_cb.status_code == 200
        data_cb = res_cb.get_json()
        assert "state" in data_cb
        assert data_cb["state"] in ("CLEAR", "COOLDOWN", "TRIPPED", "EMERGENCY_HALT")

        # 3. POST /api/telemetry/circuit-breaker/reset
        res_reset = client.post("/api/telemetry/circuit-breaker/reset", json={"note": "Reset de teste"})
        assert res_reset.status_code == 200
        assert res_reset.get_json()["status"] == "SUCCESS"
