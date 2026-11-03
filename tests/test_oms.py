"""
tests/test_oms.py - Testes de Unidade e Integração do Order Management System (OMS).
Verifica a State Machine, Idempotência Determinística, Reconciliação Forense e Kill Switch.
"""

import os
import tempfile
import pytest
import sqlite3
import time
from unittest.mock import patch

from oms import (
    OrderManagementSystem,
    OrderState,
    OrderIntent,
    VALID_STATE_TRANSITIONS,
)
from smart_trade import PaperTradingEngine, TradeSignal


@pytest.fixture
def temp_oms_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def temp_kill_switch():
    fd, path = tempfile.mkstemp(suffix=".switch")
    os.close(fd)
    if os.path.exists(path):
        os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def oms_instance(temp_oms_db, temp_kill_switch):
    return OrderManagementSystem(db_path=temp_oms_db, kill_switch_path=temp_kill_switch)


class TestOMSStateMachine:
    """Validação da máquina de estados do OMS."""

    def test_valid_state_transitions(self, oms_instance):
        oms = oms_instance
        intent, is_new = oms.register_order_intent(
            symbol="BTCUSDT",
            side="BUY",
            price=50000.0,
            quantity=0.1,
            notional=5000.0,
            leverage=2.0,
            strategy_type="DIRECTIONAL",
            timestamp_ms=1700000000000,
            position_id="POS_BTC_01",
        )
        assert is_new is True
        assert intent is not None
        assert intent.state == OrderState.ORDER_INTENT

        # ORDER_INTENT -> SENT
        assert oms.transition_state(intent.client_order_id, OrderState.SENT, "Sent to venue") is True
        # SENT -> ACKED
        assert oms.transition_state(intent.client_order_id, OrderState.ACKED, "Ack received") is True
        # ACKED -> PARTIALLY_FILLED
        assert oms.transition_state(
            intent.client_order_id,
            OrderState.PARTIALLY_FILLED,
            "Partial fill 50%",
            filled_qty=0.05,
            avg_fill_price=50001.0,
        ) is True
        # PARTIALLY_FILLED -> FILLED
        assert oms.transition_state(
            intent.client_order_id,
            OrderState.FILLED,
            "Complete fill 100%",
            filled_qty=0.1,
            avg_fill_price=50000.5,
        ) is True
        # FILLED -> CLOSED
        assert oms.transition_state(intent.client_order_id, OrderState.CLOSED, "Position closed") is True
        # CLOSED -> RECONCILED
        assert oms.transition_state(intent.client_order_id, OrderState.RECONCILED, "Reconciled") is True

    def test_invalid_state_transition_is_rejected(self, oms_instance):
        oms = oms_instance
        intent, _ = oms.register_order_intent(
            symbol="ETHUSDT",
            side="BUY",
            price=3000.0,
            quantity=1.0,
            notional=3000.0,
            leverage=1.0,
            strategy_type="SCALP",
            timestamp_ms=1700000001000,
            position_id="POS_ETH_01",
        )
        assert intent is not None
        # Transição inválida: ORDER_INTENT -> CLOSED não é permitida sem passar por SENT/ACKED/FILLED
        success = oms.transition_state(intent.client_order_id, OrderState.CLOSED, "Tentativa de pular estados")
        assert success is False

        # Verifica que o estado no banco permanece ORDER_INTENT
        with oms._get_connection() as conn:
            row = conn.execute("SELECT state FROM oms_orders WHERE client_order_id = ?", (intent.client_order_id,)).fetchone()
            assert row["state"] == OrderState.ORDER_INTENT.value

    def test_cancellation_and_rejection_flows(self, oms_instance):
        oms = oms_instance
        intent, _ = oms.register_order_intent(
            symbol="SOLUSDT",
            side="SELL",
            price=150.0,
            quantity=10.0,
            notional=1500.0,
            leverage=1.0,
            strategy_type="SCALP",
            timestamp_ms=1700000002000,
            position_id="POS_SOL_01",
        )
        assert intent is not None
        # ORDER_INTENT -> REJECTED
        assert oms.transition_state(intent.client_order_id, OrderState.REJECTED, "Rejeitada pelo risk engine") is True
        # REJECTED -> RECONCILED
        assert oms.transition_state(intent.client_order_id, OrderState.RECONCILED, "Auditada e arquivada") is True


class TestOMSIdempotency:
    """Validação de idempotência determinística de ordens."""

    def test_duplicate_submission_returns_existing_order(self, oms_instance):
        oms = oms_instance
        ts = 1700000003000
        # Primeira submissão
        intent1, is_new1 = oms.register_order_intent(
            symbol="BTCUSDT",
            side="BUY",
            price=50000.0,
            quantity=0.1,
            notional=5000.0,
            leverage=2.0,
            strategy_type="DIRECTIONAL",
            timestamp_ms=ts,
            position_id="POS_DUP_01",
        )
        assert is_new1 is True
        assert intent1 is not None

        # Segunda submissão idêntica (mesmo symbol, side, strategy e timestamp)
        intent2, is_new2 = oms.register_order_intent(
            symbol="BTCUSDT",
            side="BUY",
            price=50000.0,
            quantity=0.1,
            notional=5000.0,
            leverage=2.0,
            strategy_type="DIRECTIONAL",
            timestamp_ms=ts,
            position_id="POS_DUP_01",
        )
        assert is_new2 is False
        assert intent2 is not None
        assert intent1.client_order_id == intent2.client_order_id
        assert intent1.idempotency_key == intent2.idempotency_key

        # Verifica que apenas uma linha existe na tabela oms_orders
        with oms._get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM oms_orders").fetchone()[0]
            assert count == 1


class TestOMSKillSwitch:
    """Validação do Kill Switch local e remoto."""

    def test_file_kill_switch_blocks_orders(self, oms_instance):
        oms = oms_instance
        active, _ = oms.is_kill_switch_active()
        assert active is False

        # Ativa kill switch via arquivo
        oms.activate_kill_switch("Risco de volatilidade extrema detectado")
        active, reason = oms.is_kill_switch_active()
        assert active is True
        assert "volatilidade" in reason

        # Tentativa de registrar ordem deve retornar None, False
        intent, is_new = oms.register_order_intent(
            symbol="BTCUSDT",
            side="BUY",
            price=50000.0,
            quantity=0.1,
            notional=5000.0,
            leverage=1.0,
            strategy_type="DIRECTIONAL",
            timestamp_ms=1700000004000,
            position_id="POS_KS_01",
        )
        assert intent is None
        assert is_new is False

        # Desativa kill switch
        oms.deactivate_kill_switch()
        active, _ = oms.is_kill_switch_active()
        assert active is False

        # Agora ordem pode ser registrada
        intent_ok, is_new_ok = oms.register_order_intent(
            symbol="BTCUSDT",
            side="BUY",
            price=50000.0,
            quantity=0.1,
            notional=5000.0,
            leverage=1.0,
            strategy_type="DIRECTIONAL",
            timestamp_ms=1700000004000,
            position_id="POS_KS_01",
        )
        assert is_new_ok is True
        assert intent_ok is not None

    def test_env_kill_switch_blocks_orders(self, oms_instance, monkeypatch):
        oms = oms_instance
        monkeypatch.setenv("IA_TRADER_KILL_SWITCH", "1")
        active, reason = oms.is_kill_switch_active()
        assert active is True
        assert "IA_TRADER_KILL_SWITCH" in reason

        intent, is_new = oms.register_order_intent(
            symbol="BTCUSDT",
            side="BUY",
            price=50000.0,
            quantity=0.1,
            notional=5000.0,
            leverage=1.0,
            strategy_type="DIRECTIONAL",
            timestamp_ms=1700000005000,
            position_id="POS_KS_ENV",
        )
        assert intent is None
        assert is_new is False


class TestOMSStartupReconciliation:
    """Validação da reconciliação de inicialização."""

    def test_reconciliation_cleans_unconfirmed_orders(self, oms_instance):
        oms = oms_instance
        # Cria ordem pendente no banco simulando crash antes de finalização
        intent, _ = oms.register_order_intent(
            symbol="BTCUSDT",
            side="BUY",
            price=50000.0,
            quantity=0.1,
            notional=5000.0,
            leverage=1.0,
            strategy_type="DIRECTIONAL",
            timestamp_ms=1700000006000,
            position_id="POS_RECON_01",
        )
        oms.transition_state(intent.client_order_id, OrderState.SENT, "Sent to venue")

        # Roda reconciliação de boot
        report = oms.reconcile_on_startup(active_positions={}, trade_history=[])
        assert report.orphan_orders_found >= 1
        assert report.resolved_count >= 1
        assert report.status == "RECONCILED_WITH_WARNINGS"

        # Verifica que a ordem no banco passou para EXPIRED
        with oms._get_connection() as conn:
            row = conn.execute("SELECT state FROM oms_orders WHERE client_order_id = ?", (intent.client_order_id,)).fetchone()
            assert row["state"] == OrderState.EXPIRED.value

    def test_reconciliation_detects_orphan_positions(self, oms_instance):
        oms = oms_instance
        # Cria uma posição OPEN no banco sem nenhum trade correspondente
        with oms._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_positions (
                    position_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    side TEXT,
                    status TEXT,
                    strategy_type TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_trades (
                    trade_id TEXT PRIMARY KEY,
                    position_id TEXT,
                    action TEXT
                )
            """)
            conn.execute("""
                INSERT INTO paper_positions (position_id, symbol, side, status, strategy_type)
                VALUES ('POS_ORPHAN_99', 'SOLUSDT', 'LONG', 'OPEN', 'DIRECTIONAL')
            """)
            conn.commit()

        report = oms.reconcile_on_startup(active_positions={}, trade_history=[])
        assert report.orphan_positions_found == 1
        assert "POS_ORPHAN_99" in report.divergence_details[0]


class TestOMSIntegrationWithSmartTrade:
    """Validação da integração de ponta a ponta do OMS com o PaperTradingEngine."""

    def test_smart_trade_registers_oms_order_on_open(self, temp_oms_db):
        engine = PaperTradingEngine(
            initial_capital=10000.0,
            db_path=temp_oms_db,
            simulation_mode=True,
        )
        signal = TradeSignal(
            signal_id="SIG_OMS_TEST",
            timestamp=1700000010000,
            symbol="BTCUSDT",
            direction="BUY",
            entry_price=50000.0,
            confidence=0.85,
            consensus_score=0.80,
            position_size=1000.0,
            risk_dollars=100.0,
            rationale="OMS integration test",
            stop_loss=49000.0,
            take_profit=52000.0,
            atr=500.0,
        )

        pos = engine._open_ladder_position(
            signal=signal,
            current_price=50000.0,
            timestamp=1700000010000,
            strategy_type="DIRECTIONAL",
            leverage=2.0,
        )

        assert pos is not None
        assert pos.status == "OPEN"

        # Verifica que a ordem foi registrada e avançou no OMS
        with engine.oms._get_connection() as conn:
            orders = conn.execute("SELECT * FROM oms_orders WHERE symbol = 'BTCUSDT'").fetchall()
            assert len(orders) >= 1
            order = orders[0]
            assert order["position_id"] == pos.position_id
            assert order["state"] == OrderState.FILLED.value
            assert float(order["filled_quantity"]) > 0

            # Verifica histórico de transições
            transitions = conn.execute(
                "SELECT * FROM oms_state_transitions WHERE client_order_id = ? ORDER BY id ASC",
                (order["client_order_id"],)
            ).fetchall()
            states = [t["to_state"] for t in transitions]
            assert OrderState.ORDER_INTENT.value in states
            assert OrderState.SENT.value in states
            assert OrderState.FILLED.value in states

        # Agora fecha a posição e verifica que o OMS transita para CLOSED
        engine.close_position(
            position_id=pos.position_id,
            exit_price=51000.0,
            timestamp=1700000020000,
            reason="TAKE_PROFIT_COMPLETE",
        )

        with engine.oms._get_connection() as conn:
            closed_order = conn.execute(
                "SELECT state FROM oms_orders WHERE position_id = ?",
                (pos.position_id,)
            ).fetchone()
            assert closed_order["state"] == OrderState.CLOSED.value
