"""Regressão: o motor ao vivo deve alimentar agent_learning_memory / agent_regime_weights / performance_snapshots."""
import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from smart_trade import PaperTradingEngine
from signal_generator import TradeSignal

T0 = 1_700_000_000_000


@pytest.fixture
def engine():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    eng = PaperTradingEngine(initial_capital=10000.0, db_path=path, max_concurrent_positions=20)
    yield eng
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(path + ext):
            os.remove(path + ext)


def _sig(direction="BUY", price=1000.0, atr=10.0):
    agents = [
        {"agent_name": "TechnicalAgent", "direction": direction, "confidence": 0.8},
        {"agent_name": "FlowAgent", "direction": direction, "confidence": 0.7},
    ]
    return TradeSignal(
        signal_id="t", timestamp=T0, symbol="BTCUSDT", direction=direction, entry_price=price,
        stop_loss=price - 2 * atr if direction == "BUY" else price + 2 * atr,
        take_profit=price + 4 * atr, atr=atr, confidence=0.9, consensus_score=0.8,
        position_size=1000.0, risk_dollars=50.0, rationale="wiring", contributing_agents=agents,
    )


def test_open_records_decision_and_close_updates_outcome(engine):
    pos = engine.execute_directional_signal(_sig(), 1000.0, T0)
    assert pos is not None
    con = sqlite3.connect(engine.db_path)
    row = con.execute("SELECT outcome, regime, agent_signals_json FROM agent_learning_memory WHERE position_id=?",
                      (pos.position_id,)).fetchone()
    assert row is not None and row[0] == "PENDING" and "TechnicalAgent" in row[2]
    engine.close_position(pos.position_id, 1100.0, T0 + 60_000, reason="MANUAL_CLOSE")
    out = con.execute("SELECT outcome FROM agent_learning_memory WHERE position_id=?", (pos.position_id,)).fetchone()[0]
    assert out in ("WIN", "LOSS", "BREAKEVEN") and out != "PENDING"
    con.close()


def test_portfolio_snapshot_feeds_performance_snapshots(engine):
    pos = engine.execute_directional_signal(_sig(), 1000.0, T0)
    assert pos is not None
    engine._record_portfolio_snapshot(T0, force=True)
    engine._record_portfolio_snapshot(T0 + 400_000, force=False)
    con = sqlite3.connect(engine.db_path)
    row = con.execute("SELECT count(*), max(open_positions) FROM performance_snapshots").fetchone()
    con.close()
    assert row[0] >= 1 and row[1] >= 1


def test_close_records_consolidated_pnl_and_core_uses_calibrated_weights(engine):
    pos = engine.execute_directional_signal(_sig(), 1000.0, T0)
    assert pos is not None
    # Fecha exatamente no preço de entrada: PnL bruto = 0, mas exit_fee + slippage tornam pos.realized_pnl < 0 (LOSS)
    engine.close_position(pos.position_id, pos.entry_price, T0 + 60_000, reason="MANUAL_CLOSE")
    con = sqlite3.connect(engine.db_path)
    outcome, realized = con.execute(
        "SELECT outcome, realized_pnl FROM agent_learning_memory WHERE position_id=?", (pos.position_id,)
    ).fetchone()
    con.close()
    assert outcome == "LOSS" and realized == pytest.approx(pos.realized_pnl) and realized < 0.0
    assert engine.core._resolve_regime_weights({"regime": "SIDEWAYS_RANGE"}) is not None


def test_hold_votes_not_penalized_and_close_flushes_snapshot(engine):
    # Simula 6 decisões onde RiskSentinelAgent e SentimentAgent votam HOLD (abstenção/sem veto)
    for i in range(6):
        sig = _sig()
        sig.contributing_agents = [
            {"agent_name": "RiskSentinelAgent", "direction": "HOLD", "confidence": 0.5},
            {"agent_name": "SentimentAgent", "direction": "HOLD", "confidence": 0.5},
            {"agent_name": "TechnicalAgent", "direction": "BUY", "confidence": 0.85},
        ]
        pos = engine.execute_directional_signal(sig, 1000.0, T0 + i * 120_000)
        assert pos is not None
        engine.close_position(pos.position_id, 1050.0, T0 + i * 120_000 + 60_000, reason="MANUAL_CLOSE")

    weights = engine.learning_tracker.get_calibrated_weights_for_regime("SIDEWAYS_RANGE")
    # Agentes com voto HOLD preservam seu peso base (não caem para 0.40x por falsa acurácia 0%)
    assert weights["RiskSentinelAgent"] == pytest.approx(1.5)
    assert weights["SentimentAgent"] == pytest.approx(0.6)
    # TechnicalAgent (6/6 WIN em BUY) aumenta acima do base 1.0
    assert weights["TechnicalAgent"] > 1.0

    # close_position deve ter gravado performance_snapshots sincronizado com o saldo final
    con = sqlite3.connect(engine.db_path)
    snap_eq, snap_trades = con.execute(
        "SELECT equity, total_trades FROM performance_snapshots ORDER BY id DESC LIMIT 1"
    ).fetchone()
    con.close()
    assert snap_trades == 6 and snap_eq == pytest.approx(engine.equity, rel=1e-4)
