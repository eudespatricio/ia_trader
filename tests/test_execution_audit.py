"""
Regressão da auditoria de execução e risco (paper trading, carry).
Todos os testes usam DB temporário (nunca o market_cache.db real).
"""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from smart_trade import PaperTradingEngine
from signal_generator import TradeSignal
from funding_carry import FundingCarrySimulator
from tests.test_carry_simulator import create_synthetic_data

DAY1 = 1_700_000_000_000          # 2023-11-14 UTC
DAY2_00H = 1_700_006_400_000 + 1  # início do dia UTC seguinte (2023-11-15 00:00:00.001)


@pytest.fixture
def engine():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    eng = PaperTradingEngine(initial_capital=10000.0, db_path=path, max_concurrent_positions=20)
    yield eng
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(path + ext):
            os.remove(path + ext)


def _sig(symbol, direction, price, atr=10.0, size=1000.0):
    return TradeSignal(
        signal_id="t", timestamp=DAY1, symbol=symbol, direction=direction,
        entry_price=price,
        stop_loss=price - 2 * atr if direction == "BUY" else price + 2 * atr,
        take_profit=price + 4 * atr if direction == "BUY" else price - 4 * atr,
        atr=atr, confidence=0.9, consensus_score=0.8,
        position_size=size, risk_dollars=50.0, rationale="audit",
    )


@pytest.mark.parametrize("direction", ["BUY", "SELL"])
def test_directional_roundtrip_fee_charged_exactly_once_each_side(engine, direction):
    sig = _sig("BTCUSDT", direction, 1000.0)
    pos = engine.execute_directional_signal(sig, 1000.0, DAY1)
    assert pos is not None
    entry_fee = pos.fees_paid
    trade = engine.close_position(pos.position_id, 1000.0, DAY1 + 1000, reason="MANUAL_CLOSE")
    gross = ((trade.price - pos.entry_price) if direction == "BUY" else (pos.entry_price - trade.price)) * pos.quantity
    expected = 10000.0 + gross - entry_fee - trade.fee
    assert engine.balance == pytest.approx(expected, abs=1e-9)
    assert trade.pnl == pytest.approx(gross - trade.fee)
    # taxas totais = entrada + saída, cada uma uma única vez
    assert engine.total_fees_paid == pytest.approx(entry_fee + trade.fee)
    # equity final == balance (sem posições) e consistente com realized - entry fee
    assert engine.equity == pytest.approx(10000.0 + engine.total_realized_pnl - entry_fee)


@pytest.mark.parametrize("direction", ["BUY", "SELL"])
def test_scalp_partial_tp1_and_close_balance_reconciles(engine, direction):
    sig = _sig("ETHUSDT", direction, 100.0, atr=1.0, size=1000.0)
    pos = engine.execute_scalp_signal(sig, 100.0, DAY1, tp1_atr_mult=1.0, tp2_atr_mult=2.0, sl_atr_mult=1.0)
    assert pos is not None
    entry_fee = pos.fees_paid
    sign = 1 if direction == "BUY" else -1
    # TP1 é atingido, TP2 e SL não
    # (mínima/máxima da barra não toca o stop de breakeven recém-movido)
    hi = pos.tp1_price + 0.01 if sign == 1 else pos.entry_price - 0.5
    lo = pos.entry_price + 0.5 if sign == 1 else pos.tp1_price - 0.01
    trades = engine.update_market_tick("ETHUSDT", hi, lo, pos.tp1_price, DAY1 + 1000)
    assert [t.action for t in trades] == ["TP1_PARTIAL"]
    tp1 = trades[0]
    assert engine.balance == pytest.approx(10000.0 - entry_fee + tp1.pnl, abs=1e-9)
    # unrealized deve refletir apenas a quantidade remanescente
    assert pos.unrealized_pnl == pytest.approx(sign * (pos.tp1_price - pos.entry_price) * pos.quantity)
    close = engine.close_position(pos.position_id, pos.tp1_price, DAY1 + 2000)
    assert engine.balance == pytest.approx(10000.0 - entry_fee + tp1.pnl + close.pnl, abs=1e-9)
    assert engine.total_fees_paid == pytest.approx(entry_fee + tp1.fee + close.fee)


def test_exposure_never_exceeds_margin_cap_with_many_orders(engine):
    """Regressão do bug das 11 ordens de 1200 num capital de 10000."""
    opened = 0
    for i in range(30):
        s = _sig(f"COIN{i}USDT", "BUY", 100.0, size=1200.0)
        if engine.execute_directional_signal(s, 100.0, DAY1) is not None:
            opened += 1
        s2 = _sig(f"GP{i}USDT", "SELL", 100.0, size=1200.0)
        engine.execute_scalp_signal(s2, 100.0, DAY1)
        assert engine.margin_used <= engine.balance * engine.max_margin_utilization_pct + 1e-6
    assert opened < 11


def test_carry_allocation_respects_margin_cap(engine):
    for i in range(6):
        engine.execute_directional_signal(_sig(f"X{i}USDT", "BUY", 100.0, size=1500.0), 100.0, DAY1)
    engine.execute_carry_allocation("BTCUSDT", 50000.0, DAY1, allocation_usd=5000.0)
    # tolerância: o teto é relativo ao saldo, que cai levemente com as taxas de entrada
    assert engine.margin_used <= engine.balance * engine.max_margin_utilization_pct + 1.0


def test_daily_quota_does_not_reset_twice_on_stale_candle_timestamp(engine):
    engine.max_daily_scalps = 5
    engine.can_execute_scalp(DAY2_00H)            # now() já no dia novo
    engine.daily_scalps_count = 3
    engine.can_execute_scalp(DAY2_00H - 20 * 60 * 1000)  # vela fechada 23:4x do dia anterior
    assert engine.daily_scalps_count == 3          # não pode zerar
    engine.can_execute_scalp(DAY2_00H)
    assert engine.daily_scalps_count == 3


def test_carry_simulator_equity_includes_entry_fees():
    df = create_synthetic_data(n_bars=60, base_price=50000.0, price_trend=0.0, funding_rate=0.0)
    sim = FundingCarrySimulator(
        initial_capital=10000.0, entry_funding_threshold_ann=-1.0, exit_funding_threshold_ann=-2.0,
    )
    res = sim.run(df)
    assert len(res.trades) == 1
    assert res.net_profit == pytest.approx(sum(t.net_pnl for t in res.trades), rel=1e-6, abs=1e-6)
    assert res.net_profit < 0  # sem funding, só custos
