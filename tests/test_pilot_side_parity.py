"""Testes do LADO PILOTO da paridade backtest x piloto (P2/P4/P5/P6a, slippage, gap de SL, poll fechado)."""
from __future__ import annotations

import os
import sqlite3
import tempfile

import pytest

from signal_generator import TradeSignal
from smart_trade import (PaperTradingEngine, STRATEGY_PROFILES, FUNDING_INTERVAL_MS, MARKET_SLIPPAGE_PCT,
                         DIRECTIONAL_MAX_HOLDING_BARS)

TS0 = 1_700_000_000_000 - (1_700_000_000_000 % FUNDING_INTERVAL_MS)   # alinhado a 8h (irrelevante p/ regra elapsed)
H = 3_600_000


@pytest.fixture
def eng():
    d = tempfile.mkdtemp()
    return PaperTradingEngine(initial_capital=10_000.0, db_path=os.path.join(d, "t.db"),
                              max_concurrent_positions=20, restore=False)


def _sig(direction="BUY", price=100.0, atr=1.0):
    sl = price - atr if direction == "BUY" else price + atr
    return TradeSignal(signal_id="s", timestamp=TS0, symbol="BTCUSDT", direction=direction, entry_price=price,
                       stop_loss=sl, take_profit=price + 3 * atr, atr=atr, confidence=0.9, consensus_score=0.8,
                       position_size=100.0, risk_dollars=1.0, rationale="t")


def _open(eng, direction="BUY", price=100.0, atr=1.0):
    return eng.execute_directional_signal(_sig(direction, price, atr), current_price=price, timestamp=TS0, leverage=1.0)


def test_entry_recorded_at_next_bar_open_timestamp(eng):
    """P2: ts(decisão t) -> entry_timestamp = ts(t+1); preço = close(t) + slippage de mercado simétrico."""
    p = _open(eng)
    assert p.entry_timestamp == TS0 + STRATEGY_PROFILES["DIRECTIONAL"].bar_ms
    assert p.entry_price == pytest.approx(100.0 * (1 + MARKET_SLIPPAGE_PCT))
    s = _open(eng, "SELL")   # mesmo símbolo/estratégia -> rejeitado; abre em outro
    assert s is None
    sh = eng.execute_directional_signal(
        TradeSignal(**{**_sig("SELL").__dict__, "symbol": "ETHUSDT"}), 100.0, TS0, 1.0)
    assert sh.entry_price == pytest.approx(100.0 * (1 - MARKET_SLIPPAGE_PCT))     # long == short
    sc = eng.execute_scalp_signal(TradeSignal(**{**_sig().__dict__, "symbol": "SOLUSDT"}), 100.0, TS0)
    assert sc.entry_timestamp == TS0 + STRATEGY_PROFILES["SCALP"].bar_ms


def test_first_post_entry_bar_uses_full_range(eng):
    p = _open(eng)
    t = eng.update_market_tick("BTCUSDT", 101.0, 90.0, 95.0, p.entry_timestamp)   # barra t+1: SL pega
    assert [x.action for x in t] == ["STOP_LOSS"]


def test_stop_fill_uses_worse_of_stop_and_open_on_gap(eng):
    p = _open(eng)
    sl = p.stop_loss
    t = eng.update_market_tick("BTCUSDT", sl - 1.0, sl - 3.0, sl - 2.0, p.entry_timestamp + H, open_price=sl - 2.5)
    assert t[0].action == "STOP_LOSS"
    assert t[0].price == pytest.approx((sl - 2.5) * (1 - MARKET_SLIPPAGE_PCT))   # gap: pior entre SL e open


def test_horizon_timeout_after_n_bars_taker_at_bar_open(eng):
    """P4: 48 barras 1h; fecha a mercado (taker+slippage) no open da barra do prazo, antes de SL/TP."""
    p = _open(eng)
    t0 = p.entry_timestamp
    assert DIRECTIONAL_MAX_HOLDING_BARS == 48
    for i in range(47):   # barras pós-entrada sem tocar SL/TP
        assert eng.update_market_tick("BTCUSDT", 100.4, 99.8, 100.1, t0 + i * H) == []
    t = eng.update_market_tick("BTCUSDT", 100.4, 99.8, 100.2, t0 + 48 * H)    # prazo: sai no open (== close anterior)
    assert [x.action for x in t] == ["HORIZON_TIMEOUT"]
    assert t[0].price == pytest.approx(100.1 * (1 - MARKET_SLIPPAGE_PCT))
    assert t[0].fee == pytest.approx(t[0].quantity * t[0].price * eng.taker_fee)
    assert p.position_id not in eng.positions


def test_funding_long_pays_short_receives_every_8h_from_cache(eng):
    with sqlite3.connect(eng.db_path) as c:
        c.execute("CREATE TABLE IF NOT EXISTS funding_rates (exchange TEXT, symbol TEXT, timestamp INTEGER, "
                  "funding_rate REAL, mark_price REAL)")
        c.execute("INSERT INTO funding_rates VALUES ('binance','BTCUSDT',?,0.0005,100)", (TS0 + 2 * H,))
    p = _open(eng)
    t0 = p.entry_timestamp
    bal0 = eng.balance
    for i in range(10):
        eng.update_market_tick("BTCUSDT", 100.4, 99.8, 100.0, t0 + i * H)
    fund = [t for t in eng.trade_history if t.action == "FUNDING"]
    assert len(fund) == 1                       # 8h decorridas desde a entrada (barra 8)
    assert fund[0].pnl == pytest.approx(-0.0005 * p.quantity * 100.0)      # long paga (taxa real do cache)
    assert eng.balance == pytest.approx(bal0 + fund[0].pnl)
    assert p.funding_collected == pytest.approx(fund[0].pnl)
    # short recebe (taxa de fallback 0.01% - sem registro p/ ETH)
    s = eng.execute_directional_signal(TradeSignal(**{**_sig("SELL").__dict__, "symbol": "ETHUSDT"}), 100.0, TS0 + 10 * H, 1.0)
    for i in range(9):
        eng.update_market_tick("ETHUSDT", 100.4, 99.8, 100.0, s.entry_timestamp + i * H)
    f2 = [t for t in eng.trade_history if t.action == "FUNDING" and t.symbol == "ETHUSDT"]
    assert len(f2) == 1 and f2[0].pnl > 0 and "FALLBACK" in f2[0].rationale


def test_funding_is_idempotent_and_survives_restart(eng):
    p = _open(eng)
    t0 = p.entry_timestamp
    for _ in range(3):
        eng.update_market_tick("BTCUSDT", 100.4, 99.8, 100.0, t0 + 9 * H)       # mesma barra repetida
    assert len([t for t in eng.trade_history if t.action == "FUNDING"]) == 1
    e2 = PaperTradingEngine(initial_capital=10_000.0, db_path=eng.db_path, max_concurrent_positions=20, restore=True)
    assert e2.balance == pytest.approx(eng.balance)
    e2.update_market_tick("BTCUSDT", 100.4, 99.8, 100.0, t0 + 9 * H)
    assert len([t for t in e2.trade_history if t.action == "FUNDING"]) == 1       # cursor restaurado do ledger


def test_directional_entry_blocked_by_circuit_breaker(eng):
    eng.rollback_guard.trigger_emergency_stop("teste")
    assert _open(eng) is None


def test_mark_to_market_never_triggers_exits(eng):
    p = _open(eng)
    eng.mark_to_market("BTCUSDT", p.stop_loss - 10.0)     # preço da barra em formação muito abaixo do SL
    assert p.position_id in eng.positions and p.current_price == p.stop_loss - 10.0 and p.unrealized_pnl < 0


def test_daemon_poll_feeds_only_closed_bars(monkeypatch):
    import asyncio
    import pandas as pd
    from scripts.run_paper_simulation import PaperTradingDaemon, POLL_BAR_MS

    d = tempfile.mkdtemp()
    e = PaperTradingEngine(initial_capital=10_000.0, db_path=os.path.join(d, "t.db"), restore=False)
    p = e.execute_directional_signal(_sig(), 100.0, TS0, 1.0)
    now = p.entry_timestamp + 3 * POLL_BAR_MS + 1000       # 3 barras fechadas após a entrada + uma em formação
    bars = [{"timestamp": p.entry_timestamp + i * POLL_BAR_MS, "open": 100.0, "high": 100.2, "low": 99.9, "close": 100.0}
            for i in range(4)]
    bars[3].update(low=50.0, close=50.0)                    # vela em formação com low espúrio

    class L:
        async def get_candles(self, *a, **k):
            return pd.DataFrame(bars)

        async def get_funding_history(self, *a, **k):
            return []

        async def close(self):
            pass

    class S:
        async def close(self):
            pass

    dm = PaperTradingDaemon(engine=e, loader=L(), golden_pot_agent=S())
    monkeypatch.setattr("scripts.run_paper_simulation.time.time", lambda: now / 1000.0)
    asyncio.run(dm.update_active_positions())
    assert p.position_id in e.positions                       # SL espúrio da barra aberta NÃO disparou
    assert dm._last_tick_bar_ts["BTCUSDT"] == bars[2]["timestamp"]
    assert p.current_price == 50.0                            # só marcação a mercado
