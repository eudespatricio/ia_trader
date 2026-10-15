"""
Regressões das correções de revisão do motor de paper trading:
A5 (restore_state), BIAS (warmup sma_200 / 400 candles), A3 (vela pré-entrada),
A1 (taker em SL/BE/MANUAL) e win_rate por position_id.
Todos os testes usam DB temporário.
"""
import asyncio
import os
import sqlite3
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from smart_trade import PaperTradingEngine  # noqa: E402
from signal_generator import TradeSignal  # noqa: E402

NOW_MS = 1_700_000_000_000
DAY_MS = 86_400_000


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    for suf in ("", "-wal", "-shm"):
        if os.path.exists(path + suf):
            os.remove(path + suf)


def make_engine(db, **kw):
    return PaperTradingEngine(initial_capital=10000.0, db_path=db, max_concurrent_positions=20, **kw)


def sig(symbol="BTCUSDT", direction="BUY", size=1000.0, atr=100.0, price=50000.0):
    return TradeSignal(
        signal_id=f"t-{symbol}", timestamp=NOW_MS, symbol=symbol, direction=direction,
        entry_price=price,
        stop_loss=price - 500 if direction == "BUY" else price + 500,
        take_profit=price + 1000 if direction == "BUY" else price - 1000,
        atr=atr, confidence=0.9, consensus_score=0.8, position_size=size,
        risk_dollars=10.0, rationale="teste",
    )


# ----------------------------------------------------------------------------
# A5 - restore_state
# ----------------------------------------------------------------------------
class TestRestoreState:
    def _populate(self, e):
        e.execute_directional_signal(sig("BTCUSDT", "BUY", 1000), 50000.0, NOW_MS)
        s = e.execute_scalp_signal(sig("ETHUSDT", "BUY", 800, atr=10, price=3000), 3000.0, NOW_MS + 1000)
        e.execute_golden_pot_signal(sig("SOLUSDT", "SELL", 600, atr=2, price=150), 150.0, NOW_MS + 2000)
        # TP1 parcial no scalp (high alcança tp1)
        e.update_market_tick("ETHUSDT", s.tp1_price + 1, s.entry_price, s.entry_price, NOW_MS + 60_000)
        return s

    def test_restart_preserves_balance_margin_positions_and_quotas(self, temp_db):
        e1 = make_engine(temp_db)
        self._populate(e1)
        closed = e1.close_position(next(p.position_id for p in e1.positions.values() if p.symbol == "SOLUSDT"),
                                   149.0, NOW_MS + 70_000, reason="MANUAL")
        assert closed is not None

        e2 = make_engine(temp_db)  # "reinício do daemon"
        assert e2.balance == pytest.approx(e1.balance, abs=1e-9)
        assert e2.margin_used == pytest.approx(e1.margin_used, abs=1e-9)
        assert e2.available_margin == pytest.approx(e1.available_margin, abs=1e-9)
        assert set(e2.positions) == set(e1.positions)
        for pid, p in e1.positions.items():
            q = e2.positions[pid]
            assert (q.symbol, q.side, q.tp1_hit) == (p.symbol, p.side, p.tp1_hit)
            assert q.quantity == pytest.approx(p.quantity)
            assert q.stop_loss == pytest.approx(p.stop_loss)
            assert q.initial_quantity == pytest.approx(p.initial_quantity)
        assert e2.total_fees_paid == pytest.approx(e1.total_fees_paid)
        assert e2.total_realized_pnl == pytest.approx(e1.total_realized_pnl)
        assert len(e2.trade_history) == len(e1.trade_history)
        assert e2.daily_scalps_count == e1.daily_scalps_count == 1
        assert e2.daily_golden_pots_count == e1.daily_golden_pots_count == 1
        assert e2.current_day_utc == e1.current_day_utc != ""
        assert e2.balance != 10000.0

    def test_quota_not_counted_on_new_day(self, temp_db):
        e1 = make_engine(temp_db)
        e1.execute_scalp_signal(sig("ETHUSDT", "BUY", 800, atr=10, price=3000), 3000.0, NOW_MS)
        e2 = make_engine(temp_db)
        assert e2.daily_scalps_count == 1
        # próximo dia UTC: a lógica de cota diária zera
        ok, _ = e2.can_execute_scalp(NOW_MS + 2 * DAY_MS)
        assert ok and e2.daily_scalps_count == 0

    def test_manual_reset_flag_starts_clean_and_persists(self, temp_db):
        e1 = make_engine(temp_db)
        self._populate(e1)
        with sqlite3.connect(temp_db) as con:
            con.execute("INSERT OR REPLACE INTO simulation_settings (key, value) VALUES ('MANUAL_RESET', '1')")
        e2 = make_engine(temp_db)
        assert e2.balance == 10000.0 and not e2.positions and not e2.trade_history
        assert e2.daily_scalps_count == 0
        with sqlite3.connect(temp_db) as con:
            flag = con.execute("SELECT value FROM simulation_settings WHERE key='MANUAL_RESET'").fetchone()[0]
        assert flag == "0"
        # novo restart NÃO ressuscita o passado anterior ao reset
        e3 = make_engine(temp_db)
        assert e3.balance == 10000.0 and not e3.positions
        # ...mas trades posteriores ao reset são restaurados
        e3.execute_directional_signal(sig("BTCUSDT", "BUY", 1000), 50000.0, NOW_MS + 5000)
        e4 = make_engine(temp_db)
        assert len(e4.positions) == 1 and e4.balance == pytest.approx(e3.balance)

    def test_manual_quota_reset_survives_restart(self, temp_db):
        e1 = make_engine(temp_db)
        e1.execute_scalp_signal(sig("ETHUSDT", "BUY", 800, atr=10, price=3000), 3000.0, NOW_MS)
        e1.update_simulation_settings({}, reset_quotas=True)
        assert e1.daily_scalps_count == 0
        e2 = make_engine(temp_db)
        assert e2.daily_scalps_count == 0
        assert len(e2.positions) == 1  # posição preservada, só a cota foi zerada

    def test_restore_false_and_empty_db(self, temp_db):
        e1 = make_engine(temp_db)
        e1.execute_directional_signal(sig(), 50000.0, NOW_MS)
        assert make_engine(temp_db, restore=False).balance == 10000.0
        assert make_engine(tempfile.mktemp(suffix=".db")).balance == 10000.0


# ----------------------------------------------------------------------------
# A3 - vela pré-entrada
# ----------------------------------------------------------------------------
class TestPreEntryCandle:
    def test_pre_entry_extremes_do_not_trigger_sl(self, temp_db):
        e = make_engine(temp_db)
        p = e.execute_directional_signal(sig("BTCUSDT", "BUY", 1000), 50000.0, NOW_MS)
        # vela com timestamp <= entrada e low muito abaixo do SL
        trig = e.update_market_tick("BTCUSDT", 50100.0, 40000.0, 50050.0, NOW_MS)
        assert trig == [] and p.position_id in e.positions
        assert e.positions[p.position_id].lowest_price >= 49000.0
        trig = e.update_market_tick("BTCUSDT", 50100.0, 40000.0, 50050.0, NOW_MS - 1000)
        assert trig == []
        # vela posterior usa os extremos normalmente
        trig = e.update_market_tick("BTCUSDT", 50100.0, 40000.0, 50050.0, NOW_MS + 1)
        assert [t.action for t in trig] == ["STOP_LOSS"]

    def test_pre_entry_candle_does_not_trigger_tp_short_scalp(self, temp_db):
        e = make_engine(temp_db)
        p = e.execute_scalp_signal(sig("ETHUSDT", "SELL", 800, atr=10, price=3000), 3000.0, NOW_MS)
        trig = e.update_market_tick("ETHUSDT", 3100.0, 2000.0, 3000.0, NOW_MS)
        assert trig == [] and not e.positions[p.position_id].tp1_hit


# ----------------------------------------------------------------------------
# A1 - taxas de saída
# ----------------------------------------------------------------------------
class TestExitFees:
    def _scalp(self, e, side="BUY"):
        return e.execute_scalp_signal(sig("ETHUSDT", side, 1000, atr=10, price=3000), 3000.0, NOW_MS)

    @pytest.mark.parametrize("side", ["BUY", "SELL"])
    def test_stop_loss_is_taker_with_perp_slippage(self, temp_db, side):
        e = make_engine(temp_db)
        p = self._scalp(e, side)
        qty, sl = p.quantity, p.stop_loss
        t = e.close_position(p.position_id, sl, NOW_MS + 1000, reason="STOP_LOSS")
        exec_px = sl * (1 - 0.0003) if side == "BUY" else sl * (1 + 0.0003)
        assert t.price == pytest.approx(exec_px)
        assert t.fee == pytest.approx(qty * exec_px * 0.0006)

    def test_breakeven_and_manual_taker_tp2_maker(self, temp_db):
        e = make_engine(temp_db)
        for reason, fee_rate, slip in [("BREAKEVEN_STOP", 0.0006, 0.0003), ("MANUAL", 0.0006, 0.0003),
                                       ("TAKE_PROFIT_2", 0.0002, 0.0001)]:
            p = self._scalp(e)
            qty = p.quantity
            t = e.close_position(p.position_id, 3010.0, NOW_MS + 1000, reason=reason)
            assert t.price == pytest.approx(3010.0 * (1 - slip))
            assert t.fee == pytest.approx(qty * t.price * fee_rate)

    def test_tick_driven_stop_uses_taker(self, temp_db):
        e = make_engine(temp_db)
        p = self._scalp(e)
        trig = e.update_market_tick("ETHUSDT", 3000.0, p.stop_loss - 1, p.stop_loss, NOW_MS + 1000)
        assert trig[0].action == "STOP_LOSS"
        assert trig[0].fee == pytest.approx(trig[0].quantity * trig[0].price * 0.0006)


# ----------------------------------------------------------------------------
# win_rate por position_id
# ----------------------------------------------------------------------------
class TestWinRateByPosition:
    def test_tp1_partial_not_separate_trade(self, temp_db):
        e = make_engine(temp_db)
        # Posição A: TP1 positivo + saída final negativa, soma líquida negativa => 1 loss
        a = e.execute_scalp_signal(sig("ETHUSDT", "BUY", 2000, atr=10, price=3000), 3000.0, NOW_MS)
        e.update_market_tick("ETHUSDT", a.tp1_price + 0.5, a.tp1_price - 1.0, a.tp1_price - 1.0, NOW_MS + 1000)
        e.close_position(a.position_id, a.entry_price - 300.0, NOW_MS + 2000, reason="MANUAL")
        # Posição B: ganho simples
        b = e.execute_directional_signal(sig("BTCUSDT", "BUY", 1000), 50000.0, NOW_MS)
        e.close_position(b.position_id, 51000.0, NOW_MS + 3000, reason="TAKE_PROFIT")
        actions = [t.action for t in e.trade_history if t.position_id == a.position_id]
        assert "TP1_PARTIAL" in actions
        m = e.get_summary_metrics()
        assert m["win_rate_pct"] == 50.0  # 1 win (B) / 2 posições (A perdeu na soma)

    def test_open_position_with_partial_not_counted(self, temp_db):
        e = make_engine(temp_db)
        a = e.execute_scalp_signal(sig("ETHUSDT", "BUY", 2000, atr=10, price=3000), 3000.0, NOW_MS)
        e.update_market_tick("ETHUSDT", a.tp1_price + 0.5, a.tp1_price - 1.0, a.tp1_price - 1.0, NOW_MS + 1000)
        assert e.get_summary_metrics()["win_rate_pct"] == 0.0


# ----------------------------------------------------------------------------
# BIAS - warmup / limit=400 / skip de séries curtas
# ----------------------------------------------------------------------------
def _candles(drift, n, noise=0.01, seed=0):
    r = np.random.default_rng(seed)
    c = 50000 * np.exp(np.cumsum(drift + r.normal(0, noise, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({
        "timestamp": NOW_MS + np.arange(n) * 3_600_000, "open": o,
        "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999,
        "close": c, "volume": r.uniform(100, 200, n),
    })


class _StubLoader:
    def __init__(self, df):
        self.df, self.calls = df, []

    async def get_candles(self, sym, tf, limit=100, exchange=None, **kw):
        self.calls.append((sym, tf, limit, exchange))
        return self.df.tail(limit).copy()

    async def close(self):
        pass


def _daemon(db, df):
    from run_paper_simulation import PaperTradingDaemon
    return PaperTradingDaemon(engine=make_engine(db), loader=_StubLoader(df))


class TestSwingWarmup:
    def test_requests_at_least_400_candles_and_sma200_valid(self, temp_db):
        d = _daemon(temp_db, _candles(0.0, 500))
        asyncio.run(d.evaluate_swing_opportunities())
        assert d.loader.calls and all(c[2] >= 400 for c in d.loader.calls)
        feat = d._compute_swing_features(_candles(0.0, 400))
        assert not feat.empty
        assert (feat["sma_200"] > 0).all() and np.isfinite(feat["sma_200"]).all()

    @pytest.mark.parametrize("n", [60, 150, 249])
    def test_short_series_skipped(self, temp_db, n):
        d = _daemon(temp_db, _candles(-0.002, n))
        called = []
        orig = d.engine.core.evaluate_multi_strategy
        d.engine.core.evaluate_multi_strategy = lambda **kw: (called.append(kw), orig(**kw))[1]
        asyncio.run(d.evaluate_swing_opportunities())
        assert called == [] and not d.engine.positions

    def test_context_never_falls_back_to_zero_sma200(self, temp_db):
        d = _daemon(temp_db, _candles(0.0, 400))
        # série curta no helper: nenhuma linha (em vez de sma_200 = 0.0)
        assert d._compute_swing_features(_candles(0.0, 150)).empty
        assert not d._has_valid_swing_context(pd.DataFrame({"sma_200": [0.0]}))

    def test_synthetic_drift_scenarios(self, temp_db):
        """-0.2%/candle gera SELL(short); +0.2% nunca SELL."""
        sides = {}
        for drift in (-0.002, 0.0, 0.002):
            seen = set()
            for seed in range(8):
                fd, p = tempfile.mkstemp(suffix=".db"); os.close(fd)
                try:
                    d = _daemon(p, _candles(drift, 400, seed=seed))
                    asyncio.run(d.evaluate_swing_opportunities())
                    seen |= {pos.side for pos in d.engine.positions.values()}
                finally:
                    for suf in ("", "-wal", "-shm"):
                        if os.path.exists(p + suf):
                            os.remove(p + suf)
            sides[drift] = seen
        assert "SHORT" in sides[-0.002] and "LONG" not in sides[-0.002]
        assert "SHORT" not in sides[0.002]
