"""
Behavioral contract tests for restart catch-up, causal monotonicity, gap slicing, and UI telemetry.
Deterministic, network-free, isolated temp SQLite databases with explicit connection closing.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import urllib.request
from typing import List

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from db_conn import connect as _db_connect  # noqa: E402
from data.data_integrations import CandleData  # noqa: E402
from data.data_loader import DataLoader  # noqa: E402
import flask_interface as fi  # noqa: E402
import scripts.run_paper_simulation as rps  # noqa: E402
from signal_generator import TradeSignal  # noqa: E402
from smart_trade import BREAKEVEN_FEE_BUFFER_PCT, PaperTradingEngine  # noqa: E402

TF_5M_MS = 5 * 60 * 1000
TF_1H_MS = 60 * 60 * 1000
DAY_0_MS = 1_764_028_800_000  # 2025-11-25 00:00:00 UTC


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def _no_net(*_args, **_kwargs):
        raise AssertionError("Network access forbidden in unit tests")

    monkeypatch.setattr(urllib.request, "urlopen", _no_net)


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    for suf in ("", "-wal", "-shm"):
        if os.path.exists(path + suf):
            try:
                os.remove(path + suf)
            except OSError:
                pass


def _make_signal(symbol: str = "SOLUSDT", direction: str = "SELL", price: float = 100.0, atr: float = 2.0) -> TradeSignal:
    sgn = 1.0 if direction == "BUY" else -1.0
    return TradeSignal(
        signal_id=f"sig_{symbol}_{direction}", timestamp=DAY_0_MS, symbol=symbol, direction=direction,
        entry_price=price, stop_loss=price - sgn * atr, take_profit=price + sgn * 2.5 * atr,
        atr=atr, confidence=0.90, consensus_score=0.85, position_size=1e9, risk_dollars=1.0,
        rationale="hotfix_contract_test",
    )


def _candle(ts: int, price: float = 150.0, sym: str = "SOLUSDT", tf: str = "5m") -> CandleData:
    return CandleData(
        timestamp=ts, open=price, high=price + 0.5, low=price - 0.5, close=price + 0.1,
        volume=100.0, quote_volume=15000.0, symbol=sym, timeframe=tf,
    )


def _bars_df(rows: List[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "quote_volume"])


class _StubLoader:
    def __init__(self, candle_fn):
        self._candle_fn = candle_fn

    async def get_candles(self, symbol: str, timeframe: str = "5m", limit: int = 10, exchange: str = "binance", **_kw):
        res = self._candle_fn(symbol, timeframe, limit, exchange)
        if asyncio.iscoroutine(res):
            return await res
        return res

    async def get_funding_history(self, *_args, **_kwargs):
        return []

    async def close(self):
        return None


# ============================================================================
# 1. DataLoader.get_candles: fetch_limit rules & zero stale backfill
# ============================================================================

@pytest.mark.parametrize("exchange", ["binance", "bitget"])
def test_dataloader_get_candles_drops_forming_without_stale_fill(temp_db, monkeypatch, exchange):
    loader = DataLoader(db_path=temp_db)
    now_ms = int(time.time() * 1000)
    t_forming = now_ms - 10_000  # started 10s ago -> still open for ~290s
    t_stale = t_forming - 8 * TF_1H_MS
    closed_ts = [t_forming - (11 - i) * TF_5M_MS for i in range(1, 11)]
    available = [_candle(ts, price=140.0 + i) for i, ts in enumerate(closed_ts)] + [_candle(t_forming, price=155.0)]

    with _db_connect(temp_db, timeout=5.0) as con:
        con.execute(
            "INSERT INTO candles (exchange, symbol, timeframe, timestamp, open, high, low, close, volume, quote_volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (exchange, "SOLUSDT", "5m", t_stale, 90.0, 91.0, 89.0, 90.5, 50.0, 4500.0),
        )

    seen_limits: List[int] = []
    pool = available

    async def _fake_fetch(*_args, **kwargs):
        req_limit = int(kwargs["limit"])
        seen_limits.append(req_limit)
        return pool[-req_limit:]

    if exchange == "binance":
        monkeypatch.setattr(loader.manager.binance, "get_klines", _fake_fetch)
    else:
        monkeypatch.setattr(loader.manager.bitget, "get_candles", _fake_fetch)

    df = asyncio.run(loader.get_candles(symbol="SOLUSDT", timeframe="5m", limit=10, exchange=exchange))
    got_ts = [int(x) for x in df["timestamp"].tolist()]
    assert seen_limits == [11] and len(df) == 10 and got_ts == closed_ts
    assert t_stale not in got_ts and t_forming not in got_ts

    if exchange == "bitget":
        t_stale_old = t_forming - 24 * TF_1H_MS  # > 200 * 5m (16h40m) so truly stale for limit=200
        with _db_connect(temp_db, timeout=5.0) as con:
            con.execute(
                "INSERT INTO candles (exchange, symbol, timeframe, timestamp, open, high, low, close, volume, quote_volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("bitget", "ETHUSDT", "5m", t_stale_old, 90.0, 91.0, 89.0, 90.5, 50.0, 4500.0),
            )
        closed_200 = [t_forming - (201 - i) * TF_5M_MS for i in range(1, 201)]
        pool = [_candle(ts, price=120.0 + (i % 10), sym="ETHUSDT") for i, ts in enumerate(closed_200)] + [
            _candle(t_forming, price=155.0, sym="ETHUSDT")
        ]
        df_200 = asyncio.run(loader.get_candles(symbol="ETHUSDT", timeframe="5m", limit=200, exchange="bitget"))
        got_200 = [int(x) for x in df_200["timestamp"].tolist()]
        assert seen_limits[-1] == 201 and len(df_200) == 200 and got_200 == closed_200 and t_stale_old not in got_200
    else:
        asyncio.run(loader.get_candles(symbol="SOLUSDT", timeframe="5m", limit=1000, exchange="binance"))
        assert seen_limits[-1] == 1000


# ============================================================================
# 2. PaperTradingEngine: monotonic causal guard, FUNDING & LONG TP2 restart
# ============================================================================

def test_engine_short_tp1_restart_causal_guard_and_same_bar_tp1_tp2(temp_db):
    eng1 = PaperTradingEngine(initial_capital=10000.0, db_path=temp_db, max_concurrent_positions=10)
    t_dec = DAY_0_MS - TF_1H_MS
    pos1 = eng1.execute_directional_signal(_make_signal("SOLUSDT", "SELL", 100.0, 2.0), 100.0, t_dec)
    assert pos1 is not None
    t0 = pos1.entry_timestamp
    be = pos1.entry_price * (1.0 - BREAKEVEN_FEE_BUFFER_PCT)
    t_tp1 = t0 + 20 * 60_000

    assert [t.action for t in eng1.update_realtime_price("SOLUSDT", price=pos1.tp1_price, timestamp=t_tp1)] == ["TP1_PARTIAL"]
    assert pos1.tp1_hit is True and pos1.stop_loss == pytest.approx(be) and pos1.status == "OPEN"

    eng2 = PaperTradingEngine(initial_capital=10000.0, db_path=temp_db, max_concurrent_positions=10)
    pos2 = eng2.positions[pos1.position_id]
    assert pos2.tp1_hit is True and pos2.stop_loss == pytest.approx(be) and pos2.last_event_ts_ms == float(t_tp1)

    t_old = t0 + 10 * 60_000
    assert eng2.update_market_tick("SOLUSDT", high=be + 0.5, low=98.0, close=99.0, timestamp=t_old, open_price=99.0) == []
    assert eng2.update_market_tick("SOLUSDT", high=be + 0.5, low=98.0, close=99.0, timestamp=t_tp1, open_price=99.0) == []
    assert eng2.update_realtime_price("SOLUSDT", price=be + 0.5, timestamp=t_tp1 - 1_000) == []
    assert pos2.status == "OPEN"

    be_trades = eng2.update_market_tick("SOLUSDT", high=be + 0.5, low=98.5, close=99.5, timestamp=t_tp1 + TF_5M_MS, open_price=99.0)
    assert [t.action for t in be_trades] == ["BREAKEVEN_STOP"] and pos2.status == "CLOSED"

    pos_b = eng2.execute_directional_signal(_make_signal("BTCUSDT", "SELL", 100.0, 2.0), 100.0, t_dec)
    assert pos_b is not None
    tr_12 = eng2.update_market_tick(
        "BTCUSDT", high=pos_b.entry_price, low=pos_b.tp2_price - 0.1, close=pos_b.tp2_price,
        timestamp=pos_b.entry_timestamp, open_price=pos_b.entry_price,
    )
    assert [t.action for t in tr_12] == ["TP1_PARTIAL", "TP2_PARTIAL"]
    assert pos_b.tp1_hit is True and pos_b.tp2_hit is True and pos_b.trailing_active is True and pos_b.status == "OPEN"


def test_engine_funding_at_8h_restart_allows_exact_bar_without_double_charge(temp_db):
    eng1 = PaperTradingEngine(initial_capital=10000.0, db_path=temp_db, max_concurrent_positions=10)
    pos1 = eng1.execute_directional_signal(_make_signal("SOLUSDT", "SELL", 100.0, 2.0), 100.0, DAY_0_MS - TF_1H_MS)
    assert pos1 is not None
    t0 = pos1.entry_timestamp
    t_8h = t0 + 8 * TF_1H_MS
    assert eng1.update_realtime_price("SOLUSDT", price=pos1.entry_price, timestamp=t_8h + 60_000) == []
    assert pos1.funding_cursor_ms == float(t_8h) and pos1.last_event_ts_ms == 0.0
    assert sum(1 for t in eng1.trade_history if t.position_id == pos1.position_id and t.action == "FUNDING") == 1

    eng2 = PaperTradingEngine(initial_capital=10000.0, db_path=temp_db, max_concurrent_positions=10)
    pos2 = eng2.positions[pos1.position_id]
    assert pos2.funding_cursor_ms == float(t_8h) and pos2.last_event_ts_ms == 0.0

    # Exact t0 + 8h bar MUST still execute and MUST NOT double-charge FUNDING
    res = eng2.update_market_tick("SOLUSDT", high=pos2.stop_loss + 0.5, low=99.0, close=101.0, timestamp=t_8h, open_price=100.0)
    assert [t.action for t in res] == ["STOP_LOSS"] and pos2.status == "CLOSED"
    assert sum(1 for t in eng2.trade_history if t.position_id == pos2.position_id and t.action == "FUNDING") == 1


def test_engine_long_tp1_tp2_restart_causal_guard_and_trailing_close(temp_db):
    eng1 = PaperTradingEngine(initial_capital=10000.0, db_path=temp_db, max_concurrent_positions=10)
    pos1 = eng1.execute_directional_signal(_make_signal("ETHUSDT", "BUY", 100.0, 2.0), 100.0, DAY_0_MS - TF_1H_MS)
    assert pos1 is not None
    t0 = pos1.entry_timestamp
    t_tp1, t_tp2 = t0 + 10 * 60_000, t0 + 20 * 60_000

    assert [t.action for t in eng1.update_realtime_price("ETHUSDT", price=pos1.tp1_price, timestamp=t_tp1)] == ["TP1_PARTIAL"]
    assert [t.action for t in eng1.update_realtime_price("ETHUSDT", price=pos1.tp2_price, timestamp=t_tp2)] == ["TP2_PARTIAL"]
    assert pos1.trailing_active is True and pos1.stop_loss == pytest.approx(pos1.trailing_stop_price)

    eng2 = PaperTradingEngine(initial_capital=10000.0, db_path=temp_db, max_concurrent_positions=10)
    pos2 = eng2.positions[pos1.position_id]
    assert pos2.tp1_hit is True and pos2.tp2_hit is True and pos2.trailing_active is True
    assert pos2.last_event_ts_ms == float(t_tp2)

    sl_hit_low = pos2.stop_loss - 1.0
    assert eng2.update_market_tick("ETHUSDT", high=106.0, low=sl_hit_low, close=105.0, timestamp=t0 + 15 * 60_000, open_price=105.0) == []
    assert eng2.update_market_tick("ETHUSDT", high=106.0, low=sl_hit_low, close=105.0, timestamp=t_tp2, open_price=105.0) == []
    assert pos2.status == "OPEN"

    t_after = t_tp2 + TF_5M_MS
    closed_tr = eng2.update_market_tick(
        "ETHUSDT", high=pos2.stop_loss + 1.0, low=sl_hit_low, close=pos2.stop_loss - 0.2,
        timestamp=t_after, open_price=pos2.stop_loss + 0.5,
    )
    assert [t.action for t in closed_tr] == ["TRAILING_STOP"] and pos2.status == "CLOSED"


# ============================================================================
# 3. PaperTradingDaemon: 8h gap, 2-bar stale gap, 1h grid & run_loop ordering
# ============================================================================

def test_daemon_update_active_positions_gaps_and_sparse_grid(temp_db, monkeypatch):
    bar_01_25 = DAY_0_MS + 1 * TF_1H_MS + 25 * 60_000
    t_tp1 = DAY_0_MS + 7 * TF_1H_MS + 36 * 60_000 + 50_000
    bar_08_55, bar_09_00, bar_09_05 = DAY_0_MS + 8 * TF_1H_MS + 55 * 60_000, DAY_0_MS + 9 * TF_1H_MS, DAY_0_MS + 9 * TF_1H_MS + 5 * 60_000
    monkeypatch.setattr(rps.time, "time", lambda: (bar_09_05 + 10 * 60_000) / 1000.0)

    # Scenario A: 8h gap in 5m bars post-TP1 (07:36:50)
    eng = PaperTradingEngine(initial_capital=10000.0, db_path=temp_db, max_concurrent_positions=10)
    pos = eng.execute_directional_signal(_make_signal("SOLUSDT", "SELL", 100.0, 2.0), 100.0, DAY_0_MS - 3 * TF_1H_MS)
    assert pos is not None
    assert eng.update_realtime_price("SOLUSDT", price=pos.tp1_price, timestamp=t_tp1)[0].action == "TP1_PARTIAL"
    be = pos.stop_loss

    df_gap = _bars_df([
        (bar_01_25, 99.0, be + 5.0, 98.5, 99.5, 10.0, 1000.0),
        (bar_08_55, pos.tp1_price + 0.1, pos.tp1_price + 0.2, pos.tp1_price, pos.tp1_price + 0.05, 10.0, 970.0),
        (bar_09_00, pos.tp1_price + 0.05, pos.tp1_price + 0.2, pos.tp1_price, pos.tp1_price + 0.05, 10.0, 970.0),
        (bar_09_05, pos.tp1_price + 0.05, pos.tp1_price + 0.2, pos.tp1_price, pos.tp1_price + 0.05, 10.0, 970.0),
    ])
    daemon = rps.PaperTradingDaemon(
        poll_interval_seconds=15, engine=eng,
        loader=_StubLoader(lambda _s, tf, _l, ex: df_gap if (tf == "5m" and ex == "binance") else pd.DataFrame()),
        enable_realtime_monitor=False,
    )
    seen_ts: List[int] = []
    orig_tick = eng.update_market_tick

    def _spy_tick(*args, **kwargs):
        seen_ts.append(int(kwargs.get("timestamp") if "timestamp" in kwargs else args[4]))
        return orig_tick(*args, **kwargs)

    monkeypatch.setattr(eng, "update_market_tick", _spy_tick)
    asyncio.run(daemon.update_active_positions())
    assert pos.status == "OPEN" and seen_ts == [bar_08_55, bar_09_00, bar_09_05] and daemon._last_tick_bar_ts["SOLUSDT"] == bar_09_05

    # Scenario A2: 2-bar stale gap [bar_stale_8h_ago, bar_recent_live] within 3*tf_ms of now_ts
    daemon.engine.positions.pop(pos.position_id, None)
    pos_2 = eng.execute_directional_signal(_make_signal("ETHUSDT", "SELL", 100.0, 2.0), 100.0, DAY_0_MS - 2 * TF_1H_MS)
    assert pos_2 is not None
    df_2bar = _bars_df([
        (bar_01_25, 99.0, pos_2.stop_loss + 5.0, 98.5, 99.5, 10.0, 1000.0),
        (bar_09_05, 99.3, 99.6, 99.2, 99.4, 10.0, 990.0),
    ])
    seen_ts.clear()
    daemon.loader = _StubLoader(lambda s, tf, _l, ex: df_2bar if (s == "ETHUSDT" and tf == "5m" and ex == "binance") else pd.DataFrame())
    asyncio.run(daemon.update_active_positions())
    assert pos_2.status == "OPEN" and seen_ts == [bar_09_05]

    # Scenario B: 1h-spaced synthetic bars on 15m fallback (not ending <= 2*tf_ms) are NOT sliced away
    daemon.engine.positions.pop(pos_2.position_id, None)
    pos_b = eng.execute_directional_signal(_make_signal("BTCUSDT", "SELL", 100.0, 2.0), 100.0, DAY_0_MS - 2 * TF_1H_MS)
    assert pos_b is not None
    sparse_ts = [DAY_0_MS + 1 * TF_1H_MS, DAY_0_MS + 5 * TF_1H_MS, DAY_0_MS + 6 * TF_1H_MS, DAY_0_MS + 7 * TF_1H_MS]
    df_1h = _bars_df([(ts, 99.5, 99.8, 99.2, 99.4, 10.0, 990.0) for ts in sparse_ts])
    seen_ts.clear()
    daemon.loader = _StubLoader(lambda s, tf, _l, ex: df_1h if (s == "BTCUSDT" and tf == "15m" and ex == "binance") else pd.DataFrame())
    asyncio.run(daemon.update_active_positions())
    assert seen_ts == sparse_ts and pos_b.status == "OPEN"


def test_daemon_run_loop_runs_catchup_before_realtime_monitor(temp_db, monkeypatch):
    now_ms = int(time.time() * 1000)
    bar_ts = now_ms - 10 * 60_000
    eng = PaperTradingEngine(initial_capital=10000.0, db_path=temp_db, max_concurrent_positions=10)
    pos = eng.execute_directional_signal(_make_signal("SOLUSDT", "SELL", 100.0, 2.0), 100.0, now_ms - 3 * TF_1H_MS)
    assert pos is not None

    df_one = _bars_df([(bar_ts, 99.5, 99.8, 99.2, 99.4, 10.0, 990.0)])

    async def _get_candles(sym, tf, _l, ex):
        await asyncio.sleep(0)  # yield inside catch-up before finishing
        return df_one if (sym == "SOLUSDT" and tf == "5m" and ex == "binance") else pd.DataFrame()

    daemon = rps.PaperTradingDaemon(
        poll_interval_seconds=0, engine=eng, loader=_StubLoader(_get_candles), enable_realtime_monitor=True,
    )
    events: List[str] = []
    rt_done = {"v": False}

    orig_catchup = daemon.update_active_positions
    orig_rt_price = eng.update_realtime_price

    async def _wrapped_catchup():
        events.append("catchup_start")
        await orig_catchup()
        events.append("catchup_end")

    def _wrapped_rt_price(*args, **kwargs):
        events.append("rt_update_realtime_price")
        rt_done["v"] = True
        daemon.is_running = False
        return orig_rt_price(*args, **kwargs)

    async def _noop(*_a, **_kw):
        return None

    async def _wait_for_rt():
        for _ in range(100):
            if rt_done["v"]:
                break
            await asyncio.sleep(0.01)
        daemon.is_running = False

    monkeypatch.setattr(daemon, "update_active_positions", _wrapped_catchup)
    monkeypatch.setattr(eng, "update_realtime_price", _wrapped_rt_price)
    monkeypatch.setattr(daemon.context_enricher, "maybe_refresh_external_cache", _noop)
    monkeypatch.setattr(daemon, "evaluate_golden_pot_opportunities", _noop)
    monkeypatch.setattr(daemon, "evaluate_scalp_opportunities", _noop)
    monkeypatch.setattr(daemon, "evaluate_swing_opportunities", _noop)
    monkeypatch.setattr(daemon, "evaluate_swing_4h_opportunities", _wait_for_rt)
    monkeypatch.setattr(daemon.rt_tracker, "sync_symbols", lambda _syms: None)
    monkeypatch.setattr(daemon.rt_tracker, "stop", lambda: None)
    monkeypatch.setattr(daemon.rt_tracker, "drain", lambda sym, max_age_s=5.0: (99.4, 99.5, 99.3, bar_ts + 60_000) if sym == "SOLUSDT" else None)

    asyncio.run(daemon.run_loop(max_cycles=0))
    assert events[:3] == ["catchup_start", "catchup_end", "rt_update_realtime_price"]


# ============================================================================
# 4. flask_interface: events order, legacy exit, non-closing ignore & WAL race
# ============================================================================

def test_flask_closed_position_events_order_legacy_exit_and_wal_race(temp_db, monkeypatch):
    PaperTradingEngine(initial_capital=10000.0, db_path=temp_db, restore=False)
    DataLoader(db_path=temp_db)

    t0, t1, t2, t3, t4 = DAY_0_MS, DAY_0_MS + 1_000, DAY_0_MS + 2_000, DAY_0_MS + 3_000, DAY_0_MS + 4_000
    with _db_connect(temp_db, timeout=5.0) as con:
        pos_sql = (
            "INSERT INTO paper_positions (position_id, symbol, side, entry_price, current_price, quantity, notional, "
            "stop_loss, take_profit, unrealized_pnl, realized_pnl, funding_collected, fees_paid, entry_timestamp, "
            "strategy_type, status, tp1_price, tp1_hit, tp2_price, initial_quantity, leverage, tp1_fraction, tp2_fraction) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        con.executemany(pos_sql, [
            ("pos_be", "SOLUSDT", "SHORT", 69.45, 69.41, 0.0, 100.0, 69.41, 66.50, 0.0, 0.85, 0.02, 0.05,
             t0, "DIRECTIONAL", "CLOSED", 67.86, 1, 66.50, 1.44, 1.0, 0.4, 0.3),
            ("pos_tr", "ETHUSDT", "LONG", 100.0, 105.50, 0.0, 100.0, 105.50, 108.0, 0.0, 4.10, 0.0, 0.06,
             t0 + 100, "DIRECTIONAL", "CLOSED", 104.0, 1, 108.0, 1.0, 1.0, 0.4, 0.3),
            ("pos_leg", "BTCUSDT", "LONG", 100.0, 112.50, 0.0, 100.0, 95.0, 112.50, 0.0, 12.30, 0.01, 0.12,
             t0 + 200, "DIRECTIONAL", "CLOSED", 0.0, 0, 0.0, 1.0, 1.0, 0.5, 0.0),
        ])
        trades = [
            ("tr1", "pos_be", "SOLUSDT", "OPEN_SHORT", 69.45, 1.44, 0.03, 0.0, t0, "open"),
            ("tr2", "pos_be", "SOLUSDT", "FUNDING", 68.50, 1.44, 0.0, 0.02, t1, "funding"),
            ("tr3", "pos_be", "SOLUSDT", "BREAKEVEN_STOP", 69.41, 0.864, 0.01, 0.02, t2, "be"),
            ("tr4", "pos_be", "SOLUSDT", "TP1_PARTIAL", 67.86, 0.576, 0.01, 0.81, t3, "tp1"),
            ("tr5", "pos_tr", "ETHUSDT", "OPEN_LONG", 100.0, 1.0, 0.03, 0.0, t0 + 100, "open"),
            ("tr6", "pos_tr", "ETHUSDT", "TRAILING_STOP", 105.50, 0.3, 0.01, 1.50, t2, "trail"),
            ("tr7", "pos_tr", "ETHUSDT", "TP2_PARTIAL", 108.00, 0.3, 0.01, 2.30, t3, "tp2"),
            ("tr8", "pos_tr", "ETHUSDT", "FUNDING", 106.00, 0.3, 0.0, -0.01, t4, "funding"),
            ("tr9", "pos_leg", "BTCUSDT", "OPEN_LONG", 100.0, 1.0, 0.06, 0.0, t0 + 200, "open"),
            ("tr10", "pos_leg", "BTCUSDT", "FUNDING", 105.0, 1.0, 0.0, 0.01, t1, "funding"),
            ("tr11", "pos_leg", "BTCUSDT", "TAKE_PROFIT", 112.50, 1.0, 0.06, 12.29, t2, "legacy_tp"),
        ]
        con.executemany(
            "INSERT INTO paper_trades (trade_id, position_id, symbol, action, price, quantity, fee, pnl, timestamp, rationale) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            trades,
        )

    monkeypatch.setattr(fi, "DB_PATH", temp_db)
    closed_rows = {r["position_id"]: r for r in fi._query_paper_positions("closed")}

    assert closed_rows["pos_be"]["exit_price"] == pytest.approx(69.41)
    assert closed_rows["pos_be"]["closed_at_ms"] == t2
    assert closed_rows["pos_be"]["close_action"] == "BREAKEVEN_STOP"
    assert [e["action"] for e in closed_rows["pos_be"]["events"]] == ["OPEN_SHORT", "FUNDING", "BREAKEVEN_STOP", "TP1_PARTIAL"]
    assert [e["timestamp_ms"] for e in closed_rows["pos_be"]["events"]] == [t0, t1, t2, t3]

    assert closed_rows["pos_tr"]["exit_price"] == pytest.approx(105.50)
    assert closed_rows["pos_tr"]["closed_at_ms"] == t2
    assert closed_rows["pos_tr"]["close_action"] == "TRAILING_STOP"

    assert closed_rows["pos_leg"]["exit_price"] == pytest.approx(112.50)
    assert closed_rows["pos_leg"]["closed_at_ms"] == t2
    assert closed_rows["pos_leg"]["close_action"] == "TAKE_PROFIT"
    assert closed_rows["pos_leg"]["exit_ladder"] is False
    assert [e["action"] for e in closed_rows["pos_leg"]["events"]] == ["OPEN_LONG", "FUNDING", "TAKE_PROFIT"]

    wal_path = f"{temp_db}-wal"
    orig_exists, orig_getsize = os.path.exists, os.path.getsize
    monkeypatch.setattr(fi.os.path, "exists", lambda p: True if p == wal_path else orig_exists(p))

    def _racy_getsize(p):
        if p == wal_path:
            raise FileNotFoundError("WAL checkpointed concurrently")
        return orig_getsize(p)

    monkeypatch.setattr(fi.os.path, "getsize", _racy_getsize)
    metrics = fi.get_sqlite_metrics()
    assert metrics["status"] == "ONLINE" and metrics["wal_size_kb"] == 0 and "error" not in metrics
