"""
Independent QA Suite: Real-Time Paper Trading, Open-Runner Exit Ladder, LivePriceHub, Flask Live-Tick & Playwright UI.
Validates contracts with zero live network dependency and isolated temporary SQLite databases.
"""
from __future__ import annotations

import asyncio, os, random, sqlite3, sys, tempfile, threading, time, urllib.request, math
from typing import Any, Dict, List
from werkzeug.serving import make_server
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_LOCAL_LIBS = os.path.expanduser("~/.cache/pwlibs/usr/lib/x86_64-linux-gnu")
if os.path.isdir(_LOCAL_LIBS) and _LOCAL_LIBS not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _LOCAL_LIBS + ":" + os.environ.get("LD_LIBRARY_PATH", "")

CHROMIUM_1255_BIN = (
    "/home/eudespatricio/.cache/ms-playwright/chromium_headless_shell-1255/"
    "chrome-headless-shell-linux64/chrome-headless-shell"
)

from smart_trade import (  # noqa: E402
    BREAKEVEN_FEE_BUFFER_PCT, EXIT_TP1_FRACTION, EXIT_TP2_FRACTION, STRATEGY_PROFILES, PaperTradingEngine,
)
from backtesting import BacktestEngine  # noqa: E402
from signal_generator import TradeSignal  # noqa: E402
from live_price_hub import LIVE_PRICE_HUB, LivePriceHub  # noqa: E402
import flask_interface as fi  # noqa: E402
from run_paper_simulation import PaperTradingDaemon  # noqa: E402

TS0, H_MS = 1_700_000_000_000, 3_600_000


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


def _make_signal(symbol: str = "BTCUSDT", direction: str = "BUY", price: float = 100.0, atr: float = 2.0) -> TradeSignal:
    sgn = 1.0 if direction == "BUY" else -1.0
    return TradeSignal(
        signal_id=f"sig_{symbol}_{direction}", timestamp=TS0, symbol=symbol, direction=direction,
        entry_price=price, stop_loss=price - sgn * atr, take_profit=price + sgn * 2.5 * atr,
        atr=atr, confidence=0.90, consensus_score=0.85, position_size=1e9, risk_dollars=1.0,
        rationale="qa_contract_test",
    )


# ============================================================================
# 1. ENGINE & BACKTEST CONTRACTS
# ============================================================================

@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_strategy_profiles_directional_golden_pot_scalp_and_swing_4h(temp_db, side):
    p_dir, p_gp = STRATEGY_PROFILES["DIRECTIONAL"], STRATEGY_PROFILES["GOLDEN_POT"]
    p_sc, p_sw = STRATEGY_PROFILES["SCALP"], STRATEGY_PROFILES["SWING_4H"]

    assert p_dir.tp3_atr_mult == 0.0 and p_dir.tp1_atr_mult > 0.0 and p_dir.tp2_atr_mult > 0.0 and p_dir.trail_atr_mult > 0.0
    assert p_gp.tp3_atr_mult == 0.0 and p_gp.tp1_atr_mult > 0.0 and p_gp.tp2_atr_mult > 0.0 and p_gp.trail_atr_mult > 0.0
    assert p_sc.tp3_atr_mult == pytest.approx(2.2) and p_sc.trail_atr_mult == 0.0
    assert (p_sw.tp1_atr_mult, p_sw.tp2_atr_mult, p_sw.tp3_atr_mult, p_sw.trail_atr_mult) == (0.0, 0.0, 0.0, 3.5)

    eng = PaperTradingEngine(initial_capital=10000.0, db_path=temp_db, max_concurrent_positions=20)
    sgn = 1.0 if side == "BUY" else -1.0

    for fn, sym, prof in [(eng.execute_directional_signal, "BTCUSDT", p_dir), (eng.execute_golden_pot_signal, "SOLUSDT", p_gp)]:
        pos = fn(_make_signal(sym, side, 100.0, 2.0), 100.0, TS0)
        assert pos is not None and pos.tp3_price == 0.0 and pos.tp1_price > 0.0 and pos.tp2_price > 0.0
        assert pos.trailing_atr_mult == pytest.approx(prof.trail_atr_mult) and pos.take_profit == pytest.approx(pos.tp2_price)

    pos_sc = eng.execute_scalp_signal(_make_signal("ETHUSDT", side, 100.0, 2.0), 100.0, TS0)
    assert pos_sc is not None and pos_sc.tp3_price == pytest.approx(pos_sc.entry_price + sgn * 2.2 * 2.0) and pos_sc.trailing_atr_mult == 0.0
    tr_sc = eng.update_realtime_price("ETHUSDT", pos_sc.tp3_price + sgn * 0.25, pos_sc.entry_timestamp + 10_000)
    assert [t.action for t in tr_sc] == ["TAKE_PROFIT_3"] and pos_sc.status == "CLOSED"

    pos_sw = eng.execute_swing_4h_signal(_make_signal("ADAUSDT", side, 100.0, 2.0), 100.0, TS0)
    assert pos_sw is not None and (pos_sw.tp1_price, pos_sw.tp2_price, pos_sw.tp3_price) == (0.0, 0.0, 0.0)
    assert pos_sw.trailing_atr_mult == pytest.approx(3.5) and pos_sw.trailing_active is True


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_update_realtime_price_causal_ladder_open_runner_long_and_short(temp_db, side):
    eng = PaperTradingEngine(initial_capital=10000.0, db_path=temp_db, max_concurrent_positions=20)
    sgn, atr = (1.0 if side == "BUY" else -1.0), 2.0

    # Pre-tick stop_hit priority (causal, zero intra-tick lookahead)
    pos_sl = eng.execute_directional_signal(_make_signal("ETHUSDT", side, 100.0, atr), 100.0, TS0)
    hi_sl, lo_sl = max(pos_sl.stop_loss - sgn * 0.5, pos_sl.tp1_price + sgn * 0.5), min(pos_sl.stop_loss - sgn * 0.5, pos_sl.tp1_price + sgn * 0.5)
    tr_sl = eng.update_realtime_price("ETHUSDT", pos_sl.entry_price, pos_sl.entry_timestamp + 1_000, high=hi_sl, low=lo_sl)
    assert [t.action for t in tr_sl] == ["STOP_LOSS"] and pos_sl.status == "CLOSED"

    pos = eng.execute_directional_signal(_make_signal("BTCUSDT", side, 100.0, atr), 100.0, TS0)
    init_qty, entry, t_base = pos.quantity, pos.entry_price, pos.entry_timestamp

    # 1) Pre-entry timestamp (< entry_timestamp) is strictly ignored
    tr_pre = eng.update_realtime_price("BTCUSDT", entry - sgn * 20.0, t_base - 1, high=entry + 50.0, low=entry - 50.0)
    assert tr_pre == [] and pos.status == "OPEN" and not pos.tp1_hit and pos.quantity == pytest.approx(init_qty)

    # 2) TP1_PARTIAL (40%) -> SL moves to breakeven (+fee buffer)
    tr_tp1 = eng.update_realtime_price("BTCUSDT", price=pos.tp1_price + sgn * 0.1, timestamp=t_base + 5_000)
    assert [t.action for t in tr_tp1] == ["TP1_PARTIAL"] and pos.tp1_hit and not pos.tp2_hit
    assert pos.quantity == pytest.approx(init_qty * (1.0 - EXIT_TP1_FRACTION))
    assert pos.stop_loss == pytest.approx(entry * (1.0 + sgn * BREAKEVEN_FEE_BUFFER_PCT))

    # 3) TP2_PARTIAL (30%) -> trailing_active=True
    px_tp2 = pos.tp2_price + sgn * 0.1
    tr_tp2 = eng.update_realtime_price("BTCUSDT", price=px_tp2, timestamp=t_base + 10_000)
    assert [t.action for t in tr_tp2] == ["TP2_PARTIAL"] and pos.tp2_hit and pos.trailing_active
    assert pos.quantity == pytest.approx(init_qty * (1.0 - EXIT_TP1_FRACTION - EXIT_TP2_FRACTION))
    trail_at_tp2 = px_tp2 - sgn * pos.trailing_atr_mult * atr
    assert pos.stop_loss == pytest.approx(trail_at_tp2)

    # 4) Price climbs +10x ATR without TAKE_PROFIT_3; causal ratchet does not stop out on same tick
    peak_10x = entry + sgn * 10.0 * atr
    new_expected_trail = peak_10x - sgn * pos.trailing_atr_mult * atr
    safe_vs_old_stop = trail_at_tp2 + sgn * 0.40
    tr_climb = eng.update_realtime_price(
        "BTCUSDT", price=peak_10x, timestamp=t_base + 15_000,
        high=max(peak_10x, safe_vs_old_stop), low=min(peak_10x, safe_vs_old_stop),
    )
    assert tr_climb == [] and pos.status == "OPEN" and pos.stop_loss == pytest.approx(new_expected_trail)

    # 5) Mild pullback never loosens trailing stop
    mild_px = peak_10x - sgn * 0.5 * atr
    assert eng.update_realtime_price("BTCUSDT", price=mild_px, timestamp=t_base + 20_000) == []
    assert pos.stop_loss == pytest.approx(new_expected_trail)

    # 6) Reversal crossing updated trailing stop triggers TRAILING_STOP at expected price
    rev_px = new_expected_trail - sgn * 0.30
    tr_rev = eng.update_realtime_price("BTCUSDT", price=rev_px, timestamp=t_base + 25_000, high=max(mild_px, rev_px), low=min(mild_px, rev_px))
    assert [t.action for t in tr_rev] == ["TRAILING_STOP"] and pos.status == "CLOSED"
    assert tr_rev[0].price == pytest.approx(new_expected_trail * (1.0 - sgn * eng.slippage_spot))


def test_legacy_pre_e7_tp2_close_restore_migration_and_mark_to_market(temp_db):
    eng = PaperTradingEngine(initial_capital=10000.0, db_path=temp_db, max_concurrent_positions=20)

    pos_leg = eng.execute_directional_signal(_make_signal("BTCUSDT", "BUY", 100.0, 2.0), 100.0, TS0)
    pos_leg.tp3_price = pos_leg.tp2_fraction = pos_leg.trailing_atr_mult = 0.0
    eng.update_realtime_price("BTCUSDT", price=pos_leg.tp1_price + 0.1, timestamp=pos_leg.entry_timestamp + 2_000)
    assert pos_leg.tp1_hit and pos_leg.status == "OPEN"
    tr_leg2 = eng.update_realtime_price("BTCUSDT", price=pos_leg.tp2_price + 0.1, timestamp=pos_leg.entry_timestamp + 4_000)
    assert [t.action for t in tr_leg2] == ["TAKE_PROFIT_2"] and pos_leg.status == "CLOSED"

    pos_mig = eng.execute_directional_signal(_make_signal("SOLUSDT", "BUY", 100.0, 2.0), 100.0, TS0 + 5_000)
    pos_mig.tp3_price = pos_mig.take_profit = 112.0
    eng._save_position(pos_mig)

    eng2 = PaperTradingEngine(initial_capital=10000.0, db_path=temp_db, max_concurrent_positions=20)
    restored = eng2.positions[pos_mig.position_id]
    assert restored.tp3_price == 0.0 and restored.take_profit == pytest.approx(restored.tp2_price)
    with sqlite3.connect(temp_db) as con:
        row = con.execute("SELECT tp3_price, take_profit, tp2_price FROM paper_positions WHERE position_id=?", (pos_mig.position_id,)).fetchone()
    assert row[0] == 0.0 and row[1] == pytest.approx(row[2])

    eng2.mark_to_market("SOLUSDT", price=10.0, persist=False)
    assert restored.status == "OPEN" and restored.current_price == pytest.approx(10.0)
    eng2.mark_to_market("SOLUSDT", price=250.0, persist=True)
    assert restored.status == "OPEN"
    with sqlite3.connect(temp_db) as con:
        db_px, db_upnl, db_status = con.execute("SELECT current_price, unrealized_pnl, status FROM paper_positions WHERE position_id=?", (restored.position_id,)).fetchone()
    assert db_status == "OPEN" and db_px == pytest.approx(250.0)
    assert db_upnl == pytest.approx((250.0 - restored.entry_price) * restored.quantity)


@pytest.mark.parametrize("signal_val", [1, 2])
def test_backtest_pilot_parity_directional_climbs_past_6x_atr_and_exits_trailing(signal_val):
    atr, spot, sgn = 2.0, 0.0002, (1.0 if signal_val == 1 else -1.0)
    entry = 100.0 * (1.0 + sgn * spot)
    tp1, tp2, peak_7x = entry + sgn * 1.5 * atr, entry + sgn * 2.5 * atr, entry + sgn * 7.0 * atr
    trail_stop = peak_7x - sgn * 1.2 * atr

    def _bar(o, fav, adv, c):
        p1, p2 = o + sgn * fav, o - sgn * adv
        return (o, max(o, c, p1, p2), min(o, c, p1, p2), c)

    bars = [
        (100.0, 100.0, 100.0, 100.0),
        _bar(100.0, 1.6 * atr, 0.2 * atr, entry + sgn * 1.55 * atr),
        _bar(entry + sgn * 1.55 * atr, 1.1 * atr, 0.1 * atr, entry + sgn * 2.55 * atr),
        _bar(entry + sgn * 2.55 * atr, 4.45 * atr, 0.2 * atr, peak_7x - sgn * 0.2 * atr),
        (peak_7x - sgn * 0.2 * atr, max(peak_7x - sgn * 0.2 * atr, trail_stop - sgn * 0.5), min(peak_7x - sgn * 0.2 * atr, trail_stop - sgn * 0.5), trail_stop - sgn * 0.3),
    ]
    df = pd.DataFrame([{"timestamp": TS0 + i * H_MS, "open": o, "high": h, "low": l, "close": c, "atr_14": atr} for i, (o, h, l, c) in enumerate(bars)])
    sigs = np.zeros(len(df), dtype=int)
    sigs[0] = signal_val
    res = BacktestEngine(pilot_parity=True, risk_per_trade=None, strategy_type="DIRECTIONAL", funding_rate_8h=0.0).run(df, sigs)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.exit_reason == "TRAILING_STOP" and t.legs == 3
    q = 100.0 / entry
    q1, q2 = q * EXIT_TP1_FRACTION, q * EXIT_TP2_FRACTION
    px3 = trail_stop * (1.0 - sgn * spot)
    assert t.pnl_gross == pytest.approx(sgn * (q1 * (tp1 - entry) + q2 * (tp2 - entry) + (q - q1 - q2) * (px3 - entry)), rel=1e-9)


# ============================================================================
# 2. DAEMON REAL-TIME MONITOR CONTRACT
# ============================================================================

def test_daemon_realtime_monitor_executes_sl_tp1_tp2_trailing_without_5m_bar_or_network(temp_db, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Network forbidden")))
    eng = PaperTradingEngine(initial_capital=10000.0, db_path=temp_db, max_concurrent_positions=10)
    daemon = PaperTradingDaemon(poll_interval_seconds=15, engine=eng, enable_realtime_monitor=True)

    now_bar_ts = int(time.time() * 1000) - 7_200_000
    pos_long = eng.execute_directional_signal(_make_signal("BTCUSDT", "BUY", 100.0, 2.0), 100.0, now_bar_ts)
    pos_short = eng.execute_directional_signal(_make_signal("ETHUSDT", "SELL", 200.0, 4.0), 200.0, now_bar_ts)
    t_entry = pos_long.entry_timestamp

    # Verify RealtimePriceTracker.drain() returns exact trade ts_ms (never overwritten by now_ms), preserving pre-entry guard
    daemon.rt_tracker._on_trade("BTCUSDT", t_entry - 500, 80.0, 1.0, False)
    drained_pre = daemon.rt_tracker.drain("BTCUSDT", max_age_s=5.0)
    assert drained_pre is not None and drained_pre[3] == t_entry - 500
    assert eng.update_realtime_price("BTCUSDT", price=drained_pre[0], timestamp=drained_pre[3], high=drained_pre[1], low=drained_pre[2]) == []
    assert pos_long.status == "OPEN"

    monkeypatch.setattr(daemon.rt_tracker, "sync_symbols", lambda syms: None)
    monkeypatch.setattr(daemon.rt_tracker, "stop", lambda: None)

    async def _noop_funding(syms, now_ms):
        return None

    monkeypatch.setattr(daemon, "_refresh_funding_cache", _noop_funding)
    tick_batches = [
        [("BTCUSDT", t_entry + 1_000, 103.2), ("ETHUSDT", t_entry + 1_000, 204.5)],
        [("BTCUSDT", t_entry + 2_000, 105.2)],
        [("BTCUSDT", t_entry + 3_000, 120.0)],
        [("BTCUSDT", t_entry + 4_000, 117.0)],
    ]
    step_idx, orig_drain = {"i": 0}, daemon.rt_tracker.drain

    def _scripted_drain(sym: str, max_age_s: float = 5.0):
        idx = step_idx["i"]
        if idx < len(tick_batches):
            for s, ts_ms, px in tick_batches[idx]:
                if s == sym:
                    daemon.rt_tracker._on_trade(s, ts_ms, px, 1.0, False)
        return orig_drain(sym, max_age_s=max_age_s)

    monkeypatch.setattr(daemon.rt_tracker, "drain", _scripted_drain)

    async def _fast_sleep(delay: float):
        step_idx["i"] += 1
        if step_idx["i"] >= len(tick_batches):
            daemon.is_running = False

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)
    daemon.is_running = True
    asyncio.new_event_loop().run_until_complete(daemon._realtime_position_monitor_loop())

    assert pos_short.status == "CLOSED" and pos_long.status == "CLOSED"
    eth_actions = [t.action for t in eng.trade_history if t.position_id == pos_short.position_id and not t.action.startswith("OPEN_")]
    btc_actions = [t.action for t in eng.trade_history if t.position_id == pos_long.position_id and not t.action.startswith("OPEN_")]
    assert eth_actions == ["STOP_LOSS"] and btc_actions == ["TP1_PARTIAL", "TP2_PARTIAL", "TRAILING_STOP"]


# ============================================================================
# 3. LIVE PRICE HUB & FLASK API CONTRACTS
# ============================================================================

def test_live_price_hub_no_autostart_on_import_and_deterministic_injection(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Network forbidden")))
    assert LIVE_PRICE_HUB.is_running is False
    LIVE_PRICE_HUB.stop()  # Must not raise when never started
    hub = LivePriceHub()
    assert hub.is_running is False
    hub.stop()  # Must not raise when _thread is None

    hub.inject_price("BTCUSDT", 66000.0, open_24h=65000.0, high_24h=66500.0, low_24h=64800.0, volume_24h=1234.5)
    assert hub.has_fresh_price("BTCUSDT") is True
    tk = hub.get_ticker("BTCUSDT")
    assert tk is not None and tk["last_price"] == pytest.approx(66000.0)
    assert tk["change_pct_24h"] == pytest.approx(round(((66000.0 - 65000.0) / 65000.0) * 100.0, 2))
    assert hub.get_prices(["BTCUSDT"], allow_rest_fallback=False) == {"BTCUSDT": pytest.approx(66000.0)}

    hub.inject_kline("BTCUSDT", "5m", {"timestamp": TS0, "open": 65900.0, "high": 66100.0, "low": 65850.0, "close": 66000.0, "volume": 42.0})
    kc = hub.get_open_candle("BTCUSDT", "5m")
    assert kc is not None and kc["close"] == pytest.approx(66000.0) and kc["timestamp"] == TS0
    hub.clear()
    assert hub.has_fresh_price("BTCUSDT") is False


def _seed_isolated_flask_db(db_path: str) -> Dict[str, Any]:
    now_ms = int(time.time() // 3600 * 3600 * 1000)
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE candles (symbol TEXT NOT NULL, timeframe TEXT NOT NULL, timestamp INTEGER NOT NULL, open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL, quote_volume REAL NOT NULL, exchange TEXT DEFAULT 'binance', created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (symbol, timeframe, timestamp))")
    step_map, rnd, rows, last_ts_by_tf = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}, random.Random(42), [], {}
    for tf, step in step_map.items():
        px = 65000.0
        for i in range(210):
            ts = now_ms - (209 - i) * step
            o, c = px, round(px * (1.0 + rnd.gauss(0, 0.002) + 0.0003 * math.sin(i / 8.0)), 2)
            h, l, v = round(max(o, c) * 1.0012, 2), round(min(o, c) * 0.9988, 2), round(rnd.uniform(50.0, 300.0), 2)
            rows.append(("BTCUSDT", tf, ts, o, h, l, c, v, round(v * c, 2), "binance"))
            px, last_ts_by_tf[tf] = c, ts
    con.executemany("INSERT INTO candles (symbol,timeframe,timestamp,open,high,low,close,volume,quote_volume,exchange) VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    con.execute("CREATE TABLE paper_positions (position_id TEXT PRIMARY KEY, symbol TEXT, side TEXT, entry_price REAL, current_price REAL, quantity REAL, notional REAL, stop_loss REAL, take_profit REAL, unrealized_pnl REAL, realized_pnl REAL, entry_timestamp INTEGER, strategy_type TEXT, status TEXT, tp1_price REAL, tp1_hit INTEGER, tp2_price REAL, initial_quantity REAL, leverage REAL, tp3_price REAL, tp2_hit INTEGER, tp1_fraction REAL, tp2_fraction REAL, trailing_active INTEGER, trailing_stop_price REAL, margin REAL)")
    con.execute("CREATE TABLE paper_trades (trade_id TEXT PRIMARY KEY, position_id TEXT, symbol TEXT, action TEXT, price REAL, quantity REAL, fee REAL, pnl REAL, timestamp INTEGER, rationale TEXT)")
    con.execute("CREATE TABLE paper_portfolio (timestamp INTEGER PRIMARY KEY, equity REAL, balance REAL, margin_used REAL)")
    con.execute("CREATE TABLE simulation_settings (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)")
    con.execute("INSERT INTO paper_positions VALUES ('POS_DIR_1', 'BTCUSDT', 'LONG', 65000.0, 65050.0, 0.5, 32500.0, 64700.0, 65500.0, 25.0, 0.0, ?, 'DIRECTIONAL', 'OPEN', 65300.0, 0, 65500.0, 0.5, 2.0, 0.0, 0, 0.4, 0.3, 0, 0.0, 16250.0)", (now_ms - 3_600_000,))
    con.execute("INSERT INTO paper_trades VALUES ('TR_1', 'POS_DIR_1', 'BTCUSDT', 'OPEN_DIRECTIONAL_LONG', 65000.0, 0.5, 6.5, 0.0, ?, 'entry')", (now_ms - 3_600_000,))
    con.execute("INSERT INTO paper_portfolio VALUES (?, 10025.0, 10000.0, 16250.0)", (now_ms - 60_000,))
    con.execute("INSERT INTO simulation_settings VALUES ('max_concurrent_positions', '5', '2026-11-24T00:00:00Z')")
    con.commit()
    con.close()
    return {"now_ms": now_ms, "last_ts_by_tf": last_ts_by_tf}


def test_flask_api_stopped_hub_raw_db_vs_injected_live_repricing(tmp_path, monkeypatch):
    db_path = str(tmp_path / "flask_live_test.db")
    meta = _seed_isolated_flask_db(db_path)
    monkeypatch.setattr(fi, "DB_PATH", db_path)
    monkeypatch.setattr(fi, "MICRO_DB_PATH", str(tmp_path / "absent_micro.db"))
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Network forbidden")))

    if LIVE_PRICE_HUB.is_running:
        LIVE_PRICE_HUB.stop()
    LIVE_PRICE_HUB.clear()
    fi._TICKER_CACHE.clear()
    fi._OPEN_CANDLE_CACHE.clear()
    fi.app.config["TESTING"] = True
    try:
        with fi.app.test_client() as client:
            p_raw = client.get("/api/paper-trading/positions/open").get_json()["positions"][0]
            assert p_raw["current_price"] == pytest.approx(65050.0) and p_raw["unrealized_pnl"] == pytest.approx(25.0) and p_raw["tp3"] is None
            res_status_raw = client.get("/api/paper-trading/status").get_json()
            assert res_status_raw["current_balance"] == pytest.approx(10000.0) and res_status_raw["current_equity"] == pytest.approx(10025.0)

            injected_px, expected_upnl, expected_equity = 65400.0, 200.0, 10200.0
            LIVE_PRICE_HUB.inject_price("BTCUSDT", injected_px, open_24h=64800.0, high_24h=65600.0, low_24h=64500.0, volume_24h=5000.0)
            LIVE_PRICE_HUB.inject_kline("BTCUSDT", "1h", {"timestamp": meta["last_ts_by_tf"]["1h"], "open": 65100.0, "high": 65450.0, "low": 65050.0, "close": injected_px, "volume": 120.0})

            tick_data = client.get("/api/live-tick?symbol=BTCUSDT&timeframe=1h").get_json()
            assert tick_data["ticker"]["last_price"] == pytest.approx(injected_px) and tick_data["candle"]["close"] == pytest.approx(injected_px)
            assert tick_data["open_positions"][0]["current_price"] == pytest.approx(injected_px) and tick_data["open_positions"][0]["unrealized_pnl"] == pytest.approx(expected_upnl)
            assert tick_data["paper_status"]["current_equity"] == pytest.approx(expected_equity)

            open_live = client.get("/api/paper-trading/positions/open").get_json()
            assert open_live["positions"][0]["current_price"] == pytest.approx(injected_px) and open_live["summary"]["unrealized_pnl"] == pytest.approx(expected_upnl)

            status_live = client.get("/api/paper-trading/status").get_json()
            assert status_live["current_equity"] == pytest.approx(expected_equity) and status_live["unrealized_pnl"] == pytest.approx(expected_upnl)

            chart_live = client.get("/api/chart/BTCUSDT/1h?limit=40").get_json()
            assert chart_live["summary"]["last_price"] == pytest.approx(injected_px) and chart_live["tv_data"]["candles"][-1]["close"] == pytest.approx(injected_px)
            assert chart_live["active_position"]["current_price"] == pytest.approx(injected_px) and chart_live["active_position"]["unrealized_pnl"] == pytest.approx(expected_upnl)
    finally:
        LIVE_PRICE_HUB.clear()
        fi._TICKER_CACHE.clear()
        fi._OPEN_CANDLE_CACHE.clear()


# ============================================================================
# 4. BROWSER UI PLAYWRIGHT VERIFICATION (CHROMIUM-1255)
# ============================================================================

@pytest.mark.skipif(not os.path.exists(CHROMIUM_1255_BIN), reason="Chromium 1255 binary not found")
def test_browser_ui_playwright_live_tick_zoom_and_open_runner_overlay(tmp_path, monkeypatch):
    from playwright.sync_api import sync_playwright

    db_path, env_path = str(tmp_path / "ui_playwright_live.db"), str(tmp_path / ".env")
    with open(env_path, "w", encoding="utf-8") as f:
        f.write("DEFAULT_SYMBOL=BTCUSDT\n")
    meta = _seed_isolated_flask_db(db_path)

    monkeypatch.setattr(fi, "DB_PATH", db_path)
    monkeypatch.setattr(fi, "ENV_FILE_PATH", env_path)
    monkeypatch.setattr(fi, "MICRO_DB_PATH", str(tmp_path / "absent_micro.db"))

    if LIVE_PRICE_HUB.is_running:
        LIVE_PRICE_HUB.stop()
    LIVE_PRICE_HUB.clear()
    fi._TICKER_CACHE.clear()
    fi._OPEN_CANDLE_CACHE.clear()

    tick_counter = {"n": 0, "price": 65200.0}

    def _inject_current_tick(px: float) -> None:
        LIVE_PRICE_HUB.inject_price("BTCUSDT", px, open_24h=64800.0, high_24h=max(65600.0, px), low_24h=64500.0, volume_24h=5100.0)
        for tf, ts_val in meta["last_ts_by_tf"].items():
            LIVE_PRICE_HUB.inject_kline("BTCUSDT", tf, {"timestamp": ts_val, "open": 65000.0, "high": max(65300.0, px), "low": 64950.0, "close": px, "volume": 150.0 + tick_counter["n"]})

    _inject_current_tick(tick_counter["price"])
    orig_api_live_tick = fi.api_live_tick

    def _wrapped_live_tick():
        tick_counter["n"] += 1
        tick_counter["price"] = round(65200.0 + tick_counter["n"] * 20.0, 2)
        _inject_current_tick(tick_counter["price"])
        return orig_api_live_tick()

    monkeypatch.setattr(fi, "api_live_tick", _wrapped_live_tick)
    fi.app.view_functions["api_live_tick"] = _wrapped_live_tick

    server = make_server("127.0.0.1", 0, fi.app, threaded=True)
    base_url = f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.serve_forever, daemon=True).start()

    page_errors, console_errors, request_failures, http_errors, live_tick_requests = [], [], [], [], []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path=CHROMIUM_1255_BIN)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()

            page.on("pageerror", lambda e: page_errors.append(str(e)))
            page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
            page.on("requestfailed", lambda r: request_failures.append(f"{r.url} ({r.failure})"))
            page.on("response", lambda r: http_errors.append(f"{r.status} {r.url}") if r.status >= 400 else None)
            page.on("request", lambda r: live_tick_requests.append(time.monotonic()) if "/api/live-tick" in r.url else None)

            page.goto(base_url + "/", wait_until="load")
            page.wait_for_function("Boolean(window._tvChartInstance) === true", timeout=15000)
            page.wait_for_function("Array.isArray(window._openPositions) && window._openPositions.length === 1", timeout=10000)

            assert page.evaluate("Boolean(window._tvChartInstance) === true") is True

            c0 = len(live_tick_requests)
            page.wait_for_timeout(2300)
            assert (len(live_tick_requests) - c0) >= 2

            page.click("#btnAutoRefreshToggle")
            assert page.locator("#autoRefreshText").inner_text().strip() == "Pausado"
            page.wait_for_timeout(250)
            c_paused = len(live_tick_requests)
            page.wait_for_timeout(1600)
            assert len(live_tick_requests) == c_paused

            page.click("#btnAutoRefreshToggle")
            assert "Auto" in page.locator("#autoRefreshText").inner_text().strip()
            c_resume = len(live_tick_requests)
            page.wait_for_timeout(1600)
            assert len(live_tick_requests) > c_resume

            applied_range = page.evaluate(
                "async (rng) => { window._tvChartInstance.timeScale().setVisibleLogicalRange(rng); await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))); return window._tvChartInstance.timeScale().getVisibleLogicalRange(); }",
                {"from": 60.0, "to": 110.0},
            )
            t_zoom = len(live_tick_requests)
            page.wait_for_timeout(3600)
            assert (len(live_tick_requests) - t_zoom) >= 3
            after_range = page.evaluate("window._tvChartInstance.timeScale().getVisibleLogicalRange()")
            assert abs(float(after_range["from"]) - float(applied_range["from"])) < 1e-3
            assert abs(float(after_range["to"]) - float(applied_range["to"])) < 1e-3

            pos_row = page.locator("#paperPositionsTableBody tr[data-position-id='POS_DIR_1']")
            assert pos_row.count() == 1
            card_text = pos_row.inner_text()
            assert "Trailing ∞" in card_text and "TP1 / TP2 / RUNNER" in card_text.upper() and "TP1 / TP2 / TP3" not in card_text.upper()

            pos_row.click()
            page.wait_for_function("Boolean(window._activeOrderOverlay && window._activeOrderOverlay.position_id === 'POS_DIR_1')", timeout=5000)
            t_click = len(live_tick_requests)
            while len(live_tick_requests) <= t_click:
                page.wait_for_timeout(200)
            page.wait_for_timeout(300)

            sync_state = page.evaluate(
                "() => { const lines = window._tvNamedPriceLines || {}; const titles = Object.fromEntries(Object.entries(lines).map(([k, v]) => [k, v && v.spec ? v.spec.title : ''])); const pos = (window._openPositions || [])[0] || {}; return { keys: Object.keys(lines).sort(), titles, chartLastPx: Number(((window._cachedChartData || {}).summary || {}).last_price || 0), posCurrentPx: Number(pos.current_price || 0), posUpnl: Number(pos.unrealized_pnl || 0), ovUpnl: Number((window._activeOrderOverlay || {}).unrealizedPnl || 0), eqText: (document.getElementById('tradingCardEquity') || {}).innerText || '', tableText: (document.getElementById('paperPositionsTableBody') || {}).innerText || '' }; }"
            )
            assert "tp3" not in sync_state["keys"] and all(not t.startswith("TP3:") for t in sync_state["titles"].values())
            for req_key, prefix in [("entry", "ENTRY"), ("sl", "SL:"), ("tp1", "TP1:"), ("tp2", "TP2:")]:
                assert req_key in sync_state["keys"] and prefix in sync_state["titles"][req_key]

            live_px = sync_state["posCurrentPx"]
            expected_upnl = round((live_px - 65000.0) * 0.5, 2)
            assert sync_state["chartLastPx"] == pytest.approx(live_px)
            assert sync_state["posUpnl"] == pytest.approx(expected_upnl) and sync_state["ovUpnl"] == pytest.approx(expected_upnl)
            assert f"+${expected_upnl:.2f}" in sync_state["titles"]["entry"]
            assert f"${live_px:.2f}" in sync_state["tableText"]
            assert sync_state["eqText"] == f"${10000.0 + expected_upnl:,.2f}"
            assert page_errors == [] and console_errors == [] and request_failures == [] and http_errors == []

            context.close()
            browser.close()
    finally:
        fi.app.view_functions["api_live_tick"] = orig_api_live_tick
        server.shutdown()
        server.server_close()
        LIVE_PRICE_HUB.clear()
        fi._TICKER_CACHE.clear()
        fi._OPEN_CANDLE_CACHE.clear()
