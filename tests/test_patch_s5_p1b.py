"""Independent clean-window tests for Wave P1-B.

Covers:
1. TickAggregator isolation and WS trade ingestion
2. BinanceFuturesWSCollector event handling, liveness, liquidations, and symbols
3. TapeReadingAgent saturation fallback, legacy schema, and depth fallbacks
4. Directional maker-first execution logic and daemon loader parity
"""

import json
import sqlite3
import threading
import time
import pytest

from scripts.microstructure_collector import TickAggregator, SCHEMA as MICRO_SCHEMA
import scripts.microstructure_collector as mc
from scripts.binance_ws_collector import BinanceFuturesWSCollector
from agents.tape_reading_agent import TapeReadingAgent
from smart_trade import PaperTradingEngine, MAKER_SLIPPAGE_PCT, MARKET_SLIPPAGE_PCT
from signal_generator import TradeSignal
from scripts.run_paper_simulation import PaperTradingDaemon


def test_tick_aggregator_bitget_binance_isolation(monkeypatch):
    mc._BITGET_BACKOFF_UNTIL.clear()
    agg = TickAggregator()
    now_ms = 1700000000000

    def mock_get_json_bitget(url, params=None, retries=1):
        if mc.BITGET in url:
            return {"data": [{"tradeId": "1400000000000000000", "ts": str(now_ms), "price": "90000", "size": "0.5", "side": "buy"}]}
        raise RuntimeError(f"Unexpected url: {url}")

    monkeypatch.setattr(mc, "get_json", mock_get_json_bitget)
    assert agg.poll("BTCUSDT") == 1

    def mock_get_json_fallback(url, params=None, retries=1):
        if mc.BITGET in url:
            raise RuntimeError("Bitget API temporary failure")
        if mc.BINANCE_FAPI in url:
            return [{"a": 3000000000, "T": now_ms + 1000, "p": "90100", "q": "0.2", "m": False}]
        raise RuntimeError(f"Unexpected url: {url}")

    monkeypatch.setattr(mc, "get_json", mock_get_json_fallback)
    assert agg.poll("BTCUSDT") == 1


def test_tick_aggregator_ingest_ws_trade_and_flush(tmp_path):
    db_file = tmp_path / "micro.db"
    con = sqlite3.connect(db_file)
    con.executescript(MICRO_SCHEMA)

    agg = TickAggregator()
    now_ms = int(time.time() * 1000)
    past_ms = now_ms - 120_000

    agg.ingest_ws_trade("ETHUSDT", past_ms, 3000.0, 2.0, False)
    agg.ingest_ws_trade("ETHUSDT", past_ms + 500, 3000.0, 1.0, True)

    flushed = agg.flush(con)
    assert flushed == 1

    row = con.execute(
        "SELECT buy_notional, sell_notional, n_trades, saturated FROM tick_agg WHERE symbol='ETHUSDT'"
    ).fetchone()
    con.close()

    assert row is not None
    assert row[0] == pytest.approx(6000.0)
    assert row[1] == pytest.approx(3000.0)
    assert row[2] == 2
    assert row[3] == 0


def test_binance_ws_collector_aggtrade_and_liveness():
    received = []
    stop_event = threading.Event()
    col = BinanceFuturesWSCollector(
        symbols=["BTCUSDT"],
        on_trade=lambda s, t, p, q, m: received.append((s, t, p, q, m)),
        stop_event=stop_event,
    )

    msg = {
        "stream": "btcusdt@aggTrade",
        "data": {"e": "aggTrade", "s": "BTCUSDT", "T": 1700000000000, "p": "90000.5", "q": "0.25", "m": True},
    }
    ret = col.handle_message(json.dumps(msg))
    assert ret == "aggTrade"
    assert received == [("BTCUSDT", 1700000000000, 90000.5, 0.25, True)]

    assert col.is_symbol_live("BTCUSDT", max_age_s=20.0) is True
    assert col.is_symbol_live("BTCUSDT", max_age_s=0.0) is False

    stop_event.set()
    assert col.is_symbol_live("BTCUSDT", max_age_s=20.0) is False


def test_binance_ws_collector_liquidations_and_symbol_updates():
    col = BinanceFuturesWSCollector(symbols=["BTCUSDT"], on_trade=lambda *_: None)

    msg_tracked = {
        "stream": "!forceOrder@arr",
        "data": {"e": "forceOrder", "E": 1700000001000, "o": {"s": "BTCUSDT", "S": "SELL", "ap": "89900", "z": "0.5", "T": 1700000001000}},
    }
    msg_untracked = {
        "stream": "!forceOrder@arr",
        "data": {"e": "forceOrder", "E": 1700000001000, "o": {"s": "DOGEUSDT", "S": "SELL", "ap": "0.1", "z": "100", "T": 1700000001000}},
    }
    assert col.handle_message(json.dumps(msg_tracked)) == "forceOrder"
    assert col.handle_message(json.dumps(msg_untracked)) is None

    d1 = col.drain_liquidations()
    assert d1 == [(1700000001000, "BTCUSDT", "SELL", 89900.0, 0.5, 44950.0)]
    assert col.drain_liquidations() == []

    assert col.update_symbols(["ETHUSDT", "SOLUSDT"]) is True
    url = col.build_stream_url()
    assert "ethusdt@aggTrade/solusdt@aggTrade/!forceOrder@arr" in url
    assert col.update_symbols(["ETHUSDT", "SOLUSDT"]) is False


def test_tape_reading_saturated_fallback_to_binance_series(tmp_path):
    db_file = tmp_path / "micro.db"
    con = sqlite3.connect(db_file)
    con.executescript(MICRO_SCHEMA)
    now_ms = 1700000000000

    con.execute(
        "INSERT INTO book_features (ts_ms, symbol, mid, spread_bps, microprice_dev_bps, bid_n10, ask_n10) "
        "VALUES (?, 'BTCUSDT', 90000.0, 1.0, 0.2, 150000.0, 100000.0)",
        (now_ms - 5000,),
    )
    for i in range(1, 5):
        m = (now_ms - i * 60_000) - (now_ms - i * 60_000) % 60_000
        con.execute(
            "INSERT INTO tick_agg (minute_ms, symbol, buy_qty, sell_qty, buy_notional, sell_notional, n_trades, vwap, saturated) "
            "VALUES (?, 'BTCUSDT', 1.0, 10.0, 10000.0, 100000.0, 40, 90000.0, 1)",
            (m,),
        )
    con.execute(
        "INSERT INTO binance_series (ts_ms, symbol, kind, v1, v2, v3) VALUES (?, 'BTCUSDT', 'taker', 0.0, 300.0, 100.0)",
        (now_ms - 120_000,),
    )
    con.commit()
    con.close()

    agent = TapeReadingAgent(db_path=str(db_file))
    res = agent.evaluate("BTCUSDT", "BUY", now_ms=now_ms)
    assert res["status"] == "OK"
    assert res["vote"] == "BUY"
    assert res["features"]["taker_source"] == "binance_series_5m"
    assert res["features"]["tick_saturated_ratio"] == pytest.approx(1.0)
    assert res["features"]["taker_imb_5m"] == pytest.approx(0.50)


def test_tape_reading_legacy_8col_schema_and_depth_fallback(tmp_path):
    # Case 1: Legacy 8-column tick_agg without saturated column
    db_leg = tmp_path / "micro_legacy.db"
    con = sqlite3.connect(db_leg)
    con.execute(
        "CREATE TABLE book_features (ts_ms INTEGER, symbol TEXT, mid REAL, spread_bps REAL, microprice_dev_bps REAL, bid_n10 REAL, ask_n10 REAL, PRIMARY KEY (ts_ms, symbol))"
    )
    con.execute(
        "CREATE TABLE tick_agg (minute_ms INTEGER, symbol TEXT, buy_qty REAL, sell_qty REAL, buy_notional REAL, sell_notional REAL, n_trades INTEGER, vwap REAL, PRIMARY KEY (minute_ms, symbol))"
    )
    now_ms = 1700000000000
    con.execute(
        "INSERT INTO book_features VALUES (?, 'BTCUSDT', 90000.0, 1.0, 0.2, 150000.0, 100000.0)",
        (now_ms - 5000,),
    )
    for i in range(1, 5):
        m = (now_ms - i * 60_000) - (now_ms - i * 60_000) % 60_000
        con.execute(
            "INSERT INTO tick_agg VALUES (?, 'BTCUSDT', 5.0, 1.0, 50000.0, 10000.0, 40, 90000.0)",
            (m,),
        )
    con.commit()
    con.close()

    res_leg = TapeReadingAgent(db_path=str(db_leg)).evaluate("BTCUSDT", "BUY", now_ms=now_ms)
    assert res_leg["status"] == "OK"
    assert res_leg["features"]["taker_source"] == "tick_agg"

    # Case 2: bid_n10=None, ask_n10=None fallback to bid_n5/ask_n5
    db_fb = tmp_path / "micro_fb.db"
    con = sqlite3.connect(db_fb)
    con.executescript(MICRO_SCHEMA)
    con.execute(
        "INSERT INTO book_features (ts_ms, symbol, mid, spread_bps, microprice_dev_bps, bid_sz1, ask_sz1, bid_n2, ask_n2, bid_n5, ask_n5, bid_n10, ask_n10) "
        "VALUES (?, 'BTCUSDT', 90000.0, 1.0, 0.2, 10.0, 10.0, 20000.0, 10000.0, 150000.0, 100000.0, NULL, NULL)",
        (now_ms - 5000,),
    )
    for i in range(1, 5):
        m = (now_ms - i * 60_000) - (now_ms - i * 60_000) % 60_000
        con.execute(
            "INSERT INTO tick_agg (minute_ms, symbol, buy_qty, sell_qty, buy_notional, sell_notional, n_trades, vwap, saturated) "
            "VALUES (?, 'BTCUSDT', 5.0, 1.0, 50000.0, 10000.0, 40, 90000.0, 0)",
            (m,),
        )
    con.commit()
    con.close()

    res_fb = TapeReadingAgent(db_path=str(db_fb)).evaluate("BTCUSDT", "BUY", now_ms=now_ms)
    assert res_fb["status"] == "OK"
    assert res_fb["features"]["book_imb_10"] == pytest.approx(0.20)


def test_directional_maker_first_and_fallback(tmp_path):
    eng = PaperTradingEngine(db_path=str(tmp_path / "paper.db"), restore=False)
    now_ms = int(time.time() * 1000)

    # 1. Default engine: opens DIRECTIONAL as taker
    sig_default = TradeSignal(
        signal_id="sig_def", timestamp=now_ms, symbol="BTCUSDT", direction="BUY",
        entry_price=50000.0, stop_loss=49000.0, take_profit=52000.0, atr=500.0,
        confidence=0.80, consensus_score=0.5, position_size=1000.0, risk_dollars=100.0,
        rationale="base taker test", metrics={},
    )
    pos_def = eng.execute_directional_signal(sig_default, 50000.0, now_ms)
    assert pos_def is not None
    assert pos_def.fees_paid == pytest.approx(pos_def.notional * eng.taker_fee)
    assert pos_def.entry_price == pytest.approx(50000.0 * (1.0 + MARKET_SLIPPAGE_PCT))

    # 2. maker_entry=True with tight spread
    sig_maker = TradeSignal(
        signal_id="sig_mk", timestamp=now_ms + 1000, symbol="ETHUSDT", direction="BUY",
        entry_price=3000.0, stop_loss=2900.0, take_profit=3200.0, atr=50.0,
        confidence=0.80, consensus_score=0.5, position_size=1000.0, risk_dollars=100.0,
        rationale="maker post-only test", metrics={"spread_bps": 1.5},
    )
    pos_mk = eng.execute_directional_signal(sig_maker, 3000.0, now_ms + 1000, maker_entry=True)
    assert pos_mk is not None
    assert pos_mk.fees_paid == pytest.approx(pos_mk.notional * eng.maker_fee)
    assert pos_mk.entry_price == pytest.approx(3000.0 * (1.0 + MAKER_SLIPPAGE_PCT))
    assert "MAKER_POST_ONLY" in eng.trade_history[-1].rationale

    # 3. maker_entry=True with wide spread fallback to taker
    sig_wide = TradeSignal(
        signal_id="sig_wd", timestamp=now_ms + 2000, symbol="SOLUSDT", direction="BUY",
        entry_price=150.0, stop_loss=140.0, take_profit=170.0, atr=5.0,
        confidence=0.80, consensus_score=0.5, position_size=1000.0, risk_dollars=100.0,
        rationale="wide spread fallback test", metrics={"spread_bps": 8.0},
    )
    pos_wd = eng.execute_directional_signal(sig_wide, 150.0, now_ms + 2000, maker_entry=True)
    assert pos_wd is not None
    assert pos_wd.fees_paid == pytest.approx(pos_wd.notional * eng.taker_fee)
    assert pos_wd.entry_price == pytest.approx(150.0 * (1.0 + MARKET_SLIPPAGE_PCT))
    assert "TAKER_FALLBACK_WIDE_SPREAD" in eng.trade_history[-1].rationale


def test_paper_trading_daemon_parity_check(tmp_path):
    eng = PaperTradingEngine(db_path=str(tmp_path / "paper_daemon.db"), restore=False)

    class FakeLoader:
        pass

    class LiveDataLoader:
        pass

    d_fake = PaperTradingDaemon(engine=eng, loader=FakeLoader())
    assert d_fake.directional_maker_entry is False

    d_live = PaperTradingDaemon(engine=eng, loader=LiveDataLoader())
    assert d_live.directional_maker_entry is True
