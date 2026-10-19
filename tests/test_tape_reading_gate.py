"""Testes do TapeReadingAgent + gate de fluxo + flow_gate_log (ADDENDUM E8). DBs temporários; nunca toca bancos reais."""
import asyncio
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import tape_reading_agent as tra  # noqa: E402
from agents.tape_reading_agent import TapeReadingAgent  # noqa: E402
from scripts.run_paper_simulation import PaperTradingDaemon  # noqa: E402
from signal_generator import TradeSignal  # noqa: E402
from smart_trade import PaperTradingEngine  # noqa: E402

NOW = 1_800_000_000_000
SCHEMA = """
CREATE TABLE book_features (ts_ms INTEGER, symbol TEXT, mid REAL, spread_bps REAL, microprice_dev_bps REAL,
  bid_sz1 REAL, ask_sz1 REAL, bid_n2 REAL, ask_n2 REAL, bid_n5 REAL, ask_n5 REAL, bid_n10 REAL, ask_n10 REAL,
  bid_n25 REAL, ask_n25 REAL, bid_n50 REAL, ask_n50 REAL, bid_n100 REAL, ask_n100 REAL, PRIMARY KEY (ts_ms, symbol));
CREATE TABLE tick_agg (minute_ms INTEGER, symbol TEXT, buy_qty REAL, sell_qty REAL, buy_notional REAL,
  sell_notional REAL, n_trades INTEGER, vwap REAL, saturated INTEGER, PRIMARY KEY (minute_ms, symbol));
"""


def make_db(path, sym="BTCUSDT", taker=0.2, book=0.2, dev=1.0, spread=1.0, book_age_s=10, tick_age_s=30,
            minutes=5, trades=20, future=False):
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    # taker imbalance: buy=(1+t)*500, sell=(1-t)*500 por minuto
    last_min = ((NOW - tick_age_s * 1000) // 60_000) * 60_000
    for k in range(minutes):
        con.execute("INSERT INTO tick_agg VALUES (?,?,?,?,?,?,?,?,0)",
                    (last_min - k * 60_000, sym, 1, 1, (1 + taker) * 500, (1 - taker) * 500, trades, 100.0))
    for k in range(6):
        ts = NOW - book_age_s * 1000 - k * 5000
        bid, ask = 100 * (1 + book), 100 * (1 - book)
        con.execute("INSERT INTO book_features (ts_ms,symbol,mid,spread_bps,microprice_dev_bps,bid_n10,ask_n10) "
                    "VALUES (?,?,?,?,?,?,?)", (ts, sym, 100.0, spread, dev, bid, ask))
    if future:  # linhas FUTURAS que dariam voto oposto: não podem ser usadas
        con.execute("INSERT INTO book_features (ts_ms,symbol,spread_bps,microprice_dev_bps,bid_n10,ask_n10) "
                    "VALUES (?,?,?,?,?,?)", (NOW + 1000, sym, 1.0, -5.0, 1.0, 1000.0))
        con.execute("INSERT INTO tick_agg VALUES (?,?,1,1,1,1000,500,1,0)", (NOW + 60_000, sym))
    con.commit()
    con.close()
    return path


def ev(tmp_path, sym="BTCUSDT", **kw):
    path = tmp_path / "micro.db"
    if path.exists():
        path.unlink()
    db = make_db(str(path), sym=sym, **kw)
    return TapeReadingAgent(db_path=db).evaluate(sym, "BUY", now_ms=NOW)


# ---------------- agente ----------------
def test_vote_buy(tmp_path):
    r = ev(tmp_path)
    assert (r["vote"], r["status"], r["reason"]) == ("BUY", "OK", "flow_buy_confirmed")
    assert r["confidence"] > 0 and abs(r["features"]["taker_imb_5m"] - 0.2) < 1e-9


def test_vote_sell(tmp_path):
    r = ev(tmp_path, taker=-0.2, book=-0.2, dev=-1.0)
    assert (r["vote"], r["status"]) == ("SELL", "OK")


def test_boundaries_inclusive(tmp_path):
    # exatamente nos limiares (book_imb: (1+b)-(1-b) / 2 = b)
    assert ev(tmp_path, taker=0.15, book=0.10, dev=0.0)["vote"] == "BUY"


def test_hold_legacy_boundaries(tmp_path):
    path = tmp_path / "legacy.db"
    db = make_db(str(path), sym="BTCUSDT", taker=0.1499, book=0.2, dev=1.0)
    # No modo legacy booleano rígido (use_continuous_flow=False), 0.1499 < 0.15 é barrado
    r = TapeReadingAgent(db_path=db, use_continuous_flow=False).evaluate("BTCUSDT", "BUY", now_ms=NOW)
    assert (r["vote"], r["status"], r["reason"]) == ("HOLD", "OK", "no_flow_confluence")


@pytest.mark.parametrize("kw", [
    dict(taker=0.10, book=0.10, dev=0.0),    # 0.6*0.10 + 0.3*0.10 = 0.09 < 0.12
    dict(taker=0.14, book=0.08, dev=0.0),    # 0.6*0.14 + 0.3*0.08 = 0.108 < 0.12
    dict(taker=0.0, book=0.0, dev=0.0),       # 0.0 < 0.12
])
def test_hold_just_outside_continuous(tmp_path, kw):
    r = ev(tmp_path, **kw)
    assert (r["vote"], r["status"], r["reason"]) == ("HOLD", "OK", "no_flow_confluence")


def test_bullish_absorption(tmp_path):
    # Agressão vendedora extrema (-0.35) absorvida por barreira no book (+0.25)
    r = ev(tmp_path, taker=-0.35, book=0.25, dev=0.0)
    assert (r["vote"], r["status"], r["reason"]) == ("BUY", "OK", "absorption_buy_confirmed")
    assert r["confidence"] > 0.5


def test_bearish_absorption(tmp_path):
    # Agressão compradora extrema (+0.35) absorvida por barreira no book (-0.25)
    r = ev(tmp_path, taker=0.35, book=-0.25, dev=0.0)
    assert (r["vote"], r["status"], r["reason"]) == ("SELL", "OK", "absorption_sell_confirmed")
    assert r["confidence"] > 0.5


def test_continuous_flow_cross_threshold(tmp_path):
    # Taker moderado (+0.10) compensado por Book forte (+0.20): 0.6*0.1 + 0.3*0.2 = 0.12
    r = ev(tmp_path, taker=0.10, book=0.20, dev=0.0)
    assert (r["vote"], r["status"], r["reason"]) == ("BUY", "OK", "flow_buy_confirmed")


def test_sell_boundary_and_hold(tmp_path):
    assert ev(tmp_path, taker=-0.15, book=-0.10, dev=0.0)["vote"] == "SELL"


def test_spread_wide_is_hold(tmp_path):
    r = ev(tmp_path, spread=5.01)
    assert (r["vote"], r["reason"], r["status"]) == ("HOLD", "spread_wide", "OK")
    assert ev(tmp_path, spread=5.0)["vote"] == "BUY"


def test_stale_book_unavailable(tmp_path):
    r = ev(tmp_path, book_age_s=121)
    assert (r["status"], r["reason"]) == ("UNAVAILABLE", "book_stale")
    assert ev(tmp_path, book_age_s=120)["status"] == "OK"


def test_stale_tick_unavailable(tmp_path):
    # bucket de minuto mais recente com > 180 s
    r = ev(tmp_path, tick_age_s=200)
    assert (r["status"], r["reason"]) == ("UNAVAILABLE", "tick_stale")


def test_insufficient_ticks_unavailable(tmp_path):
    assert ev(tmp_path, minutes=2)["reason"] == "tick_insufficient"
    assert ev(tmp_path, trades=5, minutes=3)["reason"] == "tick_insufficient"
    assert ev(tmp_path, minutes=3, trades=10)["status"] == "OK"


def test_symbol_outside_coverage_unavailable(tmp_path):
    r = ev(tmp_path, sym="DOGEUSDT")
    assert (r["status"], r["reason"]) == ("UNAVAILABLE", "symbol_not_covered")


def test_missing_db_unavailable(tmp_path):
    r = TapeReadingAgent(db_path=str(tmp_path / "nope.db")).evaluate("BTCUSDT", "BUY", now_ms=NOW)
    assert r["status"] == "UNAVAILABLE" and r["reason"].startswith("db_unavailable")
    assert not os.path.exists(str(tmp_path / "nope.db"))  # mode=ro não cria arquivo


def test_no_lookahead(tmp_path):
    r = ev(tmp_path, future=True)
    assert r["vote"] == "BUY" and r["status"] == "OK"
    assert r["features"]["microprice_dev"] == 1.0


def test_read_only(tmp_path):
    db = make_db(str(tmp_path / "micro.db"))
    ag = TapeReadingAgent(db_path=db)
    con = ag._connect_ro()
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM book_features")
    con.close()


# ---------------- gate + flow_gate_log ----------------
class FixedTape:
    def __init__(self, vote, status="OK", reason="x"):
        self.res = {"vote": vote, "status": status, "reason": reason, "features": {
            "taker_imb_5m": 0.3, "book_imb_10": 0.2, "microprice_dev": 0.5, "spread_bps": 1.0}}

    def evaluate(self, symbol, direction, now_ms=None):
        return dict(self.res)


class Boom:
    def evaluate(self, *a, **k):
        raise RuntimeError("x")


def daemon(tmp_path, tape):
    eng = PaperTradingEngine(db_path=str(tmp_path / "m.db"), restore=False)
    return PaperTradingDaemon(engine=eng, loader=object(), golden_pot_agent=object(), tape_agent=tape)


def rows(d):
    con = sqlite3.connect(d.engine.db_path)
    con.row_factory = sqlite3.Row
    out = [dict(r) for r in con.execute("SELECT * FROM flow_gate_log ORDER BY id")]
    con.close()
    return out


@pytest.mark.parametrize("vote,status,direction,expect,decision", [
    ("BUY", "OK", "BUY", True, "PASSED"),
    ("SELL", "OK", "SELL", True, "PASSED"),
    ("SELL", "OK", "BUY", False, "BLOCKED"),     # voto oposto
    ("HOLD", "OK", "BUY", False, "BLOCKED"),     # HOLD
    ("HOLD", "OK", "SELL", False, "BLOCKED"),
    ("HOLD", "UNAVAILABLE", "BUY", True, "UNAVAILABLE_PASSED"),   # fail-open
])
def test_gate_decisions_and_log(tmp_path, vote, status, direction, expect, decision):
    d = daemon(tmp_path, FixedTape(vote, status))
    s = TradeSignal.__new__(TradeSignal)
    s.__dict__.update(direction=direction, confidence=0.8, consensus_score=0.5, stop_loss=99.0)
    ok = d._flow_gate("SCALP", "BTCUSDT", direction, s, 3.0, 100.0, 1.0)
    assert ok is expect
    r = rows(d)
    assert len(r) == 1 and r[0]["decision"] == decision
    assert r[0]["strategy_type"] == "SCALP" and r[0]["leverage"] == 3.0 and r[0]["atr"] == 1.0
    assert r[0]["entry_price"] == 100.0 and r[0]["stop_loss"] == 99.0
    assert (r[0]["tp1"] > 100.0 and r[0]["tp3"] > r[0]["tp1"]) if direction == "BUY" else (r[0]["tp1"] < 100.0)
    assert r[0]["confidence"] == 0.8 and r[0]["consensus_score"] == 0.5 and r[0]["taker_imb_5m"] == 0.3
    assert d.last_tape["vote"] == vote
    assert d.engine.flow_gate_counts()[decision] == 1
    assert decision in d.flow_gate_telemetry() or "Gate P/B/U" in d.flow_gate_telemetry()


def test_gate_sell_targets_below_and_agent_error_failopen(tmp_path):
    d = daemon(tmp_path, Boom())
    s = TradeSignal.__new__(TradeSignal)
    s.__dict__.update(direction="SELL", confidence=0.7, consensus_score=-0.4, stop_loss=101.0)
    assert d._flow_gate("GOLDEN_POT", "PEPEUSDT", "SELL", s, 2.0, 100.0, 1.0) is True
    r = rows(d)[0]
    assert r["decision"] == "UNAVAILABLE_PASSED" and r["tp1"] < 100.0 and r["tp3"] < r["tp1"]


def test_every_signal_logged(tmp_path):
    d = daemon(tmp_path, FixedTape("BUY"))
    s = TradeSignal.__new__(TradeSignal)
    s.__dict__.update(direction="BUY", confidence=0.8, consensus_score=0.5, stop_loss=99.0)
    for strat in ("SCALP", "GOLDEN_POT", "DIRECTIONAL"):
        d._flow_gate(strat, "ETHUSDT", "BUY", s, 2.0, 100.0, 1.0)
    d._flow_gate("SCALP", "ETHUSDT", "SELL", s, 2.0, 100.0, 1.0)
    assert [r["strategy_type"] for r in rows(d)] == ["SCALP", "GOLDEN_POT", "DIRECTIONAL", "SCALP"]
    assert d.engine.flow_gate_counts() == {"PASSED": 3, "BLOCKED": 1, "UNAVAILABLE_PASSED": 0}
