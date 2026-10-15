"""Testes do pilot_report com DB temporário (janela pós-reset, N insuficiente -> INCONCLUSIVO)."""
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import pilot_report as pr  # noqa: E402

SCHEMA = """
CREATE TABLE paper_trades (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp INTEGER, symbol TEXT,
  action TEXT, price REAL, quantity REAL, fee REAL, pnl REAL, position_id TEXT);
CREATE TABLE paper_positions (position_id TEXT, status TEXT, realized_pnl REAL, strategy_type TEXT);
CREATE TABLE paper_portfolio (timestamp INTEGER, equity REAL, unrealized_pnl REAL);
CREATE TABLE circuit_breaker_events (timestamp INTEGER, event_type TEXT);
CREATE TABLE simulation_settings (key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
"""


def _ms(dt):
    return int(dt.timestamp() * 1000)


def _add_closed(con, pid, t_open, t_close, pnl):
    con.execute("INSERT INTO paper_trades (timestamp,symbol,action,price,quantity,fee,pnl,position_id) "
                "VALUES (?,?,?,?,?,?,?,?)", (_ms(t_open), "BTC", "OPEN_LONG", 100.0, 1.0, 0.06, 0.0, pid))
    con.execute("INSERT INTO paper_trades (timestamp,symbol,action,price,quantity,fee,pnl,position_id) "
                "VALUES (?,?,?,?,?,?,?,?)", (_ms(t_close), "BTC", "CLOSE_LONG", 100.0, 1.0, 0.06, pnl, pid))
    con.execute("INSERT INTO paper_positions VALUES (?,?,?,?)", (pid, "CLOSED", pnl, "TSMOM"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = tmp_path / "m.db"
    monkeypatch.setattr(pr, "DB", str(db))
    monkeypatch.setattr(pr, "SUP_STATE", str(tmp_path / "none.json"))
    monkeypatch.setattr(pr, "SUP_LOG", str(tmp_path / "none.log"))
    monkeypatch.setattr(pr, "OUT_JSON", str(tmp_path / "out.json"))
    monkeypatch.setattr(sys, "argv", ["pilot_report", "--ref", str(tmp_path / "noref.json"), "--json"])
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    return con, tmp_path


def _setup(con, post_trades):
    now = datetime.now(timezone.utc)
    reset = now - timedelta(hours=2)
    # histórico pré-reset: grande prejuízo (viesado) que dispararia R1-R3 se contasse
    for i in range(40):
        _add_closed(con, f"pre{i}", reset - timedelta(days=3, minutes=i * 10 + 5),
                    reset - timedelta(days=3, minutes=i * 10), -50.0)
    con.execute("INSERT INTO simulation_settings VALUES ('ledger_epoch_trade_rowid', '80', ?)",
                (reset.strftime("%Y-%m-%d %H:%M:%S"),))
    con.execute("INSERT INTO simulation_settings VALUES ('MANUAL_RESET', '0', ?)",
                (reset.strftime("%Y-%m-%d %H:%M:%S"),))
    con.execute("INSERT INTO circuit_breaker_events VALUES (?, 'MANUAL_RESET')", (_ms(reset) + 1000,))
    for i in range(post_trades):
        _add_closed(con, f"post{i}", reset + timedelta(minutes=10 + i), reset + timedelta(minutes=12 + i), 1.0)
    con.commit()
    return reset


def test_zero_post_reset_trades_is_inconclusive(env, capsys):
    con, _ = env
    _setup(con, 0)
    con.close()
    assert pr.main() == 2
    out = json.loads(capsys.readouterr().out)
    assert out["closed_trades"] == 0
    assert out["verdict"] == "INCONCLUSIVO"
    assert out["refutations"] == []
    assert out["pre_reset_history"]["closed_trades"] == 40
    assert out["manual_resets_in_window"] == 0
    assert out["clean_days"] < 1


def test_only_post_reset_trades_counted(env, capsys):
    con, _ = env
    _setup(con, 5)
    con.close()
    assert pr.main() == 2  # N=5 < 30: sem refutação precoce
    out = json.loads(capsys.readouterr().out)
    assert out["closed_trades"] == 5
    assert out["pre_reset_history"]["closed_trades"] == 40
    assert out["pre_reset_history"]["net_pnl_stress_usd"] < 0
    assert out["net_pnl_stress_usd"] > -1  # só os 5 pós-reset
    assert out["refutations"] == []
    assert out["reset_epoch_utc"] is not None


def test_reset_epoch_helper_none_without_marker():
    import pandas as pd
    d = {"set": pd.DataFrame({"key": [], "value": [], "updated_at": []})}
    assert pr.reset_epoch(d) == (None, 0)
