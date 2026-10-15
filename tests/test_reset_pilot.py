"""Testes do scripts/reset_pilot.py com DB temporário (nunca toca o banco real)."""
import json
import os
import sqlite3
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import reset_pilot as rp  # noqa: E402
from scripts import pilot_report as pr  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    from agents.golden_pot_agent import GoldenPotAgent
    from agents.regime_classifier import MarketRegimeClassifier
    from learning_tracker import LearningTracker
    from performance_tracker import PerformanceTracker
    from rollback_guard import RollbackGuard
    from smart_trade import PaperTradingEngine

    db = str(tmp_path / "m.db")
    PaperTradingEngine(db_path=db, restore=False)
    RollbackGuard(db_path=db)
    LearningTracker(db_path=db)
    PerformanceTracker(db_path=db)
    MarketRegimeClassifier(db_path=db)
    GoldenPotAgent(db_path=db, binance_connector=MagicMock(), bitget_connector=MagicMock())._init_database_table()

    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE candles (symbol TEXT, timeframe TEXT, timestamp INTEGER, close REAL);
        CREATE TABLE funding_rates (symbol TEXT, timestamp INTEGER, rate REAL);
        CREATE TABLE macro_series (name TEXT, ts INTEGER, value REAL);
    """)
    con.executemany("INSERT INTO candles VALUES ('BTCUSDT','1h',?,1.0)", [(i,) for i in range(50)])
    con.executemany("INSERT INTO funding_rates VALUES ('BTCUSDT',?,0.0001)", [(i,) for i in range(7)])
    con.executemany("INSERT INTO macro_series VALUES ('DXY',?,100.0)", [(i,) for i in range(5)])
    for i in range(5):
        con.execute("INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (f"t{i}", f"p{i}", "BTC", "OPEN_SCALP_LONG", 100.0, 1.0, 0.06, 0.0, 1_700_000_000_000 + i, "x"))
        con.execute("INSERT INTO paper_positions (position_id,symbol,side,entry_price,current_price,quantity,"
                    "notional,status) VALUES (?,?,?,?,?,?,?,?)", (f"p{i}", "BTC", "LONG", 1, 1, 1, 1, "OPEN"))
        con.execute("INSERT INTO paper_portfolio VALUES (?,?,?,?,?,?,?,?)",
                    (1_700_000_000_000 + i, 9000.0, 9000.0, 0, 0, 0, 0, 1))
        con.execute("INSERT INTO circuit_breaker_events (timestamp,state,event_type,reason,drawdown_pct,"
                    "consecutive_losses) VALUES (?,?,?,?,?,?)", (1_700_000_000_000 + i, "CLEAR", "MANUAL_RESET", "r", 0.0, 0))
        con.execute("INSERT INTO golden_pots (symbol,setup_type,direction,score,last_price,dist_vwap_pct,"
                    "volume_24h_usd,price_change_pct_24h,range_position_pct,amplitude_pct,suggested_timeframe,"
                    "rationale,timestamp) VALUES ('BTC','a','LONG',1,1,1,1,1,1,1,'1h','r',?)", (i,))
        con.execute("INSERT INTO performance_snapshots (timestamp,equity,balance,net_pnl,net_pnl_pct,"
                    "max_drawdown_pct,win_rate,profit_factor,total_trades,open_positions,total_fees) "
                    "VALUES (?,1,1,1,1,1,1,1,1,1,1)", (i,))
        con.execute("INSERT INTO market_regime VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (i, "BULL_TREND", "BULLISH", 0.5, "a", "r", 1, 1, 1, 1, 50))
    con.execute("INSERT INTO circuit_breaker_state VALUES (1,'COOLDOWN',0,'perdas',9999999999999,1,1)")
    con.execute("INSERT INTO agent_learning_memory (decision_id,timestamp,symbol,regime,consensus_direction,"
                "consensus_confidence,agent_signals_json,position_id) VALUES ('d1',1,'BTC','BULL','LONG',0.5,'{}','p0')")
    con.execute("INSERT INTO signal_funnel VALUES (3600000, 'SCALP', 'evaluated', 9)")
    con.executemany("INSERT INTO flow_gate_log (ts_ms,symbol,decision) VALUES (?, 'BTCUSDT', 'BLOCKED')", [(i,) for i in range(4)])
    con.execute("INSERT INTO agent_regime_weights VALUES ('BULL','agentX',1.2,0.6,10,1)")
    for k, v in (("ledger_epoch_trade_rowid", "3"), ("ledger_epoch_pos_rowid", "3"),
                 ("quota_epoch_trade_rowid", "3"), ("MANUAL_RESET", "1"), ("reset_daily_quotas", "1")):
        con.execute("INSERT OR REPLACE INTO simulation_settings (key,value) VALUES (?,?)", (k, v))
    con.execute("UPDATE simulation_settings SET value='CUSTOM' WHERE key='operation_mode'")
    con.execute("UPDATE simulation_settings SET value='77' WHERE key='paper_lot_usd'")
    con.commit()
    con.close()

    reports = tmp_path / "reports"
    reports.mkdir()
    trial = reports / "trial_log.csv"
    trial.write_text("a,b\n1,2\n")
    pj = reports / "pilot_report.json"
    pj.write_text("{}")
    logs = tmp_path / "logs"
    logs.mkdir()
    fl = logs / "flask.log"
    fl.write_text("linha\n")
    monkeypatch.setattr(rp, "detect_daemons", lambda: [])
    return dict(db=db, backups=str(tmp_path / "backups"), flask_log=str(fl), pilot_report=str(pj),
                trial=trial, tmp=tmp_path)


def _run(env, **kw):
    return rp.run_reset(env["db"], env["backups"], env["flask_log"], env["pilot_report"], **kw)


def _count(db, t):
    with sqlite3.connect(db) as c:
        return c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]


def test_tables_cleaned_and_untouched_intact(env):
    r = _run(env)
    for t in rp.PILOT_TABLES + ["market_regime"]:
        assert _count(env["db"], t) == 0, t
    assert r["tables_before"]["paper_trades"] == 5
    assert r["tables_before"]["flow_gate_log"] == 4
    assert _count(env["db"], "candles") == 50
    assert _count(env["db"], "funding_rates") == 7
    assert _count(env["db"], "macro_series") == 5
    assert r["untouched_tables"]["before"] == r["untouched_tables"]["after"]
    assert os.path.exists(r["report_path"])
    assert json.load(open(r["report_path"]))["reset_id"] == r["reset_id"]


def test_learning_memory_preserved(env):
    r = _run(env)
    assert _count(env["db"], "agent_learning_memory") == 1
    assert _count(env["db"], "agent_regime_weights") == 1
    assert r["preserved_memory_tables"]["counts"] == {"agent_learning_memory": 1, "agent_regime_weights": 1}
    assert "decisão do usuário" in r["preserved_memory_tables"]["note"]


def test_keep_market_regime_option(env):
    _run(env, keep_market_regime=True)
    assert _count(env["db"], "market_regime") == 5


def test_trial_log_intact_and_files_archived(env):
    r = _run(env)
    assert env["trial"].read_text() == "a,b\n1,2\n"
    assert open(env["flask_log"]).read() == ""
    assert not os.path.exists(env["pilot_report"])
    arch = r["archived_files"]
    assert open(arch[env["flask_log"]]).read() == "linha\n"
    assert open(arch[env["pilot_report"]]).read() == "{}"


def test_backup_integrity_and_content(env):
    r = _run(env)
    b = sqlite3.connect(r["backup"])
    assert b.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert b.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0] == 5
    assert b.execute("SELECT COUNT(*) FROM market_regime").fetchone()[0] == 5
    b.close()


def test_refuses_with_daemon_running(env, monkeypatch):
    monkeypatch.setattr(rp, "detect_daemons", lambda: ["123 python scripts/run_paper_simulation.py 0 15"])
    with pytest.raises(rp.ResetRefused):
        _run(env)
    assert _count(env["db"], "paper_trades") == 5          # nada alterado
    assert not os.path.exists(env["backups"])
    r = _run(env, force=True)                              # --force ignora
    assert r["tables_after"]["paper_trades"] == 0


def test_detect_daemons_uses_pgrep(monkeypatch):
    monkeypatch.undo()
    out = MagicMock(stdout="999999 /x/python /x/scripts/run_paper_simulation.py 0 15\n")
    monkeypatch.setattr(rp.subprocess, "run", lambda *a, **k: out)
    assert any("run_paper_simulation.py" in ln for ln in rp.detect_daemons())


def test_sequences_reset_and_markers_removed(env):
    con = sqlite3.connect(env["db"])
    assert con.execute("SELECT seq FROM sqlite_sequence WHERE name='golden_pots'").fetchone()[0] == 5
    con.close()
    _run(env)
    con = sqlite3.connect(env["db"])
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_sequence")}
    assert not names & {"golden_pots", "circuit_breaker_events", "performance_snapshots"}
    con.execute("INSERT INTO circuit_breaker_events (timestamp,state,event_type,reason,drawdown_pct,consecutive_losses) "
                "VALUES (1,'CLEAR','X','r',0,0)")
    assert con.execute("SELECT id FROM circuit_breaker_events").fetchone()[0] == 1
    st = dict(con.execute("SELECT key, value FROM simulation_settings"))
    con.close()
    for k in rp.SETTINGS_TO_REMOVE:
        assert k not in st
    assert st["operation_mode"] == "CUSTOM" and st["paper_lot_usd"] == "77"
    assert st["pilot_start_utc"] and st["pilot_reset_id"].startswith("reset_")


def test_settings_snapshot_in_report(env):
    r = _run(env)
    assert r["settings_snapshot_pre_reset"]["MANUAL_RESET"] == "1"
    assert r["preserved_settings"]["operation_mode"] == "CUSTOM"
    assert set(r["removed_settings"]) == set(rp.SETTINGS_TO_REMOVE)


def test_engine_restore_after_reset_is_clean(env):
    r = _run(env)
    v = r["engine_verification"]
    assert v["ok"], v
    assert v["balance"] == v["initial_capital"] == 10000.0
    assert v["open_positions"] == 0 and v["daily_scalps_count"] == 0 and v["daily_golden_pots_count"] == 0
    from rollback_guard import RollbackGuard
    g = RollbackGuard(db_path=env["db"])
    assert g.status.state == "CLEAR" and g.status.is_trading_allowed


def test_pilot_report_uses_pilot_start_and_inconclusive(env, monkeypatch, capsys):
    _run(env)
    monkeypatch.setattr(pr, "DB", env["db"])
    monkeypatch.setattr(pr, "SUP_STATE", str(env["tmp"] / "none.json"))
    monkeypatch.setattr(pr, "SUP_LOG", str(env["tmp"] / "none.log"))
    monkeypatch.setattr(pr, "OUT_JSON", str(env["tmp"] / "out.json"))
    monkeypatch.setattr(sys, "argv", ["pilot_report", "--ref", str(env["tmp"] / "x.json"), "--json"])
    assert pr.main() == 2
    d = pr.load()
    ms, rid = pr.pilot_start(d)
    assert ms is not None and rid.startswith("reset_")
    assert {"pilot_start_utc", "pilot_reset_id"} <= pr.EPOCH_KEYS
