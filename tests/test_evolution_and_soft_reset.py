"""Independent QA tests for patch milestones, multi-reset evolution stitching, and soft-epoch reset."""

import datetime
import os
import sqlite3
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.golden_pot_agent import GoldenPotAgent
from agents.regime_classifier import MarketRegimeClassifier
from learning_tracker import LearningTracker
from performance_tracker import PerformanceTracker
from rollback_guard import RollbackGuard
from scripts import reset_pilot as rp
from smart_trade import PaperTradingEngine


def _utc_ms(year: int, month: int, day: int, hour: int = 15, minute: int = 0) -> int:
    dt = datetime.datetime(year, month, day, hour, minute, tzinfo=datetime.timezone.utc)
    return int(dt.timestamp() * 1000)


class TestPatchMilestones:
    """Contract 1: PATCH_MILESTONES contiguity and _resolve_patch_for_date boundaries."""

    def test_milestones_are_contiguous_from_2026_10_15_without_gaps_or_overlaps(self):
        milestones = PerformanceTracker.PATCH_MILESTONES
        assert len(milestones) >= 2
        assert milestones[0]["id"] == "S0"
        assert milestones[0]["start_date"] == "2026-10-15"
        assert milestones[-1]["id"] == "S5_PATCH"
        assert milestones[-1]["end_date"] is None

        for idx, m in enumerate(milestones):
            start = datetime.date.fromisoformat(m["start_date"])
            if idx < len(milestones) - 1:
                assert m["end_date"] is not None, f"{m['id']} must have end_date"
                end = datetime.date.fromisoformat(m["end_date"])
                assert start <= end, f"{m['id']} has start > end"
                next_start = datetime.date.fromisoformat(milestones[idx + 1]["start_date"])
                assert end + datetime.timedelta(days=1) == next_start, (
                    f"Gap or overlap between {m['id']} ({end}) and {milestones[idx + 1]['id']} ({next_start})"
                )

    @pytest.mark.parametrize(
        "target_date,expected_id",
        [
            (datetime.date(2026, 11, 14), "S3_PATCH"),
            (datetime.date(2026, 11, 15), "S4_2_PATCH"),
            (datetime.date(2026, 11, 19), "S4_2_PATCH"),
            (datetime.date(2026, 11, 20), "S5_PATCH"),
            (datetime.date(2026, 11, 25), "S5_PATCH"),
        ],
    )
    def test_resolve_patch_for_date_boundaries(self, target_date, expected_id):
        resolved = PerformanceTracker._resolve_patch_for_date(target_date)
        assert resolved["id"] == expected_id


class TestComputeEvolutionAcrossResets:
    """Contract 2: Continuous equity stitching and patch breakdown across two $10k resets."""

    def test_two_resets_stitch_equity_and_group_by_patch(self, tmp_path):
        db_path = str(tmp_path / "evo.db")
        PaperTradingEngine(db_path=db_path, restore=False)
        LearningTracker(db_path=db_path)
        tracker = PerformanceTracker(db_path=db_path, initial_capital=10000.0)

        t_s3_start = _utc_ms(2026, 11, 10, 15, 0)
        t_s3_win = _utc_ms(2026, 11, 10, 15, 30)
        t_reset1 = _utc_ms(2026, 11, 12, 15, 0)
        t_s42_win = _utc_ms(2026, 11, 16, 15, 0)
        t_reset2 = _utc_ms(2026, 11, 20, 15, 0)
        t_s5_win = _utc_ms(2026, 11, 20, 16, 0)

        con = sqlite3.connect(db_path)
        try:
            portfolio_rows = [
                (t_s3_start, 10000.0, 10000.0, 0.0, 0.0, 0.0, 0.0, 0),
                (t_s3_win, 11200.0, 11200.0, 0.0, 0.0, 1200.0, 0.0, 0),
                (t_reset1, 10000.0, 10000.0, 0.0, 0.0, 0.0, 0.0, 0),
                (t_s42_win, 10800.0, 10800.0, 0.0, 0.0, 800.0, 0.0, 0),
                (t_reset2, 10000.0, 10000.0, 0.0, 0.0, 0.0, 0.0, 0),
                (t_s5_win, 10150.0, 10150.0, 0.0, 0.0, 150.0, 0.0, 0),
            ]
            con.executemany(
                "INSERT INTO paper_portfolio VALUES (?,?,?,?,?,?,?,?)",
                portfolio_rows,
            )

            positions = [
                ("pos_s3", "BTCUSDT", "LONG", 80000.0, 82000.0, 0.6, 48000.0, "SCALP", "CLOSED"),
                ("pos_s42", "ETHUSDT", "LONG", 3000.0, 3160.0, 5.0, 15000.0, "SCALP", "CLOSED"),
                ("pos_s5", "SOLUSDT", "LONG", 200.0, 215.0, 10.0, 2000.0, "DIRECTIONAL", "CLOSED"),
            ]
            con.executemany(
                "INSERT INTO paper_positions (position_id,symbol,side,entry_price,current_price,"
                "quantity,notional,strategy_type,status) VALUES (?,?,?,?,?,?,?,?,?)",
                positions,
            )

            trades = [
                ("tr_s3_open", "pos_s3", "BTCUSDT", "OPEN_SCALP_LONG", 80000.0, 0.6, 0.0, 0.0, t_s3_start, "open"),
                ("tr_s3_close", "pos_s3", "BTCUSDT", "CLOSE_TAKE_PROFIT", 82000.0, 0.6, 0.0, 1200.0, t_s3_win, "tp"),
                ("tr_s42_open", "pos_s42", "ETHUSDT", "OPEN_SCALP_LONG", 3000.0, 5.0, 0.0, 0.0, t_reset1 + 60000, "open"),
                ("tr_s42_close", "pos_s42", "ETHUSDT", "CLOSE_TAKE_PROFIT", 3160.0, 5.0, 0.0, 800.0, t_s42_win, "tp"),
                ("tr_s5_open", "pos_s5", "SOLUSDT", "OPEN_DIRECTIONAL_LONG", 200.0, 10.0, 0.0, 0.0, t_reset2 + 60000, "open"),
                ("tr_s5_close", "pos_s5", "SOLUSDT", "CLOSE_TAKE_PROFIT", 215.0, 10.0, 0.0, 150.0, t_s5_win, "tp"),
            ]
            con.executemany(
                "INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?,?,?)",
                trades,
            )
            con.commit()
        finally:
            con.close()

        res = tracker.compute_evolution_from_db()

        assert res["has_data"] is True
        assert len(res["resets"]) == 2
        assert res["resets"][0]["prev_equity"] == pytest.approx(11200.0)
        assert res["resets"][1]["prev_equity"] == pytest.approx(10800.0)

        adj_series = [pt["adj_equity"] for pt in res["portfolio_series"]]
        assert adj_series == pytest.approx([10000.0, 11200.0, 11200.0, 12000.0, 12000.0, 12150.0])
        assert res["portfolio_series"][-1]["adj_equity"] == pytest.approx(12150.0)
        assert res["summary"]["cum_pnl_usd"] == pytest.approx(2150.0)
        assert res["summary"]["total_trades"] == 3

        by_patch = {b["label"]: b for b in res["by_patch"]}
        assert "Patch S3" in by_patch
        assert "Patch S4.2" in by_patch
        assert "Patch S5" in by_patch

        assert by_patch["Patch S3"]["trades"] == 1
        assert by_patch["Patch S3"]["pnl_usd"] == pytest.approx(1200.0)
        assert by_patch["Patch S4.2"]["trades"] == 1
        assert by_patch["Patch S4.2"]["pnl_usd"] == pytest.approx(800.0)
        assert by_patch["Patch S5"]["trades"] == 1
        assert by_patch["Patch S5"]["pnl_usd"] == pytest.approx(150.0)


class TestSoftEpochReset:
    """Contract 3: scripts.reset_pilot.run_reset(..., soft_epoch=True) preserves history and resets epoch."""

    def test_soft_epoch_preserves_history_closes_open_positions_and_clears_guard(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "soft_epoch.db")
        PaperTradingEngine(db_path=db_path, restore=False)
        guard = RollbackGuard(db_path=db_path)
        LearningTracker(db_path=db_path)
        PerformanceTracker(db_path=db_path)
        MarketRegimeClassifier(db_path=db_path)
        GoldenPotAgent(
            db_path=db_path,
            binance_connector=MagicMock(),
            bitget_connector=MagicMock(),
        )._init_database_table()

        t_open_closed = _utc_ms(2026, 11, 18, 15, 0)
        t_close_closed = _utc_ms(2026, 11, 18, 16, 0)
        t_open_stale = _utc_ms(2026, 11, 19, 15, 0)

        con = sqlite3.connect(db_path)
        try:
            con.executemany(
                "INSERT INTO paper_positions (position_id,symbol,side,entry_price,current_price,"
                "quantity,notional,strategy_type,status) VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    ("p_closed", "BTCUSDT", "LONG", 80000.0, 81000.0, 0.5, 40000.0, "SCALP", "CLOSED"),
                    ("p_stale_1", "ETHUSDT", "SHORT", 3200.0, 3190.0, 1.0, 3200.0, "SCALP", "OPEN"),
                    ("p_stale_2", "SOLUSDT", "LONG", 200.0, 202.0, 5.0, 1000.0, "GOLDEN_POT", "OPEN"),
                ],
            )
            con.executemany(
                "INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    ("t1", "p_closed", "BTCUSDT", "OPEN_SCALP_LONG", 80000.0, 0.5, 0.0, 0.0, t_open_closed, "open"),
                    ("t2", "p_closed", "BTCUSDT", "CLOSE_TAKE_PROFIT", 81000.0, 0.5, 0.0, 500.0, t_close_closed, "tp"),
                    ("t3", "p_stale_1", "ETHUSDT", "OPEN_SCALP_SHORT", 3200.0, 1.0, 0.0, 0.0, t_open_stale, "open"),
                ],
            )
            con.executemany(
                "INSERT INTO paper_portfolio VALUES (?,?,?,?,?,?,?,?)",
                [
                    (t_open_closed, 10000.0, 10000.0, 0.0, 0.0, 0.0, 0.0, 1),
                    (t_close_closed, 10500.0, 10500.0, 0.0, 0.0, 500.0, 0.0, 2),
                ],
            )
            con.execute("INSERT INTO signal_funnel VALUES (3600000, 'SCALP', 'evaluated', 12)")
            con.execute(
                "INSERT INTO golden_pots (symbol,setup_type,direction,score,last_price,dist_vwap_pct,"
                "volume_24h_usd,price_change_pct_24h,range_position_pct,amplitude_pct,suggested_timeframe,"
                "rationale,timestamp) VALUES ('BTCUSDT','BREAKOUT','LONG',0.9,80000,0.5,1e9,2.0,0.7,3.0,'1h','r',?)",
                (t_open_closed,),
            )
            con.execute(
                "INSERT INTO market_regime VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (t_open_closed, "BULL_TREND", "BULLISH", 0.7, "trend", "r", 1, 1, 1, 1, 55),
            )
            con.execute(
                "INSERT INTO agent_learning_memory (decision_id,timestamp,symbol,regime,"
                "consensus_direction,consensus_confidence,agent_signals_json,outcome,realized_pnl,position_id) "
                "VALUES ('dec_1',?,'BTCUSDT','BULL_TREND','LONG',0.7,'[]','WIN',500.0,'p_closed')",
                (t_close_closed,),
            )
            con.execute("INSERT INTO agent_regime_weights VALUES ('BULL_TREND','TechnicalAgent',1.25,0.65,10,1)")

            pre_max_trade_rowid = int(con.execute("SELECT MAX(rowid) FROM paper_trades").fetchone()[0])
            pre_max_pos_rowid = int(con.execute("SELECT MAX(rowid) FROM paper_positions").fetchone()[0])
            con.commit()
        finally:
            con.close()

        # Trip RollbackGuard before soft-epoch reset
        tripped = guard.evaluate_telemetry({"current_drawdown_pct": 15.0, "max_drawdown_pct": 15.0})
        assert tripped.state == "TRIPPED"
        assert tripped.is_trading_allowed is False

        monkeypatch.setattr(rp, "detect_daemons", lambda: [])
        report = rp.run_reset(
            db_path=db_path,
            backup_dir=str(tmp_path / "backups"),
            flask_log=str(tmp_path / "flask.log"),
            pilot_report=str(tmp_path / "pilot_report.json"),
            soft_epoch=True,
        )

        assert report["mode"] == "soft_epoch"
        assert report["closed_open_positions"] == 2

        preserved_tables = [
            "paper_trades",
            "paper_positions",
            "paper_portfolio",
            "signal_funnel",
            "golden_pots",
            "market_regime",
        ]
        for tbl in preserved_tables:
            assert report["tables_after"][tbl] >= report["tables_before"][tbl], tbl
        assert report["tables_after"]["paper_portfolio"] == report["tables_before"]["paper_portfolio"] + 1
        assert report["untouched_tables"]["before"]["agent_learning_memory"] == 1
        assert report["untouched_tables"]["after"]["agent_learning_memory"] == 1
        assert report["untouched_tables"]["before"]["agent_regime_weights"] == 1
        assert report["untouched_tables"]["after"]["agent_regime_weights"] == 1

        con = sqlite3.connect(db_path)
        try:
            statuses = dict(con.execute("SELECT position_id, status FROM paper_positions").fetchall())
            assert statuses == {
                "p_closed": "CLOSED",
                "p_stale_1": "CLOSED_BY_RESET",
                "p_stale_2": "CLOSED_BY_RESET",
            }

            settings = dict(con.execute("SELECT key, value FROM simulation_settings").fetchall())
            assert settings["ledger_epoch_trade_rowid"] == str(pre_max_trade_rowid)
            assert settings["ledger_epoch_pos_rowid"] == str(pre_max_pos_rowid)
            assert settings["quota_epoch_trade_rowid"] == str(pre_max_trade_rowid)
            assert settings["MANUAL_RESET"] == "0"

            latest_port = con.execute(
                "SELECT equity, balance, realized_pnl_cum, open_positions_count "
                "FROM paper_portfolio ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            assert latest_port == (10000.0, 10000.0, 0.0, 0)
        finally:
            con.close()

        guard_after = RollbackGuard(db_path=db_path)
        assert guard_after.status.state == "CLEAR"
        assert guard_after.status.is_trading_allowed is True

        engine = PaperTradingEngine(db_path=db_path, restore=True)
        assert engine.balance == pytest.approx(10000.0)
        assert len(engine.positions) == 0
        assert len(engine.trade_history) == 0

        evo = PerformanceTracker(db_path=db_path).compute_evolution_from_db()
        assert evo["has_data"] is True
        assert evo["summary"]["total_trades"] == 1
        assert len(evo["resets"]) == 1
