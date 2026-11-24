"""
tests/test_patch_s6_onda2.py - Independent Behavioral Test Suite for Patch S6 Onda 2.
Contracts:
1. Constants & Signatures on LearningTracker (CALIBRATION_STALE_TTL_MS, now_ms param)
2. Pilot Epoch Isolation (pilot_start_utc strict window, zero fallback to pre-pilot)
3. 72h Anti-Deadlock TTL & Silence-Gap Clustering (streak dampening, read expiry, gap break)
4. S5-1 Immunity (structural_sub_rule_calibration remains untouched)
5. Backward Compatibility & FD Leak Guard (DB without simulation_settings, <= 2 leaked FDs)
6. ML Platt Baseline & TechnicalAgent Compatibility (Platt ensemble, predict_proba, joint gate)
"""
from __future__ import annotations

import inspect, json, os, sqlite3, sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import joblib, numpy as np, pandas as pd, pytest
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from consensus_judge import AgentSignal, ConsensusJudge
from agents.technical_agent import TechnicalAgent
from learning_tracker import LearningTracker, ModelEvolutionManager
from model_trainer import CalibratedMulticlassModel


def _make_agent_signals_json(direction: str = "SELL", confidence: float = 0.75, agents: Optional[List[str]] = None) -> str:
    names = agents or ["TechnicalAgent", "SentimentAgent", "OrderFlowAgent", "MacroAgent"]
    return json.dumps([{"agent_name": n, "direction": direction, "confidence": confidence} for n in names])


def _insert_trade(conn: sqlite3.Connection, dec_id: str, ts: int, regime: str, outcome: str, c_dir: str = "SELL", conf: float = 0.75, pnl: float = -100.0, sig_json: Optional[str] = None) -> None:
    s_json = sig_json or _make_agent_signals_json(direction=c_dir, confidence=conf)
    conn.execute(
        "INSERT INTO agent_learning_memory (decision_id, timestamp, symbol, regime, consensus_direction, "
        "consensus_confidence, agent_signals_json, outcome, realized_pnl) VALUES (?, ?, 'BTCUSDT', ?, ?, ?, ?, ?, ?)",
        (dec_id, ts, regime, c_dir, conf, s_json, outcome, pnl),
    )


def _set_pilot_utc(conn: sqlite3.Connection, pilot_str: str, pilot_ms: int) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS simulation_settings (key TEXT PRIMARY KEY, value TEXT, updated_at INTEGER)")
    conn.execute("INSERT OR REPLACE INTO simulation_settings (key, value, updated_at) VALUES ('pilot_start_utc', ?, ?)", (pilot_str, pilot_ms))


class TestContract1ConstantsAndSignatures:
    """Contract 1: Constants and method signatures on LearningTracker."""

    def test_constants_and_method_signatures(self):
        assert LearningTracker.CALIBRATION_STALE_TTL_MS == 72 * 3_600_000

        sig_recal = inspect.signature(LearningTracker.recalibrate_agent_weights)
        assert "now_ms" in sig_recal.parameters and sig_recal.parameters["now_ms"].default is None

        sig_w = inspect.signature(LearningTracker.get_calibrated_weights_for_regime)
        assert "regime" in sig_w.parameters and "now_ms" in sig_w.parameters and sig_w.parameters["now_ms"].default is None

        sig_cal = inspect.signature(LearningTracker.get_agent_calibration_for_regime)
        assert "regime" in sig_cal.parameters and "now_ms" in sig_cal.parameters and sig_cal.parameters["now_ms"].default is None


class TestContract2PilotEpochIsolation:
    """Contract 2: pilot_start_utc strict window and zero fallback."""

    def test_pilot_epoch_ignores_pre_pilot_rows_and_overwrites_stale_cache(self, tmp_path):
        db_path = str(tmp_path / "pilot_iso.db")
        tracker = LearningTracker(db_path=db_path)
        pilot_str = "2026-11-20 13:15:34"
        pilot_ms = int(datetime.fromisoformat(pilot_str).replace(tzinfo=timezone.utc).timestamp() * 1000)

        with sqlite3.connect(db_path) as conn:
            _set_pilot_utc(conn, pilot_str, pilot_ms)
            for i in range(10):
                _insert_trade(conn, f"pre_{i}", pilot_ms - 200_000 + i * 1_000, "BEAR_TREND", "LOSS")
            conn.execute("INSERT OR REPLACE INTO agent_regime_weights (regime, agent_name, weight, accuracy, total_samples, updated_at) VALUES ('BEAR_TREND', 'TechnicalAgent', 0.15, 0.10, 10, ?)", (pilot_ms - 100_000,))
            conn.execute("INSERT OR REPLACE INTO agent_regime_calibration (regime, agent_name, ece, brier_score, updated_at) VALUES ('BEAR_TREND', 'TechnicalAgent', 0.45, 0.50, ?)", (pilot_ms - 100_000,))
            conn.commit()

        # Read-time: 0 post-pilot trades -> neutral weights & expired calibration
        w_read = tracker.get_calibrated_weights_for_regime("BEAR_TREND", now_ms=pilot_ms + 1_000)
        c_read = tracker.get_agent_calibration_for_regime("BEAR_TREND", now_ms=pilot_ms + 1_000)
        assert w_read["TechnicalAgent"] == tracker.DEFAULT_BASE_WEIGHTS["TechnicalAgent"]
        assert w_read["RiskSentinelAgent"] == 1.5
        assert c_read["TechnicalAgent"]["total_samples"] == 0
        assert c_read["TechnicalAgent"]["accuracy"] == 0.5
        assert c_read["TechnicalAgent"]["ece"] == 0.0
        assert c_read["TechnicalAgent"]["stale_expired"] is True

        # Recalibrate: persists neutral values and overwrites stale pre-pilot cache
        recal_w = tracker.recalibrate_agent_weights(now_ms=pilot_ms + 1_000)
        assert recal_w["BEAR_TREND"]["TechnicalAgent"] == tracker.DEFAULT_BASE_WEIGHTS["TechnicalAgent"]

        with sqlite3.connect(db_path) as conn:
            row_w = conn.execute("SELECT weight, accuracy, total_samples FROM agent_regime_weights WHERE regime = 'BEAR_TREND' AND agent_name = 'TechnicalAgent'").fetchone()
            row_c = conn.execute("SELECT ece, brier_score FROM agent_regime_calibration WHERE regime = 'BEAR_TREND' AND agent_name = 'TechnicalAgent'").fetchone()
            assert row_w[0] == tracker.DEFAULT_BASE_WEIGHTS["TechnicalAgent"] and row_w[1] == 0.5 and row_w[2] == 0
            assert row_c[0] == 0.0 and row_c[1] == 0.0

    def test_pilot_epoch_few_post_pilot_rows_no_fallback(self, tmp_path):
        db_path = str(tmp_path / "pilot_few.db")
        tracker = LearningTracker(db_path=db_path)
        pilot_str = "2026-11-20 13:15:34"
        pilot_ms = int(datetime.fromisoformat(pilot_str).replace(tzinfo=timezone.utc).timestamp() * 1000)

        with sqlite3.connect(db_path) as conn:
            _set_pilot_utc(conn, pilot_str, pilot_ms)
            for i in range(8):
                _insert_trade(conn, f"pre_{i}", pilot_ms - 100_000 + i * 1_000, "BEAR_TREND", "LOSS")
            for i in range(2):
                _insert_trade(conn, f"post_{i}", pilot_ms + 1_000 + i * 1_000, "BEAR_TREND", "LOSS")
            conn.commit()

        tracker.recalibrate_agent_weights(now_ms=pilot_ms + 5_000)
        with sqlite3.connect(db_path) as conn:
            row_w = conn.execute("SELECT weight, accuracy, total_samples FROM agent_regime_weights WHERE regime = 'BEAR_TREND' AND agent_name = 'TechnicalAgent'").fetchone()
            assert row_w[2] == 2  # Strictly 2 post-pilot rows, no fallback to 8 pre-pilot rows
            assert row_w[0] == tracker.DEFAULT_BASE_WEIGHTS["TechnicalAgent"]


class TestContract3AntiDeadlockAndSilenceClustering:
    """Contract 3: Streak dampening, read expiry, and silence gap clustering."""

    def test_active_losing_streak_under_72h_dampens_consensus(self, tmp_path):
        db_path = str(tmp_path / "deadlock.db")
        tracker = LearningTracker(db_path=db_path)
        pilot_str = "2026-11-20 13:15:34"
        pilot_ms = int(datetime.fromisoformat(pilot_str).replace(tzinfo=timezone.utc).timestamp() * 1000)
        t0 = pilot_ms + 10_000

        with sqlite3.connect(db_path) as conn:
            _set_pilot_utc(conn, pilot_str, pilot_ms)
            for i in range(6):
                _insert_trade(conn, f"loss_{i}", t0 + i * 3_600_000, "BEAR_TREND", "LOSS", "SELL", 0.75)
            conn.commit()

        last_trade_ts = t0 + 5 * 3_600_000
        now_active = last_trade_ts + 60_000
        tracker.recalibrate_agent_weights(now_ms=now_active)
        calib = tracker.get_agent_calibration_for_regime("BEAR_TREND", now_ms=now_active)

        tech_cal = calib["TechnicalAgent"]
        assert tech_cal["total_samples"] == 6 and tech_cal["accuracy"] < 0.35 and tech_cal["ece"] > 0.20

        judge = ConsensusJudge()
        signals = [
            AgentSignal("TechnicalAgent", "SELL", 0.80, 1.0, "bear"),
            AgentSignal("SentimentAgent", "SELL", 0.80, 0.8, "bear"),
            AgentSignal("OrderFlowAgent", "SELL", 0.80, 1.2, "bear"),
            AgentSignal("MacroAgent", "SELL", 0.80, 0.8, "bear"),
        ]
        decision = judge.judge_deterministic(signals, regime="BEAR_TREND", agent_calibration=calib)
        assert decision.confidence <= 0.55
        for s in decision.agent_signals:
            assert s.metrics.get("calibrated_dampened") is True

    def test_read_time_expiry_ge_72h_restores_consensus(self, tmp_path):
        db_path = str(tmp_path / "expiry.db")
        tracker = LearningTracker(db_path=db_path)
        pilot_str = "2026-11-20 13:15:34"
        pilot_ms = int(datetime.fromisoformat(pilot_str).replace(tzinfo=timezone.utc).timestamp() * 1000)
        t0 = pilot_ms + 10_000

        with sqlite3.connect(db_path) as conn:
            _set_pilot_utc(conn, pilot_str, pilot_ms)
            for i in range(6):
                _insert_trade(conn, f"loss_{i}", t0 + i * 3_600_000, "BEAR_TREND", "LOSS", "SELL", 0.75)
            conn.commit()

        last_trade_ts = t0 + 5 * 3_600_000
        tracker.recalibrate_agent_weights(now_ms=last_trade_ts + 60_000)

        # Query >= 72h past last trade
        now_exp = last_trade_ts + LearningTracker.CALIBRATION_STALE_TTL_MS + 1
        calib_exp = tracker.get_agent_calibration_for_regime("BEAR_TREND", now_ms=now_exp)
        w_exp = tracker.get_calibrated_weights_for_regime("BEAR_TREND", now_ms=now_exp)

        assert w_exp["TechnicalAgent"] == tracker.DEFAULT_BASE_WEIGHTS["TechnicalAgent"]
        assert w_exp["RiskSentinelAgent"] == 1.5
        assert calib_exp["TechnicalAgent"]["total_samples"] == 0
        assert calib_exp["TechnicalAgent"]["stale_expired"] is True
        assert calib_exp["TechnicalAgent"]["accuracy"] == 0.5 and calib_exp["TechnicalAgent"]["ece"] == 0.0

        judge = ConsensusJudge()
        signals = [
            AgentSignal("TechnicalAgent", "SELL", 0.80, 1.0, "bear"),
            AgentSignal("SentimentAgent", "SELL", 0.80, 0.8, "bear"),
            AgentSignal("OrderFlowAgent", "SELL", 0.80, 1.2, "bear"),
            AgentSignal("MacroAgent", "SELL", 0.80, 0.8, "bear"),
        ]
        decision = judge.judge_deterministic(signals, regime="BEAR_TREND", agent_calibration=calib_exp)
        assert decision.confidence >= 0.70
        for s in decision.agent_signals:
            assert s.metrics.get("calibrated_dampened") is not True

    def test_silence_gap_cluster_break_excludes_pre_cooldown_losses(self, tmp_path):
        db_path = str(tmp_path / "gap_break.db")
        tracker = LearningTracker(db_path=db_path)
        pilot_str = "2026-11-20 13:15:34"
        pilot_ms = int(datetime.fromisoformat(pilot_str).replace(tzinfo=timezone.utc).timestamp() * 1000)
        t0 = pilot_ms + 10_000

        with sqlite3.connect(db_path) as conn:
            _set_pilot_utc(conn, pilot_str, pilot_ms)
            for i in range(6):
                _insert_trade(conn, f"loss_{i}", t0 + i * 3_600_000, "BEAR_TREND", "LOSS")
            t_new = t0 + 80 * 3_600_000  # 75h gap >= 72h TTL
            _insert_trade(conn, "win_new", t_new, "BEAR_TREND", "WIN", "SELL", 0.75, pnl=150.0)
            conn.commit()

        tracker.recalibrate_agent_weights(now_ms=t_new + 1_000)
        with sqlite3.connect(db_path) as conn:
            row_w = conn.execute("SELECT weight, accuracy, total_samples FROM agent_regime_weights WHERE regime = 'BEAR_TREND' AND agent_name = 'TechnicalAgent'").fetchone()
            assert row_w[2] == 1, f"Expected total_samples == 1 after gap break, got {row_w[2]}"


class TestContract4S51Immunity:
    """Contract 4: structural_sub_rule_calibration is immune to recalibration."""

    def test_structural_sub_rule_calibration_immunity(self, tmp_path):
        db_path = str(tmp_path / "s5_immunity.db")
        tracker = LearningTracker(db_path=db_path)
        pilot_str = "2026-11-20 13:15:34"
        pilot_ms = int(datetime.fromisoformat(pilot_str).replace(tzinfo=timezone.utc).timestamp() * 1000)

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO structural_sub_rule_calibration (sub_rule, total_samples, wins, posterior_wr, cumulative_pnl, multiplier, is_muted, updated_at) "
                "VALUES ('sweep_trigger', 10, 2, 0.35, -50.0, 0.0, 1, ?)",
                (pilot_ms - 50_000,),
            )
            conn.execute("INSERT OR REPLACE INTO agent_regime_weights (regime, agent_name, weight, accuracy, total_samples, updated_at) VALUES ('BEAR_TREND', 'TechnicalAgent', 0.20, 0.15, 20, ?)", (pilot_ms - 50_000,))
            _set_pilot_utc(conn, pilot_str, pilot_ms)
            conn.commit()

        tracker.recalibrate_agent_weights(now_ms=pilot_ms + 1_000)

        with sqlite3.connect(db_path) as conn:
            sub = conn.execute("SELECT is_muted, multiplier, total_samples FROM structural_sub_rule_calibration WHERE sub_rule = 'sweep_trigger'").fetchone()
            assert sub == (1, 0.0, 10), "structural_sub_rule_calibration must remain untouched"
            w = conn.execute("SELECT total_samples, accuracy FROM agent_regime_weights WHERE regime = 'BEAR_TREND' AND agent_name = 'TechnicalAgent'").fetchone()
            assert w == (0, 0.5), "BEAR_TREND weights must be cleared to (0, 0.5)"


class TestContract5BackwardCompatibilityAndFDLeak:
    """Contract 5: Legacy DB without simulation_settings and zero FD leaks."""

    def test_backward_compatibility_db_without_simulation_settings(self, tmp_path):
        db_path = str(tmp_path / "legacy.db")
        tracker = LearningTracker(db_path=db_path)

        with sqlite3.connect(db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS simulation_settings")
            conn.execute("INSERT OR REPLACE INTO agent_regime_weights (regime, agent_name, weight, accuracy, total_samples, updated_at) VALUES ('HIGH_VOLATILITY_PANIC', 'TechnicalAgent', 1.85, 0.65, 15, 1700000000000)")
            recent_ts = 1800000000000
            for i in range(5):
                _insert_trade(conn, f"b_{i}", recent_ts + i * 1_000, "BULL_TREND", "WIN", "BUY", 0.70, pnl=50.0, sig_json=_make_agent_signals_json("BUY", 0.70))
            conn.commit()

        # Direct insertion preserved when no rows in memory
        w_panic = tracker.get_calibrated_weights_for_regime("HIGH_VOLATILITY_PANIC", now_ms=recent_ts + 10_000)
        assert w_panic.get("TechnicalAgent") == 1.85

        # Normal calibration on recent trades
        tracker.recalibrate_agent_weights(now_ms=recent_ts + 10_000)
        with sqlite3.connect(db_path) as conn:
            row_b = conn.execute("SELECT total_samples, accuracy FROM agent_regime_weights WHERE regime = 'BULL_TREND' AND agent_name = 'TechnicalAgent'").fetchone()
            assert row_b == (5, 1.0)

    def test_fd_leak_guard_under_consecutive_calls(self, tmp_path):
        db_path = str(tmp_path / "fd_leak.db")
        tracker = LearningTracker(db_path=db_path)
        tracker.get_calibrated_weights_for_regime("BULL_TREND")
        tracker.get_agent_calibration_for_regime("BULL_TREND")
        tracker.recalibrate_agent_weights()

        fd_proc = "/proc/self/fd"
        if not os.path.exists(fd_proc):
            pytest.skip("/proc/self/fd not supported")

        fd_start = len(os.listdir(fd_proc))
        for _ in range(40):
            tracker.get_calibrated_weights_for_regime("BULL_TREND")
            tracker.get_agent_calibration_for_regime("BULL_TREND")
            tracker.recalibrate_agent_weights()
        diff = len(os.listdir(fd_proc)) - fd_start
        assert diff <= 2, f"Expected <= 2 open FD leak, detected {diff}"


class TestContract6MLPlattBaselineAndTechnicalAgent:
    """Contract 6: CalibratedMulticlassModel loads via TechnicalAgent and joint gate."""

    def test_ml_platt_baseline_technical_agent_and_joint_gate(self, tmp_path, monkeypatch):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3), columns=["f1", "f2", "f3"])
        y = np.random.choice([0, 1, 2], size=100)

        base = LogisticRegression()
        base.fit(X, y)
        cal = CalibratedMulticlassModel(base_estimator=base, method="platt")
        cal.fit(X, y)

        bundle = {
            "model": cal,
            "feature_names": ["f1", "f2", "f3"],
            "metrics": {"log_loss": 0.6500, "roc_auc_ovr": 0.6000, "emitted_signals_accuracy": 0.5200},
            "model_type": "lgbm",
            "version": "v1.0.12",
        }
        p_lgbm = str(tmp_path / "lgbm_model_v1.joblib")
        p_rf = str(tmp_path / "rf_model_v1.joblib")
        joblib.dump(bundle, p_lgbm)
        joblib.dump(bundle, p_rf)

        monkeypatch.setattr(TechnicalAgent, "_get_model_dir_paths", classmethod(lambda cls: [p_lgbm, p_rf]))
        TechnicalAgent.invalidate_default_model_cache()

        try:
            model = TechnicalAgent._get_default_model()
            assert model is not None

            proba = model.predict_proba({"f1": 0.25, "f2": -0.15, "f3": 0.80})
            assert proba.shape == (1, 3) and np.all(np.isfinite(proba))
            assert np.isclose(float(np.sum(proba)), 1.0, atol=1e-5)

            mgr = ModelEvolutionManager()
            chal_metrics = {"log_loss": 0.6480, "roc_auc_ovr": 0.6050, "emitted_signals_accuracy": 0.4000}
            approved, msg = mgr.evaluate_challenger_joint(bundle["metrics"], chal_metrics)
            assert approved is True and "melhoria conjunta OOS validada" in msg
        finally:
            TechnicalAgent.invalidate_default_model_cache()
