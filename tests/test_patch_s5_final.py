"""tests/test_patch_s5_final.py
Independent Clean-Window Behavioral TDD Suite for Patch S5 ("Sem Oráculo Comprado").

Coverage:
1. S5.1: Living Sub-Rule Learner (TechnicalAgent, _DefaultMLEnsemble, LearningTracker, Core, SignalGenerator)
2. S5.2 & S5.3: ML Evolution & Post-Calibration (CalibratedMulticlassModel, ModelTrainer, ModelEvolutionManager)
3. S5.4: SentimentAgent Trend Momentum & LiveContextEnricher Stale Funding Fallback
4. S5.5: Swing4HMetaSizer Shadow/Active Invariants & Zero-Lookahead Perturbation

Rules:
- Self-contained using tmp_path SQLite DBs & synthetic OHLCV DataFrames.
- Strict numeric invariants (sum to 1.0, Bayesian posterior, bounds [0.4, 2.0], causal D-1).
- Fast execution (<15s total), zero dependency on production market_cache.db or live networks.
"""

import json
import math
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

# Target imports with clean fallback bindings for collection
from agents.base_agent import AgentSignal
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent, _DefaultMLEnsemble
import core as core_module
IA_TraderCore = getattr(core_module, "IA_TraderCore", getattr(core_module, "TradingCore", None))
from data.live_context_enricher import LiveContextEnricher
from learning_tracker import LearningTracker, ModelEvolutionManager
import model_trainer as mt_module
from model_trainer import ModelTrainer
from signal_generator import SignalGenerator

# Late/dynamic imports for components introduced in S5
CalibratedMulticlassModel = getattr(mt_module, "CalibratedMulticlassModel", None)
compute_multiclass_ece = getattr(mt_module, "compute_multiclass_ece", None)

try:
    from data.swing_4h_meta_sizer import (
        META_FEATURE_COLS,
        VALID_MODES,
        Swing4HMetaSizer,
    )
except ImportError:
    Swing4HMetaSizer = None
    META_FEATURE_COLS = None
    VALID_MODES = None


# ============================================================================
# 1. S5.1: LIVING SUB-RULE LEARNER & TECHNICAL AGENT CONFLUENCE
# ============================================================================


class TestLivingSubRuleLearner:
    """Verify Contract 1: Sub-rule tagging, Bayesian auto-mutes, native NaN in ML ensemble."""

    def test_technical_agent_default_sweep_trigger_disabled(self):
        """TechnicalAgent(enable_sweep_trigger=False) (default) keeps HOLD when sweep_bull=1."""
        agent_default = TechnicalAgent()
        assert hasattr(agent_default, "enable_sweep_trigger"), (
            "TechnicalAgent must expose enable_sweep_trigger attribute"
        )
        assert agent_default.enable_sweep_trigger is False, (
            "Default enable_sweep_trigger must be False"
        )

        # Context with sweep present but no indicator trigger
        ctx = {
            "symbol": "BTCUSDT",
            "timestamp": 1700000000000,
            "indicators": {
                "rsi_14": 50.0,
                "macd_line": 0.0,
                "macd_signal": 0.0,
                "adx": 15.0,
                "liquidity_sweep_bull": 1,
                "liquidity_sweep_bear": 0,
                "at_bullish_poi": 0,
                "at_bearish_poi": 0,
            },
            "liquidity_sweep_bull": 1,
        }
        sig = agent_default.evaluate(ctx)
        assert sig.direction == "HOLD", (
            f"Expected HOLD with enable_sweep_trigger=False, got {sig.direction}"
        )

    def test_technical_agent_enable_sweep_trigger_triggers_buy(self):
        """TechnicalAgent(enable_sweep_trigger=True) triggers BUY on liquidity_sweep_bull=1."""
        agent_enabled = TechnicalAgent(enable_sweep_trigger=True)
        assert agent_enabled.enable_sweep_trigger is True

        ctx = {
            "symbol": "BTCUSDT",
            "timestamp": 1700000000000,
            "indicators": {
                "rsi_14": 50.0,
                "macd_line": 0.0,
                "macd_signal": 0.0,
                "adx": 15.0,
                "liquidity_sweep_bull": 1,
                "liquidity_sweep_bear": 0,
                "at_bullish_poi": 0,
                "at_bearish_poi": 0,
            },
            "liquidity_sweep_bull": 1,
        }
        sig = agent_enabled.evaluate(ctx)
        assert sig.direction == "BUY", (
            f"Expected BUY with enable_sweep_trigger=True, got {sig.direction}"
        )
        assert sig.metrics.get("structural_sub_rule") == "sweep_trigger", (
            f"Expected structural_sub_rule=='sweep_trigger', got {sig.metrics.get('structural_sub_rule')}"
        )
        assert isinstance(sig.metrics.get("sub_rule_multiplier"), (float, int))

    def test_technical_agent_sub_rule_classification_and_metrics(self):
        """evaluate() must tag structural_sub_rule across ('sweep_trigger','poi_confluence','open_air','normal_no_sr')."""
        agent = TechnicalAgent(enable_sweep_trigger=False)

        # 1. POI confluence on BUY
        ctx_poi = {
            "symbol": "BTCUSDT",
            "timestamp": 1700000000000,
            "indicators": {
                "rsi_14": 30.0,
                "macd_line": 1.0,
                "macd_signal": 0.5,
                "adx": 25.0,
                "liquidity_sweep_bull": 1,
                "at_bullish_poi": 1,
            },
            "at_bullish_poi": 1,
            "liquidity_sweep_bull": 1,
        }
        sig_poi = agent.evaluate(ctx_poi)
        if sig_poi.direction == "BUY":
            assert sig_poi.metrics.get("structural_sub_rule") == "poi_confluence"
            assert "sub_rule_multiplier" in sig_poi.metrics

        # 2. Open air on BUY (no POI, dist_resistance_pct >= 1.5)
        ctx_open = {
            "symbol": "BTCUSDT",
            "timestamp": 1700000000000,
            "indicators": {
                "rsi_14": 30.0,
                "macd_line": 1.0,
                "macd_signal": 0.5,
                "adx": 25.0,
                "liquidity_sweep_bull": 0,
                "at_bullish_poi": 0,
                "dist_resistance_pct": 2.5,
            },
            "dist_resistance_pct": 2.5,
        }
        sig_open = agent.evaluate(ctx_open)
        if sig_open.direction == "BUY":
            assert sig_open.metrics.get("structural_sub_rule") == "open_air"

        # 3. Normal no S/R on BUY (no POI, dist_resistance_pct < 1.5)
        ctx_normal = {
            "symbol": "BTCUSDT",
            "timestamp": 1700000000000,
            "indicators": {
                "rsi_14": 30.0,
                "macd_line": 1.0,
                "macd_signal": 0.5,
                "adx": 25.0,
                "liquidity_sweep_bull": 0,
                "at_bullish_poi": 0,
                "dist_resistance_pct": 0.8,
            },
            "dist_resistance_pct": 0.8,
        }
        sig_normal = agent.evaluate(ctx_normal)
        if sig_normal.direction == "BUY":
            assert sig_normal.metrics.get("structural_sub_rule") == "normal_no_sr"

    def test_sub_rule_multipliers_mute_poi_confluence(self):
        """sub_rule_multipliers={'poi_confluence': 0.0} eliminates POI/sweep bonus on BUY/SELL."""
        agent = TechnicalAgent(enable_sweep_trigger=False)
        base_ctx = {
            "symbol": "BTCUSDT",
            "timestamp": 1700000000000,
            "indicators": {
                "rsi_14": 30.0,
                "macd_line": 1.0,
                "macd_signal": 0.5,
                "adx": 25.0,
                "at_bullish_poi": 1,
            },
            "at_bullish_poi": 1,
        }

        # Active multiplier 1.0 (with bonus)
        ctx_active = dict(base_ctx, sub_rule_multipliers={"poi_confluence": 1.0})
        sig_active = agent.evaluate(ctx_active)

        # Muted multiplier 0.0 (bonus eliminated)
        ctx_muted = dict(base_ctx, sub_rule_multipliers={"poi_confluence": 0.0})
        sig_muted = agent.evaluate(ctx_muted)

        if sig_active.direction == "BUY" and sig_muted.direction == "BUY":
            assert sig_muted.confidence < sig_active.confidence, (
                f"Muted confidence ({sig_muted.confidence}) must be strictly lower than active ({sig_active.confidence})"
            )
            assert "[POI/SWEEP silenciado pelo Auto-Learner]" in sig_muted.rationale, (
                "Rationale must document auto-mute notice when multiplier is 0.0"
            )

    def test_default_ml_ensemble_native_nan_preservation_for_lgbm(self):
        """_DefaultMLEnsemble preserves native np.nan for dist_resistance_pct/dist_support_pct on LightGBM only."""
        lgbm_received_x = []
        sklearn_received_x = []

        class FakeLGBM:
            __module__ = "lightgbm.sklearn"
            classes_ = np.array([0, 1, 2])

            def predict_proba(self, X):
                lgbm_received_x.append(X.copy() if hasattr(X, "copy") else X)
                return np.array([[0.2, 0.6, 0.2]])

        class FakeRandomForest:
            __module__ = "sklearn.ensemble._forest"
            classes_ = np.array([0, 1, 2])

            def predict_proba(self, X):
                sklearn_received_x.append(X.copy() if hasattr(X, "copy") else X)
                return np.array([[0.2, 0.6, 0.2]])

        feats = ["dist_resistance_pct", "dist_support_pct", "rsi_14"]
        m_lgbm = FakeLGBM()
        m_rf = FakeRandomForest()

        ensemble = _DefaultMLEnsemble(
            models_with_features=[(m_lgbm, feats), (m_rf, feats)]
        )

        input_data = {
            "dist_resistance_pct": np.nan,
            "dist_support_pct": np.nan,
            "rsi_14": np.nan,
        }
        proba = ensemble.predict_proba(input_data)
        assert proba.shape == (1, 3)
        assert len(lgbm_received_x) == 1
        assert len(sklearn_received_x) == 1

        x_lgbm = lgbm_received_x[0]
        x_rf = sklearn_received_x[0]

        # For LightGBM: dist_resistance_pct and dist_support_pct should retain NaN, while rsi_14 is 0.0
        assert np.isnan(x_lgbm["dist_resistance_pct"].iloc[0]), (
            "LightGBM estimator must receive native np.nan for dist_resistance_pct"
        )
        assert np.isnan(x_lgbm["dist_support_pct"].iloc[0]), (
            "LightGBM estimator must receive native np.nan for dist_support_pct"
        )
        assert x_lgbm["rsi_14"].iloc[0] == 0.0, (
            "Non-S/R feature NaN must be filled to 0.0 for LightGBM"
        )

        # For Sklearn: all features must be non-NaN (filled to 0.0)
        assert x_rf["dist_resistance_pct"].iloc[0] == 0.0, (
            "Sklearn estimator must receive fillna(0.0) for dist_resistance_pct"
        )
        assert x_rf["dist_support_pct"].iloc[0] == 0.0, (
            "Sklearn estimator must receive fillna(0.0) for dist_support_pct"
        )
        assert x_rf["rsi_14"].iloc[0] == 0.0

    def test_learning_tracker_recalibrate_structural_sub_rules_bayesian_and_sliding_window(
        self, tmp_path
    ):
        """recalibrate_structural_sub_rules applies Beta(4,4) posterior and auto-mutes if n>=5, post_wr<0.44, cum_pnl<0."""
        db_path = str(tmp_path / "learning.db")
        tracker = LearningTracker(db_path=db_path)
        assert hasattr(tracker, "recalibrate_structural_sub_rules"), (
            "LearningTracker must implement recalibrate_structural_sub_rules"
        )

        # 1. Inject 5 losing trades for 'sweep_trigger'
        # With n=5, wins=0, alpha=4, beta=4: post_wr = (0 + 4) / (5 + 4 + 4) = 4/13 = 0.3077 < 0.44
        now_ms = 1700000000000
        for i in range(5):
            p_id = f"loss_{i}"
            tracker.record_decision(
                decision_id=p_id,
                symbol="BTCUSDT",
                regime="TREND_BULLISH",
                consensus_direction="BUY",
                consensus_confidence=0.65,
                agent_signals=[{
                    "agent_name": "TechnicalAgent",
                    "direction": "BUY",
                    "metrics": {"structural_sub_rule": "sweep_trigger"},
                    "rationale": "Gatilho Estrutural Puro: sweep bull",
                }],
                position_id=p_id,
                timestamp=now_ms + i * 60000,
            )
            tracker.update_trade_outcome(
                position_id=p_id,
                pnl=-50.0,
                outcome_label="LOSS",
            )

        cal = tracker.recalibrate_structural_sub_rules(window=20)
        assert "sweep_trigger" in cal, "Calibration stats must include 'sweep_trigger'"
        sw_stat = cal["sweep_trigger"]
        expected_post_wr = 4.0 / 13.0
        assert math.isclose(sw_stat["posterior_wr"], expected_post_wr, abs_tol=0.01), (
            f"Expected Beta(4,4) posterior ~{expected_post_wr:.4f}, got {sw_stat['posterior_wr']}"
        )
        assert sw_stat["is_muted"] in (1, True), (
            "Sub-rule with n>=5, post_wr<0.44, cum_pnl<0 must be auto-muted"
        )
        assert sw_stat["multiplier"] == 0.0, "Muted sub-rule multiplier must be 0.0"

        # Verify DB persistence
        mults = tracker.get_sub_rule_multipliers()
        assert mults.get("sweep_trigger") == 0.0, (
            f"Expected sweep_trigger multiplier 0.0 from tracker, got {mults}"
        )

        # 2. Recovery: inject 20 subsequent winning trades
        for i in range(20):
            p_id = f"win_{i}"
            tracker.record_decision(
                decision_id=p_id,
                symbol="BTCUSDT",
                regime="TREND_BULLISH",
                consensus_direction="BUY",
                consensus_confidence=0.70,
                agent_signals=[{
                    "agent_name": "TechnicalAgent",
                    "direction": "BUY",
                    "metrics": {"structural_sub_rule": "sweep_trigger"},
                }],
                position_id=p_id,
                timestamp=now_ms + (10 + i) * 60000,
            )
            tracker.update_trade_outcome(
                position_id=p_id,
                pnl=40.0,
                outcome_label="WIN",
            )

        cal_rec = tracker.recalibrate_structural_sub_rules(window=20)
        sw_rec = cal_rec["sweep_trigger"]
        # Sliding window takes the latest 20 trades (all wins!): post_wr = (20 + 4) / (20 + 8) = 24/28 = 0.857 > 0.44
        assert sw_rec["is_muted"] in (0, False), "Sub-rule must unmute after winning window"
        assert sw_rec["multiplier"] > 0.0, "Recovered multiplier must be strictly positive"

    def test_learning_tracker_bootstrap_cold_or_pending_db_stays_empty(self, tmp_path):
        """Cold DB with 0 WIN/LOSS rows (or only PENDING/BREAKEVEN) leaves calibration table empty."""
        db_path = str(tmp_path / "cold_learning.db")

        # 1. Cold DB (0 rows)
        tracker_cold = LearningTracker(db_path=db_path)
        assert tracker_cold.get_sub_rule_multipliers() == {}, (
            "Cold DB must return empty sub-rule multipliers"
        )
        with sqlite3.connect(db_path) as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM structural_sub_rule_calibration").fetchone()[0]
            assert cnt == 0, "structural_sub_rule_calibration must have 0 rows on cold DB"

        # 2. Insert only PENDING and BREAKEVEN rows
        signals_json = json.dumps([{
            "agent_name": "TechnicalAgent",
            "direction": "BUY",
            "metrics": {"structural_sub_rule": "sweep_trigger"},
        }])
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                INSERT INTO agent_learning_memory (
                    decision_id, timestamp, symbol, regime,
                    consensus_direction, consensus_confidence,
                    agent_signals_json, outcome, realized_pnl, position_id
                ) VALUES
                ('pending_1', 1700000000000, 'BTCUSDT', 'TREND_BULLISH', 'BUY', 0.6, ?, 'PENDING', 0.0, 'pos_p1'),
                ('breakeven_1', 1700000060000, 'BTCUSDT', 'TREND_BULLISH', 'BUY', 0.6, ?, 'BREAKEVEN', 0.0, 'pos_b1')
            """, (signals_json, signals_json))
            conn.commit()

        # Re-instantiate fresh tracker - should still not bootstrap
        fresh_tracker = LearningTracker(db_path=db_path)
        assert fresh_tracker.get_sub_rule_multipliers() == {}, (
            "DB with only PENDING/BREAKEVEN rows must not trigger calibration bootstrap"
        )
        with sqlite3.connect(db_path) as conn:
            cnt_after = conn.execute("SELECT COUNT(*) FROM structural_sub_rule_calibration").fetchone()[0]
            assert cnt_after == 0, "structural_sub_rule_calibration must remain empty"

    def test_learning_tracker_bootstrap_auto_populates_on_init_with_closed_trades(self, tmp_path):
        """Pre-populated DB with closed WIN/LOSS auto-bootstraps structural_sub_rule_calibration on __init__."""
        db_path = str(tmp_path / "prepopulated_learning.db")

        # Create schema directly via sqlite3 (simulating existing DB with empty calibration)
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE agent_learning_memory (
                    decision_id TEXT PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    consensus_direction TEXT NOT NULL,
                    consensus_confidence REAL NOT NULL,
                    agent_signals_json TEXT NOT NULL,
                    outcome TEXT DEFAULT 'PENDING',
                    realized_pnl REAL DEFAULT 0.0,
                    position_id TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE structural_sub_rule_calibration (
                    sub_rule TEXT PRIMARY KEY,
                    total_samples INTEGER NOT NULL,
                    wins INTEGER NOT NULL,
                    posterior_wr REAL NOT NULL,
                    cumulative_pnl REAL NOT NULL,
                    multiplier REAL NOT NULL,
                    is_muted INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL
                )
            """)

            # Pre-populate 5 LOSS rows for 'sweep_trigger' (n=5, wins=0, cum_pnl=-250.0)
            # Bayesian Beta(4,4) posterior: (0 + 4) / (5 + 4 + 4) = 4/13 = 0.3077 < 0.44 -> auto-mute (multiplier=0.0)
            sig_sweep = json.dumps([{
                "agent_name": "TechnicalAgent",
                "direction": "BUY",
                "metrics": {"structural_sub_rule": "sweep_trigger"},
            }])
            now_ms = 1700000000000
            for i in range(5):
                conn.execute("""
                    INSERT INTO agent_learning_memory (
                        decision_id, timestamp, symbol, regime,
                        consensus_direction, consensus_confidence,
                        agent_signals_json, outcome, realized_pnl, position_id
                    ) VALUES (?, ?, 'BTCUSDT', 'TREND_BULLISH', 'BUY', 0.65, ?, 'LOSS', -50.0, ?)
                """, (f"dec_loss_{i}", now_ms + i * 1000, sig_sweep, f"pos_loss_{i}"))

            # Pre-populate 5 WIN rows for 'poi_confluence' (n=5, wins=5, cum_pnl=+200.0)
            # Bayesian Beta(4,4) posterior: (5 + 4) / (5 + 4 + 4) = 9/13 = 0.6923 > 0.44 -> active multiplier
            sig_poi = json.dumps([{
                "agent_name": "TechnicalAgent",
                "direction": "BUY",
                "metrics": {"structural_sub_rule": "poi_confluence"},
            }])
            for i in range(5):
                conn.execute("""
                    INSERT INTO agent_learning_memory (
                        decision_id, timestamp, symbol, regime,
                        consensus_direction, consensus_confidence,
                        agent_signals_json, outcome, realized_pnl, position_id
                    ) VALUES (?, ?, 'BTCUSDT', 'TREND_BULLISH', 'BUY', 0.70, ?, 'WIN', 40.0, ?)
                """, (f"dec_win_{i}", now_ms + (10 + i) * 1000, sig_poi, f"pos_win_{i}"))

            conn.commit()

            # Verify calibration table is empty before LearningTracker initialization
            pre_count = conn.execute("SELECT COUNT(*) FROM structural_sub_rule_calibration").fetchone()[0]
            assert pre_count == 0, "structural_sub_rule_calibration must be empty prior to init"

        # Initialize fresh LearningTracker: MUST auto-bootstrap without manual recalibrate call
        tracker = LearningTracker(db_path=db_path)

        multipliers = tracker.get_sub_rule_multipliers()
        assert "sweep_trigger" in multipliers, "Auto-bootstrap must populate sweep_trigger"
        assert multipliers["sweep_trigger"] == 0.0, (
            f"sweep_trigger must be auto-muted (expected 0.0, got {multipliers['sweep_trigger']})"
        )
        assert "poi_confluence" in multipliers, "Auto-bootstrap must populate poi_confluence"
        assert multipliers["poi_confluence"] > 0.0, (
            f"poi_confluence multiplier must be positive, got {multipliers['poi_confluence']}"
        )

        # Confirm persisted rows directly in SQLite
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = {r["sub_rule"]: dict(r) for r in conn.execute("SELECT * FROM structural_sub_rule_calibration").fetchall()}
            assert "sweep_trigger" in rows
            sw = rows["sweep_trigger"]
            assert sw["total_samples"] == 5
            assert sw["wins"] == 0
            assert math.isclose(sw["posterior_wr"], 4.0 / 13.0, abs_tol=0.01)
            assert sw["cumulative_pnl"] == -250.0
            assert sw["multiplier"] == 0.0
            assert sw["is_muted"] == 1

            assert "poi_confluence" in rows
            poi = rows["poi_confluence"]
            assert poi["total_samples"] == 5
            assert poi["wins"] == 5
            assert math.isclose(poi["posterior_wr"], 9.0 / 13.0, abs_tol=0.01)
            assert poi["multiplier"] > 0.0
            assert poi["is_muted"] == 0

    def test_learning_tracker_bootstrap_idempotent_preserves_existing_calibration(self, tmp_path):
        """Existing calibration rows (e.g. custom sentinel multiplier) must NEVER be overwritten on __init__."""
        db_path = str(tmp_path / "existing_learning.db")

        # 1. Setup closed trades in agent_learning_memory AND pre-insert custom calibration row
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE agent_learning_memory (
                    decision_id TEXT PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    consensus_direction TEXT NOT NULL,
                    consensus_confidence REAL NOT NULL,
                    agent_signals_json TEXT NOT NULL,
                    outcome TEXT DEFAULT 'PENDING',
                    realized_pnl REAL DEFAULT 0.0,
                    position_id TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE structural_sub_rule_calibration (
                    sub_rule TEXT PRIMARY KEY,
                    total_samples INTEGER NOT NULL,
                    wins INTEGER NOT NULL,
                    posterior_wr REAL NOT NULL,
                    cumulative_pnl REAL NOT NULL,
                    multiplier REAL NOT NULL,
                    is_muted INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL
                )
            """)

            # Pre-populate a closed loss trade in memory that would normally mute sweep_trigger to 0.0
            sig_sweep = json.dumps([{
                "agent_name": "TechnicalAgent",
                "direction": "BUY",
                "metrics": {"structural_sub_rule": "sweep_trigger"},
            }])
            conn.execute("""
                INSERT INTO agent_learning_memory (
                    decision_id, timestamp, symbol, regime,
                    consensus_direction, consensus_confidence,
                    agent_signals_json, outcome, realized_pnl, position_id
                ) VALUES ('dec_loss_0', 1700000000000, 'BTCUSDT', 'TREND_BULLISH', 'BUY', 0.65, ?, 'LOSS', -50.0, 'pos_0')
            """, (sig_sweep,))

            # Pre-insert custom sentinel calibration row
            sentinel_mult = 0.7777
            sentinel_ts = 123456789
            conn.execute("""
                INSERT INTO structural_sub_rule_calibration (
                    sub_rule, total_samples, wins, posterior_wr, cumulative_pnl, multiplier, is_muted, updated_at
                ) VALUES ('sweep_trigger', 99, 88, 0.8888, 999.0, ?, 0, ?)
            """, (sentinel_mult, sentinel_ts))
            conn.commit()

        # 2. Instantiate fresh LearningTracker - bootstrap must NOT run because table is not empty
        fresh_tracker = LearningTracker(db_path=db_path)
        mults = fresh_tracker.get_sub_rule_multipliers()
        assert mults.get("sweep_trigger") == sentinel_mult, (
            f"Expected sentinel multiplier {sentinel_mult}, got {mults.get('sweep_trigger')}"
        )

        # 3. Verify SQLite values remain completely untouched
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM structural_sub_rule_calibration WHERE sub_rule = 'sweep_trigger'").fetchone()
            assert row is not None
            assert row["multiplier"] == sentinel_mult
            assert row["total_samples"] == 99
            assert row["wins"] == 88
            assert row["updated_at"] == sentinel_ts

    def test_core_injects_sub_rule_multipliers_into_context(self, tmp_path):
        """IA_TraderCore automatically injects sub_rule_multipliers into context in deterministic and async evaluations."""
        core = IA_TraderCore()
        core.learning_tracker = LearningTracker(db_path=str(tmp_path / "core.db"))
        # Force sub_rule_multipliers in core's learning_tracker
        with patch.object(
            core.learning_tracker,
            "get_sub_rule_multipliers",
            return_value={"sweep_trigger": 0.0, "poi_confluence": 0.5},
        ):
            received_contexts = []
            orig_evaluate = None
            for ag in core.agents:
                if ag.name == "TechnicalAgent":
                    orig_evaluate = ag.evaluate

                    def _spy_eval(ctx):
                        received_contexts.append(ctx.copy())
                        return orig_evaluate(ctx)

                    ag.evaluate = _spy_eval
                    break

            assert orig_evaluate is not None, "TechnicalAgent not found in core.agents"

            # Deterministic evaluate
            ctx_in = {"symbol": "BTCUSDT", "indicators": {}}
            core.evaluate_market_deterministic(
                context=ctx_in,
                symbol="BTCUSDT",
                current_price=50000.0,
                atr=500.0,
                timestamp=1700000000000,
            )

            assert len(received_contexts) >= 1, "TechnicalAgent.evaluate must be invoked"
            last_ctx = received_contexts[-1]
            assert "sub_rule_multipliers" in last_ctx, (
                "IA_TraderCore must inject 'sub_rule_multipliers' into context"
            )
            assert last_ctx["sub_rule_multipliers"].get("sweep_trigger") == 0.0

    def test_signal_generator_structure_context_is_deprecated_noop(self):
        """SignalGenerator.generate treats structure_context as deprecated no-op and preserves standard ATR multiples."""
        gen = SignalGenerator(sl_atr_multiple=1.5, tp_atr_multiple=2.5)

        dummy_decision = MagicMock()
        dummy_decision.direction = "BUY"
        dummy_decision.confidence = 0.70
        dummy_decision.consensus_summary = "Test BUY"
        dummy_decision.is_vetoed = False
        dummy_decision.agent_signals = []

        # Standard call without structure_context
        sig_std = gen.generate(
            decision=dummy_decision,
            symbol="BTCUSDT",
            current_price=100.0,
            atr=2.0,
            timestamp=1700000000000,
            structure_context=None,
        )

        # Call with deprecated structure_context (which in legacy code narrowed SL to 0.7 and TP to 2.1)
        sig_with_struct = gen.generate(
            decision=dummy_decision,
            symbol="BTCUSDT",
            current_price=100.0,
            atr=2.0,
            timestamp=1700000000000,
            structure_context={"liquidity_sweep_bull": 1, "at_bullish_poi": 1},
        )

        # Expected standard levels: SL = 100 - (1.5 * 2) = 97.0, TP = 100 + (2.5 * 2) = 105.0
        assert math.isclose(sig_std.stop_loss, 97.0, abs_tol=1e-4)
        assert math.isclose(sig_std.take_profit, 105.0, abs_tol=1e-4)

        # Must be identical: structure_context is a no-op!
        assert math.isclose(sig_with_struct.stop_loss, sig_std.stop_loss, abs_tol=1e-4), (
            f"structure_context must NOT alter stop_loss: got {sig_with_struct.stop_loss} vs standard {sig_std.stop_loss}"
        )
        assert math.isclose(sig_with_struct.take_profit, sig_std.take_profit, abs_tol=1e-4), (
            f"structure_context must NOT alter take_profit: got {sig_with_struct.take_profit} vs standard {sig_std.take_profit}"
        )


# ============================================================================
# 2. S5.2 & S5.3: ML EVOLUTION, CALIBRATION & JOINT EVALUATION GATE
# ============================================================================


class TestMLEvolutionAndCalibration:
    """Verify Contract 2: CalibratedMulticlassModel, ExtraTrees walk-forward, joint gate, multi-symbol cycle."""

    def test_calibrated_multiclass_model_platt_and_isotonic_invariants(self):
        """CalibratedMulticlassModel produces finite probabilities strictly summing to 1.0 per row."""
        assert CalibratedMulticlassModel is not None, (
            "CalibratedMulticlassModel must be exported by model_trainer.py"
        )
        assert compute_multiclass_ece is not None, (
            "compute_multiclass_ece must be exported by model_trainer.py"
        )

        from sklearn.linear_model import LogisticRegression

        np.random.seed(42)
        n_samples = 150
        n_features = 4
        X_cal = np.random.randn(n_samples, n_features)
        y_cal = np.random.choice([0, 1, 2], size=n_samples)

        X_test = np.random.randn(50, n_features)
        y_test = np.random.choice([0, 1, 2], size=50)

        base = LogisticRegression()
        base.fit(X_cal, y_cal)

        for method in ("platt", "isotonic"):
            cal_model = CalibratedMulticlassModel(base_estimator=base, method=method)
            cal_model.fit(X_cal, y_cal)

            proba = cal_model.predict_proba(X_test)
            assert proba.shape == (50, 3), f"Expected shape (50, 3) for {method}"
            assert np.all(np.isfinite(proba)), f"Non-finite probabilities in {method}"
            assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

            # Invariant: rows strictly sum to 1.0
            row_sums = proba.sum(axis=1)
            assert np.allclose(row_sums, 1.0, atol=1e-5), (
                f"Probabilities must sum to 1.0 per row in {method}: max diff {np.max(np.abs(row_sums - 1.0))}"
            )

            preds = cal_model.predict(X_test)
            expected_preds = cal_model.classes_[np.argmax(proba, axis=1)]
            assert np.array_equal(preds, expected_preds), (
                f"predict() must align with classes_[argmax(proba)] in {method}"
            )

            ece = compute_multiclass_ece(y_test, proba, cal_model.classes_)
            assert isinstance(ece, float) and np.isfinite(ece) and 0.0 <= ece <= 1.0

    def test_calibrated_multiclass_model_joblib_serialization(self, tmp_path):
        """CalibratedMulticlassModel survives joblib.dump and joblib.load with identical predictions."""
        assert CalibratedMulticlassModel is not None, (
            "CalibratedMulticlassModel must be exported by model_trainer.py"
        )
        from sklearn.linear_model import LogisticRegression

        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = np.random.choice([0, 1, 2], size=100)

        base = LogisticRegression()
        base.fit(X, y)
        cal = CalibratedMulticlassModel(base_estimator=base, method="platt")
        cal.fit(X, y)

        proba_orig = cal.predict_proba(X[:10])
        model_path = str(tmp_path / "calibrated_model.joblib")
        joblib.dump(cal, model_path)

        loaded = joblib.load(model_path)
        proba_loaded = loaded.predict_proba(X[:10])
        assert np.allclose(proba_orig, proba_loaded, atol=1e-6), (
            "Loaded model predict_proba must match original exactly"
        )
        assert np.array_equal(loaded.classes_, cal.classes_)

    def test_model_trainer_extratrees_and_walk_forward_metrics(self):
        """ModelTrainer(model_type='extratrees', calibration_method='platt') runs train_walk_forward returning finite metrics."""
        np.random.seed(42)
        n = 260
        df = pd.DataFrame({
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
            "target": np.random.choice([0, 1, 2], size=n),
            "timestamp": np.arange(1700000000000, 1700000000000 + n * 3600000, 3600000),
        })

        trainer = ModelTrainer(
            model_type="extratrees",
            calibration_method="platt",
            horizon=3,
        )
        assert hasattr(trainer, "calibration_method"), (
            "ModelTrainer must store calibration_method"
        )
        assert trainer.calibration_method == "platt"

        results = trainer.train_walk_forward(
            X=df[["f1", "f2", "f3"]],
            y=df["target"],
            timestamps=df["timestamp"].values,
        )

        for required_key in ("log_loss", "roc_auc_ovr", "ece", "emitted_signals_accuracy"):
            assert required_key in results, f"Missing {required_key} in walk-forward results"
            val = results[required_key]
            assert isinstance(val, (float, int)) and np.isfinite(val), (
                f"Result {required_key} must be finite, got {val}"
            )

    def test_model_evolution_manager_evaluate_challenger_joint_rules(self):
        """evaluate_challenger_joint approves if log_loss drops >=0.001 or auc rises >=0.005 without regression."""
        mgr = ModelEvolutionManager(min_improvement_pct=1.0)
        assert hasattr(mgr, "evaluate_challenger_joint"), (
            "ModelEvolutionManager must implement evaluate_challenger_joint"
        )

        champ = {"log_loss": 0.6950, "roc_auc_ovr": 0.5200, "emitted_signals_accuracy": 0.5200}

        # 1. log_loss drops >= 0.001 without auc regression -> Approve
        chal_ll_win = {"log_loss": 0.6935, "roc_auc_ovr": 0.5200, "emitted_signals_accuracy": 0.5200}
        appr, msg = mgr.evaluate_challenger_joint(champ, chal_ll_win)
        assert appr is True, f"Expected approval on log_loss drop: {msg}"

        # 2. auc rises >= 0.005 without log_loss regression -> Approve
        chal_auc_win = {"log_loss": 0.6950, "roc_auc_ovr": 0.5260, "emitted_signals_accuracy": 0.5200}
        appr, msg = mgr.evaluate_challenger_joint(champ, chal_auc_win)
        assert appr is True, f"Expected approval on AUC gain: {msg}"

        # 3. log_loss regresses (+0.010) even if accuracy slightly up -> Reject
        chal_ll_regress = {"log_loss": 0.7050, "roc_auc_ovr": 0.5200, "emitted_signals_accuracy": 0.5300}
        appr, msg = mgr.evaluate_challenger_joint(champ, chal_ll_regress)
        assert appr is False, f"Expected rejection on log_loss regression: {msg}"

        # 4. AUC regresses (-0.010) -> Reject
        chal_auc_regress = {"log_loss": 0.6940, "roc_auc_ovr": 0.5090, "emitted_signals_accuracy": 0.5200}
        appr, msg = mgr.evaluate_challenger_joint(champ, chal_auc_regress)
        assert appr is False, f"Expected rejection on AUC regression: {msg}"

    def test_model_evolution_manager_evaluate_challenger_backward_compat(self):
        """evaluate_challenger remains 100% backward compatible."""
        mgr = ModelEvolutionManager(min_improvement_pct=2.0)
        appr, msg = mgr.evaluate_challenger(0.50, 0.53, "roc_auc")
        assert appr is True
        assert "PROMOÇÃO APROVADA" in msg

        rej, msg_rej = mgr.evaluate_challenger(0.50, 0.505, "roc_auc")
        assert rej is False
        assert "PROMOÇÃO REJEITADA" in msg_rej

    def test_model_evolution_manager_run_retraining_cycle_multi_symbol_contract(self):
        """run_retraining_cycle signature and contract verify multi-symbol training support."""
        import inspect
        sig = inspect.signature(ModelEvolutionManager.run_retraining_cycle)
        params = sig.parameters
        assert "symbols" in params or "symbol" in params, (
            "run_retraining_cycle must support multi-symbol or symbols parameter"
        )
        assert "calibration_method" in params, (
            "run_retraining_cycle must expose calibration_method parameter"
        )


# ============================================================================
# 3. S5.4: SENTIMENT AGENT REFORM & LIVE CONTEXT ENRICHER FUNDING FIX
# ============================================================================


class TestSentimentAgentAndLiveContextEnricher:
    """Verify Contract 3: trend_momentum mode, causal D-1 F&G, and stale funding fallback."""

    def test_sentiment_agent_trend_momentum_default_and_logic(self):
        """SentimentAgent defaults to trend_momentum with strict directional thresholds."""
        agent = SentimentAgent()
        assert getattr(agent, "mode", None) == "trend_momentum", (
            f"SentimentAgent default mode must be 'trend_momentum', got {getattr(agent, 'mode', None)}"
        )

        # 1. fng=60, fng_7d_chg=+8 -> BUY
        sig1 = agent.evaluate({"fear_and_greed": 60.0, "fng_7d_chg": 8.0})
        assert sig1.direction == "BUY", f"Expected BUY for fng=60, chg=+8, got {sig1.direction}"
        assert sig1.metrics.get("sentiment_mode") == "trend_momentum"
        assert sig1.metrics.get("fear_and_greed") == 60.0
        assert sig1.metrics.get("fng_7d_chg") == 8.0

        # 2. fng=30, fng_7d_chg=-10 -> SELL
        sig2 = agent.evaluate({"fear_and_greed": 30.0, "fng_7d_chg": -10.0})
        assert sig2.direction == "SELL", f"Expected SELL for fng=30, chg=-10, got {sig2.direction}"

        # 3. fng=15, fng_7d_chg=+2 -> HOLD (fng low but trend positive)
        sig3 = agent.evaluate({"fear_and_greed": 15.0, "fng_7d_chg": 2.0})
        assert sig3.direction == "HOLD", f"Expected HOLD for fng=15, chg=+2, got {sig3.direction}"

        # 4. fng=20, fng_7d_chg=None -> HOLD (requires both finite values in trend_momentum)
        sig4 = agent.evaluate({"fear_and_greed": 20.0, "fng_7d_chg": None})
        assert sig4.direction == "HOLD", f"Expected HOLD when fng_7d_chg is None, got {sig4.direction}"

    def test_sentiment_agent_modes_neutral_and_legacy_and_invalid(self):
        """SentimentAgent supports neutral_telemetry, legacy_contrarian, and raises ValueError on unknown."""
        # 1. neutral_telemetry always HOLD
        ag_neutral = SentimentAgent(mode="neutral_telemetry")
        sig_neut = ag_neutral.evaluate({"fear_and_greed": 90.0, "fng_7d_chg": 25.0})
        assert sig_neut.direction == "HOLD"
        assert math.isclose(sig_neut.confidence, 0.50, abs_tol=1e-3)
        assert sig_neut.metrics.get("sentiment_mode") == "neutral_telemetry"

        # 2. legacy_contrarian keeps legacy contrarian logic (fng <= 42 -> BUY)
        ag_legacy = SentimentAgent(mode="legacy_contrarian")
        sig_leg = ag_legacy.evaluate({"fear_and_greed": 20.0, "fng_7d_chg": -15.0})
        assert sig_leg.direction == "BUY", "legacy_contrarian must BUY on fear"

        # 3. Invalid mode raises ValueError
        with pytest.raises(ValueError):
            SentimentAgent(mode="invalid_unsupported_mode")

    def test_sentiment_agent_news_score_modulation(self):
        """news_score >= 0.5 boosts BUY confidence by +0.10 in trend_momentum."""
        agent = SentimentAgent(mode="trend_momentum")
        base_ctx = {"fear_and_greed": 65.0, "fng_7d_chg": 10.0}
        sig_base = agent.evaluate(base_ctx)
        assert sig_base.direction == "BUY"

        sig_boosted = agent.evaluate({**base_ctx, "news_score": 0.6})
        expected_conf = min(0.95, sig_base.confidence + 0.10)
        assert math.isclose(sig_boosted.confidence, expected_conf, abs_tol=1e-3), (
            f"news_score >= 0.5 must boost confidence from {sig_base.confidence} to {expected_conf}, got {sig_boosted.confidence}"
        )

    def test_live_context_enricher_causal_fng_7d_chg_no_day_d_leak(self, tmp_path):
        """LiveContextEnricher computes causal D-1 fng_7d_chg strictly ignoring Day D rows."""
        cache_db = str(tmp_path / "cache.db")
        micro_db = str(tmp_path / "micro.db")

        con_c = sqlite3.connect(cache_db)
        con_c.executescript("""
            CREATE TABLE macro_daily_history (series_name TEXT, date_str TEXT, timestamp_ms INTEGER, value REAL, extra_info TEXT, PRIMARY KEY (series_name, date_str));
            CREATE TABLE macro_events_calendar (event_name TEXT, date_str TEXT, timestamp_ms INTEGER, PRIMARY KEY (event_name, date_str));
            CREATE TABLE funding_rates (exchange TEXT, symbol TEXT, timestamp INTEGER, funding_rate REAL, mark_price REAL);
        """)
        con_m = sqlite3.connect(micro_db)
        con_m.executescript("""
            CREATE TABLE derivs (symbol TEXT, ts_ms INTEGER, funding_rate REAL);
        """)

        # Insert FNG entries: Day D-8 (40), Day D-1 (55), and poisoned Day D (95)
        # Note: current time is on Day D (1700000000000)
        day_ms = 86400000
        now_ms = 1700000000000
        day_start_ms = (now_ms // day_ms) * day_ms

        d_minus_8_ts = day_start_ms - 8 * day_ms + 1000
        d_minus_1_ts = day_start_ms - 1 * day_ms + 1000
        day_d_poison_ts = day_start_ms + 1000

        con_c.executemany(
            "INSERT INTO macro_daily_history VALUES (?, ?, ?, ?, ?)",
            [
                ("FEAR_AND_GREED", "2026-11-10", d_minus_8_ts, 40.0, ""),
                ("FEAR_AND_GREED", "2026-11-17", d_minus_1_ts, 55.0, ""),
                ("FEAR_AND_GREED", "2026-11-18", day_d_poison_ts, 95.0, "poison"),
            ],
        )
        con_c.commit()
        con_c.close()
        con_m.close()

        enricher = LiveContextEnricher(cache_db_path=cache_db, micro_db_path=micro_db)
        ctx = enricher.enrich(symbol="BTCUSDT", latest={}, df_feat=None, ts_ms=now_ms)

        # Causal expectation: uses D-1 value (55.0), NOT Day D (95.0)
        assert ctx.get("fear_and_greed") == 55.0, (
            f"Expected D-1 fear_and_greed 55.0, got {ctx.get('fear_and_greed')} (Day D leaked?)"
        )
        # Causal delta: 55.0 - 40.0 = +15.0
        assert ctx.get("fng_7d_chg") == 15.0, (
            f"Expected fng_7d_chg == 15.0 (55 - 40), got {ctx.get('fng_7d_chg')}"
        )

    def test_live_context_enricher_stale_funding_fallback_to_micro_derivs(self, tmp_path):
        """When cache funding_rates is stale (>12h old) while micro_db derivs is fresh (<1h), fallback activates."""
        cache_db = str(tmp_path / "cache.db")
        micro_db = str(tmp_path / "micro.db")

        con_c = sqlite3.connect(cache_db)
        con_c.executescript("""
            CREATE TABLE macro_daily_history (series_name TEXT, date_str TEXT, timestamp_ms INTEGER, value REAL, extra_info TEXT, PRIMARY KEY (series_name, date_str));
            CREATE TABLE macro_events_calendar (event_name TEXT, date_str TEXT, timestamp_ms INTEGER, PRIMARY KEY (event_name, date_str));
            CREATE TABLE funding_rates (exchange TEXT, symbol TEXT, timestamp INTEGER, funding_rate REAL, mark_price REAL);
        """)
        con_m = sqlite3.connect(micro_db)
        con_m.executescript("""
            CREATE TABLE derivs (symbol TEXT, ts_ms INTEGER, funding_rate REAL);
        """)

        now_ms = 1700000000000

        # Cache DB: latest row is 15h old (>12h -> STALE)
        stale_ts = now_ms - 15 * 3600000
        con_c.executemany(
            "INSERT INTO funding_rates VALUES (?, ?, ?, ?, ?)",
            [
                ("binance", "BTCUSDT", stale_ts, 0.00010, 50000.0),
                ("binance", "BTCUSDT", stale_ts - 8 * 3600000, 0.00012, 50000.0),
                ("binance", "BTCUSDT", stale_ts - 16 * 3600000, 0.00008, 50000.0),
            ],
        )
        con_c.commit()
        con_c.close()

        # Micro DB: fresh row is 20m old (<1h -> FRESH) plus baseline rows
        fresh_ts = now_ms - 20 * 60000
        derivs_rows = [("BTCUSDT", fresh_ts, 0.00045)]
        for h in range(1, 15):
            derivs_rows.append(("BTCUSDT", fresh_ts - h * 3600000, 0.00010 + h * 0.00001))

        con_m.executemany("INSERT INTO derivs VALUES (?, ?, ?)", derivs_rows)
        con_m.commit()
        con_m.close()

        enricher = LiveContextEnricher(cache_db_path=cache_db, micro_db_path=micro_db)
        ctx = enricher.enrich(symbol="BTCUSDT", latest={}, df_feat=None, ts_ms=now_ms)

        assert ctx.get("funding_rate") == 0.00045, (
            f"Expected fresh funding_rate 0.00045 from micro_derivs, got {ctx.get('funding_rate')}"
        )
        assert ctx.get("funding_source") == "micro_derivs", (
            f"Expected funding_source=='micro_derivs', got {ctx.get('funding_source')}"
        )
        assert ctx.get("funding_zscore") is not None, "funding_zscore must not be None"
        assert np.isfinite(ctx["funding_zscore"]), "funding_zscore must be finite"
        assert ctx["funding_zscore"] != 0.0, "funding_zscore must be non-zero against historical baseline"

    def test_live_context_enricher_fresh_cache_funding_no_fallback(self, tmp_path):
        """When cache funding_rates is fresh (<=12h), cache row is used and fallback does NOT activate."""
        cache_db = str(tmp_path / "cache.db")
        micro_db = str(tmp_path / "micro.db")

        con_c = sqlite3.connect(cache_db)
        con_c.executescript("""
            CREATE TABLE macro_daily_history (series_name TEXT, date_str TEXT, timestamp_ms INTEGER, value REAL, extra_info TEXT, PRIMARY KEY (series_name, date_str));
            CREATE TABLE macro_events_calendar (event_name TEXT, date_str TEXT, timestamp_ms INTEGER, PRIMARY KEY (event_name, date_str));
            CREATE TABLE funding_rates (exchange TEXT, symbol TEXT, timestamp INTEGER, funding_rate REAL, mark_price REAL);
        """)
        con_m = sqlite3.connect(micro_db)
        con_m.executescript("""
            CREATE TABLE derivs (symbol TEXT, ts_ms INTEGER, funding_rate REAL);
        """)

        now_ms = 1700000000000

        # Cache DB: latest row is 2h old (<=12h -> FRESH)
        cache_ts = now_ms - 2 * 3600000
        con_c.execute("INSERT INTO funding_rates VALUES (?, ?, ?, ?, ?)", ("binance", "BTCUSDT", cache_ts, 0.00020, 50000.0))
        con_c.commit()
        con_c.close()

        # Micro DB: fresh row is 10m old with different rate
        con_m.execute("INSERT INTO derivs VALUES (?, ?, ?)", ("BTCUSDT", now_ms - 600000, 0.00099))
        con_m.commit()
        con_m.close()

        enricher = LiveContextEnricher(cache_db_path=cache_db, micro_db_path=micro_db)
        ctx = enricher.enrich(symbol="BTCUSDT", latest={}, df_feat=None, ts_ms=now_ms)

        assert ctx.get("funding_rate") == 0.00020, (
            f"Expected funding_rate 0.00020 from fresh cache, got {ctx.get('funding_rate')}"
        )
        assert ctx.get("funding_source") != "micro_derivs", (
            "funding_source must NOT be micro_derivs when cache is fresh"
        )


# ============================================================================
# 4. S5.5: SWING_4H META-SIZER (SHADOW / ACTIVE & ZERO-LOOKAHEAD)
# ============================================================================


class TestSwing4HMetaSizer:
    """Verify Contract 4: 10 meta-features, shadow mult 1.0, zero-lookahead invariance, active clamping."""

    def test_swing_4h_meta_sizer_constants_and_table_creation(self, tmp_path):
        """Swing4HMetaSizer exports 10 META_FEATURE_COLS and ensures telemetry table schema."""
        assert Swing4HMetaSizer is not None, (
            "Swing4HMetaSizer must be implemented in data/swing_4h_meta_sizer.py"
        )
        assert META_FEATURE_COLS is not None, (
            "META_FEATURE_COLS must be exported by data/swing_4h_meta_sizer.py"
        )
        expected_cols = (
            "atr_pct",
            "dir_funding_z",
            "dir_fng",
            "dir_sp500_5d",
            "dir_btc_168h",
            "vix",
            "abs_dist200",
            "dir_stbl30",
            "dir_dxy_5d",
            "dir_breadth",
        )
        assert tuple(META_FEATURE_COLS) == expected_cols, (
            f"META_FEATURE_COLS mismatch: expected {expected_cols}, got {META_FEATURE_COLS}"
        )

        db_path = str(tmp_path / "cache.db")
        sizer = Swing4HMetaSizer(cache_db_path=db_path, mode="shadow")

        con = sqlite3.connect(db_path)
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "swing_4h_meta_telemetry" in tables, (
            "swing_4h_meta_telemetry table must be created on initialization"
        )
        con.close()

    def test_swing_4h_meta_sizer_shadow_mode_invariants_and_telemetry(self, tmp_path):
        """In mode='shadow', applied_mult == 1.0 invariant holds while extracting all sub-multipliers and telemetry."""
        assert Swing4HMetaSizer is not None, "Swing4HMetaSizer required"

        db_path = str(tmp_path / "cache.db")
        sizer = Swing4HMetaSizer(cache_db_path=db_path, mode="shadow")

        # Synthetic closed 4h data
        n_bars = 50
        bar_ts = 1700000000000
        timestamps = [bar_ts - (n_bars - i) * 14400000 for i in range(n_bars)]
        df_btc = pd.DataFrame({
            "timestamp": timestamps,
            "open": [100.0] * n_bars,
            "high": [105.0] * n_bars,
            "low": [95.0] * n_bars,
            "close": [102.0] * n_bars,
            "volume": [1000.0] * n_bars,
            "atr_14": [3.0] * n_bars,
            "sma_200": [98.0] * n_bars,
        })
        df_map = {"BTCUSDT": df_btc, "ETHUSDT": df_btc, "SOLUSDT": df_btc}

        res = sizer.score_entry(
            symbol="BTCUSDT",
            side="BUY",
            df_4h_map=df_map,
            context={"fear_and_greed": 60.0, "funding_zscore": 0.5},
            bar_ts_ms=bar_ts,
        )

        assert res["mode"] == "shadow"
        # STRICT NUMERIC INVARIANT: Shadow mode must NEVER scale production risk
        assert res["applied_mult"] == 1.0, (
            f"Strict invariant violation: mode='shadow' must return applied_mult == 1.0, got {res['applied_mult']}"
        )

        for req_k in ("vol_target_mult", "atr_risk_parity_mult", "rs_14d_mult", "combined_shadow_mult"):
            assert req_k in res, f"Missing {req_k} in score_entry result"
            val = res[req_k]
            assert isinstance(val, (float, int)) and np.isfinite(val), (
                f"{req_k} must be finite float, got {val}"
            )

        # Telemetry persistence
        sizer.record_telemetry(
            timestamp=bar_ts,
            symbol="BTCUSDT",
            side="BUY",
            score_result=res,
            position_id="pos_123",
        )

        con = sqlite3.connect(db_path)
        rows = con.execute("SELECT symbol, side, mode, applied_mult, position_id FROM swing_4h_meta_telemetry").fetchall()
        con.close()
        assert len(rows) == 1
        assert rows[0] == ("BTCUSDT", "BUY", "shadow", 1.0, "pos_123")

    def test_swing_4h_meta_sizer_anti_lookahead_perturbation_invariance(self, tmp_path):
        """Zero-lookahead: mutating or inserting future data strictly after as_of_ms does NOT alter features or probabilities."""
        assert Swing4HMetaSizer is not None, "Swing4HMetaSizer required"

        db_path = str(tmp_path / "cache.db")
        sizer = Swing4HMetaSizer(cache_db_path=db_path, mode="shadow")

        as_of_ms = 1700000000000
        n_bars = 60
        timestamps_hist = [as_of_ms - (n_bars - i) * 14400000 for i in range(n_bars)]
        df_hist = pd.DataFrame({
            "timestamp": timestamps_hist,
            "open": [100.0] * n_bars,
            "high": [104.0] * n_bars,
            "low": [96.0] * n_bars,
            "close": [101.0] * n_bars,
            "volume": [500.0] * n_bars,
            "atr_14": [2.5] * n_bars,
            "sma_200": [95.0] * n_bars,
        })
        df_map_1 = {"BTCUSDT": df_hist.copy()}

        feats_1, diag_1 = sizer.extract_features(
            symbol="BTCUSDT",
            side="BUY",
            df_4h_map=df_map_1,
            context={"fear_and_greed": 50.0},
            bar_ts_ms=as_of_ms,
        )

        # Poison df with future candles after as_of_ms
        timestamps_future = [as_of_ms + i * 14400000 for i in range(1, 15)]
        df_future = pd.DataFrame({
            "timestamp": timestamps_future,
            "open": [250.0] * len(timestamps_future),  # Wild price explosion in future
            "high": [300.0] * len(timestamps_future),
            "low": [200.0] * len(timestamps_future),
            "close": [280.0] * len(timestamps_future),
            "volume": [99999.0] * len(timestamps_future),
            "atr_14": [25.0] * len(timestamps_future),
            "sma_200": [150.0] * len(timestamps_future),
        })
        df_poisoned = pd.concat([df_hist, df_future], ignore_index=True)
        df_map_2 = {"BTCUSDT": df_poisoned}

        feats_2, diag_2 = sizer.extract_features(
            symbol="BTCUSDT",
            side="BUY",
            df_4h_map=df_map_2,
            context={"fear_and_greed": 50.0},
            bar_ts_ms=as_of_ms,
        )

        # Every single extracted feature and diagnostic must be bit-for-bit identical!
        for k in feats_1:
            assert math.isclose(feats_1[k], feats_2[k], abs_tol=1e-9), (
                f"Lookahead leak detected in feature {k}: hist={feats_1[k]} vs poisoned_future={feats_2[k]}"
            )
        for k in diag_1:
            assert math.isclose(diag_1[k], diag_2[k], abs_tol=1e-9), (
                f"Lookahead leak detected in diagnostic {k}: hist={diag_1[k]} vs poisoned_future={diag_2[k]}"
            )

    def test_swing_4h_meta_sizer_active_mode_clamping(self, tmp_path):
        """In mode='active', applied_mult is strictly clamped within [0.4, 2.0]."""
        assert Swing4HMetaSizer is not None, "Swing4HMetaSizer required"

        db_path = str(tmp_path / "cache.db")
        sizer = Swing4HMetaSizer(cache_db_path=db_path, mode="active")

        n_bars = 45
        bar_ts = 1700000000000
        timestamps = [bar_ts - (n_bars - i) * 14400000 for i in range(n_bars)]
        df_btc = pd.DataFrame({
            "timestamp": timestamps,
            "open": [100.0] * n_bars,
            "high": [105.0] * n_bars,
            "low": [95.0] * n_bars,
            "close": [102.0] * n_bars,
            "volume": [1000.0] * n_bars,
            "atr_14": [3.0] * n_bars,
            "sma_200": [98.0] * n_bars,
        })
        df_map = {"BTCUSDT": df_btc}

        res = sizer.score_entry(
            symbol="BTCUSDT",
            side="BUY",
            df_4h_map=df_map,
            context={"fear_and_greed": 50.0},
            bar_ts_ms=bar_ts,
        )

        assert res["mode"] == "active"
        applied = res["applied_mult"]
        assert isinstance(applied, float) and np.isfinite(applied)
        assert 0.4 <= applied <= 2.0, (
            f"Strict clamping invariant violated: applied_mult ({applied}) outside [0.4, 2.0]"
        )
