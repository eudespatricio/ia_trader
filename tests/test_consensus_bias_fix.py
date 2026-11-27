"""
Independent behavioral test suite for ConsensusJudge bias fixes and integration
with LearningTracker and TradingCore.

Tests strictly against public interfaces and the behavioral contract without
inspecting or coupling to internal implementation details.
"""

from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import AgentSignal
from consensus_judge import ConsensusJudge
from core import TradingCore
from learning_tracker import LearningTracker

BULL_TREND_WEIGHTS = {
    "RiskSentinelAgent": 1.5,
    "TechnicalAgent": 1.0,
    "OrderFlowAgent": 0.8,
    "SentimentAgent": 0.6,
    "MacroAgent": 0.7,
    "OnChainAgent": 0.5,
}
SIDEWAYS_TRACKER_WEIGHTS = {
    "RiskSentinelAgent": 1.5,
    "TechnicalAgent": 1.5,
    "OrderFlowAgent": 0.5,
    "SentimentAgent": 0.6,
    "MacroAgent": 0.5,
    "OnChainAgent": 0.5,
}
_FLIP = {"BUY": "SELL", "SELL": "BUY", "HOLD": "HOLD", "ABSTAIN": "ABSTAIN"}


def _sig(name: str, direction: str, confidence: float, weight: float = 1.0, is_veto: bool = False) -> AgentSignal:
    return AgentSignal(
        agent_name=name, direction=direction, confidence=confidence,
        weight=weight, rationale=f"{name} {direction} ({confidence:.2f})", is_veto=is_veto,
    )


def _sigs(*rows: tuple) -> list[AgentSignal]:
    return [_sig(*row) for row in rows]


def _mirror(signals: list[AgentSignal]) -> list[AgentSignal]:
    return [_sig(s.agent_name, _FLIP[s.direction], s.confidence, weight=s.weight, is_veto=s.is_veto) for s in signals]


# ---------------------------------------------------------------------------
# 1. Non-veto RiskSentinel exclusion vs Veto priority
# ---------------------------------------------------------------------------
class TestRiskSentinelVetoAndExclusion:
    def test_non_veto_risk_sentinel_hold_does_not_dilute_score_or_confidence(self):
        core = _sigs(("TechnicalAgent", "BUY", 0.75, 1.0), ("OrderFlowAgent", "BUY", 0.65, 0.8))
        sentinel = _sig("RiskSentinelAgent", "HOLD", 0.50, weight=1.5, is_veto=False)
        judge = ConsensusJudge()
        base = judge.judge_deterministic(core)
        with_s = judge.judge_deterministic([sentinel] + core)
        assert with_s.is_vetoed is False and with_s.direction == base.direction == "BUY"
        assert with_s.aggregated_score == pytest.approx(base.aggregated_score, abs=1e-12)
        assert with_s.confidence == pytest.approx(base.confidence, abs=1e-12)

    def test_non_veto_risk_sentinel_excluded_even_with_weights_override(self):
        core = _sigs(("TechnicalAgent", "SELL", 0.80, 1.0), ("OrderFlowAgent", "SELL", 0.70, 0.8))
        sentinel = _sig("RiskSentinelAgent", "HOLD", 0.50, weight=1.5, is_veto=False)
        override = {"RiskSentinelAgent": 1.5, "TechnicalAgent": 1.0, "OrderFlowAgent": 0.8}
        judge = ConsensusJudge()
        base = judge.judge_deterministic(core, weights_override=override)
        with_s = judge.judge_deterministic([sentinel] + core, weights_override=override)
        assert with_s.direction == "SELL"
        assert with_s.aggregated_score == pytest.approx(base.aggregated_score, abs=1e-12)
        assert with_s.confidence == pytest.approx(base.confidence, abs=1e-12)

    @pytest.mark.parametrize("direction", ["BUY", "SELL"])
    def test_risk_sentinel_veto_has_absolute_priority_over_unanimous_committee(self, direction: str):
        signals = _sigs(
            ("RiskSentinelAgent", "HOLD", 1.0, 1.5, True),
            ("TechnicalAgent", direction, 0.95, 1.0), ("OrderFlowAgent", direction, 0.95, 0.8),
            ("SentimentAgent", direction, 0.90, 0.6), ("MacroAgent", direction, 0.90, 0.7),
            ("OnChainAgent", direction, 0.90, 0.5),
        )
        for judge in (ConsensusJudge(), ConsensusJudge(veto_only_agents=set())):
            dec = judge.judge_deterministic(signals, weights_override=BULL_TREND_WEIGHTS)
            assert dec.direction == "HOLD" and dec.is_vetoed is True
            assert dec.confidence == pytest.approx(1.0) and dec.aggregated_score == pytest.approx(0.0)

    def test_custom_veto_only_agents_empty_restores_legacy_dilution(self):
        core = _sigs(("TechnicalAgent", "BUY", 0.80, 1.0), ("OrderFlowAgent", "BUY", 0.70, 0.8))
        sentinel = _sig("RiskSentinelAgent", "HOLD", 0.50, weight=1.5, is_veto=False)
        override = {"RiskSentinelAgent": 1.5, "TechnicalAgent": 1.0, "OrderFlowAgent": 0.8}
        judge = ConsensusJudge(veto_only_agents=set())
        no_s = judge.judge_deterministic(core, weights_override=override)
        diluted = judge.judge_deterministic([sentinel] + core, weights_override=override)
        assert diluted.is_vetoed is False and diluted.aggregated_score < no_s.aggregated_score
        assert diluted.aggregated_score == pytest.approx((1.0 * 0.80 + 0.8 * 0.70) / (1.5 + 1.0 + 0.8), abs=1e-12)

    def test_custom_veto_only_agents_targets_configured_agent_only(self):
        core = _sigs(("TechnicalAgent", "BUY", 0.80, 1.0), ("OrderFlowAgent", "BUY", 0.80, 0.8))
        custom = _sig("CustomSentinel", "HOLD", 0.50, weight=2.0, is_veto=False)
        risk = _sig("RiskSentinelAgent", "HOLD", 0.50, weight=1.5, is_veto=False)
        judge = ConsensusJudge(veto_only_agents={"CustomSentinel"})
        base = judge.judge_deterministic(core)
        assert judge.judge_deterministic([custom] + core).aggregated_score == pytest.approx(base.aggregated_score, abs=1e-12)
        assert judge.judge_deterministic([risk] + core).aggregated_score < base.aggregated_score


# ---------------------------------------------------------------------------
# 2. Proportional contextual weight cap & ratio validation
# ---------------------------------------------------------------------------
class TestProportionalContextualWeightCap:
    RAW_WEIGHTS = {"TechnicalAgent": 1.0, "OrderFlowAgent": 0.8, "SentimentAgent": 0.6, "MacroAgent": 0.7, "OnChainAgent": 0.5}

    def test_contextual_weights_capped_proportionally_affect_score_and_confidence(self):
        signals = _sigs(
            ("TechnicalAgent", "BUY", 0.80, 1.0), ("OrderFlowAgent", "BUY", 0.75, 0.8),
            ("SentimentAgent", "SELL", 0.60, 0.6), ("MacroAgent", "SELL", 0.85, 0.7), ("OnChainAgent", "SELL", 0.78, 0.5),
        )
        capped = ConsensusJudge(max_contextual_weight_ratio=0.35).judge_deterministic(signals, weights_override=self.RAW_WEIGHTS)
        prescaled = {**self.RAW_WEIGHTS, "SentimentAgent": 0.6 * 0.35, "MacroAgent": 0.7 * 0.35, "OnChainAgent": 0.5 * 0.35}
        manual = ConsensusJudge(max_contextual_weight_ratio=None).judge_deterministic(signals, weights_override=prescaled)
        uncapped = ConsensusJudge(max_contextual_weight_ratio=None).judge_deterministic(signals, weights_override=self.RAW_WEIGHTS)

        assert capped.direction == manual.direction
        assert capped.aggregated_score == pytest.approx(manual.aggregated_score, abs=1e-12)
        assert capped.confidence == pytest.approx(manual.confidence, abs=1e-12)
        assert capped.aggregated_score > uncapped.aggregated_score and capped.confidence > uncapped.confidence

    def test_contextual_cap_only_scales_active_contextual_agents(self):
        signals = _sigs(
            ("TechnicalAgent", "BUY", 0.80, 1.0), ("OrderFlowAgent", "BUY", 0.75, 0.8),
            ("SentimentAgent", "HOLD", 0.50, 0.6), ("MacroAgent", "SELL", 0.85, 0.7), ("OnChainAgent", "SELL", 0.78, 0.5),
        )
        scale = (0.35 * 1.8) / (0.7 + 0.5)
        capped = ConsensusJudge(max_contextual_weight_ratio=0.35).judge_deterministic(signals, weights_override=self.RAW_WEIGHTS)
        expected = ConsensusJudge(max_contextual_weight_ratio=None).judge_deterministic(
            signals, weights_override={**self.RAW_WEIGHTS, "MacroAgent": 0.7 * scale, "OnChainAgent": 0.5 * scale}
        )
        assert capped.direction == expected.direction == "BUY"
        assert capped.aggregated_score == pytest.approx(expected.aggregated_score, abs=1e-12)
        assert capped.confidence == pytest.approx(expected.confidence, abs=1e-12)

        capped_na = ConsensusJudge(max_contextual_weight_ratio=0.35, allow_contextual_abstain=False).judge_deterministic(
            signals, weights_override=self.RAW_WEIGHTS
        )
        expected_na = ConsensusJudge(max_contextual_weight_ratio=None, allow_contextual_abstain=False).judge_deterministic(
            signals, weights_override={**self.RAW_WEIGHTS, "SentimentAgent": 0.6 * 0.35, "MacroAgent": 0.7 * 0.35, "OnChainAgent": 0.5 * 0.35}
        )
        assert capped_na.aggregated_score == pytest.approx(expected_na.aggregated_score, abs=1e-12)
        assert capped_na.confidence == pytest.approx(expected_na.confidence, abs=1e-12)

    def test_contextual_cap_does_not_inflate_when_below_ratio(self):
        signals = _sigs(("TechnicalAgent", "BUY", 0.70, 1.0), ("OrderFlowAgent", "BUY", 0.70, 1.0), ("MacroAgent", "SELL", 0.80, 0.4))
        weights = {"TechnicalAgent": 1.0, "OrderFlowAgent": 1.0, "MacroAgent": 0.4}
        capped = ConsensusJudge(max_contextual_weight_ratio=0.35).judge_deterministic(signals, weights_override=weights)
        uncapped = ConsensusJudge(max_contextual_weight_ratio=None).judge_deterministic(signals, weights_override=weights)
        assert capped.aggregated_score == pytest.approx(uncapped.aggregated_score, abs=1e-12)
        assert capped.confidence == pytest.approx(uncapped.confidence, abs=1e-12)

    @pytest.mark.parametrize("ratio", [0.35, None, 0.0])
    def test_buy_sell_mirror_symmetry_under_contextual_cap(self, ratio):
        buy_signals = _sigs(
            ("RiskSentinelAgent", "HOLD", 0.50, 1.5), ("TechnicalAgent", "BUY", 0.82, 1.0),
            ("OrderFlowAgent", "BUY", 0.73, 0.8), ("SentimentAgent", "SELL", 0.62, 0.6),
            ("MacroAgent", "SELL", 0.85, 0.7), ("OnChainAgent", "BUY", 0.65, 0.5),
        )
        judge = ConsensusJudge(max_contextual_weight_ratio=ratio)
        d_buy = judge.judge_deterministic(buy_signals, weights_override=BULL_TREND_WEIGHTS)
        d_sell = judge.judge_deterministic(_mirror(buy_signals), weights_override=BULL_TREND_WEIGHTS)
        assert d_buy.aggregated_score == pytest.approx(-d_sell.aggregated_score, abs=1e-12)
        assert d_buy.confidence == pytest.approx(d_sell.confidence, abs=1e-12)
        assert d_sell.direction == _FLIP[d_buy.direction]

    def test_max_contextual_weight_ratio_zero_eliminates_contextual_influence(self):
        primary_only = _sigs(("TechnicalAgent", "BUY", 0.80, 1.0), ("OrderFlowAgent", "BUY", 0.75, 0.8))
        with_ctx = primary_only + _sigs(("SentimentAgent", "SELL", 0.90, 0.6), ("MacroAgent", "SELL", 0.95, 0.7), ("OnChainAgent", "SELL", 0.95, 0.5))
        judge_zero = ConsensusJudge(max_contextual_weight_ratio=0.0)
        base = judge_zero.judge_deterministic(primary_only, weights_override=BULL_TREND_WEIGHTS)
        res = judge_zero.judge_deterministic(with_ctx, weights_override=BULL_TREND_WEIGHTS)
        assert res.direction == "BUY"
        assert res.aggregated_score == pytest.approx(base.aggregated_score, abs=1e-12)
        assert res.confidence == pytest.approx(base.confidence, abs=1e-12)

    def test_zero_ratio_quorum_and_dissent_isolation(self):
        # Zero-weight MacroAgent must NOT satisfy min_directional_agents=2 quorum
        quorum_signals = _sigs(
            ("TechnicalAgent", "BUY", 0.80, 1.0),
            ("OrderFlowAgent", "HOLD", 0.50, 0.8),
            ("MacroAgent", "BUY", 0.85, 0.7),
        )
        quorum_dec = ConsensusJudge(max_contextual_weight_ratio=0.0, min_directional_agents=2).judge_deterministic(
            quorum_signals, weights_override={"TechnicalAgent": 1.0, "OrderFlowAgent": 0.8, "MacroAgent": 0.7}
        )
        assert quorum_dec.direction == "HOLD" and quorum_dec.is_vetoed is False

        # Zero-weight MacroAgent SELL must NOT appear as dissident ('divergente') in summary
        dissent_signals = _sigs(
            ("TechnicalAgent", "BUY", 0.80, 1.0),
            ("OrderFlowAgent", "BUY", 0.75, 0.8),
            ("MacroAgent", "SELL", 0.85, 0.7),
        )
        dissent_dec = ConsensusJudge(max_contextual_weight_ratio=0.0).judge_deterministic(
            dissent_signals, weights_override={"TechnicalAgent": 1.0, "OrderFlowAgent": 0.8, "MacroAgent": 0.7}
        )
        assert dissent_dec.direction == "BUY"
        assert "Macro" not in dissent_dec.consensus_summary
        assert "divergente" not in dissent_dec.consensus_summary.lower()

    def test_only_contextual_agents_active_does_not_zero_or_divide_by_zero(self):
        ctx_only = _sigs(("RiskSentinelAgent", "HOLD", 0.50, 1.5), ("MacroAgent", "SELL", 0.80, 0.7), ("OnChainAgent", "SELL", 0.75, 0.5))
        weights = {"RiskSentinelAgent": 1.5, "MacroAgent": 0.7, "OnChainAgent": 0.5}
        capped = ConsensusJudge(max_contextual_weight_ratio=0.35).judge_deterministic(ctx_only, weights_override=weights)
        uncapped = ConsensusJudge(max_contextual_weight_ratio=None).judge_deterministic(ctx_only, weights_override=weights)
        assert capped.direction == "SELL" and capped.aggregated_score < -0.30
        assert capped.aggregated_score == pytest.approx(uncapped.aggregated_score, abs=1e-12)
        assert capped.confidence == pytest.approx(uncapped.confidence, abs=1e-12)

    @pytest.mark.parametrize("bad_ratio", [-0.1, float("nan"), float("inf"), float("-inf")])
    def test_invalid_max_contextual_weight_ratio_raises_value_error(self, bad_ratio: float):
        with pytest.raises(ValueError):
            ConsensusJudge(max_contextual_weight_ratio=bad_ratio)


# ---------------------------------------------------------------------------
# 3. Counter-macro BUY feasibility & calibration symmetry
# ---------------------------------------------------------------------------
class TestCounterMacroFeasibilityAndSymmetry:
    def test_bull_trend_strong_counter_macro_buy_and_symmetric_sell(self):
        buy_signals = _sigs(
            ("RiskSentinelAgent", "HOLD", 0.50), ("TechnicalAgent", "BUY", 0.80), ("OrderFlowAgent", "BUY", 0.75),
            ("SentimentAgent", "HOLD", 0.50), ("MacroAgent", "SELL", 0.85), ("OnChainAgent", "SELL", 0.78),
        )
        judge = ConsensusJudge(min_consensus_threshold=0.30)
        d_buy = judge.judge_deterministic(buy_signals, weights_override=BULL_TREND_WEIGHTS, regime="BULL_TREND")
        assert d_buy.direction == "BUY" and d_buy.aggregated_score >= 0.30 and d_buy.confidence >= 0.70

        d_sell = judge.judge_deterministic(_mirror(buy_signals), weights_override=BULL_TREND_WEIGHTS, regime="BULL_TREND")
        assert d_sell.direction == "SELL" and d_sell.aggregated_score <= -0.30 and d_sell.confidence >= 0.70

    def test_bull_trend_weak_counter_macro_buy_stays_hold(self):
        weak = _sigs(
            ("RiskSentinelAgent", "HOLD", 0.50), ("TechnicalAgent", "BUY", 0.60), ("OrderFlowAgent", "BUY", 0.60),
            ("SentimentAgent", "HOLD", 0.50), ("MacroAgent", "SELL", 0.85), ("OnChainAgent", "SELL", 0.78),
        )
        dec = ConsensusJudge(min_consensus_threshold=0.30).judge_deterministic(weak, weights_override=BULL_TREND_WEIGHTS, regime="BULL_TREND")
        assert dec.direction == "HOLD" and 0.0 < dec.aggregated_score < 0.30

    def test_sideways_range_with_tracker_weights_and_ece_dampening_approves_buy(self):
        signals = _sigs(
            ("RiskSentinelAgent", "HOLD", 0.50), ("TechnicalAgent", "BUY", 0.85), ("OrderFlowAgent", "BUY", 0.75),
            ("SentimentAgent", "HOLD", 0.50), ("MacroAgent", "SELL", 0.73), ("OnChainAgent", "SELL", 0.78),
        )
        calib = {
            "TechnicalAgent": {"accuracy": 0.60, "ece": 0.12, "total_samples": 5},
            "OrderFlowAgent": {"accuracy": 0.48, "ece": 0.29, "total_samples": 6},
            "MacroAgent": {"accuracy": 0.45, "ece": 0.32, "total_samples": 6},
            "OnChainAgent": {"accuracy": 0.50, "ece": 0.28, "total_samples": 6},
        }
        dec = ConsensusJudge(min_consensus_threshold=0.30).judge_deterministic(
            signals, weights_override=SIDEWAYS_TRACKER_WEIGHTS, regime="SIDEWAYS_RANGE", agent_calibration=calib
        )
        assert dec.direction == "BUY" and dec.aggregated_score >= 0.30 and dec.confidence >= 0.70


# ---------------------------------------------------------------------------
# 4. Primary technical confirmation guard (require_technical_confirmation)
# ---------------------------------------------------------------------------
class TestTechnicalConfirmationGuard:
    PHANTOM_SHORT = _sigs(
        ("RiskSentinelAgent", "HOLD", 0.50, 1.5), ("TechnicalAgent", "HOLD", 0.52, 1.0),
        ("OrderFlowAgent", "SELL", 0.70, 0.8), ("SentimentAgent", "HOLD", 0.50, 0.6),
        ("MacroAgent", "SELL", 0.73, 0.7), ("OnChainAgent", "SELL", 0.78, 0.5),
    )

    def test_phantom_short_and_symmetric_phantom_buy_blocked(self):
        judge = ConsensusJudge(min_consensus_threshold=0.30)
        short_dec = judge.judge_deterministic(self.PHANTOM_SHORT)
        assert short_dec.direction == "HOLD" and short_dec.is_vetoed is False

        phantom_buy = _sigs(
            ("RiskSentinelAgent", "HOLD", 0.50, 1.5), ("TechnicalAgent", "HOLD", 0.52, 1.0),
            ("OrderFlowAgent", "BUY", 0.85, 0.8), ("SentimentAgent", "HOLD", 0.50, 0.6),
            ("MacroAgent", "BUY", 0.85, 0.7), ("OnChainAgent", "BUY", 0.78, 0.5),
        )
        buy_dec = judge.judge_deterministic(phantom_buy)
        assert buy_dec.direction == "HOLD" and buy_dec.is_vetoed is False

    @pytest.mark.parametrize("direction", ["BUY", "SELL"])
    def test_technical_agent_absent_does_not_block_synthetic_consensus(self, direction: str):
        signals = _sigs(("A1", direction, 0.85, 1.0), ("A2", direction, 0.80, 1.0))
        assert ConsensusJudge(min_consensus_threshold=0.30).judge_deterministic(signals).direction == direction

    def test_require_technical_confirmation_false_permits_flow_plus_contextual_sell(self):
        judge = ConsensusJudge(min_consensus_threshold=0.30, require_technical_confirmation=False)
        dec = judge.judge_deterministic(self.PHANTOM_SHORT)
        assert dec.direction == "SELL" and dec.aggregated_score <= -0.30


# ---------------------------------------------------------------------------
# 5. Primary opposition guard (block_primary_opposition)
# ---------------------------------------------------------------------------
class TestPrimaryOppositionGuard:
    OPPOSED_SELL = _sigs(
        ("TechnicalAgent", "SELL", 0.91, 1.0), ("OrderFlowAgent", "BUY", 0.69, 0.8),
        ("MacroAgent", "SELL", 0.85, 0.7), ("OnChainAgent", "SELL", 0.78, 0.5),
    )

    def test_primary_opposition_blocks_both_sell_and_buy(self):
        judge = ConsensusJudge(min_consensus_threshold=0.30)
        d_sell = judge.judge_deterministic(self.OPPOSED_SELL)
        d_buy = judge.judge_deterministic(_mirror(self.OPPOSED_SELL))
        assert d_sell.direction == "HOLD" and d_sell.is_vetoed is False
        assert d_buy.direction == "HOLD" and d_buy.is_vetoed is False

    def test_orderflow_hold_is_not_primary_opposition(self):
        signals = _sigs(
            ("TechnicalAgent", "BUY", 0.90, 1.0), ("OrderFlowAgent", "HOLD", 0.50, 0.8),
            ("MacroAgent", "BUY", 0.85, 0.7), ("OnChainAgent", "BUY", 0.78, 0.5),
        )
        dec = ConsensusJudge(min_consensus_threshold=0.30).judge_deterministic(signals)
        assert dec.direction == "BUY" and dec.aggregated_score >= 0.30

    def test_block_primary_opposition_false_follows_weighted_consensus(self):
        judge = ConsensusJudge(min_consensus_threshold=0.30, block_primary_opposition=False)
        d_sell = judge.judge_deterministic(self.OPPOSED_SELL)
        d_buy = judge.judge_deterministic(_mirror(self.OPPOSED_SELL))
        assert d_sell.direction == "SELL" and d_sell.aggregated_score <= -0.30
        assert d_buy.direction == "BUY" and d_buy.aggregated_score >= 0.30

    def test_block_primary_opposition_with_no_tech_confirm_does_not_falsely_block_contextual_sell(self):
        judge = ConsensusJudge(
            min_consensus_threshold=0.30,
            require_technical_confirmation=False,
            block_primary_opposition=True,
            max_contextual_weight_ratio=None,
        )
        # Case A: Technical=HOLD vs OrderFlow=BUY -> primary agents are NOT opposing
        sigs_hold_buy = _sigs(
            ("TechnicalAgent", "HOLD", 0.50, 0.5), ("OrderFlowAgent", "BUY", 0.65, 0.5),
            ("SentimentAgent", "SELL", 0.90, 1.0), ("MacroAgent", "SELL", 0.90, 1.0), ("OnChainAgent", "SELL", 0.90, 1.0),
        )
        w_a = {"TechnicalAgent": 0.5, "OrderFlowAgent": 0.5, "SentimentAgent": 1.0, "MacroAgent": 1.0, "OnChainAgent": 1.0}
        dec_a = judge.judge_deterministic(sigs_hold_buy, weights_override=w_a)
        assert dec_a.direction == "SELL" and dec_a.aggregated_score <= -0.30

        # Case B: Technical=BUY and OrderFlow=BUY agree while contextual SELL dominates
        sigs_both_buy = _sigs(
            ("TechnicalAgent", "BUY", 0.60, 0.3), ("OrderFlowAgent", "BUY", 0.60, 0.3),
            ("SentimentAgent", "SELL", 0.95, 1.0), ("MacroAgent", "SELL", 0.95, 1.0), ("OnChainAgent", "SELL", 0.95, 1.0),
        )
        w_b = {"TechnicalAgent": 0.3, "OrderFlowAgent": 0.3, "SentimentAgent": 1.0, "MacroAgent": 1.0, "OnChainAgent": 1.0}
        dec_b = judge.judge_deterministic(sigs_both_buy, weights_override=w_b)
        assert dec_b.direction == "SELL" and dec_b.aggregated_score <= -0.30


# ---------------------------------------------------------------------------
# 6. Integration, LLM auditor non-reversal & edge cases
# ---------------------------------------------------------------------------
class TestIntegrationAndEdgeCases:
    def test_abstain_and_zero_weight_failed_agents_ignored_cleanly(self):
        base = _sigs(("TechnicalAgent", "SELL", 0.85, 1.0), ("MacroAgent", "SELL", 0.80, 0.7), ("OnChainAgent", "SELL", 0.75, 0.5))
        ignored = _sigs(("SentimentAgent", "ABSTAIN", 0.0, 0.6), ("OrderFlowAgent", "BUY", 0.90, 0.0))
        judge = ConsensusJudge(min_consensus_threshold=0.30)
        for override in (None, BULL_TREND_WEIGHTS):
            d_base = judge.judge_deterministic(base, weights_override=override)
            d_full = judge.judge_deterministic(base + ignored, weights_override=override)
            assert d_base.direction == d_full.direction == "SELL"
            assert d_full.aggregated_score == pytest.approx(d_base.aggregated_score, abs=1e-12)
            assert d_full.confidence == pytest.approx(d_base.confidence, abs=1e-12)

    def test_zero_weight_or_abstain_technical_agent_does_not_corrupt_guards(self):
        rest = _sigs(("OrderFlowAgent", "SELL", 0.85, 0.8), ("MacroAgent", "SELL", 0.85, 0.7), ("OnChainAgent", "SELL", 0.78, 0.5))
        judge = ConsensusJudge(min_consensus_threshold=0.30)
        d_zero = judge.judge_deterministic([_sig("TechnicalAgent", "BUY", 0.90, weight=0.0)] + rest, weights_override=BULL_TREND_WEIGHTS)
        d_abstain = judge.judge_deterministic([_sig("TechnicalAgent", "ABSTAIN", 0.0, weight=1.0)] + rest, weights_override=BULL_TREND_WEIGHTS)
        assert d_zero.direction == "SELL" and d_zero.aggregated_score <= -0.30
        assert d_abstain.direction == "SELL" and d_abstain.aggregated_score == pytest.approx(d_zero.aggregated_score, abs=1e-12)

    def test_non_finite_weights_override_ignored_without_nan_poisoning(self):
        signals = _sigs(
            ("TechnicalAgent", "BUY", 0.80, 1.0), ("OrderFlowAgent", "BUY", 0.75, 0.8),
            ("MacroAgent", "SELL", 0.85, 0.7), ("OnChainAgent", "SELL", 0.78, 0.5),
        )
        bad_ctx = {"TechnicalAgent": 1.0, "OrderFlowAgent": 0.8, "MacroAgent": float("nan"), "OnChainAgent": float("inf"), "SentimentAgent": float("-inf")}
        dec = ConsensusJudge(min_consensus_threshold=0.30).judge_deterministic(signals, weights_override=bad_ctx)
        assert math.isfinite(dec.aggregated_score) and math.isfinite(dec.confidence) and dec.direction == "BUY"
        bad_primary = {"TechnicalAgent": float("nan"), "OrderFlowAgent": float("inf"), "MacroAgent": float("-inf")}
        dec2 = ConsensusJudge(min_consensus_threshold=0.30).judge_deterministic(signals, weights_override=bad_primary)
        assert math.isfinite(dec2.aggregated_score) and math.isfinite(dec2.confidence)

    @pytest.mark.asyncio
    async def test_llm_cannot_override_hold_from_confirmation_or_opposition_guards(self):
        class StubLLM:
            def __init__(self, direction: str):
                self._direction = direction

            async def analyze_async(self, prompt: str):
                return {"direction": self._direction, "confidence": 0.95, "rationale": "Force"}

        judge = ConsensusJudge(min_consensus_threshold=0.30)
        cases = [
            (TestTechnicalConfirmationGuard.PHANTOM_SHORT, "SELL"),
            (_mirror(TestTechnicalConfirmationGuard.PHANTOM_SHORT), "BUY"),
            (TestPrimaryOppositionGuard.OPPOSED_SELL, "SELL"),
            (_mirror(TestPrimaryOppositionGuard.OPPOSED_SELL), "BUY"),
        ]
        for signals, llm_dir in cases:
            dec = await judge.judge_async(signals, llm_assistant=StubLLM(llm_dir), use_llm=True)
            assert dec.direction == "HOLD" and dec.llm_arbitrated is False

    @pytest.mark.asyncio
    async def test_judge_async_auditor_non_reversal_and_hold_downgrade(self):
        class StubLLM:
            def __init__(self, direction: str, confidence: float = 0.95):
                self._direction = direction
                self._confidence = confidence

            async def analyze_async(self, prompt: str):
                return {"direction": self._direction, "confidence": self._confidence, "rationale": "Audit"}

        counter_macro_buy = _sigs(
            ("TechnicalAgent", "BUY", 0.80), ("OrderFlowAgent", "BUY", 0.75),
            ("MacroAgent", "SELL", 0.85), ("OnChainAgent", "SELL", 0.78),
        )
        judge = ConsensusJudge(min_consensus_threshold=0.30)

        # Discordant counter-macro BUY: LLM returning SELL must NOT flip BUY -> SELL
        dec_no_flip_sell = await judge.judge_async(
            counter_macro_buy, weights_override=BULL_TREND_WEIGHTS,
            llm_assistant=StubLLM("SELL", 0.95), use_llm=True, regime="BULL_TREND",
        )
        assert dec_no_flip_sell.direction == "BUY" and dec_no_flip_sell.llm_arbitrated is False

        # Symmetric discordant counter-macro SELL: LLM returning BUY must NOT flip SELL -> BUY
        dec_no_flip_buy = await judge.judge_async(
            _mirror(counter_macro_buy), weights_override=BULL_TREND_WEIGHTS,
            llm_assistant=StubLLM("BUY", 0.95), use_llm=True, regime="BULL_TREND",
        )
        assert dec_no_flip_buy.direction == "SELL" and dec_no_flip_buy.llm_arbitrated is False

        # Auditor returning HOLD MAY downgrade BUY -> HOLD (and must never reverse to SELL)
        dec_hold = await judge.judge_async(
            counter_macro_buy, weights_override=BULL_TREND_WEIGHTS,
            llm_assistant=StubLLM("HOLD", 0.80), use_llm=True, regime="BULL_TREND",
        )
        assert dec_hold.direction in ("BUY", "HOLD")

    def test_learning_tracker_preserves_risk_sentinel_weight(self, tmp_path):
        tracker = LearningTracker(db_path=str(tmp_path / "t.db"))
        assert tracker.get_calibrated_weights_for_regime("BULL_TREND")["RiskSentinelAgent"] == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_trading_core_e2e_counter_macro_buy_approved(self, tmp_path):
        ctx = {
            "proba_buy": 0.80, "proba_sell": 0.05,
            "close": 108.0, "open": 100.0, "high": 108.0, "low": 100.0,
            "volume": 200.0, "volume_sma": 100.0,
            "dxy_ret_5d": 0.02, "sp500_ret_5d": -0.03, "stablecoin_growth_pct_30d": -0.05,
        }
        for use_tracker in (True, False):
            core = TradingCore(capital=10000.0, risk_per_trade=0.01, min_consensus_threshold=0.30)
            if use_tracker:
                core.learning_tracker = LearningTracker(db_path=str(tmp_path / "core_buy.db"))
            det = core.evaluate_market_deterministic(ctx, "BTCUSDT", 108.0, 2.0, 1700000000000)
            asy = await core.evaluate_market_async(ctx, "BTCUSDT", 108.0, 2.0, 1700000000000, use_llm=False)
            assert det.direction == "BUY" and det.is_vetoed is False and det.position_size > 0.0
            assert asy.direction == "BUY" and asy.is_vetoed is False
            assert asy.consensus_score == pytest.approx(det.consensus_score, abs=1e-12)

    @pytest.mark.asyncio
    async def test_trading_core_e2e_neutral_technical_blocks_phantom_short(self, tmp_path):
        core = TradingCore(capital=10000.0, risk_per_trade=0.01, min_consensus_threshold=0.30)
        core.learning_tracker = LearningTracker(db_path=str(tmp_path / "core_hold.db"))
        base_ctx = {
            "close": 100.0, "sma_200": 100.0, "open": 108.0, "high": 108.0, "low": 100.0,
            "volume": 200.0, "volume_sma": 100.0,
            "dxy_ret_5d": 0.02, "sp500_ret_5d": -0.03, "stablecoin_growth_pct_30d": -0.05,
        }
        for extra in ({"rsi_14": 50.0, "proba_buy": 0.20, "proba_sell": 0.20}, {"rsi_14": 45.0}):
            ctx = {**base_ctx, **extra}
            det = core.evaluate_market_deterministic(ctx, "SOLUSDT", 100.0, 2.0, 1700000000000)
            asy = await core.evaluate_market_async(ctx, "SOLUSDT", 100.0, 2.0, 1700000000000, use_llm=False)
            by_agent = {a["agent_name"]: a["direction"] for a in det.contributing_agents}
            assert by_agent["TechnicalAgent"] == "HOLD"
            assert by_agent["OrderFlowAgent"] == "SELL"
            assert by_agent["MacroAgent"] == "SELL"
            assert by_agent["OnChainAgent"] == "SELL"
            assert det.direction == "HOLD" and det.is_vetoed is False
            assert asy.direction == "HOLD" and asy.is_vetoed is False
