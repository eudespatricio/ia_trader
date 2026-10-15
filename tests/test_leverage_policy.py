"""Política de alavancagem do Smart Trader: limites, clamp <= 5, monotonicidade."""
import itertools
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leverage_policy import MAX_LEVERAGE, clamp_leverage, decide_leverage  # noqa: E402


@pytest.mark.parametrize("conf,exp", [(0.0, 1), (0.59, 1), (0.60, 2), (0.69, 2), (0.70, 3), (0.79, 3),
                                      (0.80, 4), (0.89, 4), (0.90, 5), (1.0, 5)])
def test_confidence_ladder(conf, exp):
    assert decide_leverage(conf) == exp


def test_max_leverage_constant():
    assert MAX_LEVERAGE == 5


def test_never_exceeds_cap_or_below_one():
    for conf, atr, reg in itertools.product([-1, 0, 0.5, 0.95, 5, float("nan"), None], [0, 0.01, 0.2, float("nan")],
                                            [True, False, None]):
        lev = decide_leverage(conf, atr, reg, max_leverage=99)
        assert 1.0 <= lev <= 5.0


def test_monotonic_in_confidence():
    for atr, reg in itertools.product([0.0, 0.05], [True, False, None]):
        levs = [decide_leverage(c / 100, atr, reg) for c in range(0, 101)]
        assert levs == sorted(levs)


def test_penalties_one_step_each():
    assert decide_leverage(0.95, atr_pct=0.05) == 4
    assert decide_leverage(0.95, regime_alignment=False) == 4
    assert decide_leverage(0.95, atr_pct=0.05, regime_alignment=False) == 3
    assert decide_leverage(0.65, atr_pct=0.05, regime_alignment=False) == 1  # piso 1x
    assert decide_leverage(0.95, regime_alignment=True) == 5


def test_veto_and_low_history_force_1x():
    assert decide_leverage(0.99, is_vetoed=True) == 1
    assert decide_leverage(0.99, low_history_or_liquidity=True) == 1


def test_clamp_half_steps_and_cap():
    assert clamp_leverage(7) == 5 and clamp_leverage(0.3) == 1 and clamp_leverage(2.74) == 2.5
    assert clamp_leverage(float("nan")) == 1 and clamp_leverage("x") == 1
