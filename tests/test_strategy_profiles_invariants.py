"""
Testes de regressão para invariantes de perfis de estratégia (C9 / I9) e
derivação dinâmica de PRIOR_TRIALS do trial_log.csv (C4 / C5 / I2).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smart_trade import STRATEGY_PROFILES
from statistical_rigor import get_prior_trials_count
import scripts.strategy_search as strategy_search
import scripts.online_ml_search as online_ml_search


def test_strategy_profiles_exact_live_invariants():
    """Garante que STRATEGY_PROFILES mantém os parâmetros vivos especificados no E10 (I9)."""
    assert set(STRATEGY_PROFILES.keys()) == {"SCALP", "DIRECTIONAL", "GOLDEN_POT", "SWING_4H"}

    scalp = STRATEGY_PROFILES["SCALP"]
    assert scalp.sl_atr_mult == pytest.approx(1.4)
    assert scalp.tp1_atr_mult == pytest.approx(1.6)
    assert scalp.tp2_atr_mult == pytest.approx(2.0)
    assert scalp.tp3_atr_mult == pytest.approx(2.2)
    assert scalp.trail_atr_mult == pytest.approx(0.0)
    assert scalp.bar_ms == 900_000
    assert scalp.max_holding_bars == 30
    assert scalp.tp1_fraction == pytest.approx(0.0)
    assert scalp.tp2_fraction == pytest.approx(0.0)

    directional = STRATEGY_PROFILES["DIRECTIONAL"]
    assert directional.sl_atr_mult == pytest.approx(1.0)
    assert directional.tp1_atr_mult == pytest.approx(1.5)
    assert directional.tp2_atr_mult == pytest.approx(2.5)
    assert directional.tp3_atr_mult == pytest.approx(0.0)
    assert directional.trail_atr_mult == pytest.approx(1.2)
    assert directional.bar_ms == 3_600_000
    assert directional.max_holding_bars == 48

    gp = STRATEGY_PROFILES["GOLDEN_POT"]
    assert gp.sl_atr_mult == pytest.approx(1.5)
    assert gp.tp1_atr_mult == pytest.approx(1.5)
    assert gp.tp2_atr_mult == pytest.approx(2.5)
    assert gp.tp3_atr_mult == pytest.approx(0.0)
    assert gp.trail_atr_mult == pytest.approx(1.5)
    assert gp.bar_ms == 300_000
    assert gp.max_holding_bars == 36

    swing_4h = STRATEGY_PROFILES["SWING_4H"]
    assert swing_4h.sl_atr_mult == pytest.approx(2.5)
    assert swing_4h.tp1_atr_mult == pytest.approx(0.0)
    assert swing_4h.tp2_atr_mult == pytest.approx(0.0)
    assert swing_4h.tp3_atr_mult == pytest.approx(0.0)
    assert swing_4h.trail_atr_mult == pytest.approx(3.5)
    assert swing_4h.min_committee_confidence == pytest.approx(0.60)
    assert swing_4h.bar_ms == 14_400_000
    assert swing_4h.max_holding_bars == 120
    assert swing_4h.tp1_fraction == pytest.approx(0.0)
    assert swing_4h.tp2_fraction == pytest.approx(0.0)


def test_prior_trials_derived_from_trial_log():
    """Garante que PRIOR_TRIALS em strategy_search e online_ml_search deriva de trial_log.csv (C4/C5)."""
    expected_count = get_prior_trials_count()
    assert expected_count >= 86
    assert strategy_search.PRIOR_TRIALS == expected_count
    assert online_ml_search.PRIOR_TRIALS == expected_count
