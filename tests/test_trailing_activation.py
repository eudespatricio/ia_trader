"""trailing_activation='tp1' (experimento offline); o padrão 'tp2' é coberto por tests/test_backtest_pilot_parity.py."""
import pytest

from backtesting import BacktestEngine
from tests.test_backtest_pilot_parity import ATR, EXIT_TP1_FRACTION, FLAT, SPOT, TP1, run


BARS = [FLAT, (100, TP1 + 0.1, 101.5, 102.5),      # TP1 executa; low 101.5 > novo stop
        (102.5, 104.0, 101.0, 103.0),               # high2 só vale como extremo na barra seguinte
        (102.0, 102.5, 100.5, 101.0)]               # low 100.5: > BE (100.06) mas <= trailing


def test_tp1_activation_trails_after_tp1_only_closed_bars():
    t = run(BARS, trailing_activation="tp1").trades[0]
    stop = 104.0 - 1.2 * ATR                         # high da barra 2 (fechada) - 1.2 x ATR de entrada
    assert t.exit_reason == "TRAILING_STOP" and t.legs == 2
    assert t.exit_price == pytest.approx(EXIT_TP1_FRACTION * TP1 + (1 - EXIT_TP1_FRACTION) * stop * (1 - SPOT))


def test_default_tp2_does_not_trail_after_tp1():
    t = run(BARS).trades[0]
    assert t.exit_reason != "TRAILING_STOP"
    assert run(BARS, trailing_activation="tp2").trades[0].exit_price == t.exit_price


def test_invalid_value_rejected():
    with pytest.raises(ValueError):
        BacktestEngine(pilot_parity=True, trailing_activation="x")
