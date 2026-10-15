"""Testes de BacktestEngine(pilot_parity=True): mesma escada de saída/custos/alavancagem do piloto (smart_trade.py)."""

from __future__ import annotations

import inspect
import os

import numpy as np
import pandas as pd
import pytest

import backtesting
from backtesting import BacktestEngine, _PilotSimulator
from smart_trade import (BREAKEVEN_FEE_BUFFER_PCT, EXIT_TP1_FRACTION, EXIT_TP2_FRACTION, STRATEGY_PROFILES,
                         PaperPosition)

H = 3_600_000
T0 = 1_700_000_000_000 - (1_700_000_000_000 % H)
MAKER, TAKER, SPOT, MSLIP = 0.0002, 0.0006, 0.0002, 0.0001   # SPOT = slippage de mercado (simétrico)


def make_df(bars, atr=2.0):
    """bars = [(open, high, low, close), ...]; barra 0 = barra do sinal (close=100)."""
    rows = [{"timestamp": T0 + i * H, "open": o, "high": h, "low": l, "close": c, "atr_14": atr}
            for i, (o, h, l, c) in enumerate(bars)]
    return pd.DataFrame(rows)


def run(bars, signal=1, atr=2.0, conf=None, **kw):
    df = make_df(bars, atr)
    sig = np.zeros(len(df), dtype=int)
    sig[0] = signal
    kw.setdefault("funding_rate_8h", 0.0)
    eng = BacktestEngine(pilot_parity=True, risk_per_trade=None, **kw)
    return eng.run(df, sig, confidences=None if conf is None else np.full(len(df), conf))


FLAT = (100.0, 100.0, 100.0, 100.0)
ENTRY = 100.0 * (1 + SPOT)            # LONG: slippage spot
ATR = 2.0
TP1, TP2, TP3 = (ENTRY + m * ATR for m in (1.5, 2.5, 4.0))   # perfil DIRECTIONAL


def test_profiles_are_imported_not_duplicated():
    src = inspect.getsource(backtesting)
    assert "STRATEGY_PROFILES" in src and "tp1_atr_mult=" not in src
    p = STRATEGY_PROFILES["DIRECTIONAL"]
    assert (p.tp1_atr_mult, p.tp2_atr_mult, p.tp3_atr_mult, p.trail_atr_mult) == (1.5, 2.5, 4.0, 1.2)


def test_full_ladder_tp1_tp2_tp3_fractions_fees_and_pnl():
    res = run([FLAT, (100, TP1 + 0.1, 100.5, 101), (101, TP2 + 0.1, 101, 105), (105, TP3 + 1, 104, 108)])
    t = res.trades[0]
    assert t.exit_reason == "TAKE_PROFIT_3" and t.legs == 3 and t.leverage == 1.0
    qty = 100.0 / ENTRY
    q1, q2 = qty * EXIT_TP1_FRACTION, qty * EXIT_TP2_FRACTION
    q3 = qty - q1 - q2
    px3 = TP3 * (1 - MSLIP)
    gross = q1 * (TP1 - ENTRY) + q2 * (TP2 - ENTRY) + q3 * (px3 - ENTRY)
    fees = 100.0 * TAKER + MAKER * (q1 * TP1 + q2 * TP2 + q3 * px3)   # TP maker; entrada taker
    assert t.pnl_gross == pytest.approx(gross, rel=1e-9)
    assert t.fee_total == pytest.approx(fees, rel=1e-9)
    assert t.pnl_net == pytest.approx(gross - fees, rel=1e-9)
    assert res.final_capital == pytest.approx(10_000 + gross - fees, rel=1e-9)


def test_stop_has_priority_over_tp_in_same_bar():
    res = run([FLAT, (100, TP1 + 1, 97.0, 100)])
    t = res.trades[0]
    assert t.exit_reason == "STOP_LOSS" and t.legs == 1


def test_stop_loss_is_taker_with_symmetric_market_slippage():
    long = run([FLAT, (100, 100.5, 97.0, 98)], signal=1).trades[0]
    assert long.exit_price == pytest.approx(98.0 * (1 - SPOT))
    assert long.fee_total == pytest.approx(100 * TAKER + (100 / ENTRY) * long.exit_price * TAKER)
    short = run([FLAT, (100, 103, 99.5, 102)], signal=2).trades[0]
    assert short.exit_reason == "STOP_LOSS"
    assert short.exit_price == pytest.approx(102.0 * (1 + SPOT))


def test_tp1_moves_stop_to_breakeven_and_same_bar_touch_is_breakeven_stop():
    be = ENTRY * (1 + BREAKEVEN_FEE_BUFFER_PCT)
    res = run([FLAT, (100, TP1 + 0.1, be - 0.05, 100)])
    t = res.trades[0]
    assert t.exit_reason == "BREAKEVEN_STOP" and t.legs == 2
    # BE fill = min(stop, high) = be, taker spot
    assert t.exit_price == pytest.approx((EXIT_TP1_FRACTION * TP1 + (1 - EXIT_TP1_FRACTION) * be * (1 - SPOT)))


def test_trailing_uses_only_closed_bars_and_never_loosens():
    trail0 = TP2 - 1.2 * ATR                                   # stop inicial do trailing (extremo = TP2)
    bars = [FLAT, (100, TP1 + 0.1, 100.5, 101), (101, TP2 + 0.1, 101, 105),
            (105, TP3 - 0.5, trail0 + 0.5, 106),               # barra 3: sem stop; extremo (high) só vale na barra 4
            (106, TP3 - 0.2, 102.9, 103)]                       # barra 4: stop = high3 - 1.2 ATR
    t = run(bars).trades[0]
    trail1 = (TP3 - 0.5) - 1.2 * ATR
    assert t.exit_reason == "TRAILING_STOP"
    assert trail1 > trail0
    leg3_px = trail1 * (1 - SPOT)
    exp_exit = (EXIT_TP1_FRACTION * TP1 + EXIT_TP2_FRACTION * TP2 + (1 - 0.7) * leg3_px)
    assert t.exit_price == pytest.approx(exp_exit)
    # sem o ratchet da barra fechada o stop (trail0) não teria sido tocado em low=102.9 > trail0=102.62 -> prova
    assert 102.9 > trail0


def test_trailing_stop_not_loosened_by_lower_bar():
    bars = [FLAT, (100, TP1 + 0.1, 100.5, 101), (101, TP2 + 0.1, 101, 105),
            (105, TP3 - 0.5, 104.0, 106), (106, 106.5, 105.0, 105.5),   # barra 4 mais baixa: stop não recua
            (105.5, 105.6, 102.0, 102.5)]
    t = run(bars).trades[0]
    expected_stop = (TP3 - 0.5) - 1.2 * ATR                              # 105.62 - ... mantido da barra 3
    assert t.exit_reason == "TRAILING_STOP"
    assert t.exit_price == pytest.approx(
        EXIT_TP1_FRACTION * TP1 + EXIT_TP2_FRACTION * TP2 + 0.3 * min(expected_stop, 105.6) * (1 - SPOT))


def test_first_post_entry_bar_is_tested_and_reentry_on_exit_bar_allowed():
    # sinal na barra 0 e na barra 1: a barra 1 (1ª pós-entrada) stopa a 1ª posição e re-abre no close (como o piloto)
    df = make_df([FLAT, (100, 100.5, 97.0, 98.0), (98, 98.5, 97.5, 98.2)])
    sig = np.array([1, 1, 0])
    res = BacktestEngine(pilot_parity=True, risk_per_trade=None, funding_rate_8h=0.0).run(df, sig)
    assert [t.exit_reason for t in res.trades] == ["STOP_LOSS", "END_OF_DATA"]
    assert res.trades[0].entry_idx == 1 and res.trades[1].entry_idx == 2


def test_leverage_comes_from_policy_with_signal_confidence_and_margin_is_lot():
    r5 = run([FLAT, (100, 100.5, 99.5, 100)], conf=0.95).trades[0]
    r3 = run([FLAT, (100, 100.5, 99.5, 100)], conf=0.75).trades[0]
    assert r5.leverage == 5.0 and r5.margin == pytest.approx(100.0) and r5.size * r5.entry_price == pytest.approx(500.0)
    assert r3.leverage == 3.0 and r3.size * r3.entry_price == pytest.approx(300.0)


def test_leverage_reduced_until_stop_is_before_liquidation():
    # atr gigante (sl = close - 25 = 75): 5x -> liq 80.5 (SL além) ... reduz até SL antes da liquidação
    t = run([(100, 100, 100, 100), (100, 100.1, 99.9, 100)], atr=25.0, conf=0.95).trades[0]
    liq = PaperPosition.calc_liquidation_price("LONG", t.entry_price, t.leverage)
    assert t.leverage < 5.0 and 75.0 > liq
    assert t.leverage == 3.5


def test_liquidation_closes_taker_at_liquidation_price():
    df = make_df([FLAT, (100, 100, 100, 100)])
    eng = BacktestEngine(pilot_parity=True, risk_per_trade=None, funding_rate_8h=0.0, leverage=5.0)
    sim = _PilotSimulator(eng)
    sim.balance = 10_000.0
    positions, state = {}, {"last_entry_idx": {}, "quota": {}}
    p = sim.try_open("X", 1, 1, T0 + H, 100.0, 100.0, 2.0, None, None, positions, state)
    assert p.lev == 5.0 and p.liq == pytest.approx(PaperPosition.calc_liquidation_price("LONG", p.entry, 5.0))
    p.sl = p.liq - 1.0   # força cenário (defesa em profundidade): liquidação vem antes do stop
    out = sim.process_bar(p, T0 + H, 100.0, 100.0, p.liq - 1.0, p.liq - 0.5)
    assert out == ("LIQUIDATION", p.liq)


def test_portfolio_position_cap_and_aggregate_margin():
    n = 4
    flat = [FLAT] * n
    data = {s: make_df(flat) for s in ("A", "B", "C")}
    sigs = {s: np.array([1, 0, 0, 0]) for s in data}
    eng = BacktestEngine(pilot_parity=True, risk_per_trade=None, funding_rate_8h=0.0, max_concurrent_positions=2)
    res = eng.run_portfolio(data, sigs)
    assert res.trades_count == 2 and {t.symbol for t in res.trades} == {"A", "B"}
    eng2 = BacktestEngine(initial_capital=200.0, pilot_parity=True, risk_per_trade=None, funding_rate_8h=0.0,
                          max_concurrent_positions=10)    # teto de margem 85% de 200 = 170 -> só 1 lote de 100
    r2 = eng2.run_portfolio(data, sigs)       # lote reduzido pela margem livre (x0.95) e 3º ativo rejeitado (<$10)
    assert r2.trades_count == 2 and r2.trades[0].margin == pytest.approx(100.0)
    assert r2.trades[1].margin == pytest.approx((170.0 - 100.0 - r2.trades[0].fee_total * 0) * 0.95, rel=0.02)


def test_daily_quota_and_cooldown_options():
    data = {s: make_df([FLAT] * 3) for s in ("A", "B", "C")}
    sigs = {s: np.array([1, 0, 0]) for s in data}
    eng = BacktestEngine(pilot_parity=True, risk_per_trade=None, funding_rate_8h=0.0, daily_quota=2)
    assert eng.run_portfolio(data, sigs).trades_count == 2           # 3 sinais no mesmo dia UTC, cota 2
    bars = [FLAT, (100, 100.5, 97.0, 98.0), (98, 98.5, 97.5, 98.2), (98, 98.5, 97.5, 98.2)]
    sig = np.array([1, 0, 1, 0])                                      # stop na barra 1; novo sinal na barra 2
    n0 = BacktestEngine(pilot_parity=True, risk_per_trade=None, funding_rate_8h=0.0).run(make_df(bars), sig).trades_count
    n4 = BacktestEngine(pilot_parity=True, risk_per_trade=None, funding_rate_8h=0.0,
                        cooldown_bars=4).run(make_df(bars), sig).trades_count
    assert (n0, n4) == (2, 1)
    sc = BacktestEngine(pilot_parity=True, strategy_type="SCALP")
    gp = BacktestEngine(pilot_parity=True, strategy_type="GOLDEN_POT")
    assert (sc.daily_quota, sc.cooldown_bars, gp.daily_quota, gp.cooldown_bars) == (5, 4, 3, 16)


def test_next_open_entry_mode_is_conservative_variant():
    bars = [FLAT, (101.0, 101.2, 100.8, 101.0), (101, 101.1, 100.9, 101)]
    t = run(bars, entry_mode="next_open").trades[0]
    assert t.entry_idx == 1 and t.entry_price == pytest.approx(101.0 * (1 + SPOT))


def test_timeout_default_is_profile_horizon_and_funding_is_utc_clock():
    from smart_trade import STRATEGY_PROFILES as SP
    assert BacktestEngine(pilot_parity=True).max_holding_bars is None   # None -> horizonte do perfil (48)
    assert SP["DIRECTIONAL"].max_holding_bars == 48
    bars = [FLAT] + [(100, 100.2, 99.8, 100)] * 10
    t = run(bars, max_holding_bars=3).trades[0]
    assert t.exit_reason == "HORIZON_TIMEOUT" and t.exit_idx == 4   # ts >= entry_ts + 3 barras (piloto)
    # funding: liquidações em 00/08/16h UTC sobre qty*close da barra; T0 alinhado ao relógio -> checa contagem
    n = 12
    f = run([FLAT] + [(100, 100.2, 99.8, 100)] * n, funding_rate_8h=0.0001, max_holding_bars=0).trades[0]
    entry_ts = T0 + H
    settles = [x for x in range(((entry_ts // (8 * H)) + 1) * 8 * H, T0 + n * H + 1, 8 * H)]
    assert f.funding_total == pytest.approx(len(settles) * 0.0001 * 100.0, rel=1e-3) and len(settles) >= 1
    series = (np.array([settles[0] + 60_000]), np.array([0.0005]))   # taxa real dentro de ±10min substitui o fallback
    df = make_df([FLAT] + [(100, 100.2, 99.8, 100)] * n)
    sig = np.zeros(len(df), dtype=int)
    sig[0] = 1
    r = BacktestEngine(pilot_parity=True, risk_per_trade=None, funding_rate_8h=0.0001, max_holding_bars=0).run(
        df, sig, funding=series)
    others = (len(settles) - 1) * 0.0001 * 100.0
    assert r.trades[0].funding_total == pytest.approx(others + 0.0005 * 100.0, rel=1e-3)


def test_legacy_mode_unchanged_default():
    eng = BacktestEngine()
    assert eng.pilot_parity is False
    with pytest.raises(ValueError):
        eng.run_portfolio({"X": make_df([FLAT])}, {"X": [0]})


@pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "datasets", "market_cache.db")),
                    reason="cache SQLite ausente")
def test_leverage_4x_matches_paper_engine_trade_by_trade():
    from scripts import parity_audit as pa
    candles = pa.load_closed_candles("BTCUSDT", 900 + pa.WARMUP_BARS)
    feat = pa.build_features(candles)
    ts0 = max(0, len(feat) - 900)
    sig = pa.forced_signals(feat, ts0)
    df = feat.iloc[ts0:].reset_index(drop=True)
    res = BacktestEngine(pilot_parity=True, risk_per_trade=None, funding_rate_8h=0.0001, leverage=4.0).run(
        df, sig[ts0:], atr_col="atr_14")
    bt = [pa.TradeView(sig_idx=ts0 + t.entry_idx - 1, direction=t.direction, entry_ts=int(t.entry_timestamp),
                       entry_price=t.entry_price, exit_ts=int(t.exit_timestamp), exit_price=t.exit_price,
                       notional=t.entry_price * t.size, fees=t.fee_total, funding=t.funding_total, pnl_net=t.pnl_net,
                       reasons=[t.exit_reason]) for t in res.trades]
    import logging
    from unittest import mock
    from smart_trade import PaperTradingEngine
    logging.disable(logging.WARNING)
    try:   # RollbackGuard usa relógio de parede: desligado no replay histórico (não modelado no backtest)
        with mock.patch.object(PaperTradingEngine, "can_execute_directional", lambda self, *a, **k: (True, "OK")):
            pp, _ = pa.run_paper("BTCUSDT", feat, sig, ts0, leverage=4.0)
    finally:
        logging.disable(logging.NOTSET)
    # FUNDING: o TradeView do harness registra funding=0 do lado piloto (limitação); o PnL líquido já o inclui
    diffs = [d for d in pa.reconcile("BTCUSDT", bt, pp) if d.category not in pa.ACCEPTED_BY_DESIGN | {"FUNDING": 1}]
    assert len(bt) > 10 and not diffs, diffs[:3]
