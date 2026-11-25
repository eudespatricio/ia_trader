"""S6.3 Onda 3 — core PnL/sim helper tests (directional/scalp outcome, metrics, causal 1h dist, S0 trigger, swing 4h).
Companion of test_patch_s6_onda3_majors.py (kept separate to respect the <=450-line cap). Expected values are derived
independently from the frozen contract constants, never from the implementation's own output."""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.eval_majors_expansion_s6_oos as ev
import scripts.majors_s6_sim as ms

H, M15, T0 = 3_600_000, 900_000, 1_704_067_200_000


def _sd(n=60, step=H, over=None, atr=1.0):
    h, l = np.full(n, 100.5), np.full(n, 99.5)
    for k, (hi, lo) in (over or {}).items():
        h[k], l[k] = hi, lo
    return SimpleNamespace(n=n, atr=np.full(n, atr), c=np.full(n, 100.0), h=h, l=l, ts=T0 + np.arange(n, dtype=np.int64) * step)


# ───────────── directional_outcome ─────────────
def test_dir_sl_wins_same_bar_tie():
    out = ms.directional_outcome(_sd(over={1: (110.0, 90.0)}), 0, 1)
    assert out["reason"] == "SL" and out["exit_idx"] == 1 and out["r"] < -0.9


def test_dir_tp1_then_breakeven():
    out = ms.directional_outcome(_sd(over={1: (101.6, 99.6), 2: (100.5, 99.9)}), 0, 1)
    assert out["reason"] == "BE" and out["exit_idx"] == 2 and 0.0 < out["r"] < 1.0


def test_dir_tp2_then_trailing_stop():
    out = ms.directional_outcome(_sd(over={1: (103.0, 100.0), 2: (100.5, 90.0)}), 0, 1)
    assert out["reason"] == "TRAIL" and out["exit_idx"] == 2 and out["r"] > 0.5


def test_dir_short_mirrors_long_and_edge_none():
    out = ms.directional_outcome(_sd(over={1: (110.0, 90.0)}), 0, -1)
    assert out["reason"] == "SL" and out["r"] < -0.9
    assert ms.directional_outcome(_sd(n=10), 9, 1) is None  # no future bar


@pytest.mark.parametrize("d,rate", [(1, 0.001), (-1, 0.001), (1, -0.001), (-1, -0.001)])
def test_dir_real_8h_funding_debit_credit(d, rate):
    sd = _sd()
    ft = (sd.ts.copy(), np.full(sd.n, rate))
    base, out = ms.directional_outcome(sd, 0, d), ms.directional_outcome(sd, 0, d, ft)
    last = min(ms.DIR_PROF.max_holding_bars, sd.n - 1)
    entry = 100.0 * (1.0 + d * ms.TAKER_SLIP)
    exp = -d * rate * (last // 8) * entry / (ms.DIR_PROF.sl_atr_mult * 1.0)  # longs pay positive funding, shorts receive
    assert out["funding_r"] == pytest.approx(exp, rel=1e-9) and math.copysign(1, exp) == math.copysign(1, -d * rate)
    assert out["r"] - base["r"] == pytest.approx(out["funding_r"], abs=1e-12) and base["funding_r"] == 0.0


# ───────────── scalp_outcome ─────────────
def test_scalp_sl_before_tp3_tie_and_tp3_maker_exact():
    out = ms.scalp_outcome(_sd(step=M15, over={1: (110.0, 90.0)}), 0, 1)
    assert out["reason"] == "SL" and out["exit_idx"] == 1 and out["r"] < -0.9
    out = ms.scalp_outcome(_sd(step=M15, over={1: (103.0, 99.9)}), 0, 1)
    entry, P = 100.0 * (1 + ms.TAKER_SLIP), ms.SCALP_PROF
    tp = entry + P.tp3_atr_mult
    net = ((tp * (1 - ms.MAKER_SLIP) - entry) / entry) - ms.TAKER_FEE - ms.MAKER_FEE
    assert out["reason"] == "TP3" and out["r"] == pytest.approx(net * entry / P.sl_atr_mult, rel=1e-9)


def test_scalp_guards_gap_and_tail():
    sd = _sd(step=M15)
    sd.ts[10:] += 2 * H  # 2h hole inside the 30-bar window
    assert ms.scalp_outcome(sd, 0, 1) is None
    assert ms.scalp_outcome(_sd(step=M15, n=10), 8, 1) is None  # n-1-i < 2


# ───────────── compute_portfolio_metrics ─────────────
def _trades(rs):
    D = 24 * H  # one trade per day so daily-return based Sharpe/DSR is meaningful
    return [{"ts": T0 + i * D, "exit_ts": T0 + i * D + H, "close_t": T0 + i * D + H, "r": r} for i, r in enumerate(rs)]


def test_metrics_pf_ex_top1_maxdd_and_var_trials():
    rs = [2.0, -1.0, 1.5, -1.0, 0.5, -1.0, 2.0, -0.5, 1.0, -1.0] * 3
    m = ms.compute_portfolio_metrics(_trades(rs), bootstrap_iters=0)
    r = np.array(rs)
    assert m["n"] == 30 and m["pf"] == pytest.approx(r[r > 0].sum() / -r[r < 0].sum())
    top = np.delete(r, int(np.argmax(r)))
    assert m["pf_ex_top1"] == pytest.approx(top[top > 0].sum() / -top[top < 0].sum()) and m["pf_ex_top1"] < m["pf"]
    eq, peak, dd = 10_000.0, 10_000.0, 0.0
    for x in rs:
        eq += eq * 0.01 * x
        peak = max(peak, eq)
        dd = max(dd, (peak - eq) / peak * 100.0)
    assert m["max_dd_pct"] == pytest.approx(dd, rel=1e-9) and m["sum_r"] == pytest.approx(r.sum())
    lo = ms.compute_portfolio_metrics(_trades(rs), n_trials=50, var_trials=1e-4, bootstrap_iters=0)
    hi = ms.compute_portfolio_metrics(_trades(rs), n_trials=50, var_trials=1.0, bootstrap_iters=0)
    assert lo["dsr"] != hi["dsr"] and lo["dsr"] > hi["dsr"]  # larger trial variance → higher deflation benchmark


# ───────────── h1_dist_aligned (causality) ─────────────
def test_h1_dist_aligned_uses_only_closed_1h_bars():
    n1 = 400
    close = 100 + np.cumsum(np.random.default_rng(5).normal(0, 1, n1))
    raw = pd.DataFrame({"timestamp": T0 + np.arange(n1, dtype=np.int64) * H, "close": close})
    s15 = SimpleNamespace(ts=T0 + np.arange(n1 * 4, dtype=np.int64) * M15)
    got = ev.h1_dist_aligned(s15, raw)
    sma = pd.Series(close).rolling(200).mean().to_numpy()
    dist = (close - sma) / sma
    for k in (0, 3, 799, 800, 801, 802, 803, 1000, 1599):
        idx = (int(s15.ts[k]) + M15 - T0) // H - 1  # last 1h bar whose close <= this 15m bar's close
        exp = dist[idx] if (idx >= 199) else np.nan
        assert (np.isnan(got[k]) and np.isnan(exp)) or got[k] == pytest.approx(exp, abs=1e-12), k
    assert got[802] != got[803]  # 15m bar closing exactly at 1h close picks the new 1h bar; 802 still the old one
    pois = raw.copy(); pois.loc[300:, "close"] *= 3.0
    got2 = ev.h1_dist_aligned(s15, pois)
    safe = (s15.ts + M15) < (T0 + 301 * H)  # 15m bars closing before 1h bar #300 closes
    assert np.array_equal(got[safe], got2[safe], equal_nan=True) and not np.array_equal(got, got2, equal_nan=True)


# ───────────── scalp_s0_trigger_dirs ─────────────
def _s0(side=1, atr=0.5, **ov):
    n = 8
    p = dict(c5=100.0 + side, o5=100.0, vz5=1.5, h15=0.01 * side, pb5=0.6 if side > 0 else 0.2, ps5=0.2 if side > 0 else 0.6, bbh3=100.3)
    p.update(ov)
    c, o, bbh = np.full(n, 100.0), np.full(n, 100.0), np.full(n, 100.3)
    c[5], o[5], bbh[3] = p["c5"], p["o5"], p["bbh3"]
    vz, h1, pb, ps = np.full(n, 1.5), np.full(n, 0.01), np.full(n, 0.2), np.full(n, 0.2)
    vz[5], h1[5], pb[5], ps[5] = p["vz5"], p["h15"], p["pb5"], p["ps5"]
    df = pd.DataFrame({"bb_high": bbh, "bb_low": np.full(n, 99.7), "volume_zscore_20": vz})
    s = SimpleNamespace(c=c, o=o, atr=np.full(n, atr), kc_mid=np.full(n, 100.0), n=n, df=df)
    return ev.scalp_s0_trigger_dirs(s, h1, pb, ps)


@pytest.mark.parametrize("kw,exp", [
    ({}, 1), ({"side": -1}, -1),
    ({"bbh3": 105.0}, 0),              # no squeeze two bars back
    ({"vz5": 0.99}, 0), ({"vz5": 1.0}, 1),
    ({"atr": 0.3}, 0),                 # natr 30 < 35 bps
    ({"pb5": 0.44}, 0), ({"pb5": 0.45}, 1), ({"pb5": 0.2, "ps5": 0.2}, 0),
    ({"h15": -0.01}, 0), ({"h15": float("nan")}, 0),
    ({"side": -1, "h15": 0.01}, 0), ({"side": -1, "ps5": 0.4}, 0),
    ({"c5": 100.5}, 0),                # close inside Keltner band: no breakout
])
def test_scalp_s0_trigger_gates(kw, exp):
    out = _s0(**kw)
    assert out[5] == exp and int(np.abs(np.delete(out, 5)).sum()) == 0


# ───────────── simulate_swing_4h ─────────────
BAR = ev.SWING_4H_BAR_MS


def _sw(syms=("A",), side="BUY", over=None, entry_idx=2, n=40, atr=1.0):
    cache = {}
    for sym in syms:
        a = {k: np.full(n, 100.0) for k in "ohlc"}
        for k, vals in (over or {}).items():
            for key, v in vals.items():
                a[key][k] = v
        cache[sym] = {"n": n, "ts": T0 + np.arange(n, dtype=np.int64) * BAR, **a, "l10": np.full(n, np.nan), "h10": np.full(n, np.nan),
                      "entries": {entry_idx: (side, atr, 1.0)}}
    return cache


def _run(cache, **kw):
    return ev.simulate_swing_4h(cache, {}, list(cache), T0, **kw)


def test_swing_enters_at_next_bar_open_not_signal_close():
    # Gap-up open at bar 3 (105). Stop distance is measured from that open; low at bar 4 only breaches a 105-based stop.
    sl_mult = ev.SWING_4H_SL_ATR_MULT
    epx = 105.0 * (1 + ms.TAKER_SLIP)
    over = {3: {"o": 105.0, "h": 105.0, "l": 105.0, "c": 105.0}, 4: {"o": 105.0, "h": 105.0, "l": epx - sl_mult - 0.01, "c": 105.0}}
    res = _run(_sw(over=over))
    assert res["n"] == 1 and -1.2 < res["sum_r"] < -0.99  # stopped out at ~-1R, would be 0 trades if entered at 100


def test_swing_trailing_stop_locks_profit():
    over = {4: {"o": 105.0, "h": 115.0, "l": 105.0, "c": 115.0}, 5: {"o": 114.0, "h": 114.0, "l": 100.0, "c": 100.0}}
    res = _run(_sw(over=over))
    assert res["n"] == 1 and res["sum_r"] > 0.0


def test_swing_ch10_exit_closes_at_bar_close_before_stop():
    cache = _sw(entry_idx=11, over={12: {"h": 100.0, "l": 99.4, "c": 99.5}})
    cache["A"]["l10"][12] = 100.0  # close 99.5 < Ch10 low → exit at close, not at the (lower) stop
    res = _run(cache)
    assert res["n"] == 1 and -1.0 < res["sum_r"] < 0.0


@pytest.mark.parametrize("kw,n_exp", [({}, 2), ({"corr_cap": 1}, 1), ({"corr_cap": 2}, 2)])
def test_swing_corr_cap_per_direction(kw, n_exp):
    crash = {4: {"l": 50.0, "h": 200.0}}  # stops out every position at bar 4
    assert _run(_sw(syms=("A", "B"), over=crash), **kw)["n"] == n_exp


def test_swing_net_risk_cap_with_beta_and_hedge():
    crash = {4: {"l": 50.0, "h": 200.0}}
    longs = _sw(syms=("A", "B"), over=crash)
    assert _run(longs, max_net_risk_r=1.5)["n"] == 1                                    # unit weights: 2.0 > 1.5
    assert _run(longs, max_net_risk_r=1.5, beta_map={"A": 1.0, "B": 0.4})["n"] == 2      # 1.4 <= 1.5
    hedge = _sw(syms=("A", "B"), over=crash)
    hedge["B"]["entries"] = {2: ("SELL", 1.0, 1.0)}
    assert _run(hedge, max_net_risk_r=1.0)["n"] == 2                                    # opposite side reduces |net|
    assert _run(_sw(syms=("A",), over={}), max_net_risk_r=1.0)["n"] == 0                # never closes → no trades
