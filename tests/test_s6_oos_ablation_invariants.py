"""Independent leakage/parity invariants for the S6 OOS ablation scripts (SCALP 15m + GOLDEN_POT 5m).

Written by an independent auditor (not the script authors). Synthetic fixtures only (tmp SQLite, hand-built
OHLC paths); the real DBs are never opened for writing and the 10-min backtests are never run here.
Expected R values are computed from the cost/exit contract, not from the functions under test.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3

import numpy as np
import pandas as pd
import pytest

import scripts.eval_golden_pot_s6_oos as G
import scripts.eval_scalp_s6_oos as S
from agents.golden_pot_agent import NON_CRYPTO_BASE_EXCLUSIONS, GoldenPotAgent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALP_JSON = os.path.join(ROOT, "datasets", "reports", "s6_scalp_oos_ablation.json")
GP_JSON = os.path.join(ROOT, "datasets", "reports", "s6_golden_pot_oos_ablation.json")
B5, B15, B1H, DAY = 300_000, 900_000, 3_600_000, 86_400_000
TF, TS_, MF, MS_ = 0.0006, 0.0002, 0.0002, 0.0001

# Bases the brief requires excluded + TradFi tickers (equities / single-stock & leveraged ETFs / pre-IPO)
# observed in market_cache.db 5m candles during this audit.
REQUIRED_TRADFI = {"MU", "MUU", "ANT", "DRAM", "EWY", "NBIS", "ZHIPU", "ZHONGJI", "BE", "MRVL", "NVDL", "CRCL",
                   "IREN", "NATGAS", "CBRS", "SPCX", "MINIMAX", "XAU", "XAG", "TSLA", "NVDA"}
OBSERVED_TRADFI_IN_DB = {"AAOI", "ARM", "ASTS", "AXTI", "BITO", "BMNR", "COHR", "CSOPSKHYNIX2L", "CXMT", "DELL",
                         "MRNA", "SMCI", "TTWO", "WDC", "INTW", "MVLL", "SKUU", "SKDD"}


# ------------------------------------------------------------------ fixtures / builders
def synth_ohlcv(n: int, t0: int, up=(), dn=(), seed: int = 7, bar: int = B5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, px = [], 100.0
    for k in range(n):
        op = px
        if k in up:
            cl = op * 1.03; hi, lo, vol = cl * 1.0005, op * 0.9995, 50_000.0
        elif k in dn:
            cl = op * 0.97; hi, lo, vol = op * 1.0005, cl * 0.9995, 50_000.0
        else:
            cl = op * (1 + rng.normal(0, 0.0005))
            hi, lo, vol = max(op, cl) * 1.0004, min(op, cl) * 0.9996, 1000.0 + rng.normal(0, 50)
        rows.append((t0 + k * bar, op, hi, lo, cl, vol)); px = cl
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def mk_sd(rows, atr=1.0, t0=10_000 * B15, bar=B15, ts=None):
    """rows: list of (o, h, l, c); bar 0 is the signal bar."""
    s = S.SD()
    a = np.asarray(rows, dtype=float)
    s.o, s.h, s.l, s.c = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    s.ts = np.asarray(ts if ts is not None else [t0 + k * bar for k in range(len(a))], dtype=np.int64)
    s.atr = np.full(len(a), atr); s.n = len(a)
    return s


def mk_gp_df(rows, atr=1.0, bar=B5):
    a = np.asarray(rows, dtype=float)
    return pd.DataFrame({"timestamp": np.arange(len(a), dtype=np.int64) * bar + 10 ** 12, "open": a[:, 0],
                         "high": a[:, 1], "low": a[:, 2], "close": a[:, 3], "atr_14": np.full(len(a), atr)})


FLAT = (100.0, 100.3, 99.8, 100.1)


def write_candles_db(path: str, series: dict) -> None:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE candles (symbol TEXT, exchange TEXT, timeframe TEXT, timestamp INTEGER, "
                "open REAL, high REAL, low REAL, close REAL, volume REAL)")
    for (sym, ex), ts in series.items():
        con.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?)",
                        [(sym, ex, "5m", int(t), 1.0, 1.1, 0.9, 1.0 if ex == "binance" else 2.0, 10.0) for t in ts])
    con.commit(); con.close()


def _load(path):
    if not os.path.exists(path):
        pytest.fail(f"report missing: {path}")
    with open(path) as fh:
        return json.load(fh)


# ------------------------------------------------------------------ 1. TradFi universe
def test_required_tradfi_bases_in_canonical_exclusions():
    assert REQUIRED_TRADFI - set(NON_CRYPTO_BASE_EXCLUSIONS) == set()


def _universe(tmp_path, bases):
    t = [10 ** 12 + k * B5 for k in range(300)]
    series = {(b + "USDT", "binance"): t for b in bases}
    db = str(tmp_path / "c.db"); write_candles_db(db, series)
    segs, meta = G.load_and_split_5m_segments(db)
    return {s for _, s, _ in segs}, meta


def test_gp_universe_excludes_majors_stables_required_tradfi(tmp_path):
    kept, meta = _universe(tmp_path, ["BTC", "ETH", "SOL", "USDC", "DOGE", "SUI"] + sorted(REQUIRED_TRADFI))
    assert kept == {"DOGEUSDT", "SUIUSDT"}
    assert meta["excluded_tradfi_count"] == len(REQUIRED_TRADFI)


def test_gp_universe_excludes_tradfi_symbols_observed_in_db(tmp_path):
    """P1 audit finding: Binance TradFi perps present in market_cache 5m candles must not reach the GP universe
    (live excludes them via exchangeInfo underlyingSubType; the OOS script only uses the static list)."""
    kept, _ = _universe(tmp_path, ["DOGE"] + sorted(OBSERVED_TRADFI_IN_DB))
    leaked = sorted(GoldenPotAgent.extract_base_asset(s) for s in kept - {"DOGEUSDT"})
    assert leaked == [], f"TradFi bases leaked into GP OOS universe: {leaked}"


# ------------------------------------------------------------------ 2. GP segments / entry horizon / embargo
def test_gp_segment_splitter_gap_threshold_min_len_and_binance_priority(tmp_path):
    t0 = 10 ** 12
    a = [t0 + k * B5 for k in range(300)]
    b = [a[-1] + 900_001 + k * B5 for k in range(301)]              # gap > 900_000 -> split
    c = [t0 + k * B5 for k in range(300)] + [t0 + 299 * B5 + 900_000 + k * B5 for k in range(300)]  # == 900_000
    series = {("DOGEUSDT", "binance"): a + b, ("DOGEUSDT", "bitget"): a[:50],
              ("SUIUSDT", "binance"): c, ("PEPEUSDT", "binance"): a[:249] + b[:100]}
    db = str(tmp_path / "c.db"); write_candles_db(db, series)
    segs, _ = G.load_and_split_5m_segments(db)
    by = {}
    for _, sym, df in segs:
        d = np.diff(df["timestamp"].to_numpy())
        assert (d <= G.GAP_THRESHOLD_5M_MS).all() and len(df) >= G.MIN_SEGMENT_BARS
        by.setdefault(sym, []).append(df)
    assert [len(x) for x in by["DOGEUSDT"]] == [300, 300]           # last (forming) candle dropped once
    assert [len(x) for x in by["SUIUSDT"]] == [599]                  # exactly 900_000 is NOT a gap
    assert "PEPEUSDT" not in by                                      # both pieces < 250 bars
    assert (by["DOGEUSDT"][0]["close"] == 1.0).all()                 # binance preferred over bitget


def _gp_cands(df, seg_id=0):
    res = G._process_segment_worker((seg_id, "DOGEUSDT", df))
    assert res is not None
    return res[2], res[3]


def test_gp_entries_need_full_holding_horizon_inside_segment():
    n, warm = 700, 199
    df = synth_ohlcv(n, G.IS_START_TS, up={300, 420, 540, warm + 464, n - 20})
    dff, cands = _gp_cands(df)
    m = len(dff)
    assert m == n - warm
    idx = {c["idx"] for c in cands}
    assert all(i + G.GP_MAX_HOLDING_BARS < m for i in idx)
    assert {300 - warm, 420 - warm, 540 - warm} <= idx               # triggers fire (test is not vacuous)
    assert m - G.GP_MAX_HOLDING_BARS - 1 in idx                      # boundary bar i+36 == len-1 allowed
    assert (n - 20) - warm not in idx                                # < 36 bars left -> blocked


def test_gp_candidates_respect_is_embargo_oos_windows():
    n = 700
    t0 = G.SPLIT_TS_GP - 400 * B5
    df = synth_ohlcv(n, t0, up={330, 380, 450})
    _, cands = _gp_cands(df)
    for c in cands:
        ts = c["ts"]
        assert not (G.SPLIT_TS_GP - G.EMBARGO_GP_MS <= ts < G.SPLIT_TS_GP)
        assert c["split"] == ("OOS" if ts >= G.SPLIT_TS_GP else "IS")
        if c["split"] == "IS":
            assert ts + (G.GP_MAX_HOLDING_BARS + 1) * B5 <= G.SPLIT_TS_GP   # IS trade cannot exit inside OOS
    ts_set = {c["ts"] for c in cands}
    assert t0 + 330 * B5 in ts_set and t0 + 450 * B5 in ts_set and t0 + 380 * B5 not in ts_set


def test_gp_candidate_features_are_causal_under_future_perturbation():
    n, cut = 700, 520
    base = synth_ohlcv(n, G.IS_START_TS, up={300, 420, 600}, dn={480})
    pert = base.copy()
    rng = np.random.default_rng(99)
    f = 1 + rng.uniform(-0.05, 0.05, n - cut)
    for col in ("open", "high", "low", "close"):
        pert.loc[cut:, col] = base.loc[cut:, col].to_numpy() * f
    pert.loc[cut:, "high"] = pert.loc[cut:, ["open", "high", "low", "close"]].max(axis=1)
    pert.loc[cut:, "low"] = pert.loc[cut:, ["open", "high", "low", "close"]].min(axis=1)
    pert.loc[cut:, "volume"] *= 7.0
    _, ca = _gp_cands(base); _, cb = _gp_cands(pert)
    keys = ("ts", "std_dir", "early_dir", "atr", "atr_exp", "rng_atr", "vol_z", "cpos", "rsi", "sl_mult", "cum_move_atr")
    pa = [tuple(c[k] for k in keys) for c in ca if c["idx"] < cut - 199]
    pb = [tuple(c[k] for k in keys) for c in cb if c["idx"] < cut - 199]
    assert pa and pa == pb


def _ramp_5m(n: int, seed: int = 0) -> pd.DataFrame:
    """Volatility ramps (6%/bar for 20 bars every hour) -> triggers whose 3-bar max range is near 1.3-1.5 ATR."""
    rng, px, rows, vol = np.random.default_rng(seed), 100.0, [], 0.001
    for k in range(n):
        ph = k % 60
        vol = 0.0008 if ph == 0 else vol * (1.06 if ph > 40 else 1.0)
        op = px
        cl = op * (1 + abs(rng.normal(0.6, 0.3)) * vol) if ph > 40 else op * (1 + rng.normal(0, vol))
        v = 1000 * (1 + (ph > 40) * (ph - 40) * 0.3) + rng.normal(0, 30)
        rows.append((G.IS_START_TS + k * B5, op, max(op, cl) * (1 + 0.1 * vol), min(op, cl) * (1 - 0.1 * vol), cl, v)); px = cl
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_gp_adaptive_sl_mult_matches_live_formula():
    """Live (run_paper_simulation.py ~L1327): max(prof, rng3*1.15) if rng3 > prof else prof (prof = 1.5)."""
    dff, cands = _gp_cands(_ramp_5m(900))
    zone = 0
    for c in cands:
        i, atr = c["idx"], c["atr"]
        rng3 = max((dff["high"].iloc[k] - dff["low"].iloc[k]) / atr for k in (i - 2, i - 1, i))
        zone += int(1.5 / 1.15 < rng3 <= 1.5)
        assert c["sl_mult"] == pytest.approx(max(1.5, rng3 * 1.15) if rng3 > 1.5 else 1.5)
    assert zone >= 1                                                 # boundary zone where formulas differ is exercised


# ------------------------------------------------------------------ 3. SCALP gap guard / cooldown / 1 per bar
def test_scalp_gap_guard_rejects_gaps_over_30m_within_holding_window():
    def sd_with_gap(at, gap):
        ts = [10_000 * B15 + k * B15 for k in range(40)]
        ts = ts[:at] + [t + gap - B15 for t in ts[at:]]
        return mk_sd([FLAT] * 40, ts=ts)
    assert S.scalp_outcome(sd_with_gap(10, 1_800_001), 0, 1) is None       # inside [i, i+30]
    assert S.scalp_outcome(sd_with_gap(30, 1_800_001), 0, 1) is None       # last step of window
    assert S.scalp_outcome(sd_with_gap(10, 1_800_000), 0, 1) is not None   # exactly 30m allowed
    assert S.scalp_outcome(sd_with_gap(31, 3_600_000), 0, 1) is not None   # gap after window


def _cand(sym, ts, reason, exit_ts, i=0):
    return {"sym": sym, "ts": ts, "i": i, "d": 1,
            "_out": {"exit_idx": i + 1, "exit_ts": exit_ts, "r": -1.0 if reason == "SL" else 1.0, "reason": reason}}


def test_scalp_cooldown_4_bars_then_8_after_two_consecutive_stops():
    T = 10_000 * B15
    cs = [_cand("ETHUSDT", T, "SL", T + 2 * B15),
          _cand("ETHUSDT", T + 5 * B15, "TP3", T + 6 * B15),       # 4-bar cooldown -> rejected
          _cand("ETHUSDT", T + 6 * B15, "SL", T + 8 * B15),        # 2nd consecutive SL
          _cand("ETHUSDT", T + 15 * B15, "TP3", T + 16 * B15),     # 8-bar cooldown -> rejected
          _cand("ETHUSDT", T + 16 * B15, "TP3", T + 17 * B15),
          _cand("ETHUSDT", T + 20 * B15, "TP3", T + 22 * B15),     # streak reset -> 4-bar cooldown -> rejected
          _cand("ETHUSDT", T + 21 * B15, "TP3", T + 22 * B15)]
    tr = S.run_scalp_portfolio(cs, None)
    assert [t["ts"] for t in tr] == [T, T + 6 * B15, T + 16 * B15, T + 21 * B15]


def test_scalp_max_one_entry_per_bar_and_directional_lock():
    T = 10_000 * B15
    tr = S.run_scalp_portfolio([_cand("SOLUSDT", T, "TP3", T + B15), _cand("ETHUSDT", T, "TP3", T + B15)], None)
    assert [t["sym"] for t in tr] == ["ETHUSDT"]
    lock = {"ETHUSDT": [(T + B15, T + 3 * B15)], "SOLUSDT": [], "BTCUSDT": []}
    tr = S.run_scalp_portfolio([_cand("ETHUSDT", T, "TP3", T + B15), _cand("ETHUSDT", T + B15, "TP3", T + 3 * B15)], lock)
    assert tr == []


def test_scalp_directional_max_same_dir_majors(monkeypatch):
    T = 10_000 * B1H
    monkeypatch.setattr(S, "directional_outcome", lambda sd, i, d: {"exit_idx": i + 5, "exit_ts": T + 5 * B1H, "r": 0.1, "reason": "TIMEOUT"})
    S.G["sd1h"] = {s: None for s in S.SYMS}
    cands = [{"sym": s, "i": 1, "ts": T, "d": 1} for s in S.SYMS]
    assert len(S.run_directional_portfolio(cands)) == 2


# ------------------------------------------------------------------ 4. causal 1h alignment
def test_scalp_h1_alignment_uses_only_closed_1h_bars():
    n1 = 260
    t1 = 1_700_000_000_000 // B1H * B1H
    raw1h = pd.DataFrame({"timestamp": t1 + np.arange(n1) * B1H, "close": 100 + np.cumsum(np.sin(np.arange(n1)) + 0.3)})
    sd = mk_sd([FLAT] * 200, t0=t1 + 205 * B1H)
    got = S.h1_dist_aligned(sd, raw1h)
    c = raw1h["close"]; sma = c.rolling(200).mean(); dist = ((c - sma) / sma).to_numpy()
    for k, ts in enumerate(sd.ts):
        j = int((ts + B15) // B1H * B1H - B1H - t1) // B1H     # last 1h bar with open+1h <= ts+15m
        assert t1 + j * B1H + B1H <= ts + B15 < t1 + (j + 1) * B1H + B1H
        assert got[k] == pytest.approx(dist[j])
    # perturbing a 1h bar that is still forming at decision time must not change the aligned value
    pert = raw1h.copy(); pert.loc[230:, "close"] *= 1.5
    got2 = S.h1_dist_aligned(sd, pert)
    dec = sd.ts + B15
    safe = dec < t1 + 230 * B1H + B1H
    assert safe.any() and np.allclose(got[safe], got2[safe], equal_nan=True)


# ------------------------------------------------------------------ 5. exit ladder / cost accounting
def test_scalp_sl_before_tp_and_exact_r_costs():
    e = 100 * (1 + TS_)
    sl, tp = e - 1.4, e + 2.2
    o = S.scalp_outcome(mk_sd([(100, 100, 100, 100), (100, 102.3, 98.5, 100)] + [FLAT] * 35), 0, 1)
    assert o["reason"] == "SL"
    assert o["r"] == pytest.approx(((sl * (1 - TS_) - e) / e - 2 * TF) * e / 1.4, abs=1e-9)
    o = S.scalp_outcome(mk_sd([(100, 100, 100, 100), (100, 102.3, 99.5, 100)] + [FLAT] * 35), 0, 1)
    assert o["reason"] == "TP3"
    assert o["r"] == pytest.approx(((tp * (1 - MS_) - e) / e - TF - MF) * e / 1.4, abs=1e-9)
    o = S.scalp_outcome(mk_sd([(100, 100, 100, 100)] + [FLAT] * 40), 0, 1)
    assert o["reason"] == "TIMEOUT" and o["exit_idx"] == 30
    assert o["r"] == pytest.approx(((100.1 * (1 - TS_) - e) / e - 2 * TF) * e / 1.4, abs=1e-9)
    es = 100 * (1 - TS_); sls = es + 1.4                                   # short tie-break
    o = S.scalp_outcome(mk_sd([(100, 100, 100, 100), (100, 101.5, 97.7, 100)] + [FLAT] * 35), 0, -1)
    assert o["reason"] == "SL"
    assert o["r"] == pytest.approx((-(sls * (1 + TS_) - es) / es - 2 * TF) * es / 1.4, abs=1e-9)


def _gp(rows, d="BUY", slm=1.5):
    return G.simulate_single_ladder_trade(mk_gp_df(rows), 0, d, slm, 1.5, 2.5, 1.5, 36, B5)


def test_gp_conservative_tie_break_and_gap_fill():
    e = 100 * (1 + TS_); sl = e - 1.5
    o = _gp([(100, 100, 100, 100), (100, 101.6, 98.4, 100)] + [FLAT] * 40)
    assert o["exit_reason"] == "STOP_LOSS" and not o["tp1_hit"]
    x = sl * (1 - TS_)
    assert o["r_mult"] == pytest.approx(((x - e) - e * TF - x * TF) / 1.5, abs=1e-9)
    es = 100 * (1 - TS_); sls = es + 1.5
    o = _gp([(100, 100, 100, 100), (100, 101.6, 98.3, 100)] + [FLAT] * 40, d="SELL")
    xs = sls * (1 + TS_)
    assert o["exit_reason"] == "STOP_LOSS"
    assert o["r_mult"] == pytest.approx((-(xs - es) - es * TF - xs * TF) / 1.5, abs=1e-9)
    o = _gp([(100, 100, 100, 100), (97, 97.5, 96.5, 97)] + [FLAT] * 40)   # gap through SL fills at open
    x = 97 * (1 - TS_)
    assert o["r_mult"] == pytest.approx(((x - e) - e * TF - x * TF) / 1.5, abs=1e-9) and o["r_mult"] < -1.0


def test_gp_tp1_breakeven_and_timeout_costs():
    e = 100 * (1 + TS_); tp1 = e + 1.5; be = e * 1.0004
    o = _gp([(100, 100, 100, 100), (100.1, 101.6, 100.1, 101.0), (100.5, 100.6, 99.9, 100.0)] + [FLAT] * 40)
    assert o["exit_reason"] == "BREAKEVEN_STOP" and o["tp1_hit"] and not o["tp2_hit"]
    x1, xb = tp1 * (1 - MS_), be * (1 - TS_)
    exp = 0.4 * (x1 - e) - 0.4 * x1 * MF + 0.6 * (xb - e) - 0.6 * xb * TF - e * TF
    assert o["r_mult"] == pytest.approx(exp / 1.5, abs=1e-9)
    o = _gp([(100, 100, 100, 100)] + [FLAT] * 40)
    assert o["exit_reason"] == "HORIZON_TIMEOUT" and o["exit_idx"] == 36
    xt = 100.1 * (1 - TS_)
    assert o["r_mult"] == pytest.approx(((xt - e) - e * TF - xt * TF) / 1.5, abs=1e-9)


def test_gp_full_ladder_tp1_tp2_trailing_runner():
    e = 100 * (1 + TS_); tp1, tp2 = e + 1.5, e + 2.5
    rows = [(100, 100, 100, 100), (100.1, 101.6, 100.5, 101.5), (101.5, 102.6, 101.5, 102.5),
            (102.0, 102.2, 100.9, 101.0)] + [FLAT] * 40
    o = _gp(rows)
    assert o["exit_reason"] == "TRAILING_STOP" and o["tp2_hit"] and o["exit_idx"] == 3
    trail = 102.6 - 1.5                                   # ratcheted from previous CLOSED bar high
    x1, x2, xr = tp1 * (1 - MS_), tp2 * (1 - MS_), trail * (1 - TS_)
    exp = (0.4 * (x1 - e) + 0.3 * (x2 - e) + 0.3 * (xr - e)) - (e * TF + 0.4 * x1 * MF + 0.3 * x2 * MF + 0.3 * xr * TF)
    assert o["r_mult"] == pytest.approx(exp / 1.5, abs=1e-9)


def test_directional_1h_ladder_parity_between_scripts():
    rows = [(100, 100, 100, 100), (100.0, 101.6, 100.3, 101.5), (101.5, 102.6, 101.4, 102.5),
            (102.5, 103.5, 102.6, 103.4), (102.9, 103.0, 102.0, 102.2)] + [FLAT] * 50
    a = S.directional_outcome(mk_sd(rows, bar=B1H), 0, 1)
    p = S.DIRP
    b = G.simulate_single_ladder_trade(mk_gp_df(rows, bar=B1H), 0, "BUY", p.sl_atr_mult, p.tp1_atr_mult,
                                       p.tp2_atr_mult, p.trail_atr_mult, p.max_holding_bars, B1H)
    assert (a["reason"], b["exit_reason"]) == ("TRAIL", "TRAILING_STOP") and a["exit_idx"] == b["exit_idx"] == 4
    assert a["r"] == pytest.approx(b["r_mult"], abs=0.01)
    tie = [(100, 100, 100, 100), (100, 101.6, 98.9, 100)] + [FLAT] * 50     # SL & TP1 same bar
    a = S.directional_outcome(mk_sd(tie, bar=B1H), 0, 1)
    b = G.simulate_single_ladder_trade(mk_gp_df(tie, bar=B1H), 0, "BUY", p.sl_atr_mult, p.tp1_atr_mult,
                                       p.tp2_atr_mult, p.trail_atr_mult, p.max_holding_bars, B1H)
    assert a["reason"] == "SL" and b["exit_reason"] == "STOP_LOSS" and a["r"] == pytest.approx(b["r_mult"], abs=0.01)


# ------------------------------------------------------------------ GP portfolio constraints / metric edges
def _sig(sym, ts, rank=1.0, hold=3):
    return {"sym": sym, "ts": ts, "rank_score": rank, "exit_ts": ts + B5 + hold * B5, "r_mult": 0.1}


def test_gp_portfolio_one_per_bar_cooldown_concurrency_quota():
    T = G.SPLIT_TS_GP + 3600_000
    tr, _ = G.simulate_gp_portfolio_split([_sig("AUSDT", T, 1.0), _sig("BUSDT", T, 2.0)], 1.0)
    assert [t["sym"] for t in tr] == ["BUSDT"]
    s0 = _sig("AUSDT", T, hold=2)                       # exits at T + 3 bars (close)
    cds = [s0, _sig("AUSDT", T + 13 * B5), _sig("AUSDT", T + 14 * B5)]
    tr, _ = G.simulate_gp_portfolio_split(cds, 1.0)
    assert [t["ts"] for t in tr] == [T, T + 14 * B5]   # decision close >= exit close + 12 bars
    many = [_sig(f"S{k}USDT", T + k * B5, hold=500) for k in range(10)]
    tr, _ = G.simulate_gp_portfolio_split(many, 1.0, daily_quota=99)
    assert len(tr) == 6
    tr, _ = G.simulate_gp_portfolio_split(many, 1.0)
    assert len(tr) == 3


def test_pf_and_metric_edge_cases():
    assert S.pf_of(np.array([1.0, 2.0])) == 99.0 and S.pf_of(np.array([])) == 0.0
    assert S.pf_of(np.array([2.0, -1.0])) == pytest.approx(2.0)
    m = G.compute_split_metrics([{"r_mult": 1.0}, {"r_mult": 0.5}], 1.0, 10, 0.1)
    assert m["pf"] == 99.0 and m["pf_ex_top1"] == 99.0 and m["n"] == 2
    z = G.compute_split_metrics([], 1.0, 10, 0.1)
    assert z["n"] == 0 and z["pf"] == 0.0
    tr = [{"r": r, "exit_ts": S.SPLIT_TS + k * DAY, "ts": S.SPLIT_TS + k * DAY} for k, r in enumerate([1, -1, 2, -0.5] * 4)]
    sm = S.metrics(tr, S.SPLIT_TS, S.SPLIT_TS + 20 * DAY, 1.0, 100)
    assert sm["n"] == 16 and sm["pf"] == pytest.approx(12 / 6) and all(np.isfinite(v) for v in (sm["dsr"], sm["avg_r"]))
    assert S.metrics([], S.SPLIT_TS, S.SPLIT_TS + DAY, 1.0, 100)["n"] == 0


# ------------------------------------------------------------------ 6. purged training / no disk models
class _FakeTrainer:
    seen: list = []

    def __init__(self, *a, **k):
        self.device = "cpu"

    def prepare_dataset(self, sub, feature_columns=None):
        _FakeTrainer.seen.append(int(sub["timestamp"].max()))
        return sub[feature_columns].astype(float), pd.Series(np.zeros(len(sub), dtype=int), index=sub.index)

    def _build_estimator(self, random_state=42):
        class E:
            def fit(self, X, y):
                self.n = len(X)
        return E()


def _feat_df(n, t0, bar):
    rng = np.random.default_rng(1)
    return pd.DataFrame({"timestamp": t0 + np.arange(n, dtype=np.int64) * bar, "close": 100 + rng.normal(0, 1, n),
                         "f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})


def test_scalp_purged_fit_never_sees_rows_at_or_after_cutoff(monkeypatch):
    monkeypatch.setattr(S, "ModelTrainer", _FakeTrainer)
    monkeypatch.setattr(S, "_disable_estimator_parallelism", lambda e: e)
    _FakeTrainer.seen = []
    t0 = S.SPLIT_TS - 300 * B15
    sds = {}
    for s in S.SYMS:
        sd = S.SD(); sd.df = _feat_df(600, t0, B15); sds[s] = sd
    cutoff = S.SPLIT_TS - 10 * B15
    _, n = S.fit_ensemble(sds, ["f1", "f2"], cutoff, 1.2, 1.2, native_nan=False)
    assert _FakeTrainer.seen and max(_FakeTrainer.seen) < cutoff
    assert n == 3 * 290


def test_gp_purged_1h_training_before_is_start_minus_7d(monkeypatch):
    monkeypatch.setattr(G, "ModelTrainer", _FakeTrainer)
    monkeypatch.setattr(G, "_disable_estimator_parallelism", lambda e: e)
    monkeypatch.setattr(G, "_DefaultMLEnsemble", lambda **k: k)
    _FakeTrainer.seen = []
    assert G.TRAIN_1H_CUTOFF_TS == G.IS_START_TS - 7 * DAY and G.EMBARGO_GP_MS == 36 * B5
    majors = {s: _feat_df(800, G.TRAIN_1H_CUTOFF_TS - 400 * B1H, B1H) for s in G.MAJORS}
    _, n, cols = G.train_purged_1h_ensemble(majors, G.TRAIN_1H_CUTOFF_TS)
    assert max(_FakeTrainer.seen) < G.TRAIN_1H_CUTOFF_TS and n == 3 * 400 and "timestamp" not in cols


def test_scripts_never_load_disk_models_and_disable_learning_tracker():
    for mod in (S, G):
        src = open(mod.__file__).read()
        assert not re.search(r"joblib\.load|import joblib|_get_default_model|pickle\.load\(open\([^)]*joblib", src)
        calls = re.findall(r"TechnicalAgent\(([^)]*)\)", src)
        assert calls and all("model=" in c for c in calls)
        assert src.count("TradingCore(") == len(re.findall(r"\.learning_tracker = None", src))
    s_src = open(S.__file__).read()
    assert "SPLIT_TS - 10 * B15" in s_src and "SPLIT_TS - 7 * DAY" in s_src
    assert 't["exit_ts"] < SPLIT_TS' in s_src                      # S7 meta-labeler trains on IS exits only


def test_technical_agent_with_injected_model_never_touches_disk(monkeypatch):
    import joblib
    from agents.technical_agent import TechnicalAgent
    from core import TradingCore

    def boom(*a, **k):
        raise AssertionError("disk model load attempted")
    monkeypatch.setattr(joblib, "load", boom)
    ta = TechnicalAgent(model=object())
    core = TradingCore(min_consensus_threshold=0.30, technical_agent=ta)
    assert ta._uses_default_model is False and core.technical_agent is ta and core.learning_tracker is None


# ------------------------------------------------------------------ 7. JSON report schemas
METRICS = ("n", "trades_per_day", "wr_pct", "pf", "pf_ex_top1", "sum_r", "avg_r", "max_dd_r", "sharpe_per_trade", "dsr")


def _check_block(b, days):
    for k in METRICS:
        assert isinstance(b.get(k), (int, float)) and np.isfinite(b[k]), k
    if b["n"] > 0:
        assert b["trades_per_day"] == pytest.approx(b["n"] / days, rel=1e-3, abs=1e-3)
        if b["pf"] not in (0.0, 99.0):
            assert (b["pf"] > 1.0) == (b["sum_r"] > 0.0)
        assert b["avg_r"] == pytest.approx(b["sum_r"] / b["n"], abs=1e-3)


def test_scalp_report_schema_variants_and_verdicts():
    r = _load(SCALP_JSON)
    m = r["metadata"]
    assert m["split_ts"] == S.SPLIT_TS and m["committee_errors"] == 0 and m["n_trials_dsr"] == m["prior_trials"] + 8
    assert m["oos_days"] == pytest.approx((m["end_ts"] - S.SPLIT_TS) / DAY)
    assert set(S.VARIANT_BASE) | {"S7_META_LABEL_SQZ"} <= set(r["variants"])
    days = {"is_standalone": (S.SPLIT_TS - m["is_start_ts"]) / DAY, "oos_standalone": m["oos_days"],
            "oos_with_dir_lock": m["oos_days"], "diagnostic_4_5d": m["diag_days"]}
    s0 = r["variants"]["S0_PROD"]
    for name, v in r["variants"].items():
        for blk, d in days.items():
            _check_block(v[blk], d)
            if v[blk]["n"] >= 10:
                lo, hi = v[blk]["bootstrap_ci95_avg_r"]
                assert lo <= v[blk]["avg_r"] <= hi
        assert v["verdict"] == S.verdict(v["oos_standalone"], v["is_standalone"], s0["oos_standalone"]["trades_per_day"])
        assert v["verdict_with_dir_lock"] == S.verdict(v["oos_with_dir_lock"], v["is_standalone"],
                                                       s0["oos_with_dir_lock"]["trades_per_day"])
    for blk in ("is", "oos"):
        _check_block(r["directional_1h"][blk], days["oos_standalone" if blk == "oos" else "is_standalone"])


def test_gp_report_schema_variants_universe_and_verdicts():
    r = _load(GP_JSON)
    m = r["metadata"]
    assert m["split_ts"] == G.SPLIT_TS_GP and m["embargo_ms"] == G.EMBARGO_GP_MS
    assert m["purged_1h_train_cutoff_ts"] == G.TRAIN_1H_CUTOFF_TS and m["n_trials_dsr"] == m["prior_trials"] + 8
    assert all(GoldenPotAgent.extract_base_asset(s) in NON_CRYPTO_BASE_EXCLUSIONS for s in m["excluded_tradfi_sample"])
    assert m["total_5m_symbols_in_db"] >= (m["excluded_majors_count"] + m["excluded_stables_count"]
                                           + m["excluded_tradfi_count"] + m["kept_altcoins_count"])
    names = [f"G{k}" for k in range(8)]
    assert sorted(x[:2] for x in r["variants"]) == names
    g0 = r["variants"]["G0_PROD"]["oos"]["trades_per_day"]
    for name, v in r["variants"].items():
        f = v["funnel_counts"]
        assert f["raw_triggers"] >= f["post_macro"] >= f["post_climax"] >= f["post_committee"] >= v["is"]["n"] + v["oos"]["n"]
        for blk, d in (("is", m["is_window_days"]), ("oos", m["oos_window_days"]),
                       ("oos_bear", m["oos_bear_days"]), ("oos_non_bear", m["oos_non_bear_days"])):
            _check_block(v[blk], d)
            lo, hi = v[blk]["bootstrap_ci95_avg_r"]
            assert lo <= hi
        assert v["oos_bear"]["n"] + v["oos_non_bear"]["n"] == v["oos"]["n"]
        o, i = v["oos"], v["is"]
        go = (o["n"] >= 15 and o["pf"] >= 1.25 and o["pf_ex_top1"] >= 1.10 and o["avg_r"] > 0 and i["pf"] >= 1.0
              and o["trades_per_day"] > g0)
        assert v["verdict"] == ("GO" if go else "NO-GO")
    d = r["directional_1h_oos_20d"]
    for k in ("n", "trades_per_day", "pf", "sum_r", "avg_r"):
        assert isinstance(d[k], (int, float)) and np.isfinite(d[k])
