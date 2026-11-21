"""Independent QA test suite for Wave P3: SWING_4H (Donchian40/ExitCh10), log hygiene, and log rotation."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sqlite3
import sys
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from data.swing_4h_strategy import (  # noqa: E402
    SWING_4H_BAR_MS,
    compute_swing_4h_indicators,
    evaluate_swing_4h_entry,
    resample_1h_to_closed_4h,
    should_exit_swing_4h_channel,
)
import flask_interface  # noqa: E402
from rollback_guard import DefensiveRiskPolicy, RiskPolicy  # noqa: E402
import scripts.pilot_report as pilot_report  # noqa: E402
import scripts.run_paper_simulation as rps  # noqa: E402
import scripts.system_supervisor as system_supervisor  # noqa: E402
from signal_generator import TradeSignal  # noqa: E402
from smart_trade import PaperTradingEngine  # noqa: E402

HOUR_MS = 3_600_000
BASE_4H_TS = (1_700_000_000_000 // SWING_4H_BAR_MS) * SWING_4H_BAR_MS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_1h_bucket(
    bucket_ts: int,
    o: float,
    h: float,
    l: float,
    c: float,
    v: float = 10.0,
    n_hours: int = 4,
) -> List[Dict[str, float]]:
    """Create `n_hours` 1h candles inside a single 4h bucket with exact target OHLCV."""
    rows: List[Dict[str, float]] = []
    mid = (o + c) / 2.0
    per_v = v / max(1, n_hours)
    for k in range(n_hours):
        bo = o if k == 0 else mid
        bc = c if k == n_hours - 1 else mid
        bh = h if k == 1 or n_hours == 1 else max(bo, bc)
        bl = l if k == 2 or n_hours <= 2 else min(bo, bc)
        bh = max(bh, bo, bc)
        bl = min(bl, bo, bc)
        rows.append(
            {
                "timestamp": bucket_ts + k * HOUR_MS,
                "open": bo,
                "high": bh,
                "low": bl,
                "close": bc,
                "volume": per_v,
            }
        )
    return rows


def _make_4h_df(
    n: int = 215,
    base_price: float = 100.0,
    start_ts: int = BASE_4H_TS,
) -> pd.DataFrame:
    """Create `n` deterministic 4h bars oscillating around `base_price`."""
    rows = []
    for i in range(n):
        offset = 0.4 * math.sin(i * 0.5)
        o = base_price + offset
        c = base_price - offset
        h = base_price + 1.0 + (i % 3) * 0.1
        l = base_price - 1.0 - (i % 2) * 0.1
        rows.append(
            {
                "timestamp": start_ts + i * SWING_4H_BAR_MS,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 100.0 + i,
            }
        )
    return pd.DataFrame(rows)


def _make_1h_series_from_4h_specs(
    specs: List[Tuple[int, float, float, float, float, float]]
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for ts, o, h, l, c, v in specs:
        rows.extend(_make_1h_bucket(ts, o, h, l, c, v, n_hours=4))
    return pd.DataFrame(rows)


def _build_1h_breakout_df(
    end_bucket_ts: int,
    n_4h: int = 210,
    breakout_close: float = 105.0,
    breakout_high: float = 106.0,
    breakout_low: float = 100.0,
    extra_4h_bars: Optional[List[Tuple[float, float, float, float]]] = None,
) -> pd.DataFrame:
    """Build 1h candles resampling to `n_4h` closed 4h bars ending in a breakout at `end_bucket_ts`."""
    start_ts = end_bucket_ts - (n_4h - 1) * SWING_4H_BAR_MS
    specs: List[Tuple[int, float, float, float, float, float]] = []
    for i in range(n_4h - 1):
        ts = start_ts + i * SWING_4H_BAR_MS
        specs.append((ts, 100.0, 101.0, 99.0, 100.0, 40.0))
    specs.append((end_bucket_ts, 100.5, breakout_high, breakout_low, breakout_close, 80.0))
    if extra_4h_bars:
        for idx, (o, h, l, c) in enumerate(extra_4h_bars, start=1):
            specs.append((end_bucket_ts + idx * SWING_4H_BAR_MS, o, h, l, c, 80.0))
    return _make_1h_series_from_4h_specs(specs)


def _make_signal(
    symbol: str = "BTCUSDT",
    direction: str = "BUY",
    price: float = 100.0,
    atr: float = 2.0,
    ts: int = BASE_4H_TS,
    confidence: float = 0.75,
) -> TradeSignal:
    sgn = 1.0 if direction == "BUY" else -1.0
    return TradeSignal(
        signal_id=f"sig_{symbol}_{ts}_{direction}",
        timestamp=ts,
        symbol=symbol,
        direction=direction,
        entry_price=price,
        stop_loss=price - sgn * 2.5 * atr,
        take_profit=0.0,
        atr=atr,
        confidence=confidence,
        consensus_score=sgn * confidence,
        position_size=1000.0,
        risk_dollars=50.0,
        rationale="SWING_4H test breakout",
        metrics={"regime_alignment": True},
    )


def _make_engine(tmp_path, name: str = "swing4h.db", restore: bool = False) -> PaperTradingEngine:
    db_file = tmp_path / name
    return PaperTradingEngine(initial_capital=10_000.0, db_path=str(db_file), restore=restore)


def _funnel_counts(db_path: str) -> Dict[Tuple[str, str], int]:
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT strategy, stage, SUM(count) FROM signal_funnel GROUP BY strategy, stage"
        ).fetchall()
    return {(str(s), str(st)): int(c) for s, st, c in rows}


# ---------------------------------------------------------------------------
# 1. data/swing_4h_strategy.py
# ---------------------------------------------------------------------------

def test_resample_1h_to_closed_4h_complete_and_drops_incomplete_buckets():
    b0 = BASE_4H_TS
    b1 = b0 + SWING_4H_BAR_MS
    b2 = b1 + SWING_4H_BAR_MS
    b3 = b2 + SWING_4H_BAR_MS
    b4 = b3 + SWING_4H_BAR_MS

    rows = []
    # Leading incomplete bucket (2 bars)
    rows.extend(_make_1h_bucket(b0, 90.0, 92.0, 89.0, 91.0, 20.0, n_hours=2))
    # Complete bucket b1
    rows.extend(
        [
            {"timestamp": b1 + 0 * HOUR_MS, "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 10.0},
            {"timestamp": b1 + 1 * HOUR_MS, "open": 101.0, "high": 108.0, "low": 100.0, "close": 105.0, "volume": 15.0},
            {"timestamp": b1 + 2 * HOUR_MS, "open": 105.0, "high": 106.0, "low": 97.5, "close": 103.0, "volume": 25.0},
            {"timestamp": b1 + 3 * HOUR_MS, "open": 103.0, "high": 104.0, "low": 102.0, "close": 103.5, "volume": 30.0},
        ]
    )
    # Middle incomplete bucket (3 bars)
    rows.extend(_make_1h_bucket(b2, 103.5, 105.0, 102.0, 104.0, 30.0, n_hours=3))
    # Complete bucket b3 with exact integer-floor timestamp offset (+500 ms inside hour)
    rows.extend(
        [
            {"timestamp": b3 + 0 * HOUR_MS + 500, "open": 110.0, "high": 111.0, "low": 109.0, "close": 110.5, "volume": 5.0},
            {"timestamp": b3 + 1 * HOUR_MS + 500, "open": 110.5, "high": 115.0, "low": 110.0, "close": 114.0, "volume": 6.0},
            {"timestamp": b3 + 2 * HOUR_MS + 500, "open": 114.0, "high": 114.5, "low": 107.0, "close": 108.0, "volume": 7.0},
            {"timestamp": b3 + 3 * HOUR_MS + 500, "open": 108.0, "high": 112.0, "low": 107.5, "close": 111.5, "volume": 8.0},
        ]
    )
    # Trailing incomplete bucket (3 bars)
    rows.extend(_make_1h_bucket(b4, 111.5, 113.0, 110.0, 112.0, 40.0, n_hours=3))

    out = resample_1h_to_closed_4h(pd.DataFrame(rows))
    assert list(out["timestamp"]) == [b1, b3]

    r1 = out.iloc[0]
    assert r1["open"] == pytest.approx(100.0)
    assert r1["high"] == pytest.approx(108.0)
    assert r1["low"] == pytest.approx(97.5)
    assert r1["close"] == pytest.approx(103.5)
    assert r1["volume"] == pytest.approx(80.0)

    r2 = out.iloc[1]
    assert r2["open"] == pytest.approx(110.0)
    assert r2["high"] == pytest.approx(115.0)
    assert r2["low"] == pytest.approx(107.0)
    assert r2["close"] == pytest.approx(111.5)
    assert r2["volume"] == pytest.approx(26.0)


def test_resample_1h_to_closed_4h_drops_forming_bucket_by_now_ms():
    b0 = BASE_4H_TS
    b1 = b0 + SWING_4H_BAR_MS
    b2 = b1 + SWING_4H_BAR_MS
    rows = (
        _make_1h_bucket(b0, 100.0, 102.0, 99.0, 101.0, 40.0)
        + _make_1h_bucket(b1, 101.0, 103.0, 100.0, 102.0, 40.0)
        + _make_1h_bucket(b2, 102.0, 104.0, 101.0, 103.0, 40.0)
    )
    df = pd.DataFrame(rows)

    # Exact close boundary of b1 keeps b0 and b1, drops b2 even though b2 has 4 bars
    out_exact = resample_1h_to_closed_4h(df, now_ms=b1 + SWING_4H_BAR_MS)
    assert list(out_exact["timestamp"]) == [b0, b1]

    # 1 ms before b1 closes drops b1 as well
    out_before = resample_1h_to_closed_4h(df, now_ms=b1 + SWING_4H_BAR_MS - 1)
    assert list(out_before["timestamp"]) == [b0]


def test_resample_1h_to_closed_4h_sorts_dedups_and_handles_invalid_inputs():
    b0 = BASE_4H_TS
    b1 = b0 + SWING_4H_BAR_MS
    b2 = b1 + SWING_4H_BAR_MS
    rows = [
        {"timestamp": b1 + 3 * HOUR_MS, "open": 203.0, "high": 204.0, "low": 202.0, "close": 203.5, "volume": 10.0},
        {"timestamp": b0 + 2 * HOUR_MS, "open": 102.0, "high": 103.0, "low": 98.0, "close": 102.5, "volume": 10.0},
        {"timestamp": b1 + 0 * HOUR_MS, "open": 200.0, "high": 201.0, "low": 199.0, "close": 200.5, "volume": 10.0},
        {"timestamp": b0 + 0 * HOUR_MS, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0},
        # Duplicate timestamp with distinct values (must be deduped, not double-counted in cnt or volume)
        {"timestamp": b0 + 0 * HOUR_MS, "open": 999.0, "high": 999.0, "low": 1.0, "close": 999.0, "volume": 500.0},
        {"timestamp": b1 + 1 * HOUR_MS, "open": 200.5, "high": 206.0, "low": 200.0, "close": 205.0, "volume": 10.0},
        {"timestamp": b0 + 3 * HOUR_MS, "open": 102.5, "high": 104.0, "low": 102.0, "close": 103.0, "volume": 10.0},
        {"timestamp": b1 + 2 * HOUR_MS, "open": 205.0, "high": 205.5, "low": 198.0, "close": 203.0, "volume": 10.0},
        {"timestamp": b0 + 1 * HOUR_MS, "open": 100.5, "high": 105.0, "low": 100.0, "close": 102.0, "volume": 10.0},
        # Bucket b2 with 1 NaN close -> only 3 valid bars -> must be dropped
        {"timestamp": b2 + 0 * HOUR_MS, "open": 300.0, "high": 301.0, "low": 299.0, "close": 300.5, "volume": 10.0},
        {"timestamp": b2 + 1 * HOUR_MS, "open": 300.5, "high": 302.0, "low": 300.0, "close": float("nan"), "volume": 10.0},
        {"timestamp": b2 + 2 * HOUR_MS, "open": 301.0, "high": 303.0, "low": 300.0, "close": 302.0, "volume": 10.0},
        {"timestamp": b2 + 3 * HOUR_MS, "open": 302.0, "high": 304.0, "low": 301.0, "close": 303.0, "volume": 10.0},
    ]
    out = resample_1h_to_closed_4h(pd.DataFrame(rows))
    assert list(out["timestamp"]) == [b0, b1]
    assert out.iloc[0]["volume"] == pytest.approx(40.0)
    assert out.iloc[0]["open"] == pytest.approx(100.0)
    assert out.iloc[0]["high"] == pytest.approx(105.0)
    assert out.iloc[0]["low"] == pytest.approx(98.0)
    assert out.iloc[0]["close"] == pytest.approx(103.0)

    expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    for bad in (
        None,
        pd.DataFrame(columns=expected_cols),
        pd.DataFrame([{"timestamp": np.nan, "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan, "volume": np.nan}]),
    ):
        res = resample_1h_to_closed_4h(bad)
        assert res.empty
        assert list(res.columns) == expected_cols


def test_compute_swing_4h_indicators_values_and_strict_causality():
    df_base = _make_4h_df(n=220)
    ind = compute_swing_4h_indicators(df_base)

    # SMA 200 warmup and exact value
    assert pd.isna(ind["sma_200"].iloc[198])
    assert ind["sma_200"].iloc[199] == pytest.approx(df_base["close"].iloc[:200].mean())
    assert ind["sma_200"].iloc[-1] == pytest.approx(df_base["close"].iloc[-200:].mean())

    # Wilder ATR14 manual recursion check
    high = df_base["high"].to_numpy(dtype=float)
    low = df_base["low"].to_numpy(dtype=float)
    close = df_base["close"].to_numpy(dtype=float)
    alpha = 1.0 / 14.0
    atr_manual = high[0] - low[0]
    for i in range(1, len(df_base)):
        tr_i = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        atr_manual = alpha * tr_i + (1.0 - alpha) * atr_manual
    assert ind["atr_14"].iloc[-1] == pytest.approx(atr_manual)

    # Shifted Donchian channels exact values
    last_idx = len(df_base) - 1
    assert ind["h40"].iloc[last_idx] == pytest.approx(df_base["high"].iloc[last_idx - 40 : last_idx].max())
    assert ind["l40"].iloc[last_idx] == pytest.approx(df_base["low"].iloc[last_idx - 40 : last_idx].min())
    assert ind["h10"].iloc[last_idx] == pytest.approx(df_base["high"].iloc[last_idx - 10 : last_idx].max())
    assert ind["l10"].iloc[last_idx] == pytest.approx(df_base["low"].iloc[last_idx - 10 : last_idx].min())

    # Strict causality: mutating final bar's high/low must NOT affect final bar's h40/l40/h10/l10
    df_mut = df_base.copy()
    df_mut.loc[last_idx, "high"] = 999_999.0
    df_mut.loc[last_idx, "low"] = 0.0001
    ind_mut = compute_swing_4h_indicators(df_mut)
    for col in ("h40", "l40", "h10", "l10"):
        assert ind_mut[col].iloc[last_idx] == ind[col].iloc[last_idx], f"Lookahead leaked into {col}"

    assert compute_swing_4h_indicators(None).empty
    assert compute_swing_4h_indicators(pd.DataFrame()).empty


def test_evaluate_swing_4h_entry_buy_sell_guards_and_warmup():
    def _row_df(**kw) -> pd.DataFrame:
        base = {
            "close": 110.0,
            "atr_14": 2.0,
            "sma_200": 100.0,
            "h40": 105.0,
            "l40": 95.0,
            "h10": 103.0,
            "l10": 97.0,
            "dist_sma_200": 0.10,
        }
        base.update(kw)
        return pd.DataFrame([base])

    # Valid BUY
    res_buy = evaluate_swing_4h_entry(_row_df(close=110.0, h40=105.0, dist_sma_200=0.10))
    assert res_buy is not None
    assert res_buy[0] == "BUY"
    assert "BUY" in res_buy[1]
    assert res_buy[2]["close"] == pytest.approx(110.0)

    # BUY boundaries: dist_sma_200 == 0.0 and == 0.20 pass; > 0.20 or < 0.0 fail; close == h40 fails
    assert evaluate_swing_4h_entry(_row_df(close=106.0, h40=105.0, dist_sma_200=0.0)) is not None
    assert evaluate_swing_4h_entry(_row_df(close=120.0, h40=105.0, dist_sma_200=0.20)) is not None
    assert evaluate_swing_4h_entry(_row_df(close=121.0, h40=105.0, dist_sma_200=0.2001)) is None
    assert evaluate_swing_4h_entry(_row_df(close=106.0, h40=105.0, dist_sma_200=-0.01)) is None
    assert evaluate_swing_4h_entry(_row_df(close=105.0, h40=105.0, dist_sma_200=0.05)) is None

    # Valid SELL
    res_sell = evaluate_swing_4h_entry(_row_df(close=90.0, l40=95.0, dist_sma_200=-0.10))
    assert res_sell is not None
    assert res_sell[0] == "SELL"
    assert "SELL" in res_sell[1]

    # SELL boundaries: dist == 0.0 and == -0.20 pass; < -0.20 or > 0.0 fail; close == l40 fails
    assert evaluate_swing_4h_entry(_row_df(close=94.0, l40=95.0, dist_sma_200=0.0)) is not None
    assert evaluate_swing_4h_entry(_row_df(close=80.0, l40=95.0, dist_sma_200=-0.20)) is not None
    assert evaluate_swing_4h_entry(_row_df(close=79.0, l40=95.0, dist_sma_200=-0.2001)) is None
    assert evaluate_swing_4h_entry(_row_df(close=94.0, l40=95.0, dist_sma_200=0.01)) is None
    assert evaluate_swing_4h_entry(_row_df(close=95.0, l40=95.0, dist_sma_200=-0.05)) is None

    # NaN / insufficient warmup / empty / None
    assert evaluate_swing_4h_entry(None) is None
    assert evaluate_swing_4h_entry(pd.DataFrame()) is None
    assert evaluate_swing_4h_entry(_row_df(sma_200=float("nan"), dist_sma_200=float("nan"))) is None
    short_ind = compute_swing_4h_indicators(_make_4h_df(n=150))
    assert evaluate_swing_4h_entry(short_ind) is None


def test_should_exit_swing_4h_channel_long_and_short():
    # LONG: exit iff close < l10
    hit, why = should_exit_swing_4h_channel(pd.DataFrame([{"close": 96.9, "l10": 97.0, "h10": 103.0}]), "LONG")
    assert hit is True and why != ""
    assert should_exit_swing_4h_channel(pd.DataFrame([{"close": 97.0, "l10": 97.0, "h10": 103.0}]), "LONG") == (False, "")
    assert should_exit_swing_4h_channel(pd.DataFrame([{"close": 98.5, "l10": 97.0, "h10": 103.0}]), "LONG") == (False, "")

    # SHORT: exit iff close > h10
    hit_s, why_s = should_exit_swing_4h_channel(pd.DataFrame([{"close": 103.1, "l10": 97.0, "h10": 103.0}]), "SHORT")
    assert hit_s is True and why_s != ""
    assert should_exit_swing_4h_channel(pd.DataFrame([{"close": 103.0, "l10": 97.0, "h10": 103.0}]), "SHORT") == (False, "")
    assert should_exit_swing_4h_channel(pd.DataFrame([{"close": 101.0, "l10": 97.0, "h10": 103.0}]), "SHORT") == (False, "")

    # Empty / None / NaN
    assert should_exit_swing_4h_channel(None, "LONG") == (False, "")
    assert should_exit_swing_4h_channel(pd.DataFrame(), "SHORT") == (False, "")
    assert should_exit_swing_4h_channel(pd.DataFrame([{"close": 90.0, "l10": float("nan")}]), "LONG") == (False, "")


# ---------------------------------------------------------------------------
# 2. smart_trade.py (PaperTradingEngine in tmp_path SQLite DB)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "direction,side,expected_action,sgn",
    [
        ("BUY", "LONG", "OPEN_SWING_4H_LONG", 1.0),
        ("SELL", "SHORT", "OPEN_SWING_4H_SHORT", -1.0),
    ],
)
def test_execute_swing_4h_signal_contract(tmp_path, direction, side, expected_action, sgn):
    engine = _make_engine(tmp_path, name=f"exec_{side.lower()}.db")
    price = 100.0
    atr = 2.0
    sig = _make_signal(direction=direction, price=price, atr=atr, ts=BASE_4H_TS, confidence=0.75)

    pos = engine.execute_swing_4h_signal(sig, current_price=price, timestamp=BASE_4H_TS, leverage=2.0)
    assert pos is not None
    assert pos.strategy_type == "SWING_4H"
    assert pos.side == side
    assert pos.leverage <= 2.0

    expected_entry = price * (1.0 + sgn * engine.slippage_market)
    assert pos.entry_price == pytest.approx(expected_entry)
    assert pos.fees_paid == pytest.approx(pos.notional * engine.taker_fee)

    assert pos.tp1_price == 0.0
    assert pos.tp2_price == 0.0
    assert pos.tp3_price == 0.0
    assert pos.tp1_fraction == 0.0
    assert pos.tp2_fraction == 0.0

    assert pos.trailing_active is True
    assert pos.trailing_atr_mult == pytest.approx(3.5)
    assert pos.trailing_stop_price == 0.0
    assert pos.stop_loss == pytest.approx(pos.entry_price - sgn * 2.5 * atr)

    assert len(engine.trade_history) == 1
    assert engine.trade_history[0].action == expected_action
    assert engine.trade_history[0].fee == pytest.approx(pos.notional * engine.taker_fee)


def test_execute_swing_4h_signal_caps_leverage_at_2x(tmp_path):
    engine = _make_engine(tmp_path, name="lev_cap.db")
    sig = _make_signal(direction="BUY", price=100.0, atr=0.5, ts=BASE_4H_TS, confidence=0.95)
    pos = engine.execute_swing_4h_signal(sig, current_price=100.0, timestamp=BASE_4H_TS, leverage=4.0)
    assert pos is not None
    assert pos.leverage <= 2.0


def test_swing_4h_pure_trailing_case_a_initial_stop_loss(tmp_path):
    engine = _make_engine(tmp_path, name="case_a.db")
    sig = _make_signal(direction="BUY", price=100.0, atr=2.0, ts=BASE_4H_TS)
    pos = engine.execute_swing_4h_signal(sig, current_price=100.0, timestamp=BASE_4H_TS, leverage=2.0)
    assert pos is not None
    t1 = pos.entry_timestamp

    # Before any closed 4h bar ratchet, hitting initial SL closes as STOP_LOSS (not TRAILING_STOP / BREAKEVEN_STOP)
    triggered = engine.update_market_tick(
        "BTCUSDT", high=101.0, low=94.0, close=94.5, timestamp=t1, open_price=100.0
    )
    assert len(triggered) == 1
    assert triggered[0].action == "STOP_LOSS"
    assert pos.status == "CLOSED"


def test_swing_4h_pure_trailing_case_b_closed_4h_ratchet_and_trailing_stop(tmp_path):
    engine = _make_engine(tmp_path, name="case_b.db")
    atr = 2.0
    sig = _make_signal(direction="BUY", price=100.0, atr=atr, ts=BASE_4H_TS)
    pos = engine.execute_swing_4h_signal(sig, current_price=100.0, timestamp=BASE_4H_TS, leverage=2.0)
    assert pos is not None
    initial_sl = pos.stop_loss
    t1 = pos.entry_timestamp

    # Feed closed sub-bars inside 1st 4h bucket making new high (112.0) -> must NOT ratchet intra-bucket
    for offset, (h, l, c) in enumerate([(106.0, 99.5, 105.0), (112.0, 104.0, 110.0), (111.0, 108.0, 109.0)]):
        tr = engine.update_market_tick(
            "BTCUSDT", high=h, low=l, close=c, timestamp=t1 + offset * HOUR_MS, open_price=100.5
        )
        assert tr == []
        assert pos.trailing_stop_price == 0.0
        assert pos.stop_loss == pytest.approx(initial_sl)

    # Next 4h bucket arrives -> ratchets trailing_stop_price & stop_loss upward to 112.0 - 3.5 * 2.0 = 105.0
    t2 = t1 + SWING_4H_BAR_MS
    expected_trail = 112.0 - 3.5 * atr
    tr2 = engine.update_market_tick(
        "BTCUSDT", high=110.0, low=106.0, close=108.0, timestamp=t2, open_price=109.0
    )
    assert tr2 == []
    assert pos.trailing_stop_price == pytest.approx(expected_trail)
    assert pos.stop_loss == pytest.approx(expected_trail)
    assert pos.stop_loss > initial_sl

    # Pullback hits ratcheted stop -> closes as TRAILING_STOP with zero partial TP trades
    tr3 = engine.update_market_tick(
        "BTCUSDT", high=108.0, low=104.0, close=104.5, timestamp=t2 + HOUR_MS, open_price=108.0
    )
    closing = [t for t in tr3 if t.action != "FUNDING"]
    assert len(closing) == 1
    assert closing[0].action == "TRAILING_STOP"
    assert pos.status == "CLOSED"

    all_actions = [t.action for t in engine.trade_history]
    assert "TP1_PARTIAL" not in all_actions
    assert "TP2_PARTIAL" not in all_actions


def test_swing_4h_case_c_channel_exit_10_taker_fee_and_slippage(tmp_path):
    engine = _make_engine(tmp_path, name="case_c.db")
    sig = _make_signal(direction="BUY", price=100.0, atr=2.0, ts=BASE_4H_TS)
    pos = engine.execute_swing_4h_signal(sig, current_price=100.0, timestamp=BASE_4H_TS, leverage=2.0)
    assert pos is not None

    exit_px = 108.0
    close_ts = pos.entry_timestamp + SWING_4H_BAR_MS
    trade = engine.close_position(pos.position_id, exit_px, close_ts, reason="CHANNEL_EXIT_10")
    assert trade is not None
    assert pos.status == "CLOSED"
    assert trade.action == "CHANNEL_EXIT_10"

    expected_exec = exit_px * (1.0 - engine.slippage_market)
    expected_fee = pos.quantity * expected_exec * engine.taker_fee
    expected_net = (expected_exec - pos.entry_price) * pos.quantity - expected_fee
    assert trade.price == pytest.approx(expected_exec)
    assert trade.fee == pytest.approx(expected_fee)
    assert trade.pnl == pytest.approx(expected_net)


def test_swing_4h_restore_state_roundtrip_before_and_after_ratchet(tmp_path):
    # Case 1: trailing_stop_price == 0 -> trail_ref_extreme restored to entry_price
    db1 = tmp_path / "restore_pre.db"
    eng1 = PaperTradingEngine(initial_capital=10_000.0, db_path=str(db1), restore=False)
    pos1 = eng1.execute_swing_4h_signal(
        _make_signal(direction="BUY", price=100.0, atr=2.0, ts=BASE_4H_TS),
        current_price=100.0,
        timestamp=BASE_4H_TS,
        leverage=2.0,
    )
    assert pos1 is not None

    eng1_reloaded = PaperTradingEngine(initial_capital=10_000.0, db_path=str(db1), restore=True)
    r1 = eng1_reloaded.positions[pos1.position_id]
    assert r1.strategy_type == "SWING_4H"
    assert r1.trailing_active is True
    assert r1.trailing_atr_mult == pytest.approx(3.5)
    assert r1.trailing_stop_price == 0.0
    assert r1.trail_ref_extreme == pytest.approx(pos1.entry_price)

    # Case 2: trailing_stop_price > 0 -> trail_ref_extreme restored to trailing_stop_price +/- d_trail
    for direction, side, sgn, b1_h, b1_l, b2_h, b2_l, b2_c in [
        ("BUY", "LONG", 1.0, 114.0, 99.5, 112.0, 108.0, 110.0),
        ("SELL", "SHORT", -1.0, 100.5, 86.0, 92.0, 88.0, 90.0),
    ]:
        db2 = tmp_path / f"restore_post_{side.lower()}.db"
        eng2 = PaperTradingEngine(initial_capital=10_000.0, db_path=str(db2), restore=False)
        atr = 2.0
        pos2 = eng2.execute_swing_4h_signal(
            _make_signal(direction=direction, price=100.0, atr=atr, ts=BASE_4H_TS),
            current_price=100.0,
            timestamp=BASE_4H_TS,
            leverage=2.0,
        )
        assert pos2 is not None
        t1 = pos2.entry_timestamp
        eng2.update_market_tick("BTCUSDT", high=b1_h, low=b1_l, close=100.0 + sgn * 10.0, timestamp=t1, open_price=100.0)
        eng2.update_market_tick("BTCUSDT", high=b2_h, low=b2_l, close=b2_c, timestamp=t1 + SWING_4H_BAR_MS, open_price=b2_c)
        assert pos2.trailing_stop_price > 0.0
        d_trail = 3.5 * pos2.entry_atr

        eng2_reloaded = PaperTradingEngine(initial_capital=10_000.0, db_path=str(db2), restore=True)
        r2 = eng2_reloaded.positions[pos2.position_id]
        assert r2.trailing_active is True
        assert r2.trailing_atr_mult == pytest.approx(3.5)
        assert r2.trailing_stop_price == pytest.approx(pos2.trailing_stop_price)
        assert r2.trail_ref_extreme == pytest.approx(pos2.trailing_stop_price + sgn * d_trail)


# ---------------------------------------------------------------------------
# 3. Allowlists & Routing
# ---------------------------------------------------------------------------

def test_swing_4h_allowlists_and_routing():
    assert "SWING_4H" in RiskPolicy().allowed_strategies
    assert "SWING_4H" in DefensiveRiskPolicy().allowed_strategies
    assert "SWING_4H" in flask_interface.FUNNEL_STRATEGIES
    for action in ("OPEN_SWING_4H_LONG", "OPEN_SWING_4H_SHORT", "CHANNEL_EXIT_10"):
        assert action in flask_interface.LEDGER_ACTION_LABELS
        assert isinstance(flask_interface.LEDGER_ACTION_LABELS[action], str)
        assert flask_interface.LEDGER_ACTION_LABELS[action].strip() != ""
    assert "SWING_4H" in pilot_report.PREREG_ALLOW


# ---------------------------------------------------------------------------
# 4. scripts/run_paper_simulation.py & scripts/system_supervisor.py
# ---------------------------------------------------------------------------

def test_daemon_swing_4h_e2e_entry_funnel_single_pos_dedupe_and_cooldown(tmp_path):
    class FakeLoader:
        def __init__(self, data_by_sym: Dict[str, pd.DataFrame]):
            self.data_by_sym = data_by_sym

        async def get_candles(self, symbol: str, timeframe: str, limit: int = 1000, exchange: str = "binance"):
            return self.data_by_sym.get(symbol, pd.DataFrame()).copy()

    # Use recent closed 4h bucket well before now_ts so advancing +1/+2/+3 remains strictly closed
    now_bucket = (int(time.time() * 1000) // SWING_4H_BAR_MS) * SWING_4H_BAR_MS
    b0 = now_bucket - 10 * SWING_4H_BAR_MS
    df_b0 = _build_1h_breakout_df(end_bucket_ts=b0, n_4h=210, breakout_close=105.0, breakout_high=106.0)

    loader = FakeLoader({"BTCUSDT": df_b0})
    engine = _make_engine(tmp_path, name="daemon_e2e.db")
    daemon = rps.PaperTradingDaemon(
        engine=engine,
        loader=loader,
        tape_agent=SimpleNamespace(db_path=str(tmp_path / "micro.db")),
    )
    daemon.core_symbols = ["BTCUSDT"]

    # 1) End-to-end entry + signal_funnel stages
    asyncio.run(daemon.evaluate_swing_4h_opportunities())
    assert len(engine.positions) == 1
    pos = next(iter(engine.positions.values()))
    assert pos.strategy_type == "SWING_4H"
    assert pos.side == "LONG"
    assert pos.leverage <= 2.0
    assert any(t.action == "OPEN_SWING_4H_LONG" for t in engine.trade_history)

    engine.funnel_flush()
    fc = _funnel_counts(engine.db_path)
    assert fc.get(("SWING_4H", "evaluated"), 0) >= 1
    assert fc.get(("SWING_4H", "candidate"), 0) >= 1
    assert fc.get(("SWING_4H", "executed"), 0) == 1

    # 2) Single-open-position-per-symbol guard
    asyncio.run(daemon.evaluate_swing_4h_opportunities())
    assert len(engine.positions) == 1
    engine.funnel_flush()
    assert _funnel_counts(engine.db_path).get(("SWING_4H", "reject:margin"), 0) >= 1

    # 3) bar_id dedupe (both in-memory and persisted ledger after clearing set)
    close_ts = b0 + SWING_4H_BAR_MS  # bar_id of close = b0//4h + 1
    engine.close_position(pos.position_id, 105.0, close_ts, reason="MANUAL_CLOSE")
    assert len(engine.positions) == 0

    asyncio.run(daemon.evaluate_swing_4h_opportunities())
    assert len(engine.positions) == 0
    daemon._swing_4h_entered_bars.clear()
    asyncio.run(daemon.evaluate_swing_4h_opportunities())
    assert len(engine.positions) == 0
    engine.funnel_flush()
    assert _funnel_counts(engine.db_path).get(("SWING_4H", "reject:bar_limit"), 0) >= 1

    # 4) Cooldown = 2 bars: close occurred at bar_id (B+1), so B+1 (diff=0) and B+2 (diff=1) blocked
    loader.data_by_sym["BTCUSDT"] = _build_1h_breakout_df(
        end_bucket_ts=b0, n_4h=210, breakout_close=105.0, breakout_high=106.0,
        extra_4h_bars=[(105.0, 109.0, 104.5, 108.0)],
    )
    asyncio.run(daemon.evaluate_swing_4h_opportunities())
    assert len(engine.positions) == 0

    loader.data_by_sym["BTCUSDT"] = _build_1h_breakout_df(
        end_bucket_ts=b0, n_4h=210, breakout_close=105.0, breakout_high=106.0,
        extra_4h_bars=[(105.0, 109.0, 104.5, 108.0), (108.0, 112.0, 107.5, 111.0)],
    )
    asyncio.run(daemon.evaluate_swing_4h_opportunities())
    assert len(engine.positions) == 0
    engine.funnel_flush()
    assert _funnel_counts(engine.db_path).get(("SWING_4H", "reject:cooldown"), 0) >= 1

    # B+3 (diff = 2 == cooldown_bars) -> cooldown expired, opens new position
    loader.data_by_sym["BTCUSDT"] = _build_1h_breakout_df(
        end_bucket_ts=b0, n_4h=210, breakout_close=105.0, breakout_high=106.0,
        extra_4h_bars=[
            (105.0, 109.0, 104.5, 108.0),
            (108.0, 112.0, 107.5, 111.0),
            (111.0, 115.0, 110.5, 114.0),
        ],
    )
    asyncio.run(daemon.evaluate_swing_4h_opportunities())
    assert len(engine.positions) == 1


def test_daemon_swing_4h_channel_exit_10_runs_under_circuit_breaker_and_updates_last_tick(tmp_path):
    class FakeLoader:
        def __init__(self, data_by_sym: Dict[str, pd.DataFrame]):
            self.data_by_sym = data_by_sym

        async def get_candles(self, symbol: str, timeframe: str, limit: int = 1000, exchange: str = "binance"):
            return self.data_by_sym.get(symbol, pd.DataFrame()).copy()

    now_bucket = (int(time.time() * 1000) // SWING_4H_BAR_MS) * SWING_4H_BAR_MS
    b0 = now_bucket - 6 * SWING_4H_BAR_MS
    btc_entry_df = _build_1h_breakout_df(end_bucket_ts=b0, n_4h=210, breakout_close=105.0, breakout_high=106.0)
    eth_breakout_df = _build_1h_breakout_df(end_bucket_ts=b0 + SWING_4H_BAR_MS, n_4h=210, breakout_close=105.0, breakout_high=106.0)

    loader = FakeLoader({"BTCUSDT": btc_entry_df, "ETHUSDT": pd.DataFrame()})
    engine = _make_engine(tmp_path, name="daemon_ch10_cb.db")
    daemon = rps.PaperTradingDaemon(
        engine=engine,
        loader=loader,
        tape_agent=SimpleNamespace(db_path=str(tmp_path / "micro.db")),
    )
    daemon.core_symbols = ["BTCUSDT"]

    asyncio.run(daemon.evaluate_swing_4h_opportunities())
    assert len(engine.positions) == 1
    pos = next(iter(engine.positions.values()))
    assert pos.symbol == "BTCUSDT"

    # Trip RollbackGuard circuit breaker -> must block new entries, must NOT block CHANNEL_EXIT_10
    assert engine.rollback_guard is not None
    engine.rollback_guard.status.state = "TRIPPED"
    engine.rollback_guard.status.is_trading_allowed = False
    engine.rollback_guard.status.reason = "Test circuit breaker active"

    # Advance BTCUSDT by 1 closed 4h bar breaching l10 (l10 == 99.0, new close == 95.0 < 99.0)
    b1 = b0 + SWING_4H_BAR_MS
    expected_bar_close_ts = b1 + SWING_4H_BAR_MS
    assert expected_bar_close_ts > pos.entry_timestamp
    loader.data_by_sym["BTCUSDT"] = _build_1h_breakout_df(
        end_bucket_ts=b0,
        n_4h=210,
        breakout_close=105.0,
        breakout_high=106.0,
        extra_4h_bars=[(105.0, 105.5, 94.0, 95.0)],
    )
    loader.data_by_sym["ETHUSDT"] = eth_breakout_df
    daemon.core_symbols = ["BTCUSDT", "ETHUSDT"]

    asyncio.run(daemon.evaluate_swing_4h_opportunities())

    # BTCUSDT closed via CHANNEL_EXIT_10; ETHUSDT blocked by circuit breaker
    assert pos.status == "CLOSED"
    assert len(engine.positions) == 0
    assert any(t.action == "CHANNEL_EXIT_10" and t.symbol == "BTCUSDT" for t in engine.trade_history)
    assert not any(t.symbol == "ETHUSDT" for t in engine.trade_history)
    assert daemon._last_tick_bar_ts["BTCUSDT"] == expected_bar_close_ts - 1


def test_option_b_gp_and_scalp_stale_candle_logs_debug_not_info(tmp_path, caplog):
    def _stale_ohlcv(bar_ms: int, n: int = 220) -> pd.DataFrame:
        # Latest bar closed ~2 hours ago -> stale for both GP (5m) and Scalp (15m)
        now_ms = int(time.time() * 1000)
        end_ts = ((now_ms - 2 * HOUR_MS) // bar_ms) * bar_ms
        start_ts = end_ts - (n - 1) * bar_ms
        rows = []
        for i in range(n):
            p = 100.0 + 0.5 * math.sin(i * 0.3)
            rows.append(
                {
                    "timestamp": start_ts + i * bar_ms,
                    "open": p,
                    "high": p + 0.4,
                    "low": p - 0.4,
                    "close": p + 0.1,
                    "volume": 500.0 + (i % 7) * 20.0,
                }
            )
        return pd.DataFrame(rows)

    class _StaleCandleLoader:
        async def get_candles(self, symbol: str, timeframe: str, limit: int = 400, exchange: str = "binance"):
            bar_ms = 5 * 60 * 1000 if timeframe == "5m" else 15 * 60 * 1000
            return _stale_ohlcv(bar_ms, n=230)

    class _FakeScanner:
        async def scan_market(self, force_refresh: bool = False):
            return {"top_swings": [{"symbol": "AAAUSDT", "volume_24h": 100_000_000.0}], "top_scalps": []}

    engine = _make_engine(tmp_path, name="stale_logs.db")
    daemon = rps.PaperTradingDaemon(
        engine=engine,
        loader=_StaleCandleLoader(),
        golden_pot_agent=_FakeScanner(),
        tape_agent=SimpleNamespace(db_path=str(tmp_path / "micro.db")),
        require_strict_tape_majors=False,
        use_scalp_squeeze_ml15=False,
        enforce_gp_btc_macro=False,
    )
    daemon.enable_live_context_enrichment = False

    with caplog.at_level(logging.DEBUG, logger="ia_trader.paper_simulation"):
        asyncio.run(daemon.evaluate_golden_pot_opportunities())
        asyncio.run(daemon.evaluate_scalp_opportunities())

    stale_records = [r for r in caplog.records if "stale candle" in r.getMessage().lower()]
    gp_debug = [r for r in stale_records if r.levelno == logging.DEBUG and "GP ignorado" in r.getMessage()]
    scalp_debug = [r for r in stale_records if r.levelno == logging.DEBUG and "Scalp ignorado" in r.getMessage()]
    info_or_higher = [r for r in stale_records if r.levelno >= logging.INFO]

    assert len(gp_debug) >= 1, "Expected GP stale candle skip to log at DEBUG"
    assert len(scalp_debug) >= 1, "Expected Scalp stale candle skip to log at DEBUG"
    assert info_or_higher == [], f"Unexpected INFO+ stale candle logs: {[r.getMessage() for r in info_or_higher]}"


def test_system_supervisor_rotates_oversized_log_file_before_launch(tmp_path, monkeypatch):
    oversized_bytes = 50 * 1024 * 1024 + 4096
    log_path = tmp_path / "logs" / "paper_daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as fh:
        fh.truncate(oversized_bytes)

    popen_calls = []

    class _DummyProc:
        pid = 98765

        def poll(self):
            return None

    def _fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return _DummyProc()

    monkeypatch.setattr(system_supervisor.subprocess, "Popen", _fake_popen)

    mgr = system_supervisor.ServiceManager(
        name="paper_daemon_test",
        cmd=["python3", "-c", "pass"],
        log_file=str(log_path),
    )
    mgr.start()

    rotated_path = tmp_path / "logs" / "paper_daemon.log.1"
    assert len(popen_calls) == 1
    assert mgr.pid == 98765
    assert rotated_path.exists()
    assert rotated_path.stat().st_size == oversized_bytes
    assert log_path.exists()
    assert log_path.stat().st_size == 0

    # Non-oversized log file must NOT be rotated
    small_log = tmp_path / "logs" / "small.log"
    small_log.write_bytes(b"small log line\n")
    mgr_small = system_supervisor.ServiceManager(
        name="small_svc",
        cmd=["python3", "-c", "pass"],
        log_file=str(small_log),
    )
    mgr_small.start()
    assert not (tmp_path / "logs" / "small.log.1").exists()
    assert small_log.stat().st_size == len(b"small log line\n")


# ---------------------------------------------------------------------------
# 5. docs/pilot_backtest_reference.json (scripts/eval_swing_4h_reference.py)
# ---------------------------------------------------------------------------

def test_pilot_backtest_reference_json_contract():
    ref_path = os.path.join(ROOT, "docs", "pilot_backtest_reference.json")
    assert os.path.isfile(ref_path), f"Missing {ref_path}"

    with open(ref_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert data["strategy"] == "SWING_4H"
    for key in ("expectancy_net_pct", "sharpe_net", "trades_per_day", "max_dd_pct"):
        assert key in data, f"Missing key {key}"
        assert isinstance(data[key], (int, float)) and math.isfinite(data[key])

    oos = data["windows"]["OOS"]
    assert oos["trades_count"] >= 50
    assert oos["profit_factor"] > 1.0
