#!/usr/bin/env python3
"""
scripts/eval_swing_4h_reference.py - Referência offline e registro de trials para SWING_4H.

Reutiliza `data/swing_4h_strategy.py` com custos estresse (0,06% taker + 0,02% slip/lado)
e funding real 8h de `funding_rates` em SQLite read-only (`?mode=ro`), gerando
`docs/pilot_backtest_reference.json` e opcionalmente registrando trials via `--log-trial`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.swing_4h_strategy import (  # noqa: E402
    SWING_4H_BAR_MS, SWING_4H_COOLDOWN_BARS, SWING_4H_DONCHIAN_ENTRY, SWING_4H_DONCHIAN_EXIT,
    SWING_4H_MAX_DIST_SMA200, SWING_4H_MAX_HOLDING_BARS, SWING_4H_SL_ATR_MULT, SWING_4H_TRAIL_ATR_MULT,
    compute_swing_4h_indicators, evaluate_swing_4h_entry, resample_1h_to_closed_4h, should_exit_swing_4h_channel,
)
from leverage_policy import decide_leverage  # noqa: E402
from statistical_rigor import calculate_annualized_sharpe, deflated_sharpe_ratio, log_trial  # noqa: E402

DB_PATH = os.path.join(BASE_DIR, "datasets", "market_cache.db")
OUT_REF = os.path.join(BASE_DIR, "docs", "pilot_backtest_reference.json")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
OOS_SPLIT_MS = int(pd.Timestamp("2025-11-20", tz="UTC").timestamp() * 1000)
INITIAL_CAPITAL, TAKER_FEE, SLIPPAGE, FUNDING_FALLBACK = 10_000.0, 0.0006, 0.0002, 0.0001
MAX_LEVERAGE, MAX_MARGIN_UTIL, RISK_PCT, CAP_PCT = 2.0, 0.85, 0.01, 0.25


@dataclass
class _Pos:
    symbol: str; side: str; entry_ts: int; entry_price: float; qty: float; margin: float
    notional: float; stop_loss: float; entry_atr: float; trail_ref: float; entry_fee: float
    funding_pnl: float = 0.0; bars_held: int = 0


def _load_db_ro(db_path: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    con = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True, timeout=5.0)
    candles_4h, funding_map = {}, {}
    try:
        q_c = "SELECT timestamp, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe='1h' "
        for sym in SYMBOLS:
            df1 = pd.read_sql(q_c + "AND exchange='binance' ORDER BY timestamp ASC", con, params=(sym,))
            if df1.empty:
                df1 = pd.read_sql(q_c + "ORDER BY timestamp ASC", con, params=(sym,))
            candles_4h[sym] = compute_swing_4h_indicators(resample_1h_to_closed_4h(df1))
            fdf = pd.read_sql("SELECT timestamp, funding_rate FROM funding_rates WHERE symbol=? ORDER BY timestamp ASC", con, params=(sym,)).drop_duplicates(subset=["timestamp"], keep="last")
            funding_map[sym] = ((fdf["timestamp"].to_numpy(dtype=np.int64), fdf["funding_rate"].to_numpy(dtype=np.float64)) if not fdf.empty else (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)))
    finally:
        con.close()
    return candles_4h, funding_map


def _lookup_funding(ftuple: Tuple[np.ndarray, np.ndarray], ts_ms: int) -> float:
    ts_arr, rate_arr = ftuple
    idx = int(np.searchsorted(ts_arr, ts_ms, side="right")) - 1 if len(ts_arr) else -1
    if idx >= 0 and (ts_ms - int(ts_arr[idx])) <= 29_400_000 and math.isfinite(float(rate_arr[idx])):
        return float(rate_arr[idx])
    return FUNDING_FALLBACK


def simulate_portfolio(
    candles_4h: Dict[str, pd.DataFrame],
    funding_map: Dict[str, Tuple[np.ndarray, np.ndarray]],
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Simula portfólio SWING_4H em timeline 4h sincronizada sem lookahead."""
    rows_by_sym: Dict[str, Dict[int, int]] = {}
    all_ts = set()
    for sym, df in candles_4h.items():
        if df.empty:
            rows_by_sym[sym] = {}
            continue
        imap = {
            int(t): i for i, t in enumerate(df["timestamp"].to_numpy(dtype=np.int64))
            if (start_ms is None or int(t) >= start_ms) and (end_ms is None or int(t) < end_ms)
        }
        rows_by_sym[sym] = imap
        all_ts.update(imap.keys())
    timeline = sorted(all_ts)
    if not timeline:
        return {"trades_count": 0, "days": 0.0, "trades_per_day": 0.0, "win_rate_pct": 0.0,
                "profit_factor": 0.0, "expectancy_net_pct": 0.0, "sharpe_net": 0.0,
                "max_dd_pct": 0.0, "net_pnl_usd": 0.0, "net_pnl_pct": 0.0, "daily_returns": np.empty(0)}

    balance, positions, pending, cooldown_bar, trades, daily_eq = INITIAL_CAPITAL, {}, {}, {}, [], {}

    def _close_pos(sym: str, raw_px: float, close_ts: int, reason: str) -> None:
        nonlocal balance
        pos = positions.pop(sym)
        sgn = 1.0 if pos.side == "LONG" else -1.0
        exit_px = raw_px * (1.0 - SLIPPAGE if pos.side == "LONG" else 1.0 + SLIPPAGE)
        exit_fee = pos.qty * exit_px * TAKER_FEE
        gross = sgn * (exit_px - pos.entry_price) * pos.qty
        net_usd = gross - pos.entry_fee - exit_fee + pos.funding_pnl
        balance += pos.funding_pnl + gross - exit_fee
        trades.append({"net_usd": net_usd, "net_pct": (net_usd / pos.notional) * 100.0 if pos.notional else 0.0, "reason": reason})
        cooldown_bar[sym] = (close_ts // SWING_4H_BAR_MS) + SWING_4H_COOLDOWN_BARS

    for ts in timeline:
        bar_id = ts // SWING_4H_BAR_MS
        for sym in SYMBOLS:
            if sym not in pending or sym in positions:
                continue
            direction, atr, lev = pending.pop(sym)
            idx = rows_by_sym[sym].get(ts)
            if idx is None:
                continue
            open_px = float(candles_4h[sym].iloc[idx]["open"])
            if open_px <= 0.0 or not math.isfinite(open_px):
                continue
            side = "LONG" if direction == "BUY" else "SHORT"
            sgn = 1.0 if side == "LONG" else -1.0
            exec_px = open_px * (1.0 + SLIPPAGE if side == "LONG" else 1.0 - SLIPPAGE)
            sl_price = exec_px - sgn * SWING_4H_SL_ATR_MULT * atr
            m_used = sum(p.margin for p in positions.values())
            free_m = max(0.0, balance * MAX_MARGIN_UTIL - m_used)
            margin = min((balance * RISK_PCT) / max(1e-4, (abs(exec_px - sl_price) / exec_px) * lev), free_m * 0.95, balance * CAP_PCT)
            notional = margin * lev
            entry_fee = notional * TAKER_FEE
            if margin < 10.0 or (m_used + margin + entry_fee) > balance * MAX_MARGIN_UTIL:
                continue
            balance -= entry_fee
            positions[sym] = _Pos(sym, side, ts, exec_px, notional / exec_px, margin, notional, sl_price, atr, exec_px, entry_fee)

        for sym in SYMBOLS:
            idx = rows_by_sym[sym].get(ts)
            if idx is None:
                continue
            df_sym = candles_4h[sym]
            row = df_sym.iloc[idx]
            o_px, h_px, l_px, c_px = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            if sym in positions:
                pos = positions[sym]
                pos.bars_held += 1
                if pos.bars_held % 2 == 0:
                    rate = _lookup_funding(funding_map[sym], ts + SWING_4H_BAR_MS)
                    pos.funding_pnl += (-1.0 if pos.side == "LONG" else 1.0) * pos.notional * rate
                if pos.bars_held >= SWING_4H_MAX_HOLDING_BARS:
                    _close_pos(sym, o_px, ts, "HORIZON_TIMEOUT")
                elif (pos.side == "LONG" and l_px <= pos.stop_loss) or (pos.side == "SHORT" and h_px >= pos.stop_loss):
                    _close_pos(sym, min(pos.stop_loss, o_px) if pos.side == "LONG" else max(pos.stop_loss, o_px), ts, "STOP_OR_TRAIL")
                else:
                    if pos.side == "LONG":
                        pos.trail_ref = max(pos.trail_ref, h_px)
                        pos.stop_loss = max(pos.stop_loss, pos.trail_ref - SWING_4H_TRAIL_ATR_MULT * pos.entry_atr)
                    else:
                        pos.trail_ref = min(pos.trail_ref, l_px)
                        pos.stop_loss = min(pos.stop_loss, pos.trail_ref + SWING_4H_TRAIL_ATR_MULT * pos.entry_atr)
                    if idx >= 11 and should_exit_swing_4h_channel(df_sym.iloc[idx : idx + 1], pos.side)[0]:
                        _close_pos(sym, c_px, ts + SWING_4H_BAR_MS, "CHANNEL_EXIT_10")

            if sym not in positions and bar_id >= cooldown_bar.get(sym, -1):
                trig = evaluate_swing_4h_entry(df_sym.iloc[idx : idx + 1])
                if trig is not None:
                    direction, _, metrics = trig
                    atr, c_px = float(metrics["atr_14"]), float(metrics["close"])
                    lev = min(MAX_LEVERAGE, decide_leverage(0.75, atr_pct=atr / c_px, regime_alignment=True, timeframe="4h", max_leverage=MAX_LEVERAGE))
                    pending[sym] = (direction, atr, lev)

        mtm = sum(
            ((1.0 if p.side == "LONG" else -1.0) * (float(candles_4h[s].iloc[rows_by_sym[s][ts]]["close"]) - p.entry_price) * p.qty) + p.funding_pnl
            for s, p in positions.items() if ts in rows_by_sym[s]
        )
        daily_eq[(ts + SWING_4H_BAR_MS) // 86_400_000] = balance + mtm

    d_keys = sorted(daily_eq.keys())
    eq_vals = pd.Series([daily_eq[d] for d in d_keys], index=d_keys).reindex(np.arange(d_keys[0], d_keys[-1] + 1)).ffill().fillna(INITIAL_CAPITAL).to_numpy(float)
    prev_eq = np.insert(eq_vals[:-1], 0, INITIAL_CAPITAL)
    d_rets = (eq_vals - prev_eq) / np.maximum(1.0, prev_eq)
    peaks = np.maximum.accumulate(np.insert(eq_vals, 0, INITIAL_CAPITAL))[1:]
    max_dd = float(np.max((peaks - eq_vals) / np.maximum(1.0, peaks)) * 100.0) if len(eq_vals) else 0.0
    total_days = max(1.0, (timeline[-1] - timeline[0] + SWING_4H_BAR_MS) / 86_400_000.0)
    n_tr = len(trades)
    nets = np.array([t["net_usd"] for t in trades], dtype=float) if n_tr else np.empty(0)
    net_pcts = np.array([t["net_pct"] for t in trades], dtype=float) if n_tr else np.empty(0)
    g_pos = float(nets[nets > 0].sum()) if n_tr else 0.0
    g_neg = float(abs(nets[nets <= 0].sum())) if n_tr else 0.0
    sharpe = calculate_annualized_sharpe(d_rets, 365.0)
    net_pnl = float(eq_vals[-1] - INITIAL_CAPITAL) if len(eq_vals) else 0.0
    return {
        "trades_count": n_tr, "days": round(total_days, 2), "trades_per_day": round(n_tr / total_days, 4),
        "win_rate_pct": round((int((nets > 0).sum()) / n_tr) * 100.0, 2) if n_tr else 0.0,
        "profit_factor": round((g_pos / g_neg) if g_neg > 0 else (999.0 if g_pos > 0 else 0.0), 3),
        "expectancy_net_pct": round(float(net_pcts.mean()), 4) if n_tr else 0.0,
        "sharpe_net": round(sharpe, 4), "max_dd_pct": round(max_dd, 2),
        "net_pnl_usd": round(net_pnl, 2), "net_pnl_pct": round((net_pnl / INITIAL_CAPITAL) * 100.0, 2),
        "daily_returns": d_rets,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="SWING_4H reference backtest & trial logger")
    ap.add_argument("--log-trial", action="store_true", help="Registra E_SWING_4H_FULL e E_SWING_4H_OOS em trial_log.csv")
    args = ap.parse_args()
    candles_4h, funding_map = _load_db_ro(DB_PATH)
    res_is = simulate_portfolio(candles_4h, funding_map, end_ms=OOS_SPLIT_MS)
    res_oos = simulate_portfolio(candles_4h, funding_map, start_ms=OOS_SPLIT_MS)
    res_full = simulate_portfolio(candles_4h, funding_map)

    for k, r in (("IS", res_is), ("OOS", res_oos), ("FULL", res_full)):
        print(f"[{k}] trades={r['trades_count']} days={r['days']} tpd={r['trades_per_day']:.4f} "
              f"WR={r['win_rate_pct']:.2f}% PF={r['profit_factor']:.3f} exp={r['expectancy_net_pct']:+.4f}% "
              f"Sharpe={r['sharpe_net']:.4f} MaxDD={r['max_dd_pct']:.2f}% PnL=${r['net_pnl_usd']:+.2f}")

    ref_src = res_oos if res_oos["trades_count"] >= 30 else res_full
    ref_payload = {
        "strategy": "SWING_4H", "source_window": "OOS" if ref_src is res_oos else "FULL",
        "expectancy_net_pct": ref_src["expectancy_net_pct"], "sharpe_net": ref_src["sharpe_net"],
        "trades_per_day": ref_src["trades_per_day"], "max_dd_pct": ref_src["max_dd_pct"],
        "windows": {w: {k: v for k, v in r.items() if k != "daily_returns"} for w, r in (("IS", res_is), ("OOS", res_oos), ("FULL", res_full))},
    }
    os.makedirs(os.path.dirname(OUT_REF), exist_ok=True)
    with open(OUT_REF, "w", encoding="utf-8") as fh:
        json.dump(ref_payload, fh, indent=2)
    print(f"Written {OUT_REF}")

    if args.log_trial:
        params = {
            "donchian_entry": SWING_4H_DONCHIAN_ENTRY, "donchian_exit": SWING_4H_DONCHIAN_EXIT,
            "max_dist_sma200": SWING_4H_MAX_DIST_SMA200, "sl_atr": SWING_4H_SL_ATR_MULT,
            "trail_atr": SWING_4H_TRAIL_ATR_MULT, "max_holding_bars": SWING_4H_MAX_HOLDING_BARS,
            "cooldown_bars": SWING_4H_COOLDOWN_BARS, "max_leverage": MAX_LEVERAGE,
        }
        tl_path = os.path.join(BASE_DIR, "datasets", "reports", "trial_log.csv")
        tl_df = pd.read_csv(tl_path) if os.path.exists(tl_path) else pd.DataFrame()
        var_tr = float(np.nanvar(pd.to_numeric(tl_df["sharpe_ratio"], errors="coerce"), ddof=1)) if len(tl_df) > 1 else 1.0
        n_trials = len(tl_df) + 2
        for exp, res, period, is_oos in (("E_SWING_4H_FULL", res_full, "2020-01-01..2026-11-21", False), ("E_SWING_4H_OOS", res_oos, "2025-11-20..2026-11-21", True)):
            dsr = deflated_sharpe_ratio(res["sharpe_net"], res["daily_returns"], n_trials, var_tr, 365.0)
            label = "CANDIDATE" if (res["expectancy_net_pct"] > 0 and res["sharpe_net"] >= 1.0) else "OBSERVATION"
            tid = log_trial(
                phase="Wave_P3", experiment=exp, strategy_name="SWING_4H_DONCHIAN40_CH10",
                symbol="BTC/ETH/SOL", timeframe="4h", params=params, sample_period=period,
                is_oos=is_oos, trades_count=res["trades_count"], net_pnl=res["net_pnl_usd"],
                net_pnl_pct=res["net_pnl_pct"], sharpe_ratio=res["sharpe_net"],
                max_drawdown_pct=res["max_dd_pct"], win_rate_pct=res["win_rate_pct"],
                status_label=label, notes=f"PF={res['profit_factor']};exp={res['expectancy_net_pct']}%;DSR={dsr['dsr']:.4f};tpd={res['trades_per_day']}",
            )
            print(f"Logged {exp}: {tid} (DSR={dsr['dsr']:.4f}, label={label})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
