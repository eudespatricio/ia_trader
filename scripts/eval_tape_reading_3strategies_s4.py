#!/usr/bin/env python3
"""
scripts/eval_tape_reading_3strategies_s4.py

Simulação empírica do impacto real do TapeReadingAgent (gate binário vs ponderação contínua)
nas 3 estratégias ativas do sistema (DIRECTIONAL 1h, GOLDEN_POT 5m, SCALP 15m):
- Bloco A: Auditoria contrafactual dos 804 registros ao vivo em `flow_gate_log` (`market_cache.db`).
- Bloco B: Replay 29d causal em L2 real (`microstructure.db`, 2026-10-15..2026-11-13) comparando 5 modos:
    M0_NO_TAPE             : sem TapeReadingAgent (puro backtest)
    M1_GATE_FAIL_OPEN      : produção atual (_flow_gate pós-comitê, UNAVAILABLE passa)
    M2_GATE_STRICT_OK      : gate estrito fail-closed (exige status=='OK' e vote==direction)
    M3_TAPE_WEIGHTED_SCORE : ponderação contínua no consenso (0.65*base + 0.35*tape) + sizing 0.50x..1.25x
    M4_STRICT_OK_PLUS_WEIGHT: gate estrito + escala de mão por confiança do Tape (0.85x..1.30x; altcoin não coberta 0.50x no GP)

Somente leitura em SQLites (`mode=ro`). Não altera código nem tabelas de produção.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.tape_reading_agent import TapeReadingAgent  # noqa: E402
from agents.technical_agent import TechnicalAgent, _DefaultMLEnsemble  # noqa: E402
from backtesting import BacktestEngine, BacktestResult, _PilotPos, _PilotSimulator  # noqa: E402
from core import TradingCore  # noqa: E402
from data.feature_engineering import FeatureEngineer  # noqa: E402
import scripts.eval_s4_calibration_and_open as s4  # noqa: E402
import scripts.eval_scalp_research_s4 as sr  # noqa: E402
from scripts.run_paper_simulation import PaperTradingDaemon  # noqa: E402
import smart_trade as st_mod  # noqa: E402
from smart_trade import STRATEGY_PROFILES  # noqa: E402
from statistical_rigor import deflated_sharpe_ratio, get_prior_trials_count  # noqa: E402

DB_PATH = s4.DB_PATH
MICRO_DB_PATH = s4.MICRO_DB_PATH
REPORT_JSON = os.path.join(ROOT, "datasets", "reports", "tape_reading_3strategies_s4_20261113.json")
REPORT_MD = os.path.join(ROOT, "docs", "NOTAS_IMPACTO_TAPE_READING_3_ESTRATEGIAS.md")

MAJORS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MODES = (
    "M0_NO_TAPE",
    "M1_GATE_FAIL_OPEN",
    "M2_GATE_STRICT_OK",
    "M3_TAPE_WEIGHTED_SCORE",
    "M4_STRICT_OK_PLUS_WEIGHT",
)
GP_BAR_MS = 300_000
HOUR_MS = 3_600_000
BAR_15M_MS = 900_000
TS_29D_START = int(pd.Timestamp("2026-10-15 12:30:00", tz="UTC").timestamp() * 1000)
TS_TRAIN_CUTOFF = int(pd.Timestamp("2026-10-14 00:00:00", tz="UTC").timestamp() * 1000)


def _ro_conn(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0)


def _ts_iso(ms: int) -> str:
    return dt.datetime.fromtimestamp(int(ms) / 1000.0, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# =====================================================================
# BLOCO A: Contrafactual Ao Vivo em `flow_gate_log` (N=804)
# =====================================================================

def _simulate_logged_signal_forward(
    row: Dict[str, Any],
    df_c: pd.DataFrame,
    ts_arr: np.ndarray,
    lot_usd: float = 100.0,
) -> Optional[Dict[str, Any]]:
    """Simula forward o sinal gravado em flow_gate_log usando a ladder oficial do STRATEGY_PROFILES."""
    strat = str(row["strategy_type"])
    prof = STRATEGY_PROFILES.get(strat)
    if prof is None:
        return None
    entry = float(row.get("entry_price") or 0.0)
    sl_init = float(row.get("stop_loss") or 0.0)
    tp1 = float(row.get("tp1") or 0.0)
    tp2 = float(row.get("tp2") or 0.0)
    tp3 = float(row.get("tp3") or 0.0)
    atr = float(row.get("atr") or 0.0)
    lev = float(row.get("leverage") or 1.0)
    side = str(row.get("direction") or "").upper()
    if entry <= 0.0 or sl_init <= 0.0 or side not in ("BUY", "SELL"):
        return None
    is_long = side == "BUY"
    sgn = 1.0 if is_long else -1.0
    if (is_long and sl_init >= entry) or ((not is_long) and sl_init <= entry):
        return None

    bar_ms = int(prof.bar_ms)
    max_hold = int(prof.max_holding_bars)
    first_bar_open = (int(row["ts_ms"]) // bar_ms) * bar_ms
    i0 = int(np.searchsorted(ts_arr, first_bar_open, side="left"))
    if i0 >= len(ts_arr):
        return None

    # Paridade de taxa de entrada (_ENTRY_META) + saída real maker/taker
    meta = st_mod.PaperTradingEngine._ENTRY_META.get(strat, {"maker_entry": False})
    maker_in = bool(meta.get("maker_entry", False))
    in_fee_r = 0.0002 if maker_in else 0.0006
    in_slip_r = 0.0001 if maker_in else 0.0002
    exec_entry = entry * (1.0 + sgn * in_slip_r)

    qty0 = (lot_usd * max(1.0, lev)) / exec_entry
    qty = qty0
    risk_usd = abs(exec_entry - sl_init) * qty0
    if risk_usd <= 0.0:
        return None

    sl = sl_init
    tp1_hit = False
    tp2_hit = False
    trail_active = False
    trail_ref = 0.0
    trail_stop = 0.0
    trail_bar: Optional[Tuple[float, float]] = None
    gross = 0.0
    fees = qty0 * exec_entry * in_fee_r

    def close_leg(level: float, q_leg: float, maker_exit: bool, partial: bool) -> float:
        nonlocal qty, gross, fees
        if q_leg <= 0.0:
            return 0.0
        if partial:
            px = level
        else:
            slip = 0.0001 if maker_exit else 0.0002
            px = level * (1.0 - slip) if is_long else level * (1.0 + slip)
        fee = q_leg * px * (0.0002 if maker_exit else 0.0006)
        g = sgn * (px - exec_entry) * q_leg
        gross += g
        fees += fee
        qty -= q_leg
        return g - fee

    def ratchet(bh: float, bl: float) -> None:
        nonlocal sl, trail_ref, trail_stop
        dist = prof.trail_atr_mult * atr
        if dist <= 0.0:
            return
        if is_long:
            trail_ref = max(trail_ref, bh)
            cand = trail_ref - dist
            if cand > trail_stop:
                trail_stop = cand
            if trail_stop > sl:
                sl = trail_stop
        else:
            trail_ref = bl if trail_ref <= 0.0 else min(trail_ref, bl)
            cand = trail_ref + dist
            if trail_stop <= 0.0 or cand < trail_stop:
                trail_stop = cand
            if trail_stop > 0.0 and (sl <= 0.0 or trail_stop < sl):
                sl = trail_stop

    opens = df_c["open"].to_numpy(float)
    highs = df_c["high"].to_numpy(float)
    lows = df_c["low"].to_numpy(float)
    closes = df_c["close"].to_numpy(float)

    reason = "OPEN_END"
    end_idx = min(len(ts_arr), i0 + max_hold + 1)
    for k in range(i0, end_idx):
        o, h, l, c = opens[k], highs[k], lows[k], closes[k]
        if k - i0 >= max_hold:
            close_leg(o, qty, maker_exit=False, partial=False)
            reason = "HORIZON_TIMEOUT"
            break
        if trail_active and trail_bar is not None:
            ratchet(trail_bar[0], trail_bar[1])
            trail_bar = None

        stop_hit = (l <= sl) if is_long else (h >= sl)
        if stop_hit:
            fill = min(sl, h, o) if is_long else max(sl, l, o)
            reason = "STOP_LOSS" if not tp1_hit else ("TRAILING_STOP" if trail_active and sl == trail_stop else "BREAKEVEN_STOP")
            close_leg(fill, qty, maker_exit=False, partial=False)
            break

        if not tp1_hit and tp1 > 0.0 and ((h >= tp1) if is_long else (l <= tp1)):
            tp1_hit = True
            close_leg(tp1, min(qty0 * st_mod.EXIT_TP1_FRACTION, qty), maker_exit=True, partial=True)
            be = exec_entry * (1.0 + sgn * st_mod.BREAKEVEN_FEE_BUFFER_PCT)
            sl = max(sl, be) if is_long else min(sl, be)
            same_safe = o > 0.0 and ((c > sl) if is_long else (c < sl))
            if ((l <= sl) if is_long else (h >= sl)) and not same_safe:
                fill = min(sl, h) if is_long else max(sl, l)
                close_leg(fill, qty, maker_exit=False, partial=False)
                reason = "BREAKEVEN_STOP"
                break

        if tp1_hit and not tp2_hit and tp2 > 0.0 and ((h >= tp2) if is_long else (l <= tp2)):
            tp2_hit = True
            close_leg(tp2, min(qty0 * st_mod.EXIT_TP2_FRACTION, qty), maker_exit=True, partial=True)
            if prof.trail_atr_mult > 0.0 and atr > 0.0:
                trail_active = True
                trail_ref = tp2
                ratchet(tp2, tp2)

        if tp2_hit and tp3 > 0.0 and ((h >= tp3) if is_long else (l <= tp3)):
            close_leg(tp3, qty, maker_exit=True, partial=False)
            reason = "TAKE_PROFIT_3"
            break

        if trail_active:
            trail_bar = (h, l)

    if qty > 1e-12:
        last_k = min(len(ts_arr) - 1, max(i0, end_idx - 1))
        close_leg(closes[last_k], qty, maker_exit=False, partial=False)

    pnl = gross - fees
    return {
        "pnl": float(pnl),
        "r_mult": float(pnl / risk_usd),
        "win": pnl > 0.0,
        "reason": reason,
    }


def run_block_a_flow_gate_log() -> Dict[str, Any]:
    """Audita todos os registros de flow_gate_log em market_cache.db (simulação forward 100% causal)."""
    with _ro_conn(DB_PATH) as conn:
        df_fg = pd.read_sql_query(
            """
            SELECT id, ts_ms, symbol, strategy_type, direction, decision,
                   tape_vote, tape_status, tape_reason, confidence, leverage,
                   entry_price, stop_loss, tp1, tp2, tp3, atr
            FROM flow_gate_log
            ORDER BY ts_ms ASC
            """,
            conn,
        )

    tf_by_strat = {"DIRECTIONAL": "1h", "SCALP": "15m", "GOLDEN_POT": "5m"}
    needed_pairs = sorted({(str(r["symbol"]), tf_by_strat[str(r["strategy_type"])]) for _, r in df_fg.iterrows() if str(r["strategy_type"]) in tf_by_strat})
    candle_cache: Dict[Tuple[str, str], Tuple[pd.DataFrame, np.ndarray]] = {}
    for sym, tf in needed_pairs:
        df_c = s4.load_candles_clean(sym, tf, limit=None)
        if not df_c.empty:
            candle_cache[(sym, tf)] = (df_c, df_c["timestamp"].to_numpy(np.int64))

    rows_out: List[Dict[str, Any]] = []
    missing_candles = 0
    for _, row in df_fg.iterrows():
        strat = str(row["strategy_type"])
        tf = tf_by_strat.get(strat)
        if tf is None or (str(row["symbol"]), tf) not in candle_cache:
            missing_candles += 1
            continue
        df_c, ts_arr = candle_cache[(str(row["symbol"]), tf)]
        sim = _simulate_logged_signal_forward(row.to_dict(), df_c, ts_arr)
        if sim is None:
            missing_candles += 1
            continue
        sim["strategy_type"] = strat
        sim["decision"] = str(row["decision"])
        sim["tape_status"] = str(row["tape_status"])
        sim["tape_reason"] = str(row["tape_reason"])
        rows_out.append(sim)

    def _agg(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not items:
            return {"count": 0, "win_rate_pct": 0.0, "pf": 0.0, "expectancy_r": 0.0, "pnl_usd": 0.0}
        pnls = np.array([x["pnl"] for x in items], dtype=float)
        rs = np.array([x["r_mult"] for x in items], dtype=float)
        wins = pnls[pnls > 0.0].sum()
        losses = abs(pnls[pnls < 0.0].sum())
        pf = wins / losses if losses > 0 else (99.0 if wins > 0 else 0.0)
        return {
            "count": int(len(items)),
            "win_rate_pct": round(float((pnls > 0.0).mean() * 100.0), 2),
            "pf": round(float(min(pf, 99.0)), 3),
            "expectancy_r": round(float(rs.mean()), 4),
            "pnl_usd": round(float(pnls.sum()), 2),
        }

    summary: Dict[str, Any] = {
        "total_logged": int(len(df_fg)),
        "evaluated": int(len(rows_out)),
        "missing_or_invalid": int(missing_candles),
        "by_strategy": {},
    }
    for strat in ("DIRECTIONAL", "SCALP", "GOLDEN_POT"):
        sub = [x for x in rows_out if x["strategy_type"] == strat]
        all_no_gate = _agg(sub)
        executed_now = _agg([x for x in sub if x["decision"] in ("PASSED", "UNAVAILABLE_PASSED")])
        strict_ok = _agg([x for x in sub if x["decision"] == "PASSED"])
        blocked = _agg([x for x in sub if x["decision"] == "BLOCKED"])
        unavail = _agg([x for x in sub if x["decision"] == "UNAVAILABLE_PASSED"])
        summary["by_strategy"][strat] = {
            "all_candidates_no_gate": all_no_gate,
            "current_fail_open_executed": executed_now,
            "strict_ok_only": strict_ok,
            "unavail_passed": unavail,
            "blocked_by_gate": blocked,
            "delta_wr_strict_vs_nogate_pp": round(strict_ok["win_rate_pct"] - all_no_gate["win_rate_pct"], 2) if strict_ok["count"] else None,
            "delta_wr_failopen_vs_nogate_pp": round(executed_now["win_rate_pct"] - all_no_gate["win_rate_pct"], 2) if executed_now["count"] else None,
        }
    return summary


# =====================================================================
# BLOCO B: Simuladores com Ponderação + Aplicação dos 5 Modos
# =====================================================================

class _WeightedPilotSim(_PilotSimulator):
    """Extende _PilotSimulator com sizing ponderado, teto por direção e 1 entrada por barra."""

    def __init__(
        self,
        eng: BacktestEngine,
        size_mult: Optional[Dict[str, np.ndarray]] = None,
        max_same_dir: Optional[int] = None,
        max_entries_per_bar: Optional[int] = None,
        maker_entry_override: Optional[bool] = None,
    ):
        super().__init__(eng)
        self.size_mult = size_mult or {}
        self.max_same_dir = max_same_dir
        self.max_entries_per_bar = max_entries_per_bar
        if maker_entry_override is not None:
            self.meta = dict(self.meta)
            self.meta["maker_entry"] = bool(maker_entry_override)
        self._entries_by_ts: Dict[int, int] = {}
        self._init_risk_by_key: Dict[Tuple[str, int], float] = {}

    def try_open(self, sym, idx, direction, entry_ts, sig_close, entry_raw, atr, conf, regime_ok, positions, state):
        if self.max_same_dir is not None:
            side = "LONG" if direction == 1 else "SHORT"
            if sum(1 for q in positions.values() if q.side == side) >= self.max_same_dir:
                return None
        if self.max_entries_per_bar is not None and self._entries_by_ts.get(int(entry_ts), 0) >= self.max_entries_per_bar:
            return None

        sig_idx = max(0, idx - 1)
        mult = 1.0
        if sym in self.size_mult and sig_idx < len(self.size_mult[sym]):
            mult = float(np.clip(self.size_mult[sym][sig_idx], 0.35, 1.50))

        orig_lot = self.e.lot_margin
        self.e.lot_margin = orig_lot * mult
        try:
            pos = super().try_open(sym, idx, direction, entry_ts, sig_close, entry_raw, atr, conf, regime_ok, positions, state)
        finally:
            self.e.lot_margin = orig_lot

        if pos is not None:
            self._entries_by_ts[int(entry_ts)] = self._entries_by_ts.get(int(entry_ts), 0) + 1
            self._init_risk_by_key[(sym, int(pos.entry_ts))] = abs(pos.entry - pos.sl) * pos.qty0
        return pos


class _WeightedScalpSim(sr._ScalpResearchSimulator):
    """Extende _ScalpResearchSimulator com multiplicador de mão por barra de sinal."""

    def __init__(
        self,
        eng: BacktestEngine,
        spec: sr.ScalpSpec,
        raw_signals: Dict[str, np.ndarray],
        size_mult: Optional[Dict[str, np.ndarray]] = None,
    ):
        super().__init__(eng, spec)
        self._raw_sigs = raw_signals
        self.size_mult = size_mult or {}

    def try_open(self, sym, idx, direction, entry_ts, sig_close, entry_raw, atr, conf, regime_ok, positions, state):
        mult = 1.0
        if sym in self.size_mult:
            arr_s = self._raw_sigs[sym]
            max_off = self.spec.limit_ttl_bars + 1 if self.spec.entry_exec == "limit_pullback" else 1
            for off in range(1, max_off + 1):
                j = idx - off
                if 0 <= j < len(arr_s) and arr_s[j] != 0:
                    mult = float(np.clip(self.size_mult[sym][j], 0.35, 1.50))
                    break

        orig_lot = self.e.lot_margin
        self.e.lot_margin = orig_lot * mult
        try:
            return super().try_open(sym, idx, direction, entry_ts, sig_close, entry_raw, atr, conf, regime_ok, positions, state)
        finally:
            self.e.lot_margin = orig_lot


def apply_tape_mode(
    mode: str,
    data_map: Dict[str, pd.DataFrame],
    sig_raw: Dict[str, np.ndarray],
    conf_raw: Dict[str, np.ndarray],
    bar_ms: int,
    tape_agent: TapeReadingAgent,
    tape_cache: Dict[Tuple[str, int, str], Dict[str, Any]],
    allow_unavail_alt: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:
    """Aplica exatamente os 5 modos (M0..M4) de TapeReadingAgent com estrita causalidade em now_ms."""
    sig_out: Dict[str, np.ndarray] = {}
    conf_out: Dict[str, np.ndarray] = {}
    mult_out: Dict[str, np.ndarray] = {}
    stats = {
        "candidates": 0,
        "kept": 0,
        "ok_dir_agree": 0,
        "ok_hold": 0,
        "ok_opp_or_spread": 0,
        "unavail_kept": 0,
        "unavail_blocked": 0,
        "score_blocked": 0,
    }

    for sym, df in data_map.items():
        s_in = sig_raw[sym].copy()
        c_in = conf_raw[sym].copy()
        m_in = np.ones(len(df), dtype=np.float64)
        if mode == "M0_NO_TAPE":
            nz = np.where(s_in != 0)[0]
            stats["candidates"] += int(len(nz))
            stats["kept"] += int(len(nz))
            sig_out[sym], conf_out[sym], mult_out[sym] = s_in, c_in, m_in
            continue

        ts_open = df["timestamp"].to_numpy(np.int64)
        for idx in np.where(s_in != 0)[0]:
            stats["candidates"] += 1
            direction = "BUY" if int(s_in[idx]) == 1 else "SELL"
            now_ms = int(ts_open[idx]) + bar_ms
            key = (sym, now_ms, direction)
            ev = tape_cache.get(key)
            if ev is None:
                ev = tape_agent.evaluate(sym, direction, now_ms=now_ms, allow_live_fallback=False)
                tape_cache[key] = ev

            status = str(ev.get("status", "UNAVAILABLE"))
            vote = str(ev.get("vote", "HOLD"))
            reason = str(ev.get("reason", ""))
            tape_conf = float(ev.get("confidence", 0.5) or 0.5)
            fscore = float((ev.get("features") or {}).get("flow_score", 0.0) or 0.0)
            base_conf = float(c_in[idx]) if np.isfinite(c_in[idx]) and c_in[idx] > 0 else 0.70

            if mode == "M1_GATE_FAIL_OPEN":
                if status != "OK":
                    stats["unavail_kept"] += 1
                    stats["kept"] += 1
                elif vote == direction:
                    stats["ok_dir_agree"] += 1
                    stats["kept"] += 1
                else:
                    stats["ok_opp_or_spread" if (vote in ("BUY", "SELL") or reason == "spread_wide") else "ok_hold"] += 1
                    s_in[idx] = 0
                    c_in[idx] = 0.0

            elif mode == "M2_GATE_STRICT_OK":
                if status == "OK" and vote == direction:
                    stats["ok_dir_agree"] += 1
                    stats["kept"] += 1
                elif status != "OK":
                    stats["unavail_blocked"] += 1
                    s_in[idx] = 0
                    c_in[idx] = 0.0
                else:
                    stats["ok_opp_or_spread" if (vote in ("BUY", "SELL") or reason == "spread_wide") else "ok_hold"] += 1
                    s_in[idx] = 0
                    c_in[idx] = 0.0

            elif mode == "M3_TAPE_WEIGHTED_SCORE":
                if status == "OK" and (reason == "spread_wide" or (vote in ("BUY", "SELL") and vote != direction)):
                    stats["ok_opp_or_spread"] += 1
                    s_in[idx] = 0
                    c_in[idx] = 0.0
                elif status != "OK":
                    if allow_unavail_alt and sym not in MAJORS and base_conf >= 0.72:
                        stats["unavail_kept"] += 1
                        stats["kept"] += 1
                        m_in[idx] = 0.50
                    else:
                        stats["unavail_blocked"] += 1
                        s_in[idx] = 0
                        c_in[idx] = 0.0
                else:
                    sgn = 1.0 if direction == "BUY" else -1.0
                    if vote == direction:
                        stats["ok_dir_agree"] += 1
                        tape_unit = min(1.0, 0.65 + 0.35 * tape_conf)
                        mult = 1.25 if tape_conf >= 0.60 else 1.00
                    else:
                        stats["ok_hold"] += 1
                        tape_unit = float(np.clip(0.50 + 1.25 * (sgn * fscore), 0.0, 0.64))
                        mult = 0.65
                    blended = 0.65 * base_conf + 0.35 * tape_unit
                    if blended < 0.62:
                        stats["score_blocked"] += 1
                        s_in[idx] = 0
                        c_in[idx] = 0.0
                    else:
                        stats["kept"] += 1
                        c_in[idx] = blended
                        m_in[idx] = mult

            elif mode == "M4_STRICT_OK_PLUS_WEIGHT":
                if status == "OK" and vote == direction:
                    stats["ok_dir_agree"] += 1
                    stats["kept"] += 1
                    m_in[idx] = float(0.85 + 0.45 * tape_conf)
                elif status != "OK" and allow_unavail_alt and sym not in MAJORS and reason == "symbol_not_covered":
                    stats["unavail_kept"] += 1
                    stats["kept"] += 1
                    m_in[idx] = 0.50
                elif status != "OK":
                    stats["unavail_blocked"] += 1
                    s_in[idx] = 0
                    c_in[idx] = 0.0
                else:
                    stats["ok_opp_or_spread" if (vote in ("BUY", "SELL") or reason == "spread_wide") else "ok_hold"] += 1
                    s_in[idx] = 0
                    c_in[idx] = 0.0

        sig_out[sym], conf_out[sym], mult_out[sym] = s_in, c_in, m_in
    return sig_out, conf_out, mult_out, stats


def _compute_metrics_full(
    res: BacktestResult,
    data_map: Dict[str, pd.DataFrame],
    sl_atr: float,
    start_ms: int,
    end_ms: int,
    init_risk_by_key: Optional[Dict[Tuple[str, int], float]] = None,
    min_atr_pct: float = 0.0,
    n_trials_dsr: int = 2,
    var_sharpes: float = 0.0,
) -> Dict[str, Any]:
    """Calcula métricas completas incluindo H1/H2 e Expectancy R normalizada pelo risco real de entrada."""
    r_mults: List[float] = []
    daily_pnl: Dict[str, float] = {}
    exit_reasons: Dict[str, int] = {}

    mid_ms = start_ms + (end_ms - start_ms) // 2
    h1_pnls: List[float] = []
    h2_pnls: List[float] = []

    for t in res.trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        key = (t.symbol, int(t.entry_timestamp))
        risk_usd = init_risk_by_key.get(key, 0.0) if init_risk_by_key else 0.0
        if risk_usd <= 0.0:
            df = data_map[t.symbol]
            sig_idx = max(0, min(len(df) - 1, t.entry_idx - 1))
            atr_sig = float(df["atr_14"].iloc[sig_idx])
            if not (np.isfinite(atr_sig) and atr_sig > 0):
                atr_sig = float(t.entry_price) * 0.005
            if min_atr_pct > 0.0:
                atr_sig = max(atr_sig, float(df["close"].iloc[sig_idx]) * min_atr_pct)
            risk_usd = t.size * sl_atr * atr_sig
        r_mults.append(float(t.pnl_net / risk_usd) if risk_usd > 0 else 0.0)

        day_key = pd.Timestamp(int(t.exit_timestamp), unit="ms", tz="UTC").strftime("%Y-%m-%d")
        daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + float(t.pnl_net)
        if int(t.entry_timestamp) < mid_ms:
            h1_pnls.append(float(t.pnl_net))
        else:
            h2_pnls.append(float(t.pnl_net))

    d0 = pd.Timestamp(int(start_ms), unit="ms", tz="UTC").floor("D")
    d1 = pd.Timestamp(int(end_ms), unit="ms", tz="UTC").floor("D")
    idx_days = pd.date_range(d0, d1, freq="D", tz="UTC")
    daily_rets = np.array([daily_pnl.get(d.strftime("%Y-%m-%d"), 0.0) / 10000.0 for d in idx_days], dtype=np.float64)
    from statistical_rigor import calculate_annualized_sharpe
    sr_daily = calculate_annualized_sharpe(daily_rets, periods_per_year=365.0)
    dsr_info = deflated_sharpe_ratio(sr_daily, daily_rets, n_trials=max(2, n_trials_dsr), variance_of_trials=var_sharpes, periods_per_year=365.0)

    r_arr = np.asarray(r_mults, dtype=np.float64)
    if len(r_arr) > 0:
        rng = np.random.default_rng(42)
        boot = rng.choice(r_arr, size=(1500, len(r_arr)), replace=True).mean(axis=1)
        ci_lo, ci_hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    else:
        ci_lo, ci_hi = 0.0, 0.0

    def _half(ps: List[float]) -> Dict[str, Any]:
        if not ps:
            return {"trades": 0, "wr": 0.0, "pf": 0.0, "pnl": 0.0}
        arr = np.asarray(ps, dtype=float)
        w = arr[arr > 0.0].sum()
        l = abs(arr[arr < 0.0].sum())
        pf = w / l if l > 0 else (99.0 if w > 0 else 0.0)
        return {"trades": int(len(arr)), "wr": round(float((arr > 0.0).mean() * 100.0), 1), "pf": round(float(min(pf, 99.0)), 3), "pnl": round(float(arr.sum()), 2)}

    return {
        "trades": int(res.trades_count),
        "win_rate_pct": round(float(res.win_rate * 100.0), 2),
        "profit_factor": round(float(res.profit_factor), 3),
        "net_pnl_usd": round(float(res.net_profit), 2),
        "net_pnl_pct": round(float(res.net_profit_pct), 2),
        "expectancy_r": round(float(np.mean(r_arr)) if len(r_arr) else 0.0, 4),
        "expectancy_r_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        "max_dd_pct": round(float(res.max_drawdown_pct), 2),
        "sharpe_daily_ann": round(float(sr_daily), 3),
        "dsr": round(float(dsr_info["dsr"]), 4),
        "total_fees": round(float(res.total_fees), 2),
        "exit_reasons": exit_reasons,
        "h1": _half(h1_pnls),
        "h2": _half(h2_pnls),
        "_daily_sr": float(sr_daily),
    }


def _enrich_mode_deltas_and_dsr(mode_map: Dict[str, Dict[str, Any]], n_prior: int) -> Dict[str, Dict[str, Any]]:
    base_wr = mode_map["M0_NO_TAPE"]["win_rate_pct"]
    srs = [m["_daily_sr"] for m in mode_map.values()]
    var_sh = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    out: Dict[str, Dict[str, Any]] = {}
    for k, m in mode_map.items():
        mc = {kk: vv for kk, vv in m.items() if not kk.startswith("_")}
        mc["delta_wr_pp"] = round(float(mc["win_rate_pct"] - base_wr), 2)
        mc["dsr_pool_n"] = int(n_prior + len(mode_map))
        mc["var_trials_pool"] = round(var_sh, 4)
        out[k] = mc
    return out


# =====================================================================
# GOLDEN_POT 5m: Loader batch + features paralelas + comitê determinístico
# =====================================================================

def _fe_worker(item: Tuple[str, pd.DataFrame]) -> Tuple[str, Optional[pd.DataFrame], List[Tuple[int, str]]]:
    sym, df_raw = item
    df = FeatureEngineer.compute_all_features(df_raw, drop_na=False)
    df = df[df["sma_200"].notna()].reset_index(drop=True)
    if df.empty or len(df) < 200:
        return sym, None, []
    df["atr_median"] = df["atr_14"].shift(1).rolling(100, min_periods=20).median()
    df["volume_sma"] = df["vol_sma_20"].shift(1) if "vol_sma_20" in df.columns else df["volume"].shift(1).rolling(20, min_periods=10).mean()

    ts_arr = df["timestamp"].to_numpy(np.int64)
    atr_v = df["atr_14"].to_numpy(float)
    rng_v = df["high"].to_numpy(float) - df["low"].to_numpy(float)
    vz_v = df["volume_zscore_20"].to_numpy(float)
    i_start = max(48, int(np.searchsorted(ts_arr, TS_29D_START, side="left")))
    pre_mask = (
        (np.arange(len(df)) >= i_start)
        & (np.arange(len(df)) < len(df) - 1)
        & np.isfinite(atr_v) & (atr_v > 0.0) & (rng_v > 0.0)
        & ((rng_v / np.where(atr_v > 0.0, atr_v, 1.0)) >= 1.1)
        & np.isfinite(vz_v) & (vz_v >= 1.0)
    )
    cands: List[Tuple[int, str]] = []
    for k in np.where(pre_mask)[0]:
        sub = df.iloc[max(0, int(k) - 50) : int(k) + 1]
        trig = PaperTradingDaemon.gp_momentum_trigger(sub)
        if trig is None:
            continue
        direction, _ = trig
        is_climax, _ = PaperTradingDaemon._gp_check_climax_exhaustion(sub, direction, float(atr_v[k]))
        if not is_climax:
            cands.append((int(k), direction))
    if not cands:
        return sym, None, []
    return sym, df[df["timestamp"] >= TS_29D_START].copy().reset_index(drop=True), [
        (k - int(np.searchsorted(ts_arr, TS_29D_START, side="left")), d) for k, d in cands if int(ts_arr[k]) >= TS_29D_START
    ]


def _load_all_5m_gp_universe() -> Dict[str, pd.DataFrame]:
    """Carrega todas as velas 5m válidas (>=400 barras, fora da lista de exclusão GP) em 1 query RO."""
    with _ro_conn(DB_PATH) as conn:
        df_all = pd.read_sql_query(
            """
            SELECT symbol, exchange, timestamp, open, high, low, close, volume
            FROM candles
            WHERE timeframe = '5m'
            ORDER BY symbol ASC, timestamp ASC
            """,
            conn,
        )
    if df_all.empty:
        return {}
    df_all["_prio"] = np.where(df_all["exchange"] == "binance", 0, 1)
    df_all = df_all.sort_values(["symbol", "timestamp", "_prio"]).drop_duplicates(["symbol", "timestamp"], keep="first")

    groups: Dict[str, pd.DataFrame] = {}
    for sym, grp in df_all.groupby("symbol", sort=True):
        s_str = str(sym)
        if PaperTradingDaemon.is_gp_excluded_symbol(s_str):
            continue
        if len(grp) < 400:
            continue
        g = grp[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        # Descarta última vela possivelmente em formação
        groups[s_str] = g.iloc[:-1].copy().reset_index(drop=True)
    return groups


def evaluate_golden_pot_29d(
    btc_1h_29d: pd.DataFrame,
    ens_model: _DefaultMLEnsemble,
    static_lt: Any,
    tape_hist: TapeReadingAgent,
    tape_cache: Dict[Tuple[str, int, str], Dict[str, Any]],
    all_micro_syms: set,
    funding: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_prior: int,
) -> Dict[str, Any]:
    raw_5m = _load_all_5m_gp_universe()
    items = list(raw_5m.items())
    gp_data: Dict[str, pd.DataFrame] = {}
    gp_raw_cands: Dict[str, List[Tuple[int, str]]] = {}

    with ProcessPoolExecutor(max_workers=8) as ex:
        for sym, df_feat, cands in ex.map(_fe_worker, items, chunksize=8):
            if df_feat is not None and cands:
                gp_data[sym] = df_feat
                gp_raw_cands[sym] = cands

    # Mapa causal 1h do BTCUSDT (para filtro macro Item 1 ±1.5% SMA200)
    btc_ts_close = (btc_1h_29d["timestamp"].to_numpy(np.int64) + HOUR_MS)
    btc_dist_1h = ((btc_1h_29d["close"] - btc_1h_29d["sma_200"]) / btc_1h_29d["sma_200"]).to_numpy(float)

    ta = TechnicalAgent(model=ens_model)
    core = TradingCore(min_consensus_threshold=0.30, technical_agent=ta)
    core.learning_tracker = static_lt

    sigs_raw: Dict[str, np.ndarray] = {}
    confs_raw: Dict[str, np.ndarray] = {}
    sigs_macro: Dict[str, np.ndarray] = {}
    confs_macro: Dict[str, np.ndarray] = {}
    diag = {
        "symbols_scanned_5m": int(len(raw_5m)),
        "symbols_with_trigger": int(len(gp_data)),
        "raw_momentum_triggers": 0,
        "committee_agreed_conf70": 0,
        "macro_btc_blocked": 0,
        "covered_in_micro_db": int(sum(1 for s in gp_data if s in all_micro_syms)),
    }

    for sym, df in gp_data.items():
        n = len(df)
        s_r = np.zeros(n, dtype=np.int32)
        c_r = np.zeros(n, dtype=np.float64)
        s_m = np.zeros(n, dtype=np.int32)
        c_m = np.zeros(n, dtype=np.float64)
        for k, direction in gp_raw_cands[sym]:
            diag["raw_momentum_triggers"] += 1
            rec = df.iloc[k].to_dict()
            ts_sig = core.evaluate_market_deterministic(
                context=rec,
                symbol=sym,
                current_price=float(rec["close"]),
                atr=float(rec["atr_14"]),
                timestamp=int(rec["timestamp"]),
            )
            if ts_sig.is_vetoed or ts_sig.direction != direction or ts_sig.confidence < 0.70:
                continue
            diag["committee_agreed_conf70"] += 1
            code = 1 if direction == "BUY" else 2
            s_r[k] = code
            c_r[k] = float(ts_sig.confidence)

            # Macro alignment BTC 1h ±1.5% SMA200 (causal)
            now_close = int(rec["timestamp"]) + GP_BAR_MS
            bi = int(np.searchsorted(btc_ts_close, now_close, side="right")) - 1
            b_dist = float(btc_dist_1h[bi]) if 0 <= bi < len(btc_dist_1h) else 0.0
            macro_ok = not ((direction == "BUY" and b_dist < -0.015) or (direction == "SELL" and b_dist > 0.015))
            if macro_ok:
                s_m[k] = code
                c_m[k] = float(ts_sig.confidence)
            else:
                diag["macro_btc_blocked"] += 1

        if np.any(s_r != 0):
            sigs_raw[sym] = s_r
            confs_raw[sym] = c_r
            sigs_macro[sym] = s_m
            confs_macro[sym] = c_m

    data_active = {sym: gp_data[sym] for sym in sigs_raw}
    start_ms = TS_29D_START
    end_ms = max(int(df["timestamp"].iloc[-1]) for df in data_active.values()) if data_active else TS_29D_START

    def _run_gp_suite(base_sigs: Dict[str, np.ndarray], base_confs: Dict[str, np.ndarray], subset_covered_only: bool) -> Dict[str, Any]:
        sub_syms = sorted(s for s in data_active if (not subset_covered_only or s in all_micro_syms) and np.any(base_sigs[s] != 0))
        if not sub_syms:
            return {"note": "zero_signals"}
        d_sub = {s: data_active[s] for s in sub_syms}
        s_sub = {s: base_sigs[s] for s in sub_syms}
        c_sub = {s: base_confs[s] for s in sub_syms}

        eng_gp = BacktestEngine(
            initial_capital=10000.0,
            timeframe="5m",
            pilot_parity=True,
            strategy_type="GOLDEN_POT",
            max_leverage=2.0,
            daily_quota=3,
            cooldown_bars=12,
            risk_per_trade=None,
            fee_rate=0.0006,
            slippage=0.0002,
            maker_fee=0.0002,
            maker_slippage=0.0001,
        )
        modes_out: Dict[str, Any] = {}
        for mode in MODES:
            allow_alt = mode in ("M3_TAPE_WEIGHTED_SCORE", "M4_STRICT_OK_PLUS_WEIGHT")
            sm, cm, mm, st_cnt = apply_tape_mode(mode, d_sub, s_sub, c_sub, GP_BAR_MS, tape_hist, tape_cache, allow_unavail_alt=allow_alt)
            sim = _WeightedPilotSim(eng_gp, size_mult=mm, max_entries_per_bar=1, maker_entry_override=False)
            res = sim.run(d_sub, sm, cm, "atr_14", funding={})
            m = _compute_metrics_full(res, d_sub, STRATEGY_PROFILES["GOLDEN_POT"].sl_atr_mult, start_ms, end_ms, init_risk_by_key=sim._init_risk_by_key, n_trials_dsr=n_prior + 5)
            m["gate_stats"] = st_cnt
            modes_out[mode] = m
        return _enrich_mode_deltas_and_dsr(modes_out, n_prior)

    return {
        "diagnostics": diag,
        "gp_raw_all_symbols": _run_gp_suite(sigs_raw, confs_raw, False),
        "gp_macro_btc_all_symbols": _run_gp_suite(sigs_macro, confs_macro, False),
        "gp_macro_btc_covered_l2_only": _run_gp_suite(sigs_macro, confs_macro, True),
    }


# =====================================================================
# MAIN
# =====================================================================

def _render_md(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Anotações Empíricas — Impacto e Ponderação do `TapeReadingAgent` nas 3 Estratégias (`E10.10`)\n")
    lines.append(f"Gerado em `{report['generated_at_utc']}` | Janela 29d real (`microstructure.db`): `2026-10-15 12:30 UTC` .. `2026-11-13` | Conexão SQLite `mode=ro`.\n")

    lines.append("## 1. Como o Código Funciona Hoje vs O Que Testamos")
    lines.append("- **No `ConsensusJudge` ([core.py](file:///home/eudespatricio/ia_trader/core.py#L62-L69), [consensus_judge.py](file:///home/eudespatricio/ia_trader/consensus_judge.py#L57-L64)):** Apenas `OrderFlowAgent` (features OHLCV/derivativos) participa da votação ponderada com peso `0.8`. O `TapeReadingAgent` ([tape_reading_agent.py](file:///home/eudespatricio/ia_trader/agents/tape_reading_agent.py#L279-L323)) **não está dentro do consenso**.")
    lines.append("- **No daemon live ([run_paper_simulation.py](file:///home/eudespatricio/ia_trader/scripts/run_paper_simulation.py#L531-L581)):** Ele roda obrigatoriamente **depois do comitê** como barreira binária (`_flow_gate`) nas 3 estratégias (`DIRECTIONAL`, `GOLDEN_POT`, `SCALP`), **porém com política fail-open (`UNAVAILABLE_PASSED`)** quando falta dado de book/ticks.")
    lines.append("- **Simulação `M3_TAPE_WEIGHTED_SCORE` vs `M2_GATE_STRICT_OK`:** Testamos transformar o `TapeReadingAgent` em componente ponderado (`0.65 * convicção_comitê + 0.35 * tape_unit`, usando `flow_score` + `confidence` + veto em `spread_wide`/fluxo oposto e sizing `0.50x..1.25x`) versus o gate estrito binário (`status == 'OK' and vote == direction`).\n")

    lines.append("## 2. Auditoria Contrafactual Ao Vivo de `flow_gate_log` (`N=804`)")
    lines.append("| Estratégia | Corte | Sinais | WR% | ΔWR vs Sem Gate | PF | E[R] | PnL Sim (`$100`/lote) |")
    lines.append("| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    fg = report["block_a_flow_gate_log"]["by_strategy"]
    for strat in ("DIRECTIONAL", "SCALP", "GOLDEN_POT"):
        d = fg[strat]
        for label, key in [("Sem Gate (Todos Candidatos)", "all_candidates_no_gate"), ("Produção Atual (Fail-Open)", "current_fail_open_executed"), ("Gate Estrito (OK + Direção)", "strict_ok_only"), ("Passou Sem Dado (UNAVAILABLE)", "unavail_passed"), ("Bloqueados pelo Gate", "blocked_by_gate")]:
            v = d[key]
            dwr = "-"
            if key == "strict_ok_only" and d["delta_wr_strict_vs_nogate_pp"] is not None:
                dwr = f"{d['delta_wr_strict_vs_nogate_pp']:+.2f} pp"
            elif key == "current_fail_open_executed" and d["delta_wr_failopen_vs_nogate_pp"] is not None:
                dwr = f"{d['delta_wr_failopen_vs_nogate_pp']:+.2f} pp"
            lines.append(f"| `{strat}` | {label} | {v['count']} | {v['win_rate_pct']:.1f}% | {dwr} | {v['pf']:.3f} | {v['expectancy_r']:+.4f} | ${v['pnl_usd']:+.2f} |")
    lines.append("")

    def _tbl_modes(title: str, mdict: Dict[str, Any]) -> None:
        lines.append(f"### {title}")
        if not mdict or "note" in mdict:
            lines.append(f"_Sem sinais suficientes ({mdict})._\n")
            return
        lines.append("| Modo | Trades (Cand->Keep) | WR% | ΔWR | PF | E[R] | PnL (`$10k`) | MaxDD | H1 -> H2 (PF / PnL) |")
        lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
        for mode in MODES:
            v = mdict[mode]
            gs = v.get("gate_stats", {})
            ck = f"{gs.get('candidates', 0)}->{gs.get('kept', 0)}"
            h12 = f"`{v['h1']['pf']:.2f}` (${v['h1']['pnl']:+.1f}) / `{v['h2']['pf']:.2f}` (${v['h2']['pnl']:+.1f})"
            lines.append(f"| `{mode}` | {v['trades']} ({ck}) | {v['win_rate_pct']:.1f}% | {v['delta_wr_pp']:+.2f} pp | {v['profit_factor']:.3f} | {v['expectancy_r']:+.4f} | ${v['net_pnl_usd']:+.2f} | {v['max_dd_pct']:.2f}% | {h12} |")
        lines.append("")

    lines.append("## 3. Replay 29d L2 Real — Comparativo dos 5 Modos nas 3 Estratégias")
    _tbl_modes("3.1 `DIRECTIONAL` 1h (`BTC/ETH/SOL`, S4.2 + `corr_cap=2`)", report["block_b_29d"]["directional"])
    _tbl_modes("3.2a `GOLDEN_POT` 5m — Sem Filtro Macro BTC (Universo Completo 5m)", report["block_b_29d"]["golden_pot"]["gp_raw_all_symbols"])
    _tbl_modes("3.2b `GOLDEN_POT` 5m — Com Item 1 Macro BTC `±1.5% SMA200` (Universo Completo 5m)", report["block_b_29d"]["golden_pot"]["gp_macro_btc_all_symbols"])
    _tbl_modes("3.2c `GOLDEN_POT` 5m — Com Macro BTC Apenas nos 117 Símbolos Cobertos L2", report["block_b_29d"]["golden_pot"]["gp_macro_btc_covered_l2_only"])
    lines.append("### 3.3 `SCALP` 15m (`BTC/ETH/SOL`) — Produção vs Melhores Win Rates/Estruturas do Backtest")
    for sname, sres in report["block_b_29d"]["scalp"].items():
        _tbl_modes(f"`SCALP` — `{sname}`", sres)

    return "\n".join(lines) + "\n"


def main() -> int:
    t0 = time.time()
    print("=== E10.10: Auditoria Tape Reading nas 3 Estratégias (mode=ro) ===", flush=True)

    print("  [Bloco A] Auditando 804 registros reais de flow_gate_log...", flush=True)
    block_a = run_block_a_flow_gate_log()
    for s, d in block_a["by_strategy"].items():
        print(
            f"    {s:<12} | All={d['all_candidates_no_gate']['count']:>3} (WR {d['all_candidates_no_gate']['win_rate_pct']:>5.1f}%, PF {d['all_candidates_no_gate']['pf']:.2f}) "
            f"| StrictOK={d['strict_ok_only']['count']:>3} (WR {d['strict_ok_only']['win_rate_pct']:>5.1f}%, PF {d['strict_ok_only']['pf']:.2f}) "
            f"| Blocked={d['blocked_by_gate']['count']:>3} (WR {d['blocked_by_gate']['win_rate_pct']:>5.1f}%, PF {d['blocked_by_gate']['pf']:.2f}, PnL=${d['blocked_by_gate']['pnl_usd']:+.2f})",
            flush=True,
        )

    print("  [Bloco B] Carregando BTC/ETH/SOL (1h, 15m), ensemble OOS pré-2026-10-14 e TapeReadingAgent L2...", flush=True)
    funding = sr.load_funding_map(list(MAJORS))
    btc_raw_1h = sr.load_candles_clean("BTCUSDT", "1h", limit=None)
    btc_1h_full = sr.FeatureEngineer.compute_all_features(btc_raw_1h, drop_na=False)
    btc_1h_full = btc_1h_full[btc_1h_full["sma_200"].notna()].reset_index(drop=True)
    feat45 = [c for c in FeatureEngineer.FEATURE_COLUMNS if c in btc_1h_full.columns]
    lgbm_29d, rf_29d, n_tr = sr._train_pre_split_ensemble(btc_1h_full, TS_TRAIN_CUTOFF, feat45)
    ens_model = _DefaultMLEnsemble([lgbm_29d, rf_29d], feat45)
    static_lt = s4._StaticCalibrationTracker(DB_PATH)
    all_micro_syms = s4._get_all_micro_symbols()
    tape_hist = TapeReadingAgent(db_path=MICRO_DB_PATH, covered_symbols=all_micro_syms)
    tape_cache: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    n_prior = get_prior_trials_count()

    # 1) DIRECTIONAL 1h (29d)
    d29_1h, p29_1h, sig_dir_base, conf_dir_base = {}, {}, {}, {}
    for sym in MAJORS:
        df1h = btc_1h_full if sym == "BTCUSDT" else FeatureEngineer.compute_all_features(sr.load_candles_clean(sym, "1h", limit=None), drop_na=False)
        df1h = df1h[(df1h["sma_200"].notna()) & (df1h["timestamp"] >= TS_29D_START)].copy().reset_index(drop=True)
        d29_1h[sym] = df1h
        p29_1h[sym] = sr._predict_45_probas(df1h, lgbm_29d, rf_29d, feat45)
        sig_dir_base[sym], conf_dir_base[sym], _ = s4._generate_directional_core_signals(df1h, sym, p29_1h[sym], add_p2_risk_cols=True, learning_tracker=static_lt)

    start_1h = min(int(df["timestamp"].iloc[0]) for df in d29_1h.values())
    end_1h = max(int(df["timestamp"].iloc[-1]) for df in d29_1h.values())
    eng_dir = BacktestEngine(initial_capital=10000.0, timeframe="1h", pilot_parity=True, strategy_type="DIRECTIONAL", max_leverage=3.0, risk_per_trade=None)
    dir_results: Dict[str, Any] = {}
    print("\n  [DIRECTIONAL 1h 29d]", flush=True)
    for mode in MODES:
        sm, cm, mm, st_cnt = apply_tape_mode(mode, d29_1h, sig_dir_base, conf_dir_base, HOUR_MS, tape_hist, tape_cache, allow_unavail_alt=False)
        sim = _WeightedPilotSim(eng_dir, size_mult=mm, max_same_dir=2)
        res = sim.run(d29_1h, sm, cm, "atr_14", funding=funding)
        m = _compute_metrics_full(res, d29_1h, STRATEGY_PROFILES["DIRECTIONAL"].sl_atr_mult, start_1h, end_1h, init_risk_by_key=sim._init_risk_by_key, n_trials_dsr=n_prior + 5)
        m["gate_stats"] = st_cnt
        dir_results[mode] = m
        print(f"    {mode:<25} | Tr={m['trades']:>3} (cand {st_cnt['candidates']}->{st_cnt['kept']}) | WR={m['win_rate_pct']:>5.1f}% | PF={m['profit_factor']:>5.3f} | E[R]={m['expectancy_r']:>+6.3f} | PnL=${m['net_pnl_usd']:>+7.2f}", flush=True)
    dir_results = _enrich_mode_deltas_and_dsr(dir_results, n_prior)

    # 2) GOLDEN_POT 5m (29d universo completo 5m)
    print("\n  [GOLDEN_POT 5m 29d] Carregando e processando universo 5m...", flush=True)
    btc_1h_29d_sub = btc_1h_full[btc_1h_full["timestamp"] >= TS_29D_START - 20 * HOUR_MS].copy().reset_index(drop=True)
    gp_results = evaluate_golden_pot_29d(btc_1h_29d_sub, ens_model, static_lt, tape_hist, tape_cache, all_micro_syms, funding, n_prior + 5)
    print("    GP Diag:", gp_results["diagnostics"], flush=True)
    for gname in ("gp_raw_all_symbols", "gp_macro_btc_all_symbols", "gp_macro_btc_covered_l2_only"):
        print(f"    -- {gname} --", flush=True)
        grp = gp_results[gname]
        if "note" in grp:
            print("      sem sinais", flush=True)
            continue
        for mode in MODES:
            m = grp[mode]
            gs = m["gate_stats"]
            print(f"    {mode:<25} | Tr={m['trades']:>3} (cand {gs['candidates']}->{gs['kept']}) | WR={m['win_rate_pct']:>5.1f}% | PF={m['profit_factor']:>5.3f} | E[R]={m['expectancy_r']:>+6.3f} | PnL=${m['net_pnl_usd']:>+7.2f}", flush=True)

    # 3) SCALP 15m (29d) — Produção + Top Win Rates/Estruturas do Backtest
    print("\n  [SCALP 15m 29d] Preparando sinais em 29d...", flush=True)
    d29_15m, p29_15m = {}, {}
    for sym in MAJORS:
        df1h_sym = btc_1h_full if sym == "BTCUSDT" else FeatureEngineer.compute_all_features(sr.load_candles_clean(sym, "1h", limit=None), drop_na=False)
        df1h_sym = df1h_sym[df1h_sym["sma_200"].notna()].reset_index(drop=True)
        df15_sym = FeatureEngineer.compute_all_features(sr.load_candles_clean(sym, "15m", limit=None), drop_na=False)
        df15_sym = df15_sym[df15_sym["sma_200"].notna()].reset_index(drop=True)
        dfa = sr._attach_causal_features(df15_sym, df1h_sym, sym)
        mask = dfa["timestamp"].to_numpy(np.int64) >= TS_29D_START
        d29_15m[sym] = dfa[mask].copy().reset_index(drop=True)
        p29_15m[sym] = sr._predict_45_probas(dfa, lgbm_29d, rf_29d, feat45)[mask]

    specs_a = sr._build_signals_horizon_a(d29_15m, p29_15m)
    specs_b, _ = sr._build_signals_horizon_b(d29_15m, p29_15m)
    wanted = {
        "A0_LIVE_REAL_TAKER": "SCALP_PROD_BASE",
        "B4_FUNDING_WINDOW_FADE": "B4_FUNDING_WINDOW_FADE",
        "A8_RANGE_VWAP_REV_PULLBACK": "A8_RANGE_VWAP_REV_PULLBACK",
        "A9b_SWEEP_FAKEOUT_PULLBACK": "A9b_SWEEP_FAKEOUT_PULLBACK",
        "A4_BO20_DELAYED_BE_PULLBACK": "A4_BO20_DELAYED_BE_PULLBACK",
    }
    selected: Dict[str, Tuple[sr.ScalpSpec, Dict[str, np.ndarray], Dict[str, np.ndarray]]] = {}
    for sp, sg, cf in specs_a + specs_b:
        if sp.name in wanted:
            selected[wanted[sp.name]] = (sp, sg, cf)

    start_15 = min(int(df["timestamp"].iloc[0]) for df in d29_15m.values())
    end_15 = max(int(df["timestamp"].iloc[-1]) for df in d29_15m.values())
    scalp_results: Dict[str, Any] = {}
    for alias in ("SCALP_PROD_BASE", "B4_FUNDING_WINDOW_FADE", "A8_RANGE_VWAP_REV_PULLBACK", "A9b_SWEEP_FAKEOUT_PULLBACK", "A4_BO20_DELAYED_BE_PULLBACK"):
        sp, sg_base, cf_base = selected[alias]
        print(f"    -- SCALP {alias} --", flush=True)
        m_modes: Dict[str, Any] = {}
        maker_slip = 0.0 if sp.entry_exec == "limit_pullback" else 0.0001
        eng_sc = BacktestEngine(
            initial_capital=10000.0, timeframe="15m", pilot_parity=True, strategy_type="SCALP",
            max_leverage=3.0, daily_quota=15, cooldown_bars=4, risk_per_trade=None,
            fee_rate=0.0006, slippage=0.0002, maker_fee=0.0002, maker_slippage=maker_slip,
        )
        for mode in MODES:
            sm, cm, mm, st_cnt = apply_tape_mode(mode, d29_15m, sg_base, cf_base, BAR_15M_MS, tape_hist, tape_cache, allow_unavail_alt=False)
            sim = _WeightedScalpSim(eng_sc, sp, sm, size_mult=mm)
            res = sim.run(d29_15m, sm, cm, "atr_14", funding=funding)
            m = _compute_metrics_full(res, d29_15m, sp.sl_atr, start_15, end_15, min_atr_pct=0.0025, n_trials_dsr=n_prior + 25)
            m["gate_stats"] = st_cnt
            m["exec_stats"] = dict(sim.exec_stats)
            m_modes[mode] = m
            print(f"      {mode:<25} | Tr={m['trades']:>3} (cand {st_cnt['candidates']}->{st_cnt['kept']}) | WR={m['win_rate_pct']:>5.1f}% | PF={m['profit_factor']:>5.3f} | E[R]={m['expectancy_r']:>+6.3f} | PnL=${m['net_pnl_usd']:>+7.2f}", flush=True)
        scalp_results[alias] = _enrich_mode_deltas_and_dsr(m_modes, n_prior)

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "window_29d_start_utc": _ts_iso(TS_29D_START),
        "window_29d_end_utc": _ts_iso(end_15),
        "block_a_flow_gate_log": block_a,
        "block_b_29d": {
            "directional": dir_results,
            "golden_pot": gp_results,
            "scalp": scalp_results,
        },
    }

    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(_render_md(report))

    print(f"\n[OK] E10.10 concluída em {time.time() - t0:.1f}s.", flush=True)
    print(f"  JSON: {REPORT_JSON}", flush=True)
    print(f"  MD  : {REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
