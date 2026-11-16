#!/usr/bin/env python3
"""
scripts/eval_scalp_research_s4.py

Estudo Empírico Estrutural de Alternativas para Tornar o `SCALP` (15m) Lucrativo.
Zero Lookahead | Conexões SQLite Estritamente Read-Only (`mode=ro`) | Não altera código de produção.

Horizontes testados em `BTCUSDT`, `ETHUSDT`, `SOLUSDT` (custo real Bitget + funding 8h real):
- Horizonte A (~9-13 meses, 2025-10 .. 2026-11-13): Execução Limit Pullback vs Taker, guarda v1.0.5,
  Alvo Único Maker vs Escada BE atrasado, Squeeze Bollinger/Keltner, Liquidez Sessão,
  Reversão em Range Lateral e Sweep Fakeout (Stop-Hunt).
- Horizonte B (60 dias Binance 5m, 2026-09-15 .. 2026-11-13): `oi`, `taker`, `ls_global`, `ls_top_pos`
  (OI buildup breakout, OI flush cascade fade, divergência Smart vs Varejo, janela de funding).
- Horizonte C (29 dias L2/Ticks/Derivs, 2026-10-15 .. 2026-11-13): `book_features` + `tick_agg`
  (Absorção L2 Maker Fade, Ignição OBI/Taker, Filtro anti-fluxo tóxico nos candidatos A/B).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.tape_reading_agent import TapeReadingAgent  # noqa: E402
from backtesting import BacktestEngine, BacktestResult, _PilotPos, _PilotSimulator  # noqa: E402
from data.feature_engineering import FeatureEngineer  # noqa: E402
from scripts.eval_s4_calibration_and_open import (  # noqa: E402
    BAR_15M_MS,
    DAY_MS,
    DB_PATH,
    HOUR_MS,
    MAJORS,
    MICRO_DB_PATH,
    _generate_scalp_live_signals,
    _predict_45_probas,
    _ro_conn,
    _tape_decision,
    _train_pre_split_ensemble,
    _ts_iso,
    load_candles_clean,
    load_funding_map,
)
from smart_trade import StrategyProfile  # noqa: E402
from statistical_rigor import (  # noqa: E402
    bootstrap_performance_metrics,
    calculate_annualized_sharpe,
    deflated_sharpe_ratio,
    get_prior_trials_count,
)

REPORT_JSON = os.path.join(ROOT, "datasets", "reports", "scalp_research_s4_20261113.json")
REPORT_MD = os.path.join(ROOT, "docs", "ESTUDO_SCALP_S4_ALTERNATIVAS.md")


@dataclass(frozen=True)
class ScalpSpec:
    """Especificação paramétrica de cada experimento offline do SCALP."""

    name: str
    horizon: str  # "A", "B", "C"
    entry_exec: str  # "sim_parity", "real_taker", "limit_pullback"
    sl_atr: float
    tp1_atr: float
    tp2_atr: float
    tp3_atr: float
    trail_atr: float
    tp1_frac: float = 0.40
    tp2_frac: float = 0.30
    be_mode: str = "tp1_legacy"  # "tp1_legacy", "tp1_v105", "tp2_delayed", "none"
    be_buffer_pct: float = 0.0004
    max_hold_bars: int = 30
    pullback_atr: float = 0.0
    limit_ttl_bars: int = 2
    trade_through_bps: float = 2.0
    min_conf: float = 0.50
    desc: str = ""

    @property
    def single_tp(self) -> bool:
        return self.tp1_frac == 0.0 and self.tp2_frac == 0.0


class _ScalpResearchSimulator(_PilotSimulator):
    """Simulador offline derivado de _PilotSimulator para testar execução/saídas sem editar smart_trade/backtesting."""

    def __init__(self, eng: BacktestEngine, spec: ScalpSpec):
        super().__init__(eng)
        self.spec = spec
        self.max_hold = int(spec.max_hold_bars)
        self.prof = StrategyProfile(
            sl_atr_mult=spec.sl_atr,
            tp1_atr_mult=spec.tp1_atr,
            tp2_atr_mult=spec.tp2_atr,
            tp3_atr_mult=spec.tp3_atr,
            trail_atr_mult=spec.trail_atr,
            min_committee_confidence=spec.min_conf,
            bar_ms=BAR_15M_MS,
            max_holding_bars=spec.max_hold_bars,
        )
        # sim_parity mantém entrada 2bps+1bp como produção; real_taker cobra 6bps+2bps; limit_pullback usa 2bps+0bp
        self.meta = dict(self.meta)
        self.meta["maker_entry"] = spec.entry_exec in ("sim_parity", "limit_pullback")
        maker_slip = 0.0 if spec.entry_exec == "limit_pullback" else self.st.MAKER_SLIPPAGE_PCT
        st_dict = {
            **vars(self.st),
            "MAKER_SLIPPAGE_PCT": maker_slip,
            "EXIT_TP1_FRACTION": spec.tp1_frac,
            "EXIT_TP2_FRACTION": spec.tp2_frac,
            "BREAKEVEN_FEE_BUFFER_PCT": spec.be_buffer_pct,
        }
        self.st = SimpleNamespace(**st_dict)
        self._filled_bar_idx: Dict[str, int] = {}
        self.exec_stats: Dict[str, int] = {
            "signals_seen": 0,
            "orders_queued": 0,
            "orders_filled": 0,
            "orders_expired": 0,
        }

    def process_bar_ctx(
        self, p: _PilotPos, ts: int, idx: int, o: float, h: float, l: float, c: float
    ) -> Optional[Tuple[str, float]]:
        st, sp = self.st, self.spec
        long, sgn = p.long, p.sgn
        self._apply_funding(p, int(ts), c)
        if self.max_hold > 0 and int(ts) >= int(p.entry_ts) + self.max_hold * self.bar_ms:
            return "HORIZON_TIMEOUT", o
        if p.liq > 0.0 and ((long and l <= p.liq and p.liq >= p.sl) or ((not long) and h >= p.liq and p.liq <= p.sl)):
            return "LIQUIDATION", p.liq
        if p.trail_active and p.trail_bar is not None:
            self._ratchet(p, p.trail_bar[0], p.trail_bar[1])
            p.trail_bar = None

        def stop_hit() -> bool:
            return (l <= p.sl) if long else (h >= p.sl)

        def reached(level: float) -> bool:
            return level > 0.0 and ((h >= level) if long else (l <= level))

        def stop_close(pre_existing: bool) -> Tuple[str, float]:
            if long:
                fill = min(p.sl, h)
                if pre_existing and o > 0.0:
                    fill = min(fill, o)
            else:
                fill = max(p.sl, l)
                if pre_existing and o > 0.0:
                    fill = max(fill, o)
            return self._stop_reason(p), fill

        # Barra de fill intra-barra (limit_pullback): avalia APENAS risco adverso (LIQ/SL); TPs só valem a partir da barra seguinte
        if self._filled_bar_idx.get(p.sym) == idx:
            if stop_hit():
                return stop_close(False)
            return None

        if stop_hit():
            return stop_close(True)

        if sp.single_tp:
            if reached(p.tp3):
                return "TAKE_PROFIT_3", p.tp3
            return None

        if not p.tp1_hit and reached(p.tp1):
            p.tp1_hit = True
            self.balance += self._leg(p, p.tp1, min(p.qty0 * sp.tp1_frac, p.qty), "TAKE_PROFIT_1", True)
            if sp.be_mode in ("tp1_legacy", "tp1_v105"):
                be = p.entry * (1.0 + sgn * sp.be_buffer_pct)
                p.sl = max(p.sl, be) if long else min(p.sl, be)
                if sp.be_mode == "tp1_legacy":
                    if stop_hit():
                        return stop_close(False)
                else:  # tp1_v105: só estopa na mesma barra se fechar violando o BE
                    closed_safe = (c > p.sl) if long else (c < p.sl)
                    same_bar_safe = o > 0.0 and closed_safe
                    if stop_hit() and not same_bar_safe:
                        return stop_close(False)

        if p.tp1_hit and not p.tp2_hit and reached(p.tp2):
            p.tp2_hit = True
            self.balance += self._leg(p, p.tp2, min(p.qty0 * sp.tp2_frac, p.qty), "TAKE_PROFIT_2", True)
            if sp.be_mode == "tp2_delayed":
                be = p.entry * (1.0 + sgn * sp.be_buffer_pct)
                p.sl = max(p.sl, be) if long else min(p.sl, be)
            if p.trail_mult > 0.0 and p.entry_atr > 0.0:
                p.trail_active = True
                p.trail_ref = p.tp2
                p.trail_bar = None
                self._ratchet(p, p.tp2, p.tp2)
            if sp.be_mode == "tp2_delayed":
                closed_safe = (c > p.sl) if long else (c < p.sl)
                if stop_hit() and not (o > 0.0 and closed_safe):
                    return stop_close(False)

        if p.tp2_hit and reached(p.tp3):
            return "TAKE_PROFIT_3", p.tp3
        if p.trail_active:
            p.trail_bar = (h, l)
        return None

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        signals: Dict[str, np.ndarray],
        confidences: Optional[Dict[str, np.ndarray]],
        atr_col: str,
        funding: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> BacktestResult:
        e, sp = self.e, self.spec
        self.funding = funding or {}
        self.balance = float(e.initial_capital)
        self._filled_bar_idx.clear()
        self.exec_stats = {"signals_seen": 0, "orders_queued": 0, "orders_filled": 0, "orders_expired": 0}

        arr: Dict[str, Dict[str, Any]] = {}
        for s, df in data.items():
            closes = df["close"].to_numpy(float)
            arr[s] = {
                "ts": df["timestamp"].to_numpy(),
                "o": df["open"].to_numpy(float),
                "h": df["high"].to_numpy(float),
                "l": df["low"].to_numpy(float),
                "c": closes,
                "sig": np.asarray(signals[s]).astype(int),
                "atr": df[atr_col].to_numpy(float) if atr_col in df.columns else closes * 0.005,
                "sma": df["sma_200"].to_numpy(float) if "sma_200" in df.columns else None,
                "conf": (np.asarray(confidences[s], dtype=float) if confidences and s in confidences else None),
            }
        index = {s: {int(t): k for k, t in enumerate(a["ts"])} for s, a in arr.items()}
        all_ts = sorted({t for m in index.values() for t in m})
        positions: Dict[str, _PilotPos] = {}
        pending: Dict[str, Dict[str, Any]] = {}
        state: Dict[str, Any] = {"last_entry_idx": {}, "quota": {}}
        trades: List[Any] = []
        equity_curve: List[float] = [self.balance]

        from backtesting import TradeRecord  # import local limpo

        def finalize(p: _PilotPos, reason: str, ref: float, ts: int, idx: int) -> None:
            if p.qty > 0:
                self.balance += self._leg(p, ref, p.qty, reason, partial=False)
            self.balance = max(0.0, self.balance)
            exit_qty = p.qty0 - p.qty
            pnl_net = p.gross - p.fees - p.funding
            trades.append(TradeRecord(
                trade_id=len(trades) + 1,
                direction=p.side,
                entry_idx=p.entry_idx,
                entry_timestamp=int(p.entry_ts),
                entry_price=p.entry,
                exit_idx=idx,
                exit_timestamp=int(ts),
                exit_price=(p.exit_qty_px / exit_qty) if exit_qty > 0 else p.entry,
                size=p.qty0,
                pnl_gross=p.gross,
                fee_total=p.fees,
                funding_total=p.funding,
                pnl_net=pnl_net,
                return_pct=pnl_net / (p.margin + 1e-8),
                exit_reason=reason,
                symbol=p.sym,
                leverage=p.lev,
                margin=p.margin,
                legs=p.n_legs,
            ))
            self._filled_bar_idx.pop(p.sym, None)
            del positions[p.sym]

        last_idx: Dict[str, int] = {}
        tt = float(sp.trade_through_bps) * 1e-4
        is_limit = sp.entry_exec == "limit_pullback"

        for t in all_ts:
            present = [s for s in arr if t in index[s]]

            # 1) Posições abertas existentes
            for s in present:
                k = index[s][t]
                last_idx[s] = k
                a = arr[s]
                p = positions.get(s)
                if p is None or k < p.entry_idx:
                    continue
                out = self.process_bar_ctx(p, t, k, a["o"][k], a["h"][k], a["l"][k], a["c"][k])
                if out is not None:
                    finalize(p, out[0], out[1], t, k)

            # 1b) Ordens Limit pendentes (apenas limit_pullback)
            if is_limit:
                for s in present:
                    ord_info = pending.get(s)
                    if ord_info is None:
                        continue
                    k = index[s][t]
                    if k < ord_info["start_idx"]:
                        continue
                    if s in positions:
                        del pending[s]
                        continue
                    a = arr[s]
                    lpx = float(ord_info["limit_px"])
                    long_ord = ord_info["dir"] == 1
                    hit = (a["l"][k] <= lpx * (1.0 - tt)) if long_ord else (a["h"][k] >= lpx * (1.0 + tt))
                    if hit:
                        opened = self.try_open(
                            s, k, ord_info["dir"], int(t), float(ord_info["sig_close"]),
                            lpx, float(ord_info["atr"]), ord_info["conf"], ord_info["regime"], positions, state,
                        )
                        del pending[s]
                        if opened is not None:
                            self.exec_stats["orders_filled"] += 1
                            self._filled_bar_idx[s] = k
                            out = self.process_bar_ctx(opened, t, k, a["o"][k], a["h"][k], a["l"][k], a["c"][k])
                            if out is not None:
                                finalize(opened, out[0], out[1], t, k)
                    elif k >= ord_info["expire_idx"]:
                        self.exec_stats["orders_expired"] += 1
                        del pending[s]

            # 2) Novos sinais no close da barra k
            for s in present:
                k = index[s][t]
                a = arr[s]
                sg = int(a["sig"][k])
                if sg not in (1, 2, -1) or k + 1 >= len(a["ts"]):
                    continue
                self.exec_stats["signals_seen"] += 1
                if s in positions:
                    continue
                sg_norm = 1 if sg == 1 else 2
                dir_int = 1 if sg_norm == 1 else -1
                conf = None if a["conf"] is None else float(a["conf"][k])
                regime = None
                if a["sma"] is not None:
                    sma, cl = float(a["sma"][k]), float(a["c"][k])
                    if np.isfinite(sma) and np.isfinite(cl) and sma > 0:
                        regime = (cl >= sma) if sg_norm == 1 else (cl <= sma)
                atr_k = float(a["atr"][k])
                if is_limit:
                    if s in pending:
                        continue
                    limit_px = float(a["c"][k]) - dir_int * sp.pullback_atr * atr_k
                    if limit_px <= 0:
                        continue
                    self.exec_stats["orders_queued"] += 1
                    pending[s] = {
                        "dir": dir_int,
                        "limit_px": limit_px,
                        "sig_close": float(a["c"][k]),
                        "atr": atr_k,
                        "conf": conf,
                        "regime": regime,
                        "start_idx": k + 1,
                        "expire_idx": k + sp.limit_ttl_bars,
                    }
                else:
                    self.exec_stats["orders_queued"] += 1
                    raw = float(a["o"][k + 1]) if e.entry_mode == "next_open" else float(a["c"][k])
                    opened = self.try_open(
                        s, k + 1, dir_int, int(a["ts"][k + 1]), float(a["c"][k]), raw,
                        atr_k, conf, regime, positions, state,
                    )
                    if opened is not None:
                        self.exec_stats["orders_filled"] += 1

            unreal = sum(p.sgn * (arr[s]["c"][last_idx[s]] - p.entry) * p.qty for s, p in positions.items() if s in last_idx)
            equity_curve.append(self.balance + unreal)

        for s in list(positions):
            a = arr[s]
            k = len(a["ts"]) - 1
            finalize(positions[s], "END_OF_DATA", float(a["c"][k]), int(a["ts"][k]), k)
        if equity_curve:
            equity_curve[-1] = self.balance
        return e._pilot_result(trades, equity_curve, self.balance)


def run_spec(
    spec: ScalpSpec,
    data_map: Dict[str, pd.DataFrame],
    signals: Dict[str, np.ndarray],
    confidences: Optional[Dict[str, np.ndarray]],
    funding: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[BacktestResult, Dict[str, int]]:
    maker_slip = 0.0 if spec.entry_exec == "limit_pullback" else 0.0001
    eng = BacktestEngine(
        initial_capital=10000.0,
        timeframe="15m",
        pilot_parity=True,
        strategy_type="SCALP",
        max_leverage=3.0,
        daily_quota=15,
        cooldown_bars=4,
        risk_per_trade=None,
        fee_rate=0.0006,
        slippage=0.0002,
        maker_fee=0.0002,
        maker_slippage=maker_slip,
    )
    sim = _ScalpResearchSimulator(eng, spec)
    res = sim.run(data_map, signals, confidences, "atr_14", funding=funding)
    return res, dict(sim.exec_stats)


def compute_scalp_metrics(
    res: BacktestResult,
    data_map: Dict[str, pd.DataFrame],
    sl_atr: float,
    start_ms: int,
    end_ms: int,
    exec_stats: Optional[Dict[str, int]] = None,
    n_trials_dsr: int = 1,
    var_sharpes: float = 0.0,
) -> Dict[str, Any]:
    """Calcula métricas normalizadas pelo sl_atr real do ScalpSpec + Bootstrap/DSR."""
    r_mults: List[float] = []
    daily_pnl: Dict[str, float] = {}
    exit_reasons: Dict[str, int] = {}

    for t in res.trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        sym_df = data_map.get(t.symbol)
        if sym_df is None:
            sym_df = next(iter(data_map.values()))
        sig_idx = max(0, min(len(sym_df) - 1, t.entry_idx - 1))
        atr_sig = float(sym_df["atr_14"].iloc[sig_idx])
        if not (np.isfinite(atr_sig) and atr_sig > 0):
            atr_sig = float(t.entry_price) * 0.005
        atr_sig = max(atr_sig, float(sym_df["close"].iloc[sig_idx]) * 0.0025)
        risk_usd = t.size * sl_atr * atr_sig
        r_mults.append(float(t.pnl_net / risk_usd) if risk_usd > 0 else 0.0)
        day_key = pd.Timestamp(int(t.exit_timestamp), unit="ms", tz="UTC").strftime("%Y-%m-%d")
        daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + float(t.pnl_net)

    d0 = pd.Timestamp(int(start_ms), unit="ms", tz="UTC").floor("D")
    d1 = pd.Timestamp(int(end_ms), unit="ms", tz="UTC").floor("D")
    idx_days = pd.date_range(d0, d1, freq="D", tz="UTC")
    daily_rets = np.array([daily_pnl.get(d.strftime("%Y-%m-%d"), 0.0) / 10000.0 for d in idx_days], dtype=np.float64)

    sr_daily_ann = calculate_annualized_sharpe(daily_rets, periods_per_year=365.0)
    dsr_info = deflated_sharpe_ratio(
        observed_sr_ann=sr_daily_ann,
        returns=daily_rets,
        n_trials=max(2, n_trials_dsr),
        variance_of_trials=var_sharpes,
        periods_per_year=365.0,
    )
    boot_info = bootstrap_performance_metrics(daily_rets, n_iterations=800, periods_per_year=365.0, seed=42, block_size=7)
    sh_ci = boot_info.get("sharpe_ci", (0.0, 0.0))

    r_arr = np.asarray(r_mults, dtype=np.float64)
    if len(r_arr) > 0:
        rng = np.random.default_rng(42)
        boot_r = rng.choice(r_arr, size=(2000, len(r_arr)), replace=True).mean(axis=1)
        r_ci_lo, r_ci_hi = float(np.quantile(boot_r, 0.025)), float(np.quantile(boot_r, 0.975))
    else:
        r_ci_lo, r_ci_hi = 0.0, 0.0

    fill_rate = 0.0
    if exec_stats and exec_stats.get("orders_queued", 0) > 0:
        fill_rate = exec_stats["orders_filled"] / exec_stats["orders_queued"] * 100.0

    return {
        "trades": int(res.trades_count),
        "win_rate_pct": round(float(res.win_rate * 100.0), 2),
        "profit_factor": round(float(res.profit_factor), 3),
        "net_pnl_usd": round(float(res.net_profit), 2),
        "net_pnl_pct": round(float(res.net_profit_pct), 2),
        "expectancy_r": round(float(np.mean(r_arr)) if len(r_arr) else 0.0, 4),
        "expectancy_r_ci95": [round(r_ci_lo, 4), round(r_ci_hi, 4)],
        "max_dd_pct": round(float(res.max_drawdown_pct), 2),
        "sharpe_daily_ann": round(float(sr_daily_ann), 3),
        "dsr": round(float(dsr_info["dsr"]), 4),
        "daily_boot_sharpe_ci95": [round(float(sh_ci[0]), 3), round(float(sh_ci[1]), 3)],
        "total_fees_funding": round(float(res.total_fees), 2),
        "exit_reasons": exit_reasons,
        "exec_stats": exec_stats or {},
        "fill_rate_pct": round(fill_rate, 1),
        "_daily_rets": daily_rets,
        "_sr_daily_ann": float(sr_daily_ann),
    }


def _strip_internal(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if not k.startswith("_")}


# =====================================================================
# CARGA & ALINHAMENTO CAUSAL (1H, 5M DERIVS BINANCE, 15S/1M L2 MICRO)
# =====================================================================

def _pivot_binance_series(sym: str) -> pd.DataFrame:
    with _ro_conn(MICRO_DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT ts_ms, kind, v1, v2, v3 FROM binance_series WHERE symbol=? ORDER BY ts_ms ASC",
            conn,
            params=(sym,),
        )
    if df.empty:
        return pd.DataFrame(columns=["ts_ms"])

    wide_frames: List[pd.DataFrame] = []
    # Cada ponto 5m aberto em ts_ms fecha em ts_5m_close = ts_ms + 300_000
    for kind, cols in [
        ("oi", ("oi_contracts", "oi_usd")),
        ("taker", ("taker_ratio", "taker_buy_vol", "taker_sell_vol")),
        ("ls_global", ("ls_global_ratio", "ls_global_long")),
        ("ls_top_pos", ("ls_top_ratio", "ls_top_long")),
    ]:
        sub = df[df["kind"] == kind].drop_duplicates(subset=["ts_ms"], keep="last").sort_values("ts_ms")
        out = pd.DataFrame({"ts_5m": sub["ts_ms"].to_numpy(np.int64)})
        if kind == "oi":
            out["oi_usd"] = np.where(sub["v2"].notna() & (sub["v2"] > 0), sub["v2"].to_numpy(float), sub["v1"].to_numpy(float))
        elif kind == "taker":
            out["taker_buy_vol"] = sub["v2"].fillna(0.0).to_numpy(float)
            out["taker_sell_vol"] = sub["v3"].fillna(0.0).to_numpy(float)
        elif kind == "ls_global":
            out["ls_global_long"] = sub["v2"].to_numpy(float)
        elif kind == "ls_top_pos":
            out["ls_top_long"] = sub["v2"].to_numpy(float)
        wide_frames.append(out.set_index("ts_5m"))

    merged = pd.concat(wide_frames, axis=1).sort_index().reset_index()
    merged["ts_5m_close"] = merged["ts_5m"] + 300_000
    return merged


def _attach_causal_features(df15: pd.DataFrame, df1h_feat: pd.DataFrame, sym: str) -> pd.DataFrame:
    df_w = df15.copy().reset_index(drop=True)
    df_w["atr_14"] = np.maximum(df_w["atr_14"].to_numpy(float), df_w["close"].to_numpy(float) * 0.0025)
    df_w["natr_bps"] = df_w["atr_14"] / df_w["close"] * 1e4
    df_w["ts_close"] = df_w["timestamp"].astype(np.int64) + BAR_15M_MS

    # Alinhamento estritamente causal 1h -> 15m (usa apenas velas 1h JÁ FECHADAS)
    h1_c = pd.DataFrame({
        "ts_1h_close": df1h_feat["timestamp"].astype(np.int64) + HOUR_MS,
        "h1_dist_sma200": df1h_feat["dist_sma_200"].to_numpy(float),
        "h1_rsi_14": df1h_feat["rsi_14"].to_numpy(float),
    }).sort_values("ts_1h_close")
    df_w = pd.merge_asof(
        df_w.sort_values("ts_close"),
        h1_c,
        left_on="ts_close",
        right_on="ts_1h_close",
        direction="backward",
    ).reset_index(drop=True)

    # Indicadores estruturais 15m
    c = df_w["close"].to_numpy(float)
    o = df_w["open"].to_numpy(float)
    h = df_w["high"].to_numpy(float)
    l = df_w["low"].to_numpy(float)
    rng = np.where((h - l) > 0, h - l, 1e-9)
    df_w["cpos"] = (c - l) / rng
    df_w["lower_wick_frac"] = (np.minimum(o, c) - l) / rng
    df_w["upper_wick_frac"] = (h - np.maximum(o, c)) / rng
    df_w["prior_high20"] = df_w["high"].shift(1).rolling(20).max()
    df_w["prior_low20"] = df_w["low"].shift(1).rolling(20).min()
    df_w["ret_3b_pct"] = (df_w["close"] / df_w["close"].shift(3) - 1.0) * 100.0

    # Squeeze: Bollinger(20, 2.0) dentro de Keltner(EMA20, 1.5 ATR_14)
    kc_mid = df_w["close"].ewm(span=20, adjust=False).mean()
    kc_hi = kc_mid + 1.5 * df_w["atr_14"]
    kc_lo = kc_mid - 1.5 * df_w["atr_14"]
    in_sqz = (df_w["bb_high"] < kc_hi) & (df_w["bb_low"] > kc_lo)
    # Squeeze ativo na barra anterior (t-1 e t-2) -> disparo no rompimento de t
    df_w["prior_squeeze_2b"] = in_sqz.shift(1).fillna(False) & in_sqz.shift(2).fillna(False)
    df_w["kc_hi"] = kc_hi
    df_w["kc_lo"] = kc_lo

    # Horário UTC & dia útil para filtros de liquidez / janela funding
    dt_utc = pd.to_datetime(df_w["ts_close"], unit="ms", utc=True)
    df_w["utc_hour"] = dt_utc.dt.hour.to_numpy(int)
    df_w["utc_min"] = dt_utc.dt.minute.to_numpy(int)
    df_w["utc_dow"] = dt_utc.dt.dayofweek.to_numpy(int)  # 0=Mon .. 6=Sun
    min_of_day = df_w["utc_hour"] * 60 + df_w["utc_min"]
    # Distância em minutos até 00:00, 08:00, 16:00 UTC
    d_fund = np.minimum.reduce([
        np.minimum(min_of_day, 1440 - min_of_day),
        np.abs(min_of_day - 480),
        np.abs(min_of_day - 960),
    ])
    df_w["mins_to_funding"] = d_fund

    # Alinhamento Binance 5m derivatives -> 15m (ts_5m_close <= ts_close)
    b5 = _pivot_binance_series(sym)
    for col in ("oi_usd", "oi_chg_15m_pct", "oi_chg_30m_pct", "oi_chg_1h_pct", "taker_ratio_15m", "smart_div", "global_long_z"):
        df_w[col] = np.nan

    if not b5.empty:
        # OI snapshot mais recente <= ts_close (tolerância 15m = 900_000ms)
        oi_df = b5[["ts_5m_close", "oi_usd", "ls_global_long", "ls_top_long"]].dropna(subset=["oi_usd"]).sort_values("ts_5m_close")
        m_oi = pd.merge_asof(
            df_w[["ts_close"]].sort_values("ts_close"),
            oi_df,
            left_on="ts_close",
            right_on="ts_5m_close",
            direction="backward",
            tolerance=900_000,
        )
        df_w["oi_usd"] = m_oi["oi_usd"].to_numpy(float)
        df_w["oi_chg_15m_pct"] = (df_w["oi_usd"] / df_w["oi_usd"].shift(1) - 1.0) * 100.0
        df_w["oi_chg_30m_pct"] = (df_w["oi_usd"] / df_w["oi_usd"].shift(2) - 1.0) * 100.0
        df_w["oi_chg_1h_pct"] = (df_w["oi_usd"] / df_w["oi_usd"].shift(4) - 1.0) * 100.0

        gl = m_oi["ls_global_long"].astype(float)
        tp = m_oi["ls_top_long"].astype(float)
        df_w["smart_div"] = (tp - gl).to_numpy(float)
        rm = gl.rolling(96, min_periods=24).mean()
        rs = gl.rolling(96, min_periods=24).std().replace(0.0, np.nan)
        df_w["global_long_z"] = ((gl - rm) / rs).to_numpy(float)

        # Taker volume agregado nos 15m encerrados em ts_close: (ts_close - 900_000 < ts_5m_close <= ts_close)
        tk = b5[["ts_5m_close", "taker_buy_vol", "taker_sell_vol"]].dropna().sort_values("ts_5m_close")
        tk_t = tk["ts_5m_close"].to_numpy(np.int64)
        tk_b = tk["taker_buy_vol"].to_numpy(float)
        tk_s = tk["taker_sell_vol"].to_numpy(float)
        ts_c_arr = df_w["ts_close"].to_numpy(np.int64)
        hi_i = np.searchsorted(tk_t, ts_c_arr, side="right")
        lo_i = np.searchsorted(tk_t, ts_c_arr - BAR_15M_MS, side="right")
        cs_b = np.concatenate([[0.0], np.cumsum(tk_b)])
        cs_s = np.concatenate([[0.0], np.cumsum(tk_s)])
        sb = cs_b[hi_i] - cs_b[lo_i]
        ss = cs_s[hi_i] - cs_s[lo_i]
        with np.errstate(divide="ignore", invalid="ignore"):
            df_w["taker_ratio_15m"] = np.where((sb > 0) & (ss > 0), sb / ss, np.nan)

    # Alinhamento L2 book_features + tick_agg + derivs (para Horizonte C 29d)
    for col in ("l2_taker_imb_5m", "l2_book_imb_10", "l2_book_imb_25", "l2_spread_bps", "l2_mdev_bps", "deriv_basis_bps", "deriv_funding"):
        df_w[col] = np.nan

    with _ro_conn(MICRO_DB_PATH) as conn:
        ticks = pd.read_sql_query(
            "SELECT minute_ms, buy_notional, sell_notional, n_trades FROM tick_agg WHERE symbol=? ORDER BY minute_ms ASC",
            conn, params=(sym,),
        )
        books = pd.read_sql_query(
            "SELECT ts_ms, spread_bps, microprice_dev_bps, bid_n10, ask_n10, bid_n25, ask_n25 FROM book_features WHERE symbol=? ORDER BY ts_ms ASC",
            conn, params=(sym,),
        )
        derivs = pd.read_sql_query(
            "SELECT ts_ms, funding_rate, mark_price, index_price FROM derivs WHERE symbol=? ORDER BY ts_ms ASC",
            conn, params=(sym,),
        )

    ts_c_arr = df_w["ts_close"].to_numpy(np.int64)
    if not ticks.empty:
        t_ms = ticks["minute_ms"].to_numpy(np.int64)
        b_not = ticks["buy_notional"].fillna(0.0).to_numpy(float)
        s_not = ticks["sell_notional"].fillna(0.0).to_numpy(float)
        n_tr = ticks["n_trades"].fillna(0).to_numpy(np.int64)
        # Cada minute_ms fecha em minute_ms + 60_000; exige fechamento <= ts_close (minute_ms <= ts_close - 60_000)
        hi = np.searchsorted(t_ms, ts_c_arr - 60_000, side="right")
        lo = np.searchsorted(t_ms, ts_c_arr - 300_000, side="left")
        cb = np.concatenate([[0.0], np.cumsum(b_not)])
        cs = np.concatenate([[0.0], np.cumsum(s_not)])
        cn = np.concatenate([[0], np.cumsum(n_tr)])
        buy5 = cb[hi] - cb[lo]
        sel5 = cs[hi] - cs[lo]
        ntr5 = cn[hi] - cn[lo]
        ok_t = ((buy5 + sel5) > 0) & (ntr5 >= 20) & ((hi - lo) >= 3)
        df_w["l2_taker_imb_5m"] = np.where(ok_t, (buy5 - sel5) / np.maximum(buy5 + sel5, 1e-9), np.nan)

    if not books.empty:
        b_ms = books["ts_ms"].to_numpy(np.int64)
        imb10 = ((books["bid_n10"] - books["ask_n10"]) / np.maximum(books["bid_n10"] + books["ask_n10"], 1e-9)).to_numpy(float)
        imb25 = ((books["bid_n25"] - books["ask_n25"]) / np.maximum(books["bid_n25"] + books["ask_n25"], 1e-9)).to_numpy(float)
        spr = books["spread_bps"].to_numpy(float)
        mdev = books["microprice_dev_bps"].to_numpy(float)
        hi_b = np.searchsorted(b_ms, ts_c_arr, side="right")
        lo_b = np.searchsorted(b_ms, ts_c_arr - 150_000, side="left")
        ci10 = np.concatenate([[0.0], np.cumsum(np.nan_to_num(imb10, nan=0.0))])
        ci25 = np.concatenate([[0.0], np.cumsum(np.nan_to_num(imb25, nan=0.0))])
        cspr = np.concatenate([[0.0], np.cumsum(np.nan_to_num(spr, nan=0.0))])
        cmdev = np.concatenate([[0.0], np.cumsum(np.nan_to_num(mdev, nan=0.0))])
        cnt_b = hi_b - lo_b
        ok_b = cnt_b >= 4
        df_w["l2_book_imb_10"] = np.where(ok_b, (ci10[hi_b] - ci10[lo_b]) / np.maximum(cnt_b, 1), np.nan)
        df_w["l2_book_imb_25"] = np.where(ok_b, (ci25[hi_b] - ci25[lo_b]) / np.maximum(cnt_b, 1), np.nan)
        df_w["l2_spread_bps"] = np.where(ok_b, (cspr[hi_b] - cspr[lo_b]) / np.maximum(cnt_b, 1), np.nan)
        df_w["l2_mdev_bps"] = np.where(ok_b, (cmdev[hi_b] - cmdev[lo_b]) / np.maximum(cnt_b, 1), np.nan)

    if not derivs.empty:
        derivs["basis_bps"] = (derivs["mark_price"] - derivs["index_price"]) / np.maximum(derivs["index_price"], 1e-9) * 1e4
        m_drv = pd.merge_asof(
            df_w[["ts_close"]].sort_values("ts_close"),
            derivs[["ts_ms", "funding_rate", "basis_bps"]].rename(columns={"ts_ms": "ts_d"}),
            left_on="ts_close",
            right_on="ts_d",
            direction="backward",
            tolerance=300_000,
        )
        df_w["deriv_funding"] = m_drv["funding_rate"].to_numpy(float)
        df_w["deriv_basis_bps"] = m_drv["basis_bps"].to_numpy(float)

    return df_w


def _pack_masks(buy_mask: np.ndarray, sell_mask: np.ndarray, warmup: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    n = len(buy_mask)
    sig = np.zeros(n, dtype=np.int32)
    bm = np.asarray(buy_mask, dtype=bool)
    sm = np.asarray(sell_mask, dtype=bool)
    sig[bm & (~sm)] = 1
    sig[sm & (~bm)] = 2
    sig[:warmup] = 0
    conf = np.where(sig != 0, 0.80, 0.0)
    return sig, conf


def _evaluate_horizon(
    horizon_tag: str,
    specs_and_signals: List[Tuple[ScalpSpec, Dict[str, np.ndarray], Dict[str, np.ndarray]]],
    data_map: Dict[str, pd.DataFrame],
    funding: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_prior_trials: int,
) -> Dict[str, Any]:
    start_ms = min(int(df["timestamp"].iloc[0]) for df in data_map.values())
    end_ms = max(int(df["timestamp"].iloc[-1]) for df in data_map.values())
    mid_ms = start_ms + (end_ms - start_ms) // 2

    raw_cache: Dict[str, Tuple[Dict[str, Any], BacktestResult, Dict[str, np.ndarray], Dict[str, np.ndarray]]] = {}
    for spec, sig_m, conf_m in specs_and_signals:
        res, st_ex = run_spec(spec, data_map, sig_m, conf_m, funding)
        m = compute_scalp_metrics(res, data_map, spec.sl_atr, start_ms, end_ms, exec_stats=st_ex)
        raw_cache[spec.name] = (m, res, sig_m, conf_m)

    srs = [v[0]["_sr_daily_ann"] for v in raw_cache.values()]
    var_sh = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    n_tot = n_prior_trials + len(specs_and_signals)

    results_out: Dict[str, Any] = {}
    for spec, sig_m, conf_m in specs_and_signals:
        m, res, _, _ = raw_cache[spec.name]
        dsr_info = deflated_sharpe_ratio(
            observed_sr_ann=m["_sr_daily_ann"],
            returns=m["_daily_rets"],
            n_trials=max(2, n_tot),
            variance_of_trials=var_sh,
            periods_per_year=365.0,
        )
        m_clean = _strip_internal(m)
        m_clean["dsr"] = round(float(dsr_info["dsr"]), 4)
        m_clean["spec"] = asdict(spec)

        # Breakdown por ativo + estabilidade temporal H1/H2 para todo candidato positivo ou referência principal
        if m_clean["expectancy_r"] > 0 or m_clean["profit_factor"] >= 0.95 or "BASE" in spec.name or "LIVE" in spec.name:
            per_asset = {}
            for sym in MAJORS:
                f_sym = {sym: funding[sym]} if sym in funding else {}
                r_s, ex_s = run_spec(spec, {sym: data_map[sym]}, {sym: sig_m[sym]}, {sym: conf_m[sym]}, f_sym)
                m_s = compute_scalp_metrics(
                    r_s, {sym: data_map[sym]}, spec.sl_atr,
                    int(data_map[sym]["timestamp"].iloc[0]), int(data_map[sym]["timestamp"].iloc[-1]),
                    exec_stats=ex_s,
                )
                per_asset[sym] = _strip_internal(m_s)
            m_clean["per_asset"] = per_asset

            # H1 vs H2 temporal split
            h1_h2 = {}
            for tag_half, cond in [("H1", lambda ts: ts < mid_ms), ("H2", lambda ts: ts >= mid_ms)]:
                d_sub, s_sub, c_sub = {}, {}, {}
                for sym, df in data_map.items():
                    mask = cond(df["timestamp"].to_numpy(np.int64))
                    if mask.sum() >= 50:
                        d_sub[sym] = df[mask].copy().reset_index(drop=True)
                        s_sub[sym] = sig_m[sym][mask]
                        c_sub[sym] = conf_m[sym][mask]
                if d_sub:
                    r_h, ex_h = run_spec(spec, d_sub, s_sub, c_sub, funding)
                    h1_s = min(int(x["timestamp"].iloc[0]) for x in d_sub.values())
                    h1_e = max(int(x["timestamp"].iloc[-1]) for x in d_sub.values())
                    m_h = compute_scalp_metrics(r_h, d_sub, spec.sl_atr, h1_s, h1_e, exec_stats=ex_h)
                    h1_h2[tag_half] = _strip_internal(m_h)
            m_clean["temporal_h1_h2"] = h1_h2

        results_out[spec.name] = m_clean
        ex_r = m_clean["exit_reasons"]
        print(
            f"  [{horizon_tag}] {spec.name:<34} | Tr={m_clean['trades']:>4} (Fill {m_clean['fill_rate_pct']:>5.1f}%) "
            f"| WR={m_clean['win_rate_pct']:>5.1f}% | PF={m_clean['profit_factor']:>5.3f} "
            f"| E[R]={m_clean['expectancy_r']:>+6.3f} | PnL=${m_clean['net_pnl_usd']:>+7.2f} "
            f"| DD={m_clean['max_dd_pct']:>5.2f}% | SL/BE/TP3={ex_r.get('STOP_LOSS',0)}/{ex_r.get('BREAKEVEN_STOP',0)}/{ex_r.get('TAKE_PROFIT_3',0)}",
            flush=True,
        )

    return {
        "window_start": _ts_iso(start_ms),
        "window_end": _ts_iso(end_ms),
        "mid_split": _ts_iso(mid_ms),
        "results": results_out,
    }


def _build_signals_horizon_a(
    data_a: Dict[str, pd.DataFrame],
    probs_a: Dict[str, np.ndarray],
) -> List[Tuple[ScalpSpec, Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
    base_sig, base_conf = {}, {}
    for sym, df in data_a.items():
        base_sig[sym], base_conf[sym], _ = _generate_scalp_live_signals(df, sym, probs_a[sym], 0.8, 30.0, 70.0, -0.5, None)

    def _make_gen(fn: Callable[[pd.DataFrame], Tuple[np.ndarray, np.ndarray]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        sm, cm = {}, {}
        for sym, df in data_a.items():
            sm[sym], cm[sym] = fn(df)
        return sm, cm

    # A2/A3/A4: Breakout 20 alinhado a SMA200 1h + filtro volatilidade NATR >= 40bps
    def _bo20_aligned(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        c = df["close"].to_numpy(float)
        o = df["open"].to_numpy(float)
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        atr = df["atr_14"].to_numpy(float)
        rng_atr = (h - l) / atr
        vz = df["volume_zscore_20"].to_numpy(float)
        cpos = df["cpos"].to_numpy(float)
        natr = df["natr_bps"].to_numpy(float)
        h1d = df["h1_dist_sma200"].to_numpy(float)
        ph20 = df["prior_high20"].to_numpy(float)
        pl20 = df["prior_low20"].to_numpy(float)
        buy = (vz >= 0.8) & (rng_atr >= 1.0) & (natr >= 40.0) & (c > o) & (cpos >= 0.70) & (c > ph20) & (h1d >= 0.003)
        sell = (vz >= 0.8) & (rng_atr >= 1.0) & (natr >= 40.0) & (c < o) & (cpos <= 0.30) & (c < pl20) & (h1d <= -0.003)
        return _pack_masks(buy, sell)

    # A5/A6/A7: Volatility Squeeze (BB dentro de Keltner) + rompimento alinhado H1 + NATR >= 35bps
    def _squeeze_sig(session_only: bool) -> Callable[[pd.DataFrame], Tuple[np.ndarray, np.ndarray]]:
        def _fn(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            c = df["close"].to_numpy(float)
            o = df["open"].to_numpy(float)
            vz = df["volume_zscore_20"].to_numpy(float)
            natr = df["natr_bps"].to_numpy(float)
            h1d = df["h1_dist_sma200"].to_numpy(float)
            sqz = df["prior_squeeze_2b"].to_numpy(bool)
            kch = df["kc_hi"].to_numpy(float)
            kcl = df["kc_lo"].to_numpy(float)
            sess_ok = np.ones(len(df), dtype=bool)
            if session_only:
                hr = df["utc_hour"].to_numpy(int)
                dow = df["utc_dow"].to_numpy(int)
                sess_ok = (hr >= 12) & (hr < 21) & (dow < 5)
            buy = sqz & sess_ok & (vz >= 1.0) & (natr >= 35.0) & (c > o) & (c > kch) & (h1d >= 0.0)
            sell = sqz & sess_ok & (vz >= 1.0) & (natr >= 35.0) & (c < o) & (c < kcl) & (h1d <= 0.0)
            return _pack_masks(buy, sell)
        return _fn

    # A8: Reversão VWAP pura em regime lateral (sem Breakout20)
    def _range_vwap_rev(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        vwap = df["dist_vwap_pct"].to_numpy(float)
        rsi = df["rsi_14"].to_numpy(float)
        h1d = df["h1_dist_sma200"].to_numpy(float)
        cpos = df["cpos"].to_numpy(float)
        vz = df["volume_zscore_20"].to_numpy(float)
        natr = df["natr_bps"].to_numpy(float)
        lat = np.abs(h1d) <= 0.02
        buy = lat & (vwap <= -1.2) & (rsi <= 28.0) & (vz >= 0.2) & (cpos >= 0.50) & (natr >= 30.0)
        sell = lat & (vwap >= 1.2) & (rsi >= 72.0) & (vz >= 0.2) & (cpos <= 0.50) & (natr >= 30.0)
        return _pack_masks(buy, sell)

    # A9: Stop-Hunt / Liquidity Sweep Fakeout de 20 barras
    def _sweep_fakeout(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        c = df["close"].to_numpy(float)
        h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float)
        ph20 = df["prior_high20"].to_numpy(float)
        pl20 = df["prior_low20"].to_numpy(float)
        lwf = df["lower_wick_frac"].to_numpy(float)
        uwf = df["upper_wick_frac"].to_numpy(float)
        vz = df["volume_zscore_20"].to_numpy(float)
        natr = df["natr_bps"].to_numpy(float)
        buy = (l < pl20) & (c > pl20) & (lwf >= 0.35) & (vz >= 0.8) & (natr >= 30.0)
        sell = (h > ph20) & (c < ph20) & (uwf >= 0.35) & (vz >= 0.8) & (natr >= 30.0)
        return _pack_masks(buy, sell)

    bo_s, bo_c = _make_gen(_bo20_aligned)
    sq_s, sq_c = _make_gen(_squeeze_sig(False))
    sqs_s, sqs_c = _make_gen(_squeeze_sig(True))
    rv_s, rv_c = _make_gen(_range_vwap_rev)
    sw_s, sw_c = _make_gen(_sweep_fakeout)

    return [
        (ScalpSpec("A0_LIVE_SIM_PARITY", "A", "sim_parity", 0.9, 1.2, 2.0, 3.6, 0.8, 0.40, 0.30, "tp1_legacy", 0.0004, min_conf=0.70, desc="Baseline S4 atual (simulação maker_entry close±1bp)"), base_sig, base_conf),
        (ScalpSpec("A0_LIVE_REAL_TAKER", "A", "real_taker", 0.9, 1.2, 2.0, 3.6, 0.8, 0.40, 0.30, "tp1_legacy", 0.0004, min_conf=0.70, desc="Baseline S4 penalizado com entrada Taker real (8bps)"), base_sig, base_conf),
        (ScalpSpec("A1_V105_BE_GUARD_TAKER", "A", "real_taker", 0.9, 1.2, 2.0, 3.6, 0.8, 0.40, 0.30, "tp1_v105", 0.0004, min_conf=0.70, desc="A0 com guarda v1.0.5 (closed_safe) + BE +4bps"), base_sig, base_conf),
        (ScalpSpec("A2_BO20_SINGLE_TP_TAKER", "A", "real_taker", 1.4, 2.2, 2.2, 2.2, 0.0, 0.0, 0.0, "none", 0.0, desc="BO20 alinhado H1 + NATR>=40bps + Alvo Único Maker 2.2 ATR (Taker in)"), bo_s, bo_c),
        (ScalpSpec("A3_BO20_SINGLE_TP_PULLBACK", "A", "limit_pullback", 1.4, 2.2, 2.2, 2.2, 0.0, 0.0, 0.0, "none", 0.0, pullback_atr=0.25, limit_ttl_bars=2, desc="A2 com Limit Pullback 0.25 ATR (TTL=2, Maker 2bps)"), bo_s, bo_c),
        (ScalpSpec("A4_BO20_DELAYED_BE_PULLBACK", "A", "limit_pullback", 1.4, 1.8, 3.0, 5.0, 1.0, 0.30, 0.30, "tp2_delayed", 0.0012, max_hold_bars=48, pullback_atr=0.25, limit_ttl_bars=2, desc="BO20 alinhado + Limit Pullback + BE só no TP2 (+12bps) + runner 5.0 ATR"), bo_s, bo_c),
        (ScalpSpec("A5_SQUEEZE_KELTNER_TAKER", "A", "real_taker", 1.4, 2.2, 2.2, 2.2, 0.0, 0.0, 0.0, "none", 0.0, desc="Squeeze BB/KC + rompimento H1 + NATR>=35bps + Alvo Único Maker (Taker)"), sq_s, sq_c),
        (ScalpSpec("A6_SQUEEZE_KELTNER_PULLBACK", "A", "limit_pullback", 1.4, 2.2, 2.2, 2.2, 0.0, 0.0, 0.0, "none", 0.0, pullback_atr=0.20, limit_ttl_bars=2, desc="A5 com Limit Pullback 0.20 ATR (Maker 2bps)"), sq_s, sq_c),
        (ScalpSpec("A7_SQUEEZE_US_EU_PULLBACK", "A", "limit_pullback", 1.4, 2.2, 2.2, 2.2, 0.0, 0.0, 0.0, "none", 0.0, pullback_atr=0.20, limit_ttl_bars=2, desc="Squeeze Limit Pullback restrito à sessão Londres/NY (12-21h UTC Seg-Sex)"), sqs_s, sqs_c),
        (ScalpSpec("A8_RANGE_VWAP_REV_PULLBACK", "A", "limit_pullback", 1.4, 1.6, 1.6, 1.6, 0.0, 0.0, 0.0, "none", 0.0, pullback_atr=0.15, limit_ttl_bars=2, desc="Reversão VWAP>=1.2% em lateral H1 + Limit Pullback 0.15 ATR + Alvo 1.6 ATR"), rv_s, rv_c),
        (ScalpSpec("A9_SWEEP_FAKEOUT_TAKER", "A", "real_taker", 1.3, 1.95, 1.95, 1.95, 0.0, 0.0, 0.0, "none", 0.0, desc="Liquidity Sweep Fakeout 20 barras + pavio>=35% + Alvo Único Maker 1.95 ATR (Taker)"), sw_s, sw_c),
        (ScalpSpec("A9b_SWEEP_FAKEOUT_PULLBACK", "A", "limit_pullback", 1.3, 1.95, 1.95, 1.95, 0.0, 0.0, 0.0, "none", 0.0, pullback_atr=0.15, limit_ttl_bars=2, desc="A9 com Limit Pullback 0.15 ATR (TTL=2, Maker 2bps)"), sw_s, sw_c),
    ]


def _build_signals_horizon_b(
    data_b: Dict[str, pd.DataFrame],
    probs_b: Dict[str, np.ndarray],
) -> Tuple[List[Tuple[ScalpSpec, Dict[str, np.ndarray], Dict[str, np.ndarray]]], Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]]:
    base_sig, base_conf = {}, {}
    for sym, df in data_b.items():
        base_sig[sym], base_conf[sym], _ = _generate_scalp_live_signals(df, sym, probs_b[sym], 0.8, 30.0, 70.0, -0.5, None)

    def _make_gen(fn: Callable[[pd.DataFrame], Tuple[np.ndarray, np.ndarray]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        sm, cm = {}, {}
        for sym, df in data_b.items():
            sm[sym], cm[sym] = fn(df)
        return sm, cm

    # B1: OI Buildup + Taker Aggression Momentum
    def _oi_buildup(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        c = df["close"].to_numpy(float)
        o = df["open"].to_numpy(float)
        vz = df["volume_zscore_20"].to_numpy(float)
        natr = df["natr_bps"].to_numpy(float)
        h1d = df["h1_dist_sma200"].to_numpy(float)
        ph20 = df["prior_high20"].to_numpy(float)
        pl20 = df["prior_low20"].to_numpy(float)
        oi1h = df["oi_chg_1h_pct"].to_numpy(float)
        tkr = df["taker_ratio_15m"].to_numpy(float)
        buy = (c > ph20) & (c > o) & (oi1h >= 0.30) & (tkr >= 1.12) & (vz >= 0.6) & (natr >= 30.0) & (h1d >= 0.0)
        sell = (c < pl20) & (c < o) & (oi1h >= 0.30) & (tkr <= 0.89) & (vz >= 0.6) & (natr >= 30.0) & (h1d <= 0.0)
        return _pack_masks(buy, sell)

    # B2: OI Flush Cascade Fade (exaustão pós-liquidação forçada)
    def _oi_flush_fade(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        ret3 = df["ret_3b_pct"].to_numpy(float)
        vwap = df["dist_vwap_pct"].to_numpy(float)
        oi15 = df["oi_chg_15m_pct"].to_numpy(float)
        oi30 = df["oi_chg_30m_pct"].to_numpy(float)
        lwf = df["lower_wick_frac"].to_numpy(float)
        uwf = df["upper_wick_frac"].to_numpy(float)
        natr = df["natr_bps"].to_numpy(float)
        flush = (oi15 <= -0.25) | (oi30 <= -0.40)
        buy = flush & ((ret3 <= -0.70) | (vwap <= -0.90)) & (lwf >= 0.25) & (natr >= 28.0)
        sell = flush & ((ret3 >= 0.70) | (vwap >= 0.90)) & (uwf >= 0.25) & (natr >= 28.0)
        return _pack_masks(buy, sell)

    # B3: Divergência Smart Money vs Varejo
    def _smart_retail_div(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        sdiv = df["smart_div"].to_numpy(float)
        glz = df["global_long_z"].to_numpy(float)
        vwap = df["dist_vwap_pct"].to_numpy(float)
        rsi = df["rsi_14"].to_numpy(float)
        natr = df["natr_bps"].to_numpy(float)
        buy = (sdiv >= 0.03) & (glz <= -1.2) & (vwap <= -0.7) & (rsi <= 38.0) & (natr >= 25.0)
        sell = (sdiv <= -0.03) & (glz >= 1.2) & (vwap >= 0.7) & (rsi >= 62.0) & (natr >= 25.0)
        return _pack_masks(buy, sell)

    # B4: Funding Window (±30m de 00/08/16 UTC) Fade contra crowding
    def _funding_fade(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        mtf = df["mins_to_funding"].to_numpy(int)
        glz = df["global_long_z"].to_numpy(float)
        vwap = df["dist_vwap_pct"].to_numpy(float)
        natr = df["natr_bps"].to_numpy(float)
        win = mtf <= 30
        buy = win & (vwap <= -0.7) & (glz <= -0.8) & (natr >= 25.0)
        sell = win & (vwap >= 0.7) & (glz >= 0.8) & (natr >= 25.0)
        return _pack_masks(buy, sell)

    b1_s, b1_c = _make_gen(_oi_buildup)
    b2_s, b2_c = _make_gen(_oi_flush_fade)
    b3_s, b3_c = _make_gen(_smart_retail_div)
    b4_s, b4_c = _make_gen(_funding_fade)

    specs = [
        (ScalpSpec("B0_BASE_60D_TAKER", "B", "real_taker", 0.9, 1.2, 2.0, 3.6, 0.8, 0.40, 0.30, "tp1_legacy", 0.0004, min_conf=0.70, desc="Baseline S4 nos últimos 60 dias (real_taker)"), base_sig, base_conf),
        (ScalpSpec("B1_OI_BUILDUP_TAKER", "B", "real_taker", 1.4, 2.2, 2.2, 2.2, 0.0, 0.0, 0.0, "none", 0.0, desc="Breakout + OI 1h>=+0.30% + Taker>=1.12 H1-alinhado (Taker in)"), b1_s, b1_c),
        (ScalpSpec("B1b_OI_BUILDUP_PULLBACK", "B", "limit_pullback", 1.4, 2.2, 2.2, 2.2, 0.0, 0.0, 0.0, "none", 0.0, pullback_atr=0.20, limit_ttl_bars=2, desc="B1 com Limit Pullback 0.20 ATR (TTL=2, Maker 2bps)"), b1_s, b1_c),
        (ScalpSpec("B2_OI_FLUSH_FADE_TAKER", "B", "real_taker", 1.4, 1.8, 1.8, 1.8, 0.0, 0.0, 0.0, "none", 0.0, desc="Fade pós-cascata OI (oi15<=-0.25% ou oi30<=-0.40%) + pavio (Taker in)"), b2_s, b2_c),
        (ScalpSpec("B2b_OI_FLUSH_FADE_PULLBACK", "B", "limit_pullback", 1.4, 1.8, 1.8, 1.8, 0.0, 0.0, 0.0, "none", 0.0, pullback_atr=0.10, limit_ttl_bars=2, desc="B2 com Limit Pullback 0.10 ATR (TTL=2, Maker 2bps)"), b2_s, b2_c),
        (ScalpSpec("B3_SMART_RETAIL_DIV_PULLBACK", "B", "limit_pullback", 1.4, 1.8, 1.8, 1.8, 0.0, 0.0, 0.0, "none", 0.0, pullback_atr=0.10, limit_ttl_bars=2, desc="Divergência Top Traders vs Varejo + VWAP>=0.7% (Limit Pullback)"), b3_s, b3_c),
        (ScalpSpec("B4_FUNDING_WINDOW_FADE", "B", "limit_pullback", 1.3, 1.6, 1.6, 1.6, 0.0, 0.0, 0.0, "none", 0.0, pullback_atr=0.10, limit_ttl_bars=2, desc="Janela ±30m Funding (00/08/16 UTC) contra crowding (Limit Pullback)"), b4_s, b4_c),
    ]
    signal_cache = {"OI_BUILDUP": (b1_s, b1_c), "OI_FLUSH": (b2_s, b2_c)}
    return specs, signal_cache


def _build_signals_horizon_c(
    data_c: Dict[str, pd.DataFrame],
    probs_c: Dict[str, np.ndarray],
    best_ab_name: str,
    best_ab_spec: Optional[ScalpSpec],
    best_ab_sig_fn: Callable[[pd.DataFrame], Tuple[np.ndarray, np.ndarray]],
) -> List[Tuple[ScalpSpec, Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
    tape_agent = TapeReadingAgent(db_path=MICRO_DB_PATH, covered_symbols=set(MAJORS))

    # C0: Baseline atual + TapeReadingAgent (como em Block 4 anterior, porém com real_taker)
    c0_sig, c0_conf = {}, {}
    for sym, df in data_c.items():
        sg, cf, _ = _generate_scalp_live_signals(df, sym, probs_c[sym], 0.8, 30.0, 70.0, -0.5, None)
        ts_arr = df["timestamp"].to_numpy(np.int64)
        for idx in np.where(sg != 0)[0]:
            direction = "BUY" if sg[idx] == 1 else "SELL"
            res_t = tape_agent.evaluate(sym, direction, now_ms=int(ts_arr[idx]) + BAR_15M_MS, allow_live_fallback=False)
            passed, _, _ = _tape_decision(res_t, direction)
            if not passed:
                sg[idx] = 0
                cf[idx] = 0.0
        c0_sig[sym], c0_conf[sym] = sg, cf

    def _make_gen(fn: Callable[[pd.DataFrame], Tuple[np.ndarray, np.ndarray]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        sm, cm = {}, {}
        for sym, df in data_c.items():
            sm[sym], cm[sym] = fn(df)
        return sm, cm

    # C1: Absorção real no L2 (agressão contra book profundo) + VWAP estendido
    def _l2_absorption(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        vwap = df["dist_vwap_pct"].to_numpy(float)
        tkr5 = df["l2_taker_imb_5m"].to_numpy(float)
        bi10 = df["l2_book_imb_10"].to_numpy(float)
        spr = df["l2_spread_bps"].to_numpy(float)
        natr = df["natr_bps"].to_numpy(float)
        ok_spr = spr <= 4.5
        buy = ok_spr & (vwap <= -0.6) & (tkr5 <= -0.20) & (bi10 >= 0.12) & (natr >= 25.0)
        sell = ok_spr & (vwap >= 0.6) & (tkr5 >= 0.20) & (bi10 <= -0.12) & (natr >= 25.0)
        return _pack_masks(buy, sell)

    # C2: Ignição de OBI + Agressão Taker L2 alinhada H1
    def _l2_toxic_obi(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        tkr5 = df["l2_taker_imb_5m"].to_numpy(float)
        bi10 = df["l2_book_imb_10"].to_numpy(float)
        bi25 = df["l2_book_imb_25"].to_numpy(float)
        spr = df["l2_spread_bps"].to_numpy(float)
        h1d = df["h1_dist_sma200"].to_numpy(float)
        natr = df["natr_bps"].to_numpy(float)
        ok_spr = spr <= 4.5
        buy = ok_spr & (tkr5 >= 0.22) & (bi10 >= 0.18) & (bi25 >= 0.10) & (natr >= 35.0) & (h1d >= 0.0)
        sell = ok_spr & (tkr5 <= -0.22) & (bi10 <= -0.18) & (bi25 <= -0.10) & (natr >= 35.0) & (h1d <= 0.0)
        return _pack_masks(buy, sell)

    c1_s, c1_c = _make_gen(_l2_absorption)
    c2_s, c2_c = _make_gen(_l2_toxic_obi)

    # C3: Melhor candidato estrutural de A/B sem vs com filtro L2 anti-fluxo tóxico contrário
    c3_raw_s, c3_raw_c = _make_gen(best_ab_sig_fn)
    c3_filt_s, c3_filt_c = {}, {}
    for sym, df in data_c.items():
        s_out = c3_raw_s[sym].copy()
        bi10 = df["l2_book_imb_10"].to_numpy(float)
        spr = df["l2_spread_bps"].to_numpy(float)
        # Bloqueia somente se houver dado L2 válido mostrando book fortemente contrário (>0.20) ou spread > 5bps
        bad_buy = (s_out == 1) & ((np.isfinite(spr) & (spr > 5.0)) | (np.isfinite(bi10) & (bi10 < -0.20)))
        bad_sell = (s_out == 2) & ((np.isfinite(spr) & (spr > 5.0)) | (np.isfinite(bi10) & (bi10 > 0.20)))
        s_out[bad_buy | bad_sell] = 0
        c3_filt_s[sym] = s_out
        c3_filt_c[sym] = np.where(s_out != 0, 0.80, 0.0)

    ref_sp = best_ab_spec or ScalpSpec("REF", "C", "limit_pullback", 1.3, 1.95, 1.95, 1.95, 0.0, 0.0, 0.0, "none", 0.0, pullback_atr=0.15, limit_ttl_bars=2)
    specs = [
        (ScalpSpec("C0_BASE_29D_WITH_TAPE_TAKER", "C", "real_taker", 0.9, 1.2, 2.0, 3.6, 0.8, 0.40, 0.30, "tp1_legacy", 0.0004, min_conf=0.70, desc="Baseline S4 + TapeReadingAgent em 29d (real_taker)"), c0_sig, c0_conf),
        (ScalpSpec("C1_L2_ABSORPTION_PULLBACK", "C", "limit_pullback", 1.3, 1.7, 1.7, 1.7, 0.0, 0.0, 0.0, "none", 0.0, pullback_atr=0.08, limit_ttl_bars=2, desc="Absorção real L2 (taker<= -0.20 & book10>= +0.12) + VWAP>=0.6% (Limit)"), c1_s, c1_c),
        (ScalpSpec("C2_L2_TOXIC_OBI_TAKER", "C", "real_taker", 1.4, 2.1, 2.1, 2.1, 0.0, 0.0, 0.0, "none", 0.0, desc="Ignição L2 (taker>=0.22 & book10>=0.18 & book25>=0.10) + H1 (Taker in)"), c2_s, c2_c),
        (ScalpSpec("C2b_L2_TOXIC_OBI_PULLBACK", "C", "limit_pullback", 1.4, 2.1, 2.1, 2.1, 0.0, 0.0, 0.0, "none", 0.0, pullback_atr=0.15, limit_ttl_bars=2, desc="C2 com Limit Pullback 0.15 ATR (TTL=2, Maker 2bps)"), c2_s, c2_c),
        (ScalpSpec(f"C3_RAW_{best_ab_name}", "C", ref_sp.entry_exec, ref_sp.sl_atr, ref_sp.tp1_atr, ref_sp.tp2_atr, ref_sp.tp3_atr, ref_sp.trail_atr, ref_sp.tp1_frac, ref_sp.tp2_frac, ref_sp.be_mode, ref_sp.be_buffer_pct, ref_sp.max_hold_bars, ref_sp.pullback_atr, ref_sp.limit_ttl_bars, desc=f"Melhor A/B ({best_ab_name}) nos 29d SEM filtro L2"), c3_raw_s, c3_raw_c),
        (ScalpSpec(f"C3b_L2FILTER_{best_ab_name}", "C", ref_sp.entry_exec, ref_sp.sl_atr, ref_sp.tp1_atr, ref_sp.tp2_atr, ref_sp.tp3_atr, ref_sp.trail_atr, ref_sp.tp1_frac, ref_sp.tp2_frac, ref_sp.be_mode, ref_sp.be_buffer_pct, ref_sp.max_hold_bars, ref_sp.pullback_atr, ref_sp.limit_ttl_bars, desc=f"Melhor A/B ({best_ab_name}) nos 29d COM trava anti-fluxo L2 contrário & spread<=5bps"), c3_filt_s, c3_filt_c),
    ]
    return specs


def _render_md(report: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Estudo Estrutural de Viabilidade do `SCALP` 15m — Pesquisa S4 (2026-11-13)",
        "",
        f"Gerado em `{report['generated_at_utc']}` em `BTCUSDT`, `ETHUSDT`, `SOLUSDT` usando `_ScalpResearchSimulator` (paridade Smart Trade + funding real 8h + simulação causal de Limit Pullback com verificação de Stop-Loss/seleção adversa na barra do preenchimento) e conexões `mode=ro`.",
        "",
        "## 1. Diagnóstico Matemático do Problema Atual (`SCALP` Produção)",
        "- **Atrito intradiário vs 1R:** No 15m (`1R = 0.9 * ATR_14`), a mediana de `1R` é `~29.5 bps` (`24.0 bps` em BTC). Um round-trip Taker real (`8 bps` in + `8 bps` out = `16 bps`) consome **`54%–67%` de 1R**.",
        "- **Inversão de Payoff na Escada `40/30/30` + `BE +4 bps`:** Ao atingir `TP1 (+1.2 ATR = +1.33R)`, `79.8%` das operações retornam antes do `TP2 (+2.0 ATR)` e são estopadas em `BREAKEVEN_STOP` pagando `8 bps` de saída taker sobre `60%` da mão -> ganho líquido médio **`+0.27R`** por acerto de TP1 contra **`-1.34R a -1.42R`** nos `STOP_LOSS`.",
        "",
    ]

    for tag, blk_key, title in [
        ("A", "horizon_a", "## 2. Horizonte A — Longo Prazo 15m (`9-13 meses`: Execução, Saídas & Estrutura)"),
        ("B", "horizon_b", "## 3. Horizonte B — 60 Dias Derivativos Binance 5m (`oi`, `taker`, `ls_global`, `ls_top_pos`)"),
        ("C", "horizon_c", "## 4. Horizonte C — 29 Dias Microestrutura Real (`book_features` + `tick_agg` L2)"),
    ]:
        blk = report[blk_key]
        lines += [
            f"{title}",
            f"Janela: `{blk['window_start']}` .. `{blk['window_end']}` (Split H1/H2: `{blk['mid_split']}`)",
            "",
            "| Estratégia | Execução | Trades (Fill%) | WR% | PF | E[R] (IC 95%) | PnL (`$10k`) | MaxDD | Sharpe Diário | DSR | Taxas/Fund |",
            "| :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
        positives = []
        for name, m in blk["results"].items():
            ci = m["expectancy_r_ci95"]
            ex_mode = m["spec"]["entry_exec"]
            lines.append(
                f"| `{name}` | `{ex_mode}` | {m['trades']} ({m['fill_rate_pct']:.1f}%) | {m['win_rate_pct']:.1f}% | "
                f"{m['profit_factor']:.3f} | {m['expectancy_r']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                f"${m['net_pnl_usd']:+.2f} | {m['max_dd_pct']:.2f}% | {m['sharpe_daily_ann']:+.3f} | {m['dsr']:.4f} | ${m['total_fees_funding']:.2f} |"
            )
            if m["expectancy_r"] > 0 and m["profit_factor"] > 1.0:
                positives.append((name, m))

        if positives:
            lines += [
                "",
                f"### Candidatos Positivos no Horizonte {tag} (`E[R] > 0` e `PF > 1.0`) — Por Ativo & H1/H2",
                "| Estratégia | BTC (Tr / WR / PF / E[R]) | ETH (Tr / WR / PF / E[R]) | SOL (Tr / WR / PF / E[R]) | H1 (Tr / PF / E[R] / PnL) | H2 (Tr / PF / E[R] / PnL) |",
                "| :--- | :--- | :--- | :--- | :--- | :--- |",
            ]
            for name, m in positives:
                pa = m.get("per_asset", {})
                hh = m.get("temporal_h1_h2", {})
                def _fa(s):
                    d = pa.get(s, {})
                    return f"{d.get('trades',0)} / {d.get('win_rate_pct',0):.1f}% / {d.get('profit_factor',0):.2f} / {d.get('expectancy_r',0):+.3f}"
                def _fh(k):
                    d = hh.get(k, {})
                    return f"{d.get('trades',0)} / {d.get('profit_factor',0):.2f} / {d.get('expectancy_r',0):+.3f} / ${d.get('net_pnl_usd',0):+.2f}"
                lines.append(f"| `{name}` | {_fa('BTCUSDT')} | {_fa('ETHUSDT')} | {_fa('SOLUSDT')} | {_fh('H1')} | {_fh('H2')} |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    t0 = time.time()
    n_prior = get_prior_trials_count()
    funding = load_funding_map(MAJORS)

    print("=== SCALP RESEARCH S4: Carregando e alinhando bases (mode=ro) ===", flush=True)
    bundle_45 = joblib.load(os.path.join(ROOT, "models", "lgbm_model_v1.joblib"))
    feature_names_45 = list(bundle_45["feature_names"])

    # Ensemble ML treinado estritamente ANTES do primeiro dado 15m (2025-09-01 UTC) -> ZERO vazamento no baseline
    pre_15m_cutoff = int(pd.Timestamp("2025-09-01 00:00:00", tz="UTC").timestamp() * 1000)
    btc_1h = FeatureEngineer.compute_all_features(load_candles_clean("BTCUSDT", "1h", limit=None), drop_na=False)
    btc_1h = btc_1h[btc_1h["sma_200"].notna()].reset_index(drop=True)
    lgbm_pre15, rf_pre15, n_tr_ml = _train_pre_split_ensemble(btc_1h, pre_15m_cutoff, feature_names_45)
    print(f"  Ensemble ML 1h treinado pre-2025-09-01 com {n_tr_ml} barras BTCUSDT.", flush=True)

    data_a: Dict[str, pd.DataFrame] = {}
    probs_a: Dict[str, np.ndarray] = {}
    for sym in MAJORS:
        df1h = btc_1h if sym == "BTCUSDT" else FeatureEngineer.compute_all_features(load_candles_clean(sym, "1h", limit=None), drop_na=False)
        df1h = df1h[df1h["sma_200"].notna()].reset_index(drop=True)
        df15_raw = load_candles_clean(sym, "15m", limit=None)
        df15_feat = FeatureEngineer.compute_all_features(df15_raw, drop_na=False)
        df15_feat = df15_feat[df15_feat["sma_200"].notna()].reset_index(drop=True)
        df_aligned = _attach_causal_features(df15_feat, df1h, sym)
        data_a[sym] = df_aligned
        probs_a[sym] = _predict_45_probas(df_aligned, lgbm_pre15, rf_pre15, feature_names_45)
        print(f"  {sym}: {len(df_aligned)} velas 15m alinhadas ({_ts_iso(int(df_aligned['timestamp'].iloc[0]))} .. {_ts_iso(int(df_aligned['timestamp'].iloc[-1]))})", flush=True)

    # Horizonte A
    print("\n=== HORIZONTE A: Longo Prazo 15m (~9-13 meses) ===", flush=True)
    specs_a = _build_signals_horizon_a(data_a, probs_a)
    res_a = _evaluate_horizon("H-A", specs_a, data_a, funding, n_prior_trials=n_prior)

    # Horizonte B (60 dias: 2026-09-15 12:00 UTC ..)
    ts_b0 = int(pd.Timestamp("2026-09-15 12:00:00", tz="UTC").timestamp() * 1000)
    data_b, probs_b = {}, {}
    for sym in MAJORS:
        m = data_a[sym]["timestamp"].to_numpy(np.int64) >= ts_b0
        data_b[sym] = data_a[sym][m].copy().reset_index(drop=True)
        probs_b[sym] = probs_a[sym][m]
    print("\n=== HORIZONTE B: 60 Dias Binance Derivativos 5m (oi, taker, ls_global, ls_top_pos) ===", flush=True)
    specs_b, sig_cache_b = _build_signals_horizon_b(data_b, probs_b)
    res_b = _evaluate_horizon("H-B", specs_b, data_b, funding, n_prior_trials=n_prior + len(specs_a))

    # Seleciona melhor candidato não-baseline dos Horizontes A/B para o filtro C3 em 29d
    non_base_a = [(k, v) for k, v in res_a["results"].items() if "A0_" not in k and "A1_" not in k and v["trades"] >= 20]
    best_a_name, best_a_m = max(non_base_a, key=lambda kv: kv[1]["expectancy_r"])
    non_base_b = [(k, v) for k, v in res_b["results"].items() if "B0_" not in k and v["trades"] >= 10]
    best_b_name, best_b_m = max(non_base_b, key=lambda kv: kv[1]["expectancy_r"])

    if best_b_m["expectancy_r"] > best_a_m["expectancy_r"] and "B2" in best_b_name:
        best_ab_name = best_b_name
        best_ab_spec = next(sp for sp, _, _ in specs_b if sp.name == best_b_name)
        def _best_fn(df):
            ret3 = df["ret_3b_pct"].to_numpy(float)
            vwap = df["dist_vwap_pct"].to_numpy(float)
            oi15 = df["oi_chg_15m_pct"].to_numpy(float)
            oi30 = df["oi_chg_30m_pct"].to_numpy(float)
            lwf = df["lower_wick_frac"].to_numpy(float)
            uwf = df["upper_wick_frac"].to_numpy(float)
            natr = df["natr_bps"].to_numpy(float)
            flush = (oi15 <= -0.25) | (oi30 <= -0.40)
            return _pack_masks(
                flush & ((ret3 <= -0.70) | (vwap <= -0.90)) & (lwf >= 0.25) & (natr >= 28.0),
                flush & ((ret3 >= 0.70) | (vwap >= 0.90)) & (uwf >= 0.25) & (natr >= 28.0),
            )
    else:
        best_ab_name = best_a_name
        best_ab_spec = next(sp for sp, _, _ in specs_a if sp.name == best_a_name)
        if "A9" in best_a_name:
            def _best_fn(df):
                return _pack_masks(
                    (df["low"].to_numpy(float) < df["prior_low20"].to_numpy(float)) & (df["close"].to_numpy(float) > df["prior_low20"].to_numpy(float)) & (df["lower_wick_frac"].to_numpy(float) >= 0.35) & (df["volume_zscore_20"].to_numpy(float) >= 0.8) & (df["natr_bps"].to_numpy(float) >= 30.0),
                    (df["high"].to_numpy(float) > df["prior_high20"].to_numpy(float)) & (df["close"].to_numpy(float) < df["prior_high20"].to_numpy(float)) & (df["upper_wick_frac"].to_numpy(float) >= 0.35) & (df["volume_zscore_20"].to_numpy(float) >= 0.8) & (df["natr_bps"].to_numpy(float) >= 30.0),
                )
        elif "A8" in best_a_name:
            def _best_fn(df):
                lat = np.abs(df["h1_dist_sma200"].to_numpy(float)) <= 0.02
                return _pack_masks(
                    lat & (df["dist_vwap_pct"].to_numpy(float) <= -1.2) & (df["rsi_14"].to_numpy(float) <= 28.0) & (df["volume_zscore_20"].to_numpy(float) >= 0.2) & (df["cpos"].to_numpy(float) >= 0.50) & (df["natr_bps"].to_numpy(float) >= 30.0),
                    lat & (df["dist_vwap_pct"].to_numpy(float) >= 1.2) & (df["rsi_14"].to_numpy(float) >= 72.0) & (df["volume_zscore_20"].to_numpy(float) >= 0.2) & (df["cpos"].to_numpy(float) <= 0.50) & (df["natr_bps"].to_numpy(float) >= 30.0),
                )
        elif "A6" in best_a_name or "A7" in best_a_name or "A5" in best_a_name:
            def _best_fn(df):
                return _pack_masks(
                    df["prior_squeeze_2b"].to_numpy(bool) & (df["volume_zscore_20"].to_numpy(float) >= 1.0) & (df["natr_bps"].to_numpy(float) >= 35.0) & (df["close"].to_numpy(float) > df["open"].to_numpy(float)) & (df["close"].to_numpy(float) > df["kc_hi"].to_numpy(float)) & (df["h1_dist_sma200"].to_numpy(float) >= 0.0),
                    df["prior_squeeze_2b"].to_numpy(bool) & (df["volume_zscore_20"].to_numpy(float) >= 1.0) & (df["natr_bps"].to_numpy(float) >= 35.0) & (df["close"].to_numpy(float) < df["open"].to_numpy(float)) & (df["close"].to_numpy(float) < df["kc_lo"].to_numpy(float)) & (df["h1_dist_sma200"].to_numpy(float) <= 0.0),
                )
        else:
            def _best_fn(df):
                return _pack_masks(
                    (df["volume_zscore_20"].to_numpy(float) >= 0.8) & ((df["high"].to_numpy(float) - df["low"].to_numpy(float)) / df["atr_14"].to_numpy(float) >= 1.0) & (df["natr_bps"].to_numpy(float) >= 40.0) & (df["close"].to_numpy(float) > df["open"].to_numpy(float)) & (df["cpos"].to_numpy(float) >= 0.70) & (df["close"].to_numpy(float) > df["prior_high20"].to_numpy(float)) & (df["h1_dist_sma200"].to_numpy(float) >= 0.003),
                    (df["volume_zscore_20"].to_numpy(float) >= 0.8) & ((df["high"].to_numpy(float) - df["low"].to_numpy(float)) / df["atr_14"].to_numpy(float) >= 1.0) & (df["natr_bps"].to_numpy(float) >= 40.0) & (df["close"].to_numpy(float) < df["open"].to_numpy(float)) & (df["cpos"].to_numpy(float) <= 0.30) & (df["close"].to_numpy(float) < df["prior_low20"].to_numpy(float)) & (df["h1_dist_sma200"].to_numpy(float) <= -0.003),
                )

    # Horizonte C (29 dias: 2026-10-15 12:30 UTC ..)
    ts_c0 = int(pd.Timestamp("2026-10-15 12:30:00", tz="UTC").timestamp() * 1000)
    data_c, probs_c = {}, {}
    for sym in MAJORS:
        m = data_a[sym]["timestamp"].to_numpy(np.int64) >= ts_c0
        data_c[sym] = data_a[sym][m].copy().reset_index(drop=True)
        probs_c[sym] = probs_a[sym][m]
    print(f"\n=== HORIZONTE C: 29 Dias Microestrutura Real L2/Ticks (best_ab={best_ab_name}) ===", flush=True)
    specs_c = _build_signals_horizon_c(data_c, probs_c, best_ab_name, best_ab_spec, _best_fn)
    res_c = _evaluate_horizon("H-C", specs_c, data_c, funding, n_prior_trials=n_prior + len(specs_a) + len(specs_b))

    report = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t0, 2),
        "best_ab_selected_for_c3": best_ab_name,
        "horizon_a": res_a,
        "horizon_b": res_b,
        "horizon_c": res_c,
    }

    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(_render_md(report))

    print(f"\n[OK] Pesquisa SCALP concluída em {time.time() - t0:.1f}s.", flush=True)
    print(f"  JSON: {REPORT_JSON}", flush=True)
    print(f"  MD  : {REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
