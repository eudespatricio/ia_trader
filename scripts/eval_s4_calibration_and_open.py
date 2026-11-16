#!/usr/bin/env python3
"""
scripts/eval_s4_calibration_and_open.py

Auditoria Empírica S4 Completa (Zero Lookahead, Zero Vazamento In-Sample, Pilot-Parity):
- Bloco 1: Impacto ML Walk-Forward Antes vs Depois da Deduplicação de Candles + Auditoria .joblib
- Bloco 2: Reavaliação da Abertura 21:00 BRT (00:00 UTC) e Reversão de Fluxo com BacktestEngine(pilot_parity=True) + ML estritamente OOS (1h e 15m)
- Bloco 3: A/B de Calibrações S4.2 para DIRECTIONAL (1h) e SCALP (15m) em Holdout OOS (split 60% + embargo 10d)
- Bloco 4: Replay 29 dias em microstructure.db real (2026-10-15..2026-11-13) + Contrafactual nos N=24 trades fechados S4 (2026-11-09..2026-11-13)

Regra de ouro: abertura estritamente read-only (`mode=ro`) nos SQLites de produção e backup; não altera produção.
"""

from __future__ import annotations

import copy
import datetime as dt
import json
import math
import os
import sqlite3
import sys
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.tape_reading_agent import TapeReadingAgent  # noqa: E402
from agents.technical_agent import TechnicalAgent  # noqa: E402
from backtesting import BacktestEngine, BacktestResult, _PilotSimulator  # noqa: E402
from core import TradingCore  # noqa: E402
from data.feature_engineering import FeatureEngineer  # noqa: E402
from model_trainer import ModelTrainer, PurgedTimeSeriesSplit  # noqa: E402
from smart_trade import STRATEGY_PROFILES  # noqa: E402
from statistical_rigor import (  # noqa: E402
    bootstrap_performance_metrics,
    calculate_annualized_sharpe,
    deflated_sharpe_ratio,
    get_prior_trials_count,
)

DB_PATH = os.path.join(ROOT, "datasets", "market_cache.db")
BACKUP_DB_PATH = os.path.join(ROOT, "datasets", "backups", "candles_overlap_backup_20261113.sqlite")
MICRO_DB_PATH = os.path.join(ROOT, "datasets", "microstructure.db")
REPORT_JSON = os.path.join(ROOT, "datasets", "reports", "s4_empirical_audit_20261113.json")
REPORT_MD = os.path.join(ROOT, "docs", "S4_AUDITORIA_EMPIRICA_E_CALIBRACAO.md")

MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
HOUR_MS = 3_600_000
BAR_15M_MS = 900_000
DAY_MS = 86_400_000
EMBARGO_DAYS = 10
TRAIN_FRAC = 0.60

NON_STATIONARY_COLS = frozenset({
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume",
    "sma_20", "sma_50", "sma_200", "ema_9", "ema_21", "bb_high", "bb_mid",
    "bb_low", "obv", "vol_sma_20", "atr_14", "nearest_bsl", "nearest_ssl",
})

MODEL_PATHS = [
    ("lgbm_archive_nov08", os.path.join(ROOT, "models", "archive", "lgbm_model_v1_20261111T231642Z.joblib")),
    ("lgbm_prod_nov11", os.path.join(ROOT, "models", "lgbm_model_v1.joblib")),
    ("rf_prod_nov08", os.path.join(ROOT, "models", "rf_model_v1.joblib")),
]


def _ro_conn(path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0)


def _ts_iso(ms: int) -> str:
    return dt.datetime.fromtimestamp(int(ms) / 1000.0, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def load_candles_clean(symbol: str, timeframe: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Carrega candles deduplicados (priorizando `binance`), descartando a barra potencialmente em formação."""
    with _ro_conn(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT exchange, timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp ASC, CASE WHEN exchange = 'binance' THEN 0 ELSE 1 END ASC
            """,
            conn,
            params=(symbol, timeframe),
        )
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp").reset_index(drop=True)
    # Descarta última vela se potencialmente em formação; em seguida aplica limite
    df = df.iloc[:-1].drop(columns=["exchange"]).reset_index(drop=True)
    if limit is not None and limit > 0 and len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)
    return df


def load_candles_contaminated_25k(symbol: str, timeframe: str = "1h", limit: int = 25000) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Reconstrói a fatia bruta contaminada (Binance + Bitget sobrepostas) anterior ao Step 2."""
    with _ro_conn(DB_PATH) as c1:
        df_clean = pd.read_sql_query(
            "SELECT exchange, timestamp, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe=?",
            c1,
            params=(symbol, timeframe),
        )
    with _ro_conn(BACKUP_DB_PATH) as c2:
        df_bk = pd.read_sql_query(
            "SELECT exchange, timestamp, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe=?",
            c2,
            params=(symbol, timeframe),
        )
    comb = pd.concat([df_clean, df_bk], ignore_index=True)
    comb["prio"] = np.where(comb["exchange"] == "binance", 0, 1)
    comb = comb.sort_values(["timestamp", "prio"], ascending=[False, True]).head(limit)
    comb = comb.sort_values(["timestamp", "prio"], ascending=[True, True]).reset_index(drop=True)
    dup_count = int(comb["timestamp"].duplicated().sum())
    uniq_count = int(comb["timestamp"].nunique())
    # Para reproduzir o cálculo pré-Step-2 sem que o dedup interno do FeatureEngineer remova as linhas,
    # desloca timestamps duplicados em +1ms antes do FeatureEngineer e restaura em seguida.
    cum_idx = comb.groupby("timestamp").cumcount()
    orig_ts = comb["timestamp"].copy()
    comb["timestamp"] = comb["timestamp"] + cum_idx
    stats = {"raw_rows": int(len(comb)), "unique_timestamps": uniq_count, "duplicate_timestamps": dup_count}
    return comb.drop(columns=["prio", "exchange"]), orig_ts, stats  # type: ignore[return-value]


def load_funding_map(symbols: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    with _ro_conn(DB_PATH) as conn:
        for sym in symbols:
            df = pd.read_sql_query(
                """
                SELECT timestamp, funding_rate
                FROM funding_rates
                WHERE symbol = ?
                ORDER BY timestamp ASC, CASE WHEN exchange = 'binance' THEN 0 ELSE 1 END ASC
                """,
                conn,
                params=(sym,),
            )
            if not df.empty:
                df = df.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp")
                out[sym] = (df["timestamp"].to_numpy(np.int64), df["funding_rate"].to_numpy(np.float64))
    return out


def select_stationary_features(df_feat: pd.DataFrame) -> List[str]:
    num_cols = df_feat.select_dtypes(include=[np.number]).columns
    return [
        c for c in num_cols
        if c not in NON_STATIONARY_COLS
        and (c in {"dist_resistance_pct", "dist_support_pct"} or df_feat[c].notna().mean() > 0.95)
    ]


class _IndexedOOSModel:
    """Wrapper determinístico O(1) para injetar probabilidades ML pré-computadas sem lookahead no TechnicalAgent."""

    def __init__(self, probas: np.ndarray):
        self.probas = np.asarray(probas, dtype=np.float64)
        self.feature_names = ["_oos_idx"]
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, X: Any) -> np.ndarray:
        idx = int(np.asarray(X, dtype=np.float64)[0, 0])
        if 0 <= idx < len(self.probas):
            return self.probas[idx : idx + 1]
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float64)


class _StaticCalibrationTracker:
    """LearningTracker read-only em memória a partir dos pesos/calibração reparados no DB."""

    def __init__(self, db_path: str):
        import tempfile
        from learning_tracker import LearningTracker

        fd, tmp_p = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        try:
            lt = LearningTracker(db_path=tmp_p)
            with _ro_conn(db_path) as ro, sqlite3.connect(tmp_p) as dst:
                for tbl in ("agent_regime_weights", "agent_regime_calibration"):
                    rows = ro.execute(f"SELECT * FROM {tbl}").fetchall()
                    dst.execute(f"DELETE FROM {tbl}")
                    if rows:
                        qs = ",".join(["?"] * len(rows[0]))
                        dst.executemany(f"INSERT INTO {tbl} VALUES ({qs})", rows)
                dst.commit()
            regimes = ("BEAR_TREND", "BULL_TREND", "SIDEWAYS_RANGE", "HIGH_VOLATILITY_PANIC")
            self._weights = {r: lt.get_calibrated_weights_for_regime(r) for r in regimes}
            self._calib = {r: lt.get_agent_calibration_for_regime(r) for r in regimes}
        finally:
            if os.path.exists(tmp_p):
                os.unlink(tmp_p)

    def get_calibrated_weights_for_regime(self, regime: str) -> Optional[Dict[str, float]]:
        return self._weights.get(regime)

    def get_agent_calibration_for_regime(self, regime: str) -> Optional[Dict[str, Dict[str, float]]]:
        return self._calib.get(regime)


class _PilotSimWithCorrCap(_PilotSimulator):
    """Subclasse de _PilotSimulator impondo teto de posições simultâneas na mesma direção entre majors."""

    def __init__(self, eng: BacktestEngine, max_same_dir: Optional[int] = None):
        super().__init__(eng)
        self.max_same_dir = max_same_dir

    def try_open(self, sym, idx, direction, entry_ts, sig_close, entry_raw, atr, conf, regime_ok, positions, state):
        if self.max_same_dir is not None:
            side = "LONG" if direction == 1 else "SHORT"
            if sum(1 for q in positions.values() if q.side == side) >= self.max_same_dir:
                return None
        return super().try_open(sym, idx, direction, entry_ts, sig_close, entry_raw, atr, conf, regime_ok, positions, state)


def run_portfolio_parity(
    engine: BacktestEngine,
    data: Dict[str, pd.DataFrame],
    signals: Dict[str, np.ndarray],
    confidences: Optional[Dict[str, np.ndarray]],
    funding: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    max_same_dir: Optional[int] = None,
) -> BacktestResult:
    if max_same_dir is None:
        return engine.run_portfolio(data, signals, confidences=confidences, atr_col="atr_14", funding=funding)
    sim = _PilotSimWithCorrCap(engine, max_same_dir=max_same_dir)
    return sim.run(data, {k: np.asarray(v) for k, v in signals.items()}, confidences, "atr_14", funding)


def compute_enriched_metrics(
    res: BacktestResult,
    data_map: Dict[str, pd.DataFrame],
    strategy_type: str,
    start_ms: int,
    end_ms: int,
    n_trials_dsr: int = 1,
    var_sharpes: float = 0.0,
) -> Dict[str, Any]:
    """Calcula métricas consolidadas (E[R] normalizado pelo risco inicial, retornos diários, Sharpe, DSR, Bootstrap CI)."""
    prof = STRATEGY_PROFILES[strategy_type]
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
            atr_sig = float(t.entry_price) * (0.005 if strategy_type == "SCALP" else 0.01)
        if strategy_type == "SCALP":
            atr_sig = max(atr_sig, float(sym_df["close"].iloc[sig_idx]) * 0.0025)
        risk_usd = t.size * prof.sl_atr_mult * atr_sig
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
    boot_info = bootstrap_performance_metrics(daily_rets, n_iterations=1000, periods_per_year=365.0, seed=42, block_size=7)

    sh_ci = boot_info.get("sharpe_ci", (0.0, 0.0))
    ret_ci = boot_info.get("ann_ret_ci", (0.0, 0.0))
    dd_ci = boot_info.get("max_dd_ci", (0.0, 0.0))

    r_arr = np.asarray(r_mults, dtype=np.float64)
    if len(r_arr) > 0:
        rng = np.random.default_rng(42)
        boot_r = rng.choice(r_arr, size=(2000, len(r_arr)), replace=True).mean(axis=1)
        r_ci_lo, r_ci_hi = float(np.quantile(boot_r, 0.025)), float(np.quantile(boot_r, 0.975))
    else:
        r_ci_lo, r_ci_hi = 0.0, 0.0

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
        "sharpe_equity_curve": round(float(res.sharpe_ratio), 3),
        "dsr": round(float(dsr_info["dsr"]), 4),
        "daily_boot_sharpe_ci95": [round(float(sh_ci[0]), 3), round(float(sh_ci[1]), 3)],
        "daily_boot_ret_pct_ci95": [round(float(ret_ci[0] * 100.0), 2), round(float(ret_ci[1] * 100.0), 2)],
        "daily_boot_dd_pct_ci95": [round(float(dd_ci[0] * 100.0), 2), round(float(dd_ci[1] * 100.0), 2)],
        "total_fees_funding": round(float(res.total_fees), 2),
        "exit_reasons": exit_reasons,
        "_daily_rets": daily_rets,
        "_sr_daily_ann": float(sr_daily_ann),
    }


def _strip_internal(metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in metrics_dict.items() if not k.startswith("_")}


# =====================================================================
# BLOCO 1: ML Before vs After Dedup + Auditoria .joblib
# =====================================================================

def run_block1_ml_audit() -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    print("\n=== BLOCO 1: ML Walk-Forward Antes vs Depois Dedup + Auditoria .joblib ===", flush=True)
    results: Dict[str, Any] = {"contaminated_vs_clean": {}, "models_audit": {}}
    cached_clean_1h: Dict[str, Dict[str, Any]] = {}

    for sym in MAJORS:
        t0 = time.time()
        # 1A. Contaminado (antes Step 2)
        df_contam_raw, orig_ts, contam_stats = load_candles_contaminated_25k(sym, "1h", limit=25000)
        feat_contam = FeatureEngineer.compute_all_features(df_contam_raw, drop_na=False)
        feat_contam["timestamp"] = orig_ts.values[: len(feat_contam)]
        fcols_contam = select_stationary_features(feat_contam)

        res_contam_models = {}
        for mtype in ("lgbm", "rf"):
            mt = ModelTrainer(model_type=mtype, horizon=5, multiple_tp=1.2, multiple_sl=1.2, confidence_threshold=0.50, n_jobs=8)
            X_c, y_c = mt.prepare_dataset(feat_contam, feature_columns=fcols_contam)
            r_c = mt.train_walk_forward(X_c, y_c, n_splits=4)
            res_contam_models[mtype] = {
                "samples_evaluated": int(r_c["samples_evaluated"]),
                "n_features": int(len(fcols_contam)),
                "global_accuracy": round(float(r_c["global_accuracy"]), 4),
                "f1_macro": round(float(r_c["f1_macro"]), 4),
                "emitted_signals_accuracy": round(float(r_c["emitted_signals_accuracy"]), 4),
                "emitted_signals_count": int(r_c["emitted_signals_count"]),
                "coverage_pct": round(float(r_c["coverage_pct"] * 100.0), 2),
            }

        # 1B. Limpo Deduplicado (25.000 barras únicas 1h) -> reutilizado no Bloco 2
        df_clean = load_candles_clean(sym, "1h", limit=25000)
        feat_clean = FeatureEngineer.compute_all_features(df_clean, drop_na=False)
        fcols_clean = select_stationary_features(feat_clean)

        res_clean_models = {}
        oos_probs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for mtype in ("lgbm", "rf"):
            mt = ModelTrainer(model_type=mtype, horizon=5, multiple_tp=1.2, multiple_sl=1.2, confidence_threshold=0.50, n_jobs=8)
            X_cl, y_cl = mt.prepare_dataset(feat_clean, feature_columns=fcols_clean)
            r_cl = mt.train_walk_forward(X_cl, y_cl, n_splits=4)
            res_clean_models[mtype] = {
                "samples_evaluated": int(r_cl["samples_evaluated"]),
                "n_features": int(len(fcols_clean)),
                "global_accuracy": round(float(r_cl["global_accuracy"]), 4),
                "f1_macro": round(float(r_cl["f1_macro"]), 4),
                "emitted_signals_accuracy": round(float(r_cl["emitted_signals_accuracy"]), 4),
                "emitted_signals_count": int(r_cl["emitted_signals_count"]),
                "coverage_pct": round(float(r_cl["coverage_pct"] * 100.0), 2),
            }
            df_idx = X_cl.index[np.asarray(r_cl["test_indices"], dtype=int)].to_numpy()
            oos_probs[mtype] = (df_idx, np.asarray(r_cl["y_proba"], dtype=np.float64))

        cached_clean_1h[sym] = {
            "df_feat": feat_clean,
            "fcols": fcols_clean,
            "oos_probs": oos_probs,
            "ts_start": int(df_clean["timestamp"].iloc[0]),
            "ts_end": int(df_clean["timestamp"].iloc[-1]),
        }
        results["contaminated_vs_clean"][sym] = {
            "contaminated_slice": {
                **contam_stats,
                "ts_start": _ts_iso(int(orig_ts.iloc[0])),
                "ts_end": _ts_iso(int(orig_ts.iloc[-1])),
                "models": res_contam_models,
            },
            "clean_slice": {
                "rows": int(len(df_clean)),
                "unique_timestamps": int(df_clean["timestamp"].nunique()),
                "duplicate_timestamps": 0,
                "ts_start": _ts_iso(int(df_clean["timestamp"].iloc[0])),
                "ts_end": _ts_iso(int(df_clean["timestamp"].iloc[-1])),
                "models": res_clean_models,
            },
        }
        print(
            f"  {sym}: Contam LGBM={res_contam_models['lgbm']['global_accuracy']:.4f} "
            f"(dups={contam_stats['duplicate_timestamps']}) vs Clean LGBM={res_clean_models['lgbm']['global_accuracy']:.4f} "
            f"({time.time()-t0:.1f}s)",
            flush=True,
        )

    for label, path in MODEL_PATHS:
        if os.path.exists(path):
            bundle = joblib.load(path)
            m = bundle.get("metrics", {})
            results["models_audit"][label] = {
                "path": os.path.relpath(path, ROOT),
                "created_at": bundle.get("created_at"),
                "horizon": bundle.get("horizon"),
                "multiple_tp": bundle.get("multiple_tp"),
                "multiple_sl": bundle.get("multiple_sl"),
                "n_features": len(bundle.get("feature_names", [])),
                "samples_evaluated": m.get("samples_evaluated"),
                "global_accuracy": round(float(m.get("global_accuracy", 0.0)), 4),
                "f1_macro": round(float(m.get("f1_macro", 0.0)), 4),
                "emitted_signals_accuracy": round(float(m.get("emitted_signals_accuracy", 0.0)), 4),
                "emitted_signals_count": m.get("emitted_signals_count"),
                "coverage_pct": round(float(m.get("coverage_pct", 0.0) * 100.0), 2),
                "uncontaminated_reason": "ModelEvolutionManager consulta estritamente exchange='binance'",
            }
    return results, cached_clean_1h


# =====================================================================
# BLOCO 2: Reavaliação 21:00 BRT + Pilot Parity + ML Estritamente OOS
# =====================================================================

def compute_hourly_brt_profile() -> Dict[str, Any]:
    """Perfil horário BRT (00..23) em toda a base limpa 1h deduplicada (174.344 velas)."""
    frames = []
    counts_by_sym = {}
    for sym in MAJORS:
        df = load_candles_clean(sym, "1h", limit=None)
        counts_by_sym[sym] = int(len(df))
        fe_df = FeatureEngineer.compute_all_features(df, drop_na=False)
        fe_df = fe_df[fe_df["sma_200"].notna()].copy()
        fe_df["symbol"] = sym
        fe_df["ret_1h_next"] = (fe_df["close"].shift(-1) - fe_df["close"]) / fe_df["close"]
        fe_df["ret_4h_next"] = (fe_df["close"].shift(-4) - fe_df["close"]) / fe_df["close"]
        fe_df["ret_bar"] = (fe_df["close"] - fe_df["open"]) / fe_df["open"]
        fe_df["range_atr"] = (fe_df["high"] - fe_df["low"]) / fe_df["atr_14"].replace(0, np.nan)
        fe_df["vol_ratio"] = fe_df["volume"] / fe_df["volume"].rolling(20).mean()
        # Reversão nas próximas 4h: sinal oposto ao candle da barra quando |ret_bar| > 0.2%
        fe_df["rev_4h"] = ((fe_df["ret_bar"] * fe_df["ret_4h_next"]) < 0).astype(float)
        frames.append(fe_df)

    all_df = pd.concat(frames, ignore_index=True)
    dt_brt = pd.to_datetime(all_df["timestamp"], unit="ms", utc=True).dt.tz_convert("America/Sao_Paulo")
    all_df["brt_hour"] = dt_brt.dt.hour
    all_df["dow"] = dt_brt.dt.day_name()

    by_hour = {}
    for hr, grp in all_df.groupby("brt_hour"):
        big = grp[grp["ret_bar"].abs() >= 0.003]
        by_hour[f"{int(hr):02d}h"] = {
            "n_bars": int(len(grp)),
            "mean_range_atr": round(float(grp["range_atr"].mean()), 3),
            "mean_vol_ratio": round(float(grp["vol_ratio"].mean()), 3),
            "mean_abs_ret_pct": round(float(grp["ret_bar"].abs().mean() * 100.0), 3),
            "rev_4h_all_pct": round(float(grp["rev_4h"].mean() * 100.0), 2),
            "rev_4h_big_move_pct": round(float(big["rev_4h"].mean() * 100.0), 2) if len(big) else 0.0,
        }

    hr21 = all_df[all_df["brt_hour"] == 21]
    by_dow_21h = {}
    for dow, grp in hr21.groupby("dow"):
        big = grp[grp["ret_bar"].abs() >= 0.003]
        by_dow_21h[str(dow)] = {
            "n_bars": int(len(grp)),
            "mean_range_atr": round(float(grp["range_atr"].mean()), 3),
            "mean_vol_ratio": round(float(grp["vol_ratio"].mean()), 3),
            "rev_4h_big_move_pct": round(float(big["rev_4h"].mean() * 100.0), 2) if len(big) else 0.0,
        }

    return {
        "total_clean_bars": int(sum(counts_by_sym.values())),
        "bars_by_symbol": counts_by_sym,
        "by_hour_brt": by_hour,
        "hour_21_brt_by_dow": by_dow_21h,
    }


def _generate_block2_signals_1h(df: pd.DataFrame, p_buy: np.ndarray, p_sell: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(df)
    close = df["close"].to_numpy(float)
    open_p = df["open"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    sma200 = df["sma_200"].to_numpy(float)
    rsi = df["rsi_14"].to_numpy(float)
    vol_z = df["volume_zscore_20"].to_numpy(float)

    rng = np.where((high - low) == 0.0, 1e-9, high - low)
    upper_wick_pct = (high - np.maximum(open_p, close)) / open_p * 100.0
    lower_wick_pct = (np.minimum(open_p, close) - low) / open_p * 100.0
    close_pos = (close - low) / rng
    r_high_6 = df["high"].shift(1).rolling(6).max().to_numpy(float)
    r_low_6 = df["low"].shift(1).rolling(6).min().to_numpy(float)
    ret_3h = ((df["close"].shift(1) - df["open"].shift(3)) / df["open"].shift(3) * 100.0).to_numpy(float)
    hours = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("America/Sao_Paulo").dt.hour.to_numpy()

    def _pack(buy_mask: np.ndarray, sell_mask: np.ndarray) -> np.ndarray:
        sig = np.zeros(n, dtype=np.int32)
        sig[buy_mask] = 1
        sig[sell_mask] = 2
        sig[buy_mask & sell_mask] = 0
        return sig

    # Baseline Momentum
    b_buy = (close > sma200) & (rsi > 52.0) & (close > r_high_6)
    b_sell = (close < sma200) & (rsi < 48.0) & (close < r_low_6)

    # Opção A: Flow Reversal
    a_buy = (ret_3h < -0.3) & (lower_wick_pct >= 0.20) & (close_pos >= 0.55) & (vol_z >= 0.5)
    a_sell = (ret_3h > 0.3) & (upper_wick_pct >= 0.20) & (close_pos <= 0.45) & (vol_z >= 0.5)

    # Opção B: Session Sweep Fade (21h, 05h, 11h BRT)
    is_sess = np.isin(hours, [21, 5, 11])
    b_fade_buy = is_sess & (low < r_low_6) & (lower_wick_pct >= 0.15) & (close > open_p)
    b_fade_sell = is_sess & (high > r_high_6) & (upper_wick_pct >= 0.15) & (close < open_p)

    # Opção B-21h: somente 21:00 BRT
    is_21 = hours == 21
    b21_buy = is_21 & (low < r_low_6) & (lower_wick_pct >= 0.15) & (close > open_p)
    b21_sell = is_21 & (high > r_high_6) & (upper_wick_pct >= 0.15) & (close < open_p)

    ml_buy_ok = (p_buy >= 0.34) & (p_buy > p_sell)
    ml_sell_ok = (p_sell >= 0.34) & (p_sell > p_buy)

    return {
        "Baseline Momentum (1h)": _pack(b_buy, b_sell),
        "Opção A: Flow Reversal (Pura)": _pack(a_buy, a_sell),
        "Opção A: Flow Reversal (ML OOS)": _pack(a_buy & ml_buy_ok, a_sell & ml_sell_ok),
        "Opção B: Session Sweep 21/05/11h (Pura)": _pack(b_fade_buy, b_fade_sell),
        "Opção B: Session Sweep 21/05/11h (ML OOS)": _pack(b_fade_buy & ml_buy_ok, b_fade_sell & ml_sell_ok),
        "Opção B-21h: Daily Open 21h BRT (Pura)": _pack(b21_buy, b21_sell),
        "Opção B-21h: Daily Open 21h BRT (ML OOS)": _pack(b21_buy & ml_buy_ok, b21_sell & ml_sell_ok),
    }


def _generate_block2_signals_15m(df: pd.DataFrame, p_buy: np.ndarray, p_sell: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(df)
    vwap = df["dist_vwap_pct"].to_numpy(float)
    rsi = df["rsi_14"].to_numpy(float)
    vol_z = df["volume_zscore_20"].to_numpy(float)

    def _pack(buy_mask: np.ndarray, sell_mask: np.ndarray) -> np.ndarray:
        sig = np.zeros(n, dtype=np.int32)
        sig[buy_mask] = 1
        sig[sell_mask] = 2
        sig[buy_mask & sell_mask] = 0
        return sig

    c20_buy = (vwap <= -2.0) & (rsi <= 35.0) & (vol_z >= 0.2)
    c20_sell = (vwap >= 2.0) & (rsi >= 65.0) & (vol_z >= 0.2)

    c08_buy = (vwap <= -0.8) & (rsi <= 30.0) & (vol_z >= -0.5)
    c08_sell = (vwap >= 0.8) & (rsi >= 70.0) & (vol_z >= -0.5)

    c06_buy = (vwap <= -0.6) & (rsi <= 35.0) & (vol_z >= 0.8)
    c06_sell = (vwap >= 0.6) & (rsi >= 65.0) & (vol_z >= 0.8)

    ml_buy_ok = (p_buy >= 0.34) & (p_buy > p_sell)
    ml_sell_ok = (p_sell >= 0.34) & (p_sell > p_buy)

    return {
        "Opção C: Scalp VWAP (Baseline 2.0%)": _pack(c20_buy, c20_sell),
        "Opção C: Scalp VWAP (Live DB 0.8%)": _pack(c08_buy, c08_sell),
        "Opção C: Scalp VWAP (Calibrado 0.6%)": _pack(c06_buy, c06_sell),
        "Opção C: Scalp VWAP (Calibrado 0.6% + ML OOS 15m)": _pack(c06_buy & ml_buy_ok, c06_sell & ml_sell_ok),
    }


def _eval_strategies_oos_pilot(
    data_oos: Dict[str, pd.DataFrame],
    signals_map: Dict[str, Dict[str, np.ndarray]],
    funding: Dict[str, Tuple[np.ndarray, np.ndarray]],
    strategy_type: str,
    timeframe: str,
    daily_quota: Optional[int],
    cooldown_bars: Optional[int],
    n_prior_trials: int,
) -> Dict[str, Any]:
    eng = BacktestEngine(
        initial_capital=10000.0,
        timeframe=timeframe,
        pilot_parity=True,
        strategy_type=strategy_type,
        max_leverage=3.0,
        risk_per_trade=None,
        daily_quota=daily_quota,
        cooldown_bars=cooldown_bars,
    )
    start_ms = min(int(df["timestamp"].iloc[0]) for df in data_oos.values())
    end_ms = max(int(df["timestamp"].iloc[-1]) for df in data_oos.values())

    strat_names = list(signals_map[MAJORS[0]].keys())
    raw_port: Dict[str, Dict[str, Any]] = {}
    per_asset: Dict[str, Dict[str, Any]] = {name: {} for name in strat_names}

    for name in strat_names:
        sig_port = {sym: signals_map[sym][name] for sym in MAJORS}
        res_p = eng.run_portfolio(data_oos, sig_port, confidences=None, atr_col="atr_14", funding=funding)
        raw_port[name] = compute_enriched_metrics(res_p, data_oos, strategy_type, start_ms, end_ms)

        for sym in MAJORS:
            f_sym = {sym: funding[sym]} if sym in funding else None
            res_s = eng.run_portfolio({sym: data_oos[sym]}, {sym: signals_map[sym][name]}, confidences=None, atr_col="atr_14", funding=f_sym)
            m_s = compute_enriched_metrics(res_s, {sym: data_oos[sym]}, strategy_type, start_ms, end_ms)
            per_asset[name][sym] = _strip_internal(m_s)

    var_sh = float(np.var([m["_sr_daily_ann"] for m in raw_port.values()], ddof=1)) if len(raw_port) > 1 else 0.0
    n_trials_total = n_prior_trials + len(strat_names)
    consolidated: Dict[str, Any] = {}
    for name in strat_names:
        m = dict(raw_port[name])
        dsr_info = deflated_sharpe_ratio(
            observed_sr_ann=m["_sr_daily_ann"],
            returns=m["_daily_rets"],
            n_trials=max(2, n_trials_total),
            variance_of_trials=var_sh,
            periods_per_year=365.0,
        )
        m["dsr"] = round(float(dsr_info["dsr"]), 4)
        consolidated[name] = _strip_internal(m)

    return {
        "oos_start": _ts_iso(start_ms),
        "oos_end": _ts_iso(end_ms),
        "consolidated_portfolio": consolidated,
        "per_asset": per_asset,
    }


def run_block2_market_open(
    cached_clean_1h: Dict[str, Dict[str, Any]],
    funding: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_prior_trials: int,
) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    print("\n=== BLOCO 2: Perfil 21:00 BRT & Reavaliação Market Open (Pilot Parity + ML OOS) ===", flush=True)
    hourly_profile = compute_hourly_brt_profile()

    # 1h OOS window (reaproveita walk-forward limpo do Bloco 1)
    data_1h_oos: Dict[str, pd.DataFrame] = {}
    signals_1h_oos: Dict[str, Dict[str, np.ndarray]] = {}
    for sym in MAJORS:
        bundle = cached_clean_1h[sym]
        df_feat = bundle["df_feat"]
        idx_l, p_l = bundle["oos_probs"]["lgbm"]
        idx_r, p_r = bundle["oos_probs"]["rf"]
        # Mesma partição PurgedTimeSeriesSplit -> mesmos índices
        common_idx = np.intersect1d(idx_l, idx_r)
        p_buy = np.zeros(len(df_feat), dtype=np.float64)
        p_sell = np.zeros(len(df_feat), dtype=np.float64)
        map_l = {int(i): k for k, i in enumerate(idx_l)}
        map_r = {int(i): k for k, i in enumerate(idx_r)}
        for i in common_idx:
            kl, kr = map_l[int(i)], map_r[int(i)]
            p_buy[int(i)] = 0.5 * (p_l[kl, 1] + p_r[kr, 1])
            p_sell[int(i)] = 0.5 * (p_l[kl, 2] + p_r[kr, 2])
        oos_start = int(common_idx.min())
        all_sigs = _generate_block2_signals_1h(df_feat, p_buy, p_sell)
        data_1h_oos[sym] = df_feat.iloc[oos_start:].reset_index(drop=True)
        signals_1h_oos[sym] = {k: v[oos_start:] for k, v in all_sigs.items()}

    res_1h = _eval_strategies_oos_pilot(
        data_1h_oos, signals_1h_oos, funding, "DIRECTIONAL", "1h",
        daily_quota=None, cooldown_bars=0, n_prior_trials=n_prior_trials,
    )

    # 15m OOS window (modelos LGBM + RF treinados especificamente em 15m!)
    data_15m_full: Dict[str, pd.DataFrame] = {}
    data_15m_oos: Dict[str, pd.DataFrame] = {}
    signals_15m_oos: Dict[str, Dict[str, np.ndarray]] = {}
    ml_15m_stats: Dict[str, Any] = {}

    for sym in MAJORS:
        t0 = time.time()
        df15 = load_candles_clean(sym, "15m", limit=25000)
        feat15 = FeatureEngineer.compute_all_features(df15, drop_na=False)
        data_15m_full[sym] = feat15
        fcols15 = select_stationary_features(feat15)
        oos_probs15: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        ml_15m_stats[sym] = {}
        for mtype in ("lgbm", "rf"):
            mt = ModelTrainer(model_type=mtype, horizon=5, multiple_tp=1.2, multiple_sl=1.2, confidence_threshold=0.50, n_jobs=8)
            X15, y15 = mt.prepare_dataset(feat15, feature_columns=fcols15)
            r15 = mt.train_walk_forward(X15, y15, n_splits=4)
            ml_15m_stats[sym][mtype] = {
                "samples_evaluated": int(r15["samples_evaluated"]),
                "global_accuracy": round(float(r15["global_accuracy"]), 4),
                "f1_macro": round(float(r15["f1_macro"]), 4),
                "emitted_signals_accuracy": round(float(r15["emitted_signals_accuracy"]), 4),
                "emitted_signals_count": int(r15["emitted_signals_count"]),
            }
            oos_probs15[mtype] = (X15.index[np.asarray(r15["test_indices"], dtype=int)].to_numpy(), np.asarray(r15["y_proba"], dtype=np.float64))

        idx_l, p_l = oos_probs15["lgbm"]
        idx_r, p_r = oos_probs15["rf"]
        common_idx = np.intersect1d(idx_l, idx_r)
        p_buy = np.zeros(len(feat15), dtype=np.float64)
        p_sell = np.zeros(len(feat15), dtype=np.float64)
        map_l = {int(i): k for k, i in enumerate(idx_l)}
        map_r = {int(i): k for k, i in enumerate(idx_r)}
        for i in common_idx:
            kl, kr = map_l[int(i)], map_r[int(i)]
            p_buy[int(i)] = 0.5 * (p_l[kl, 1] + p_r[kr, 1])
            p_sell[int(i)] = 0.5 * (p_l[kl, 2] + p_r[kr, 2])
        oos_start = int(common_idx.min())
        all_sigs15 = _generate_block2_signals_15m(feat15, p_buy, p_sell)
        # Floor SCALP_MIN_ATR_PCT = 0.0025 como no daemon
        feat15_oos = feat15.iloc[oos_start:].copy().reset_index(drop=True)
        feat15_oos["atr_14"] = np.maximum(feat15_oos["atr_14"].to_numpy(float), feat15_oos["close"].to_numpy(float) * 0.0025)
        data_15m_oos[sym] = feat15_oos
        signals_15m_oos[sym] = {k: v[oos_start:] for k, v in all_sigs15.items()}
        print(f"  15m OOS ML {sym}: LGBM={ml_15m_stats[sym]['lgbm']['global_accuracy']:.4f} ({time.time()-t0:.1f}s)", flush=True)

    res_15m = _eval_strategies_oos_pilot(
        data_15m_oos, signals_15m_oos, funding, "SCALP", "15m",
        daily_quota=15, cooldown_bars=4, n_prior_trials=n_prior_trials + 7,
    )

    for sname, m in res_1h["consolidated_portfolio"].items():
        print(f"  [1h OOS Port] {sname:<42} | Tr={m['trades']:>4} | WR={m['win_rate_pct']:>5.1f}% | PF={m['profit_factor']:>5.2f} | E[R]={m['expectancy_r']:>+6.3f} | PnL=${m['net_pnl_usd']:>+8.2f}", flush=True)
    for sname, m in res_15m["consolidated_portfolio"].items():
        print(f"  [15m OOS Port] {sname:<48} | Tr={m['trades']:>4} | WR={m['win_rate_pct']:>5.1f}% | PF={m['profit_factor']:>5.2f} | E[R]={m['expectancy_r']:>+6.3f} | PnL=${m['net_pnl_usd']:>+8.2f}", flush=True)

    return {
        "hourly_brt_profile_174k": hourly_profile,
        "ml_15m_walk_forward": ml_15m_stats,
        "strategies_1h_oos": res_1h,
        "strategies_15m_oos": res_15m,
        "flawed_nov10_reference": {
            "baseline_bug_initial": {"win_rate_pct": 68.1, "cause": "duplicação Binance/Bitget intra-barra + scaling atr_pct + vazamento ML in-sample"},
            "e10_7_baseline_clean_toy": {"win_rate_pct": 37.7, "pf": 0.64, "exp_r": -0.27},
            "e10_7_opt_a_btc_insample_toy": {"trades": 261, "win_rate_pct": 51.3, "pf": 1.08, "exp_r": 0.05, "consolidated_wr_pct": 41.8, "consolidated_pf": 0.82},
            "e10_7_opt_b_btc_insample_toy": {"win_rate_pct": 47.5, "pf": 0.90, "exp_r": -0.07},
            "e10_7_opt_c_btc_insample_toy": {"win_rate_pct": 51.6, "pf": 0.69, "exp_r": -0.22},
        },
    }, data_15m_full


# =====================================================================
# BLOCO 3: Calibrações S4.2 DIRECTIONAL (1h) & SCALP (15m) em Holdout OOS
# =====================================================================

def _train_pre_split_ensemble(df_btc_feat: pd.DataFrame, split_ts: int, feature_names_45: List[str]) -> Tuple[Any, Any, int]:
    """Treina LGBM + RF estritamente antes do embargo (ts < split_ts) com exatamente 45 features."""
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from model_trainer import create_triple_barrier_labels

    df_tr = df_btc_feat[df_btc_feat["timestamp"] < split_ts].copy().reset_index(drop=True)
    y_ser = create_triple_barrier_labels(df_tr, horizon=5, multiple_tp=1.2, multiple_sl=1.2)
    X_full = pd.DataFrame(0.0, index=df_tr.index, columns=feature_names_45)
    for col in feature_names_45:
        if col in df_tr.columns:
            X_full[col] = df_tr[col].astype(np.float64).fillna(0.0)
    valid = y_ser.notna()
    X_tr = X_full.loc[valid]
    y_tr = y_ser.loc[valid].astype(int)

    lgbm = LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42, class_weight="balanced", n_jobs=8, verbose=-1,
    )
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, min_samples_leaf=5,
        random_state=42, class_weight="balanced", n_jobs=8,
    )
    lgbm.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr)
    return lgbm, rf, int(len(X_tr))


def _predict_45_probas(df_feat: pd.DataFrame, lgbm: Any, rf: Any, feature_names_45: List[str]) -> np.ndarray:
    X_full = pd.DataFrame(0.0, index=df_feat.index, columns=feature_names_45)
    for col in feature_names_45:
        if col in df_feat.columns:
            X_full[col] = df_feat[col].astype(np.float64).fillna(0.0)
    return 0.5 * (lgbm.predict_proba(X_full) + rf.predict_proba(X_full))


def _generate_directional_core_signals(
    df_feat: pd.DataFrame,
    sym: str,
    probas: np.ndarray,
    add_p2_risk_cols: bool = False,
    learning_tracker: Optional[Any] = None,
    conf_gate: float = 0.70,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Gera sinais e confianças do comitê TradingCore sem vazamento nem I/O."""
    df_w = df_feat.copy().reset_index(drop=True)
    df_w["_oos_idx"] = np.arange(len(df_w), dtype=float)
    if add_p2_risk_cols:
        # Past-only rolling medians/averages (shift 1) para RiskSentinelAgent
        df_w["atr_median"] = df_w["atr_14"].shift(1).rolling(100, min_periods=20).median()
        if "vol_sma_20" in df_w.columns:
            df_w["volume_sma"] = df_w["vol_sma_20"].shift(1)
        else:
            df_w["volume_sma"] = df_w["volume"].shift(1).rolling(20, min_periods=10).mean()

    ta = TechnicalAgent(model=_IndexedOOSModel(probas))
    core = TradingCore(min_consensus_threshold=0.30, technical_agent=ta)
    if learning_tracker is not None:
        core.learning_tracker = learning_tracker

    records = df_w.to_dict("records")
    n = len(records)
    sigs = np.zeros(n, dtype=np.int32)
    confs = np.zeros(n, dtype=np.float64)
    vetoes = 0
    for i, rec in enumerate(records):
        ts_sig = core.evaluate_market_deterministic(
            context=rec,
            symbol=sym,
            current_price=float(rec["close"]),
            atr=float(rec["atr_14"]),
            timestamp=int(rec["timestamp"]),
        )
        if ts_sig.is_vetoed:
            vetoes += 1
            continue
        if ts_sig.direction in ("BUY", "SELL") and ts_sig.confidence >= conf_gate:
            sigs[i] = 1 if ts_sig.direction == "BUY" else 2
            confs[i] = float(ts_sig.confidence)
    return sigs, confs, vetoes


def _generate_scalp_live_signals(
    df15_feat: pd.DataFrame,
    sym: str,
    probas15: np.ndarray,
    vwap_dev: float,
    rsi_os: float,
    rsi_ob: float,
    vol_z_min: float,
    block_counter_regime_pct: Optional[float] = None,
    conf_gate: float = 0.70,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Reproduz exatamente evaluate_scalp_opportunities (gatilho primário + fallback DB + _committee_gate)."""
    df_w = df15_feat.copy().reset_index(drop=True)
    df_w["_oos_idx"] = np.arange(len(df_w), dtype=float)
    ta = TechnicalAgent(model=_IndexedOOSModel(probas15))
    core = TradingCore(min_consensus_threshold=0.30, technical_agent=ta)

    close = df_w["close"].to_numpy(float)
    open_p = df_w["open"].to_numpy(float)
    high_p = df_w["high"].to_numpy(float)
    low_p = df_w["low"].to_numpy(float)
    atr = df_w["atr_14"].to_numpy(float)
    vol_z = df_w["volume_zscore_20"].to_numpy(float)
    vwap_dist = df_w["dist_vwap_pct"].to_numpy(float)
    rsi = df_w["rsi_14"].to_numpy(float)
    dist200 = df_w["dist_sma_200"].to_numpy(float)
    prior_high20 = df_w["high"].shift(1).rolling(20).max().to_numpy(float)
    prior_low20 = df_w["low"].shift(1).rolling(20).min().to_numpy(float)

    n = len(df_w)
    sigs = np.zeros(n, dtype=np.int32)
    confs = np.zeros(n, dtype=np.float64)
    stats = {"candidates_primary": 0, "candidates_fallback": 0, "blocked_counter_regime": 0, "committee_passed": 0}
    records = df_w.to_dict("records")

    for i in range(21, n):
        if not (np.isfinite(atr[i]) and atr[i] > 0 and high_p[i] > low_p[i]):
            continue
        rng_atr = (high_p[i] - low_p[i]) / atr[i]
        cpos = (close[i] - low_p[i]) / (high_p[i] - low_p[i])
        trig_dir: Optional[str] = None
        is_primary = False

        # 1. Gatilho primário: Breakout 20 barras
        if vol_z[i] >= 0.8 and rng_atr >= 1.0:
            if close[i] > open_p[i] and cpos >= 0.70 and close[i] > prior_high20[i]:
                trig_dir = "BUY"
                is_primary = True
            elif close[i] < open_p[i] and cpos <= 0.30 and close[i] < prior_low20[i]:
                trig_dir = "SELL"
                is_primary = True

        # 2. Fallback VWAP/RSI
        if trig_dir is None:
            if vwap_dist[i] <= -vwap_dev and rsi[i] <= rsi_os and vol_z[i] >= vol_z_min:
                trig_dir = "BUY"
            elif vwap_dist[i] >= vwap_dev and rsi[i] >= rsi_ob and vol_z[i] >= vol_z_min:
                trig_dir = "SELL"

        if trig_dir is None:
            continue

        if is_primary:
            stats["candidates_primary"] += 1
        else:
            stats["candidates_fallback"] += 1

        if block_counter_regime_pct is not None and np.isfinite(dist200[i]):
            if trig_dir == "BUY" and dist200[i] <= -block_counter_regime_pct:
                stats["blocked_counter_regime"] += 1
                continue
            if trig_dir == "SELL" and dist200[i] >= block_counter_regime_pct:
                stats["blocked_counter_regime"] += 1
                continue

        rec = records[i]
        ts_sig = core.evaluate_market_deterministic(
            context=rec, symbol=sym, current_price=float(close[i]),
            atr=float(max(atr[i], close[i] * 0.0025)), timestamp=int(rec["timestamp"]),
        )
        if not ts_sig.is_vetoed and ts_sig.direction == trig_dir and ts_sig.confidence >= conf_gate:
            sigs[i] = 1 if trig_dir == "BUY" else 2
            confs[i] = float(ts_sig.confidence)
            stats["committee_passed"] += 1

    return sigs, confs, stats


def run_block3_holdout_ab(
    funding: Dict[str, Tuple[np.ndarray, np.ndarray]],
    data_15m_full: Dict[str, pd.DataFrame],
    n_prior_trials: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    print("\n=== BLOCO 3: A/B Calibrações S4.2 em Holdout OOS (DIRECTIONAL 1h & SCALP 15m) ===", flush=True)
    bundle_45 = joblib.load(os.path.join(ROOT, "models", "lgbm_model_v1.joblib"))
    feature_names_45 = list(bundle_45["feature_names"])

    full_1h: Dict[str, pd.DataFrame] = {}
    for sym in MAJORS:
        df = load_candles_clean(sym, "1h", limit=None)
        feat = FeatureEngineer.compute_all_features(df, drop_na=False)
        feat = feat[feat["sma_200"].notna()].reset_index(drop=True)
        full_1h[sym] = feat

    start_1h = min(int(df["timestamp"].iloc[0]) for df in full_1h.values())
    end_1h = max(int(df["timestamp"].iloc[-1]) for df in full_1h.values()) + HOUR_MS
    split_1h = int(start_1h + TRAIN_FRAC * (end_1h - start_1h))
    hold0 = split_1h + EMBARGO_DAYS * DAY_MS
    pre_s3_end = int(pd.Timestamp("2026-10-30 23:59:59", tz="UTC").timestamp() * 1000)

    print(f"  Holdout 1h: {_ts_iso(hold0)} .. {_ts_iso(end_1h)} (Train BTC < {_ts_iso(split_1h)})", flush=True)
    lgbm_hold, rf_hold, n_train_bars = _train_pre_split_ensemble(full_1h["BTCUSDT"], split_1h, feature_names_45)

    hold_1h: Dict[str, pd.DataFrame] = {}
    pre_s3_1h: Dict[str, pd.DataFrame] = {}
    probs_hold: Dict[str, np.ndarray] = {}
    probs_pre_s3: Dict[str, np.ndarray] = {}
    for sym in MAJORS:
        df_h = full_1h[sym][full_1h[sym]["timestamp"] >= hold0].copy().reset_index(drop=True)
        hold_1h[sym] = df_h
        probs_hold[sym] = _predict_45_probas(df_h, lgbm_hold, rf_hold, feature_names_45)
        mask_pre = df_h["timestamp"] <= pre_s3_end
        pre_s3_1h[sym] = df_h[mask_pre].copy().reset_index(drop=True)
        probs_pre_s3[sym] = probs_hold[sym][mask_pre.to_numpy()]

    static_lt = _StaticCalibrationTracker(DB_PATH)

    # Gera vetores DIRECTIONAL nos cenários
    dir_inputs: Dict[str, Dict[str, Any]] = {}
    sigs_base, confs_base, vetoes_base = {}, {}, {}
    sigs_p2, confs_p2, vetoes_p2 = {}, {}, {}
    sigs_reg, confs_reg, vetoes_reg = {}, {}, {}
    sigs_comb, confs_comb, vetoes_comb = {}, {}, {}

    for sym in MAJORS:
        sigs_base[sym], confs_base[sym], vetoes_base[sym] = _generate_directional_core_signals(hold_1h[sym], sym, probs_hold[sym], add_p2_risk_cols=False, learning_tracker=None)
        sigs_p2[sym], confs_p2[sym], vetoes_p2[sym] = _generate_directional_core_signals(hold_1h[sym], sym, probs_hold[sym], add_p2_risk_cols=True, learning_tracker=None)
        sigs_reg[sym], confs_reg[sym], vetoes_reg[sym] = _generate_directional_core_signals(hold_1h[sym], sym, probs_hold[sym], add_p2_risk_cols=False, learning_tracker=static_lt)
        sigs_comb[sym], confs_comb[sym], vetoes_comb[sym] = _generate_directional_core_signals(hold_1h[sym], sym, probs_hold[sym], add_p2_risk_cols=True, learning_tracker=static_lt)

    dir_scenarios = [
        ("DIR_BASE", sigs_base, confs_base, None, sum(vetoes_base.values()), "Produção S4 v1.0.6-s4.1 (pesos default, sem P2)"),
        ("DIR_CAL1_P2", sigs_p2, confs_p2, None, sum(vetoes_p2.values()), "+ atr_median(100) & volume_sma (vetos RiskSentinel)"),
        ("DIR_CAL2_CORR_CAP", sigs_base, confs_base, 2, sum(vetoes_base.values()), "+ max_same_dir_majors=2 entre BTC/ETH/SOL"),
        ("DIR_CAL3_REGIME_WEIGHTS", sigs_reg, confs_reg, None, sum(vetoes_reg.values()), "+ pesos/calibração reparados por regime (nota: fit pós-S3)"),
        ("DIR_S4_2_COMBINED", sigs_comb, confs_comb, 2, sum(vetoes_comb.values()), "+ P2 + max_same_dir_majors=2 + pesos por regime"),
    ]

    eng_dir = BacktestEngine(
        initial_capital=10000.0, timeframe="1h", pilot_parity=True,
        strategy_type="DIRECTIONAL", max_leverage=3.0, risk_per_trade=None,
    )
    end_hold_ms = max(int(df["timestamp"].iloc[-1]) for df in hold_1h.values())
    dir_raw: Dict[str, Dict[str, Any]] = {}
    for name, s_map, c_map, cap_n, _vet, _desc in dir_scenarios:
        res_d = run_portfolio_parity(eng_dir, hold_1h, s_map, c_map, funding, max_same_dir=cap_n)
        dir_raw[name] = compute_enriched_metrics(res_d, hold_1h, "DIRECTIONAL", hold0, end_hold_ms)

    var_sh_dir = float(np.var([m["_sr_daily_ann"] for m in dir_raw.values()], ddof=1))
    n_trials_dir = n_prior_trials + len(dir_scenarios)
    dir_out: Dict[str, Any] = {}
    for name, s_map, c_map, cap_n, vet_cnt, desc in dir_scenarios:
        m = dict(dir_raw[name])
        dsr_info = deflated_sharpe_ratio(
            observed_sr_ann=m["_sr_daily_ann"],
            returns=m["_daily_rets"],
            n_trials=max(2, n_trials_dir),
            variance_of_trials=var_sh_dir,
            periods_per_year=365.0,
        )
        m["dsr"] = round(float(dsr_info["dsr"]), 4)
        # Subset estritamente anterior ao fit dos pesos (hold0 .. 2026-10-30) para DIR_BASE vs DIR_CAL3/COMBINED
        s_pre = {sym: s_map[sym][: len(pre_s3_1h[sym])] for sym in MAJORS}
        c_pre = {sym: c_map[sym][: len(pre_s3_1h[sym])] for sym in MAJORS}
        res_pre = run_portfolio_parity(eng_dir, pre_s3_1h, s_pre, c_pre, funding, max_same_dir=cap_n)
        m_pre = compute_enriched_metrics(res_pre, pre_s3_1h, "DIRECTIONAL", hold0, pre_s3_end, n_trials_dsr=n_trials_dir, var_sharpes=var_sh_dir)
        entry = _strip_internal(m)
        entry["vetoes"] = int(vet_cnt)
        entry["desc"] = desc
        entry["pre_s3_subwindow"] = _strip_internal(m_pre)
        dir_out[name] = entry
        print(f"  [DIR OOS] {name:<25} | Tr={entry['trades']:>4} | WR={entry['win_rate_pct']:>5.1f}% | PF={entry['profit_factor']:>5.2f} | E[R]={entry['expectancy_r']:>+6.3f} | PnL=${entry['net_pnl_usd']:>+8.2f} | DD={entry['max_dd_pct']:>5.2f}% | SR={entry['sharpe_daily_ann']:>+5.2f}", flush=True)

    # ---- SCALP 15m OOS ----
    # Usa probabilidades pré-calculadas pelo mesmo modelo pré-holdout nas colunas 15m (ou ensemble 15m)
    # Como SCALP usa TradingCore apenas como gate de confluência após gatilho VWAP/Breakout, usa ensemble pré-Holdout
    # para não vazar barras 15m futuras.
    data_15m_clean: Dict[str, pd.DataFrame] = {}
    probas_15m: Dict[str, np.ndarray] = {}
    for sym in MAJORS:
        df15 = data_15m_full[sym]
        df15_v = df15[df15["sma_200"].notna()].copy().reset_index(drop=True)
        df15_v["atr_14"] = np.maximum(df15_v["atr_14"].to_numpy(float), df15_v["close"].to_numpy(float) * 0.0025)
        data_15m_clean[sym] = df15_v
        probas_15m[sym] = _predict_45_probas(df15_v, lgbm_hold, rf_hold, feature_names_45)

    start_15 = min(int(df["timestamp"].iloc[0]) for df in data_15m_clean.values())
    end_15 = max(int(df["timestamp"].iloc[-1]) for df in data_15m_clean.values())

    scalp_configs = [
        ("SCALP_BASE_DB", 0.8, 30.0, 70.0, -0.5, None, "DB atual (vwap=0.8%, rsi=30/70, volZ=-0.5) + Breakout20 + Comitê>=0.70"),
        ("SCALP_CAL_TRIGGER", 2.0, 30.0, 70.0, 0.5, None, "Fallback restritivo (vwap=2.0%, rsi=30/70, volZ=0.5) + Breakout20"),
        ("SCALP_CAL_ALIGNED", 2.0, 30.0, 70.0, 0.5, 0.010, "Fallback restritivo + bloqueia contra-regime se |dist_sma_200|>=1.0%"),
    ]
    eng_scalp = BacktestEngine(
        initial_capital=10000.0, timeframe="15m", pilot_parity=True,
        strategy_type="SCALP", max_leverage=3.0, daily_quota=15, cooldown_bars=4, risk_per_trade=None,
    )
    scalp_built: Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, int], str]] = {}
    scalp_raw: Dict[str, Dict[str, Any]] = {}
    for name, vd, r_os, r_ob, vz, reg_blk, desc in scalp_configs:
        s_m, c_m = {}, {}
        tot_st: Counter = Counter()
        for sym in MAJORS:
            sg, cf, st_sym = _generate_scalp_live_signals(
                data_15m_clean[sym], sym, probas_15m[sym], vd, r_os, r_ob, vz, block_counter_regime_pct=reg_blk,
            )
            s_m[sym], c_m[sym] = sg, cf
            tot_st.update(st_sym)
        res_sc = eng_scalp.run_portfolio(data_15m_clean, s_m, confidences=c_m, atr_col="atr_14", funding=funding)
        scalp_raw[name] = compute_enriched_metrics(res_sc, data_15m_clean, "SCALP", start_15, end_15)
        scalp_built[name] = (s_m, c_m, dict(tot_st), desc)

    var_sh_sc = float(np.var([m["_sr_daily_ann"] for m in scalp_raw.values()], ddof=1))
    n_trials_sc = n_trials_dir + len(scalp_configs)
    scalp_out: Dict[str, Any] = {}
    for name, _vd, _ro, _rb, _vz, _rbk, desc in scalp_configs:
        _, _, st_tot, _ = scalp_built[name]
        m = dict(scalp_raw[name])
        dsr_info = deflated_sharpe_ratio(
            observed_sr_ann=m["_sr_daily_ann"],
            returns=m["_daily_rets"],
            n_trials=max(2, n_trials_sc),
            variance_of_trials=var_sh_sc,
            periods_per_year=365.0,
        )
        m["dsr"] = round(float(dsr_info["dsr"]), 4)
        entry = _strip_internal(m)
        entry["funnel_stats"] = st_tot
        entry["desc"] = desc
        scalp_out[name] = entry
        print(f"  [SCALP OOS] {name:<20} | Tr={entry['trades']:>4} | WR={entry['win_rate_pct']:>5.1f}% | PF={entry['profit_factor']:>5.2f} | E[R]={entry['expectancy_r']:>+6.3f} | PnL=${entry['net_pnl_usd']:>+8.2f} | DD={entry['max_dd_pct']:>5.2f}%", flush=True)

    runtime_cache = {
        "feature_names_45": feature_names_45,
        "pre_s3_cutoff": _ts_iso(pre_s3_end),
        "hold0_ms": hold0,
        "end_1h_ms": end_1h,
        "full_1h": full_1h,
        "static_lt": static_lt,
    }
    return {
        "holdout_window": {"start": _ts_iso(hold0), "end": _ts_iso(end_1h), "train_btc_bars": n_train_bars, "split_ts": _ts_iso(split_1h)},
        "scalp_window": {"start": _ts_iso(start_15), "end": _ts_iso(end_15)},
        "directional_ab": dir_out,
        "scalp_ab": scalp_out,
    }, runtime_cache


# =====================================================================
# BLOCO 4: Replay 29 dias Microstructure.db + Contrafactual N=24 S4
# =====================================================================

def _get_all_micro_symbols() -> set:
    with _ro_conn(MICRO_DB_PATH) as conn:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM book_features UNION SELECT DISTINCT symbol FROM tick_agg UNION SELECT DISTINCT symbol FROM derivs"
        ).fetchall()
    return {r[0] for r in rows}


def _tape_decision(res: Dict[str, Any], direction: str) -> Tuple[bool, str, str]:
    """Replica _flow_gate do daemon: fail-open quando UNAVAILABLE; HOLD ou voto oposto quando OK bloqueiam."""
    status = str(res.get("status", "UNAVAILABLE"))
    vote = str(res.get("vote", "HOLD"))
    if status != "OK":
        return True, "UNAVAILABLE_PASSED", status
    if vote == direction:
        return True, "PASSED", status
    return False, "BLOCKED", status


def _apply_historical_tape_gate(
    tape_agent: TapeReadingAgent,
    df_map: Dict[str, pd.DataFrame],
    signals: Dict[str, np.ndarray],
    bar_ms: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    out = {}
    counts = {"evaluated": 0, "passed": 0, "unavailable_passed": 0, "blocked": 0}
    for sym, df in df_map.items():
        sig_in = signals[sym].copy()
        ts_arr = df["timestamp"].to_numpy(np.int64)
        nz = np.where(sig_in != 0)[0]
        for idx in nz:
            direction = "BUY" if sig_in[idx] == 1 else "SELL"
            now_ms = int(ts_arr[idx]) + bar_ms
            res = tape_agent.evaluate(sym, direction, now_ms=now_ms, allow_live_fallback=False)
            counts["evaluated"] += 1
            passed, dec, _ = _tape_decision(res, direction)
            if not passed:
                sig_in[idx] = 0
                counts["blocked"] += 1
            elif dec == "UNAVAILABLE_PASSED":
                counts["unavailable_passed"] += 1
            else:
                counts["passed"] += 1
        out[sym] = sig_in
    return out, counts


def run_block4_replay_and_counterfactual(
    funding: Dict[str, Tuple[np.ndarray, np.ndarray]],
    data_15m_full: Dict[str, pd.DataFrame],
    block3_cache: Dict[str, Any],
) -> Dict[str, Any]:
    print("\n=== BLOCO 4: Replay 29d Microstructure Real + Contrafactual S4 (N=24) ===", flush=True)
    rep_start_ms = int(pd.Timestamp("2026-10-15 12:00:00", tz="UTC").timestamp() * 1000)
    train_cutoff_29d = int(pd.Timestamp("2026-10-14 00:00:00", tz="UTC").timestamp() * 1000)

    feature_names_45 = block3_cache["feature_names_45"]
    full_1h = block3_cache["full_1h"]
    static_lt = block3_cache["static_lt"]

    lgbm_29d, rf_29d, _ = _train_pre_split_ensemble(full_1h["BTCUSDT"], train_cutoff_29d, feature_names_45)
    all_micro_syms = _get_all_micro_symbols()
    tape_hist = TapeReadingAgent(db_path=MICRO_DB_PATH, covered_symbols=all_micro_syms)

    # 4A: 29-Day DIRECTIONAL Replay
    d29_1h: Dict[str, pd.DataFrame] = {}
    p29_1h: Dict[str, np.ndarray] = {}
    for sym in MAJORS:
        df_sub = full_1h[sym][full_1h[sym]["timestamp"] >= rep_start_ms].copy().reset_index(drop=True)
        d29_1h[sym] = df_sub
        p29_1h[sym] = _predict_45_probas(df_sub, lgbm_29d, rf_29d, feature_names_45)

    end_29d_1h = max(int(df["timestamp"].iloc[-1]) for df in d29_1h.values())
    eng_dir = BacktestEngine(initial_capital=10000.0, timeframe="1h", pilot_parity=True, strategy_type="DIRECTIONAL", max_leverage=3.0, risk_per_trade=None)

    sig_db_no_tape, conf_db = {}, {}
    sig_s42_no_tape, conf_s42 = {}, {}
    for sym in MAJORS:
        sig_db_no_tape[sym], conf_db[sym], _ = _generate_directional_core_signals(d29_1h[sym], sym, p29_1h[sym], add_p2_risk_cols=False, learning_tracker=None)
        sig_s42_no_tape[sym], conf_s42[sym], _ = _generate_directional_core_signals(d29_1h[sym], sym, p29_1h[sym], add_p2_risk_cols=True, learning_tracker=static_lt)

    sig_db_tape, tape_dir_base_cnt = _apply_historical_tape_gate(tape_hist, d29_1h, sig_db_no_tape, HOUR_MS)
    sig_s42_tape, tape_dir_s42_cnt = _apply_historical_tape_gate(tape_hist, d29_1h, sig_s42_no_tape, HOUR_MS)

    replay_dir = {}
    for lbl, s_m, c_m, cap_n, t_cnt in [
        ("DIR_BASE_NO_TAPE", sig_db_no_tape, conf_db, None, None),
        ("DIR_BASE_WITH_TAPE", sig_db_tape, conf_db, None, tape_dir_base_cnt),
        ("DIR_S42_NO_TAPE", sig_s42_no_tape, conf_s42, 2, None),
        ("DIR_S42_WITH_TAPE", sig_s42_tape, conf_s42, 2, tape_dir_s42_cnt),
    ]:
        res = run_portfolio_parity(eng_dir, d29_1h, s_m, c_m, funding, max_same_dir=cap_n)
        m = _strip_internal(compute_enriched_metrics(res, d29_1h, "DIRECTIONAL", rep_start_ms, end_29d_1h))
        if t_cnt:
            m["tape_gate_stats"] = t_cnt
        replay_dir[lbl] = m
        print(f"  [29d DIR] {lbl:<20} | Tr={m['trades']:>3} | WR={m['win_rate_pct']:>5.1f}% | PF={m['profit_factor']:>5.2f} | PnL=${m['net_pnl_usd']:>+7.2f}", flush=True)

    # 4A: 29-Day SCALP Replay
    d29_15m: Dict[str, pd.DataFrame] = {}
    p29_15m: Dict[str, np.ndarray] = {}
    for sym in MAJORS:
        df_sub = data_15m_full[sym]
        df_sub = df_sub[(df_sub["sma_200"].notna()) & (df_sub["timestamp"] >= rep_start_ms)].copy().reset_index(drop=True)
        df_sub["atr_14"] = np.maximum(df_sub["atr_14"].to_numpy(float), df_sub["close"].to_numpy(float) * 0.0025)
        d29_15m[sym] = df_sub
        p29_15m[sym] = _predict_45_probas(df_sub, lgbm_29d, rf_29d, feature_names_45)

    end_29d_15 = max(int(df["timestamp"].iloc[-1]) for df in d29_15m.values())
    eng_sc = BacktestEngine(initial_capital=10000.0, timeframe="15m", pilot_parity=True, strategy_type="SCALP", max_leverage=3.0, daily_quota=15, cooldown_bars=4, risk_per_trade=None)

    sc_base_no, sc_base_conf = {}, {}
    sc_cal_no, sc_cal_conf = {}, {}
    for sym in MAJORS:
        sc_base_no[sym], sc_base_conf[sym], _ = _generate_scalp_live_signals(d29_15m[sym], sym, p29_15m[sym], 0.8, 30.0, 70.0, -0.5, None)
        sc_cal_no[sym], sc_cal_conf[sym], _ = _generate_scalp_live_signals(d29_15m[sym], sym, p29_15m[sym], 2.0, 30.0, 70.0, 0.5, 0.010)

    sc_base_tape, tape_sc_base_cnt = _apply_historical_tape_gate(tape_hist, d29_15m, sc_base_no, BAR_15M_MS)
    sc_cal_tape, tape_sc_cal_cnt = _apply_historical_tape_gate(tape_hist, d29_15m, sc_cal_no, BAR_15M_MS)

    replay_scalp = {}
    for lbl, s_m, c_m, t_cnt in [
        ("SCALP_BASE_NO_TAPE", sc_base_no, sc_base_conf, None),
        ("SCALP_BASE_WITH_TAPE", sc_base_tape, sc_base_conf, tape_sc_base_cnt),
        ("SCALP_ALIGNED_NO_TAPE", sc_cal_no, sc_cal_conf, None),
        ("SCALP_ALIGNED_WITH_TAPE", sc_cal_tape, sc_cal_conf, tape_sc_cal_cnt),
    ]:
        res = eng_sc.run_portfolio(d29_15m, s_m, confidences=c_m, atr_col="atr_14", funding=funding)
        m = _strip_internal(compute_enriched_metrics(res, d29_15m, "SCALP", rep_start_ms, end_29d_15))
        if t_cnt:
            m["tape_gate_stats"] = t_cnt
        replay_scalp[lbl] = m
        print(f"  [29d SCALP] {lbl:<23} | Tr={m['trades']:>3} | WR={m['win_rate_pct']:>5.1f}% | PF={m['profit_factor']:>5.2f} | PnL=${m['net_pnl_usd']:>+7.2f}", flush=True)

    # 4B: Contrafactual nos trades reais S4 (2026-11-09 00:00 UTC ..)
    ts_s4 = int(pd.Timestamp("2026-11-09 00:00:00", tz="UTC").timestamp() * 1000)
    with _ro_conn(DB_PATH) as conn:
        pos_rows = pd.read_sql_query(
            """
            SELECT position_id, symbol, side, strategy_type, status, entry_timestamp,
                   entry_price, current_price, realized_pnl, fees_paid, funding_collected
            FROM paper_positions
            WHERE entry_timestamp >= ?
            ORDER BY entry_timestamp ASC
            """,
            conn,
            params=(ts_s4,),
        )
        flow_rows = pd.read_sql_query(
            """
            SELECT strategy_type, decision, COUNT(*) as n
            FROM flow_gate_log
            WHERE ts_ms >= ?
            GROUP BY strategy_type, decision
            """,
            conn,
            params=(ts_s4,),
        )
        trades_last = pd.read_sql_query(
            "SELECT position_id, action, timestamp, pnl FROM paper_trades WHERE timestamp >= ? ORDER BY timestamp ASC",
            conn,
            params=(ts_s4,),
        )

    close_ts_map: Dict[str, int] = {}
    close_reason_map: Dict[str, str] = {}
    for _, tr in trades_last.iterrows():
        close_ts_map[str(tr["position_id"])] = int(tr["timestamp"])
        close_reason_map[str(tr["position_id"])] = str(tr["action"])

    btc_1h = full_1h["BTCUSDT"]
    eth_1h = full_1h["ETHUSDT"]
    sol_1h = full_1h["SOLUSDT"]
    major_dist_map = {"BTCUSDT": btc_1h, "ETHUSDT": eth_1h, "SOLUSDT": sol_1h}
    btc_ts = btc_1h["timestamp"].to_numpy(np.int64)
    btc_d200 = btc_1h["dist_sma_200"].to_numpy(float)

    def _btc_dist_at(entry_ts: int) -> float:
        idx = int(np.searchsorted(btc_ts, entry_ts, side="right")) - 1
        return float(btc_d200[max(0, idx)])

    def _sym_dist200_at_1h(sym: str, entry_ts: int) -> Optional[float]:
        df_s = major_dist_map.get(sym)
        if df_s is None:
            return None
        ts_a = df_s["timestamp"].to_numpy(np.int64)
        idx = int(np.searchsorted(ts_a, entry_ts, side="right")) - 1
        return float(df_s["dist_sma_200"].iloc[max(0, idx)])

    closed_df = pos_rows[pos_rows["status"] != "OPEN"].copy()
    # smart_trade já debita open/partial/exit fees e acumula funding em paper_positions.realized_pnl (= SUM(paper_trades.pnl))
    closed_df["net_pnl"] = closed_df["realized_pnl"].fillna(0.0)
    closed_df["close_ts"] = closed_df["position_id"].map(lambda pid: close_ts_map.get(str(pid), ts_s4))
    closed_df["exit_reason"] = closed_df["position_id"].map(lambda pid: close_reason_map.get(str(pid), "UNKNOWN"))

    # Auditoria por trade fechado S4
    audited_trades = []
    # Para simular max_same_dir_majors=2 cronologicamente nos DIRECTIONAL S4
    open_dir_sim: List[Tuple[str, str, int]] = []  # (pid, side, close_ts)

    gp_before, gp_after_macro = [], []
    sc_before, sc_after_aligned = [], []
    dir_before, dir_after_cap2 = [], []

    for _, row in closed_df.iterrows():
        pid = str(row["position_id"])
        sym = str(row["symbol"])
        side = str(row["side"])
        strat = str(row["strategy_type"])
        ets = int(row["entry_timestamp"])
        cts = int(row["close_ts"])
        pnl = float(row["net_pnl"])
        btc_d = _btc_dist_at(ets)
        dir_str = "BUY" if side == "LONG" else "SELL"
        tape_res = tape_hist.evaluate(sym, dir_str, now_ms=ets, allow_live_fallback=False)
        tape_passed, tape_dec, tape_status = _tape_decision(tape_res, dir_str)

        item: Dict[str, Any] = {
            "position_id": pid,
            "symbol": sym,
            "side": side,
            "strategy_type": strat,
            "entry_utc": _ts_iso(ets),
            "net_pnl": round(pnl, 2),
            "exit_reason": str(row["exit_reason"]),
            "btc_dist_sma200": round(btc_d, 4),
            "tape_hist_status": tape_status,
            "tape_hist_vote": str(tape_res.get("vote", "HOLD")),
            "tape_hist_decision": tape_dec,
            "tape_hist_passed": bool(tape_passed),
        }

        if strat == "GOLDEN_POT":
            gp_before.append(pnl)
            macro_block = (side == "LONG" and btc_d <= -0.015) or (side == "SHORT" and btc_d >= 0.015)
            item["s42_candidate_action"] = "BLOCK_MACRO_COUNTER" if macro_block else "ALLOW"
            if not macro_block:
                gp_after_macro.append(pnl)
        elif strat == "SCALP":
            sc_before.append(pnl)
            sym_d200 = _sym_dist200_at_1h(sym, ets)
            # Counter-regime block if |dist_sma_200| >= 0.010, plus trigger threshold tightening note
            counter_blk = sym_d200 is not None and ((side == "LONG" and sym_d200 <= -0.010) or (side == "SHORT" and sym_d200 >= 0.010))
            item["sym_dist_sma200_1h"] = round(float(sym_d200), 4) if sym_d200 is not None else None
            item["s42_candidate_action"] = "BLOCK_COUNTER_REGIME" if counter_blk else "ALLOW_OR_TRIGGER_CHECK"
            if not counter_blk:
                sc_after_aligned.append(pnl)
        elif strat == "DIRECTIONAL":
            dir_before.append(pnl)
            open_dir_sim = [o for o in open_dir_sim if o[2] > ets]
            same_side_open = sum(1 for o in open_dir_sim if o[1] == side)
            cap_blocked = same_side_open >= 2
            item["s42_candidate_action"] = "BLOCK_CORR_CAP_2" if cap_blocked else "ALLOW"
            if not cap_blocked:
                dir_after_cap2.append(pnl)
                open_dir_sim.append((pid, side, cts))

        audited_trades.append(item)

    def _agg(vals: List[float]) -> Dict[str, Any]:
        n = len(vals)
        wins = sum(1 for v in vals if v > 0)
        return {"n": n, "wins": wins, "wr_pct": round(wins / n * 100.0, 1) if n else 0.0, "net_pnl": round(sum(vals), 2)}

    all_before = gp_before + sc_before + dir_before
    all_after = gp_after_macro + sc_after_aligned + dir_after_cap2

    print(f"  [S4 Counterfactual N={len(closed_df)}] Before PnL=${sum(all_before):+.2f} -> Candidate Filters PnL=${sum(all_after):+.2f} (N={len(all_after)})", flush=True)

    return {
        "replay_29d_window": {"start": _ts_iso(rep_start_ms), "end_1h": _ts_iso(end_29d_1h), "end_15m": _ts_iso(end_29d_15), "covered_symbols_db": len(all_micro_syms)},
        "replay_29d_directional": replay_dir,
        "replay_29d_scalp": replay_scalp,
        "s4_live_flow_gate_log": flow_rows.to_dict("records"),
        "s4_live_open_count": int((pos_rows["status"] == "OPEN").sum()),
        "s4_live_closed_counterfactual": {
            "total_closed": int(len(closed_df)),
            "summary_before_vs_after": {
                "DIRECTIONAL": {"before": _agg(dir_before), "after_corr_cap2": _agg(dir_after_cap2)},
                "GOLDEN_POT": {"before": _agg(gp_before), "after_btc_macro_15pct": _agg(gp_after_macro)},
                "SCALP": {"before": _agg(sc_before), "after_counter_regime_block": _agg(sc_after_aligned)},
                "TOTAL": {"before": _agg(all_before), "after": _agg(all_after)},
            },
            "trades": audited_trades,
        },
    }


# =====================================================================
# RELATÓRIO MARKDOWN
# =====================================================================

def _load_calib_snapshots() -> Dict[str, Any]:
    out = {}
    for key, fname in [("before", "agent_calibration_before_20261113.json"), ("after", "agent_calibration_after_20261113.json")]:
        p = os.path.join(ROOT, "datasets", "backups", fname)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                out[key] = json.load(f)
    return out


def render_markdown_report(audit: Dict[str, Any]) -> str:
    b1 = audit["block1_ml_audit"]
    b2 = audit["block2_market_open"]
    b3 = audit["block3_holdout_ab"]
    b4 = audit["block4_replay_counterfactual"]
    cal = audit["calibration_snapshots"]

    lines: List[str] = [
        "# Auditoria Empírica S4, Saneamento do Banco e Calibração S4.2 (2026-11-13)",
        "",
        f"Gerado em `{audit['generated_at_utc']}` em `BTCUSDT`, `ETHUSDT`, `SOLUSDT` com **`BacktestEngine(..., pilot_parity=True)`** (Smart Trade 40/30/30, breakeven +0,04%, trailing pós-TP2, funding real 8h e slippage real) e ML estritamente out-of-sample (`PurgedTimeSeriesSplit` com embargo).",
        "",
        "## 1. Resumo Executivo das Correções já Aplicadas no Código & Banco (Steps 1–3)",
        "- **Poluição de testes isolada:** `47` linhas de `'Reset de teste'` removidas de `circuit_breaker_events` (backup em `datasets/backups/cb_test_pollution_backup_20261113.json`); `tests/test_performance_and_rollback.py` isolado com `monkeypatch.setattr(\"flask_interface.DB_PATH\", temp_db)`.",
        "- **Deduplicação canônica de velas:** `63.008` linhas sobrepostas não-`binance` removidas de `candles` (backup em `datasets/backups/candles_overlap_backup_20261113.sqlite`, `0` duplicatas remanescentes); loaders endurecidos com prioridade `binance` + `drop_duplicates`.",
        "- **Regime, Brier e TapeReading ao vivo corrigidos:** `core.py` propaga `regime` para `TradeSignal.metrics`; `MarketRegimeClassifier` usa janela `LIMIT 250` e detecta consolidação em tendência; `learning_tracker.py` persiste `brier_score` real; `run_paper_simulation.py` passa `allow_live_fallback=True` em `_flow_gate`.",
        "",
        "### Antes vs Depois da Matriz de Calibração no Banco (`agent_regime_weights` & `agent_regime_calibration`)",
        "| Snapshot | Regime | Agente | Peso | Acurácia | Amostras | ECE | Brier |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, snap in [("Antes (pré-backfill)", cal.get("before", {})), ("Depois (reparado)", cal.get("after", {}))]:
        cal_idx = {(r["regime"], r["agent_name"]): r for r in snap.get("agent_regime_calibration", [])}
        for w in sorted(snap.get("agent_regime_weights", []), key=lambda x: (x["regime"], x["agent_name"])):
            if int(w.get("total_samples", 0)) == 0:
                continue
            c = cal_idx.get((w["regime"], w["agent_name"]), {})
            lines.append(
                f"| {label} | `{w['regime']}` | `{w['agent_name']}` | {w['weight']:.3f} | "
                f"{w['accuracy']*100:.1f}% | {w['total_samples']} | {c.get('ece', 0.0):.4f} | {c.get('brier_score', 0.0):.4f} |"
            )

    lines += [
        "",
        "## 2. Bloco 1 — ML Walk-Forward Antes vs Depois da Deduplicação (`25.000` linhas 1h)",
        "| Ativo | Cenário | Linhas Brutas | Únicas | Duplicatas | Modelo | Acc Global | F1 Macro | Acc Sinais (>=0.50) | Cobertura |",
        "| :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for sym in MAJORS:
        sc = b1["contaminated_vs_clean"][sym]
        cs, cl = sc["contaminated_slice"], sc["clean_slice"]
        for mtype in ("lgbm", "rf"):
            mc = cs["models"][mtype]
            lines.append(
                f"| `{sym}` | Contaminado | {cs['raw_rows']} | {cs['unique_timestamps']} | {cs['duplicate_timestamps']} | "
                f"{mtype.upper()} | {mc['global_accuracy']*100:.2f}% | {mc['f1_macro']:.4f} | "
                f"{mc['emitted_signals_accuracy']*100:.2f}% ({mc['emitted_signals_count']}) | {mc['coverage_pct']:.2f}% |"
            )
            ml = cl["models"][mtype]
            lines.append(
                f"| `{sym}` | Limpo Dedup | {cl['rows']} | {cl['unique_timestamps']} | 0 | "
                f"{mtype.upper()} | {ml['global_accuracy']*100:.2f}% | {ml['f1_macro']:.4f} | "
                f"{ml['emitted_signals_accuracy']*100:.2f}% ({ml['emitted_signals_count']}) | {ml['coverage_pct']:.2f}% |"
            )

    lines += [
        "",
        "### Auditoria dos Modelos `.joblib` em Produção",
        "| Artefato | Criado em | TP/SL | Amostras OOS | Features | Acc Global | F1 Macro | Acc Emissão (>=0.50) |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k, ma in b1["models_audit"].items():
        lines.append(
            f"| `{ma['path']}` | {ma['created_at']} | {ma['multiple_tp']}/{ma['multiple_sl']} | "
            f"{ma['samples_evaluated']} | {ma['n_features']} | {ma['global_accuracy']*100:.2f}% | "
            f"{ma['f1_macro']:.4f} | {ma['emitted_signals_accuracy']*100:.2f}% ({ma['emitted_signals_count']}) |"
        )

    prof = b2["hourly_brt_profile_174k"]
    hr = prof["by_hour_brt"]
    lines += [
        "",
        f"## 3. Bloco 2 — Perfil 21:00 BRT (`{prof['total_clean_bars']:,}` velas 1h limpas) & Estratégias em `pilot_parity=True`",
        "| Hora (BRT) | Velas | Range/ATR | Vol Ratio | Reversão 4h (Mov >=0.3%) |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for hkey in ("11h", "18h", "20h", "21h", "22h", "05h"):
        if hkey in hr:
            x = hr[hkey]
            lines.append(f"| `{hkey}` | {x['n_bars']} | {x['mean_range_atr']:.3f} | {x['mean_vol_ratio']:.3f} | {x['rev_4h_big_move_pct']:.2f}% |")

    lines += [
        "",
        "### Reavaliação Consolidada & Por Ativo (Somente Folds Estritamente OOS, Smart Trade Ladder + Funding Real)",
        "| Estratégia | TF | Port. Trades | Port. WR | Port. PF | Port. E[R] | Port. PnL ($10k) | Max DD | BTC WR / PF / PnL | ETH WR / PF / PnL | SOL WR / PF / PnL |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |",
    ]
    for tf_lbl, blk in [("1h", b2["strategies_1h_oos"]), ("15m", b2["strategies_15m_oos"])]:
        for sname, pm in blk["consolidated_portfolio"].items():
            pa = blk["per_asset"][sname]
            def _fmt(s):
                d = pa[s]
                return f"{d['win_rate_pct']:.1f}% / {d['profit_factor']:.2f} / ${d['net_pnl_usd']:+.0f}"
            lines.append(
                f"| {sname} | {tf_lbl} | {pm['trades']} | {pm['win_rate_pct']:.1f}% | {pm['profit_factor']:.2f} | "
                f"{pm['expectancy_r']:+.3f}R | ${pm['net_pnl_usd']:+.2f} | {pm['max_dd_pct']:.2f}% | "
                f"{_fmt('BTCUSDT')} | {_fmt('ETHUSDT')} | {_fmt('SOLUSDT')} |"
            )

    lines += [
        "",
        "## 4. Bloco 3 — A/B de Calibrações S4.2 em Holdout OOS (`DIRECTIONAL 1h` e `SCALP 15m`)",
        f"- Janela Holdout 1h (`60% treino + 10d embargo`): `{b3['holdout_window']['start']}` a `{b3['holdout_window']['end']}` (ensemble ML treinado só antes de `{b3['holdout_window']['split_ts']}`).",
        f"- Janela 15m OOS disponível: `{b3['scalp_window']['start']}` a `{b3['scalp_window']['end']}`.",
        "",
        "| Variante | Trades | WR% | PF | E[R] (IC 95%) | PnL ($10k) | Ret% | Max DD | Sharpe Diário | DSR |",
        "| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for grp in (b3["directional_ab"], b3["scalp_ab"]):
        for vname, m in grp.items():
            ci = m["expectancy_r_ci95"]
            lines.append(
                f"| `{vname}` | {m['trades']} | {m['win_rate_pct']:.1f}% | {m['profit_factor']:.3f} | "
                f"{m['expectancy_r']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | ${m['net_pnl_usd']:+.2f} | "
                f"{m['net_pnl_pct']:+.2f}% | {m['max_dd_pct']:.2f}% | {m['sharpe_daily_ann']:+.3f} | {m['dsr']:.4f} |"
            )

    cf = b4["s4_live_closed_counterfactual"]
    sb = cf["summary_before_vs_after"]
    lines += [
        "",
        f"## 5. Bloco 4 — Replay 29d em `microstructure.db` (`{b4['replay_29d_window']['covered_symbols_db']}` símbolos) & Contrafactual S4 (`N={cf['total_closed']}`)",
        "### 5.1 Replay 29 dias (`2026-10-15 .. 2026-11-13`)",
        "| Cenário Replay 29d | Trades | WR% | PF | E[R] | PnL ($10k) | Max DD | Gate Tape (Blk/Pass/Unav) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for grp in (b4["replay_29d_directional"], b4["replay_29d_scalp"]):
        for k, m in grp.items():
            tg = m.get("tape_gate_stats")
            tg_s = f"{tg['blocked']}/{tg['passed']}/{tg['unavailable_passed']}" if tg else "Sem Gate"
            lines.append(
                f"| `{k}` | {m['trades']} | {m['win_rate_pct']:.1f}% | {m['profit_factor']:.3f} | "
                f"{m['expectancy_r']:+.4f}R | ${m['net_pnl_usd']:+.2f} | {m['max_dd_pct']:.2f}% | {tg_s} |"
            )

    lines += [
        "",
        f"### 5.2 Contrafactual S4 Ao Vivo (`N={cf['total_closed']}` trades fechados desde `2026-11-09 00:00 UTC`; `{b4['s4_live_open_count']}` posições abertas intactas)",
        "| Estratégia | Antes (N / WR / PnL) | Filtro S4.2 Candidato (N / WR / PnL) |",
        "| :--- | :--- | :--- |",
    ]
    for st_name, sub in sb.items():
        b_k = "before"
        a_k = [k for k in sub.keys() if k != "before"][0]
        bv, av = sub[b_k], sub[a_k]
        lines.append(
            f"| `{st_name}` | {bv['n']} / {bv['wr_pct']:.1f}% / ${bv['net_pnl']:+.2f} | "
            f"{av['n']} / {av['wr_pct']:.1f}% / ${av['net_pnl']:+.2f} ({a_k}) |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    t_start = time.time()
    n_prior = get_prior_trials_count()
    funding = load_funding_map(MAJORS)

    b1, cached_clean_1h = run_block1_ml_audit()
    b2, data_15m_full = run_block2_market_open(cached_clean_1h, funding, n_prior_trials=n_prior)
    b3, block3_cache = run_block3_holdout_ab(funding, data_15m_full, n_prior_trials=n_prior)
    b4 = run_block4_replay_and_counterfactual(funding, data_15m_full, block3_cache)

    audit = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - t_start, 2),
        "n_prior_trials_log": n_prior,
        "calibration_snapshots": _load_calib_snapshots(),
        "block1_ml_audit": b1,
        "block2_market_open": b2,
        "block3_holdout_ab": b3,
        "block4_replay_counterfactual": b4,
    }

    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(render_markdown_report(audit))

    print(f"\n[OK] Auditoria empírica concluída em {time.time() - t_start:.1f}s.", flush=True)
    print(f"  JSON: {REPORT_JSON}", flush=True)
    print(f"  MD  : {REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
