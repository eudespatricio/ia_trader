#!/usr/bin/env python3
"""
Treina e persiste o ensemble dedicado de 15m para o SCALP universal (BTCUSDT, ETHUSDT, SOLUSDT):
  - models/lgbm_15m_v1.joblib
  - models/rf_15m_v1.joblib

Usa conexões read-only em datasets/market_cache.db, deduplicação canônica (prioridade `binance`),
features estacionárias consistentes com FeatureEngineer.compute_all_features e validação PurgedTimeSeriesSplit.
"""

import os
import sqlite3
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.feature_engineering import FeatureEngineer  # noqa: E402
from model_trainer import ModelTrainer  # noqa: E402

DB_PATH = os.path.join(ROOT, "datasets", "market_cache.db")
MODELS_DIR = os.path.join(ROOT, "models")
MAJORS: List[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
MAX_CANDLES_PER_SYMBOL = 25_000

NON_STATIONARY_COLS = frozenset({
    "timestamp", "open", "high", "low", "close", "volume", "quote_volume", "created_at",
    "exchange", "symbol", "timeframe",
    "sma_20", "sma_50", "sma_200", "ema_9", "ema_21", "bb_high", "bb_mid", "bb_low",
    "vwap", "obv", "vol_sma_20", "atr_14", "nearest_bsl", "nearest_ssl",
    "ob_bull_top", "ob_bull_bottom", "ob_bear_top", "ob_bear_bottom",
    "fvg_bull_top", "fvg_bull_bottom", "fvg_bear_top", "fvg_bear_bottom",
})


def load_clean_15m_candles(db_path: str, symbol: str, limit: int = MAX_CANDLES_PER_SYMBOL) -> pd.DataFrame:
    """Carrega velas 15m deduplicadas (priorizando `binance`) e descarta a última barra em formação."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30.0) as conn:
        df = pd.read_sql_query(
            """
            SELECT exchange, timestamp, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND timeframe = '15m'
            ORDER BY timestamp ASC, CASE WHEN exchange = 'binance' THEN 0 ELSE 1 END ASC
            """,
            conn,
            params=(symbol,),
        )
    if df.empty:
        raise RuntimeError(f"Sem candles 15m para {symbol} em {db_path}")
    df = df.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp").reset_index(drop=True)
    if len(df) > 1:
        df = df.iloc[:-1].reset_index(drop=True)
    if limit > 0 and len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)
    return df.drop(columns=["exchange"]).reset_index(drop=True)


def select_stationary_features(df_feat: pd.DataFrame) -> List[str]:
    """Seleciona colunas numéricas estacionárias com baixa taxa de NaN."""
    num_cols = df_feat.select_dtypes(include=[np.number]).columns
    return [
        c for c in num_cols
        if c not in NON_STATIONARY_COLS
        and (c in {"dist_resistance_pct", "dist_support_pct"} or float(df_feat[c].notna().mean()) > 0.95)
    ]


def build_pooled_15m_dataset(db_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepara e empilha datasets rotulados por ativo ordenados cronologicamente."""
    feats_by_sym: Dict[str, pd.DataFrame] = {}
    cols_sets: List[set] = []

    for sym in MAJORS:
        raw = load_clean_15m_candles(db_path, sym, limit=MAX_CANDLES_PER_SYMBOL)
        df_feat = FeatureEngineer.compute_all_features(raw, drop_na=False)
        valid_sma = np.isfinite(df_feat["sma_200"].astype(float)) & (df_feat["sma_200"].astype(float) > 0.0)
        df_feat = df_feat[valid_sma].reset_index(drop=True)
        feats_by_sym[sym] = df_feat
        cols_sets.append(set(select_stationary_features(df_feat)))
        print(f"  [15m prep] {sym}: {len(df_feat)} barras validas pos-SMA200", flush=True)

    feature_cols = sorted(set.intersection(*cols_sets))
    if not feature_cols:
        raise RuntimeError("Nenhuma feature estacionaria em comum encontrada para 15m.")

    prep_trainer = ModelTrainer(
        model_type="lgbm",
        horizon=5,
        multiple_tp=1.2,
        multiple_sl=1.2,
        confidence_threshold=0.50,
        n_jobs=8,
        models_dir=MODELS_DIR,
    )

    x_parts: List[pd.DataFrame] = []
    y_parts: List[pd.Series] = []
    ts_parts: List[np.ndarray] = []

    for sym in MAJORS:
        df_feat = feats_by_sym[sym]
        X_sym, y_sym = prep_trainer.prepare_dataset(df_feat, feature_columns=feature_cols)
        X_sym = X_sym.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ts_sym = df_feat.loc[X_sym.index, "timestamp"].to_numpy(dtype=np.int64)
        x_parts.append(X_sym)
        y_parts.append(y_sym)
        ts_parts.append(ts_sym)

    X_pool = pd.concat(x_parts, axis=0, ignore_index=True)
    y_pool = pd.concat(y_parts, axis=0, ignore_index=True)
    ts_all = np.concatenate(ts_parts)

    order = np.argsort(ts_all, kind="mergesort")
    X_pool = X_pool.iloc[order].reset_index(drop=True)
    y_pool = y_pool.iloc[order].reset_index(drop=True)
    return X_pool, y_pool, feature_cols


def train_and_save_all(db_path: str = DB_PATH) -> Dict[str, Dict[str, float]]:
    """Treina LGBM e RF 15m e salva os artefatos .joblib validando predict_proba."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    X_pool, y_pool, feature_cols = build_pooled_15m_dataset(db_path)
    print(f"Dataset 15m pooled: N={len(X_pool)} amostras | {len(feature_cols)} features", flush=True)

    summary: Dict[str, Dict[str, float]] = {}
    for mtype in ("lgbm", "rf"):
        trainer = ModelTrainer(
            model_type=mtype,
            horizon=5,
            multiple_tp=1.2,
            multiple_sl=1.2,
            confidence_threshold=0.50,
            n_jobs=8,
            models_dir=MODELS_DIR,
        )
        save_name = f"{mtype}_15m_v1.joblib"
        payload = trainer.train_full_model(X_pool, y_pool, save_name=save_name)
        payload["timeframe"] = "15m"
        payload["symbols"] = list(MAJORS)
        filepath = os.path.join(MODELS_DIR, save_name)
        joblib.dump(payload, filepath, compress=3)

        loaded = joblib.load(filepath)
        proba = loaded["model"].predict_proba(X_pool.iloc[:10])
        if proba.shape != (10, 3) or not np.all(np.isfinite(proba)):
            raise RuntimeError(f"predict_proba invalido para {filepath}: shape={proba.shape}")

        m = loaded["metrics"]
        summary[mtype] = {
            "global_accuracy": float(m["global_accuracy"]),
            "emitted_signals_accuracy": float(m["emitted_signals_accuracy"]),
            "emitted_signals_count": int(m["emitted_signals_count"]),
            "samples_evaluated": int(m["samples_evaluated"]),
        }
        print(
            f"  [OK] {save_name}: GlobalAcc={m['global_accuracy']:.2%} | "
            f"EmittedAcc={m['emitted_signals_accuracy']:.2%} ({m['emitted_signals_count']} sinais) | "
            f"proba_shape={proba.shape}",
            flush=True,
        )
    return summary


if __name__ == "__main__":
    train_and_save_all()
