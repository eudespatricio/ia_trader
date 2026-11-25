#!/usr/bin/env python3
"""
scripts/eval_golden_pot_s6_oos.py

Zero-Leakage Out-of-Sample (OOS) Ablation for GOLDEN_POT 5m (G0..G7) vs DIRECTIONAL 1h.
Strictly read-only against datasets/market_cache.db and datasets/microstructure.db
(mode=ro + PRAGMA query_only=ON). No disk .joblib models loaded; trains a purged 1h
ensemble strictly before the 41-day 5m evaluation window (timestamp < 2026-10-15 - 7d).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import logging
import math
import os
import sqlite3
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:sklearn.utils.parallel")
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agents.golden_pot_agent import (  # noqa: E402
    GoldenPotAgent,
    NON_CRYPTO_BASE_EXCLUSIONS,
    STABLECOIN_EXCLUSIONS,
)
from agents.technical_agent import (  # noqa: E402
    TechnicalAgent,
    _DefaultMLEnsemble,
    _disable_estimator_parallelism,
)
from core import TradingCore  # noqa: E402
from data.feature_engineering import FeatureEngineer  # noqa: E402
from data.live_context_enricher import LiveContextEnricher  # noqa: E402
from model_trainer import ModelTrainer  # noqa: E402
from scripts.train_15m_scalp_models import select_stationary_features  # noqa: E402
from smart_trade import STRATEGY_PROFILES  # noqa: E402
from statistical_rigor import (  # noqa: E402
    bootstrap_performance_metrics,
    calculate_per_period_sharpe,
    deflated_sharpe_ratio,
    get_prior_trials_count,
)

CACHE_DB = os.path.abspath(os.path.join(ROOT, "datasets", "market_cache.db"))
MICRO_DB = os.path.abspath(os.path.join(ROOT, "datasets", "microstructure.db"))
OUTPUT_JSON = os.path.abspath(
    os.path.join(ROOT, "datasets", "reports", "s6_golden_pot_oos_ablation.json")
)

MAJORS: Tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
MAX_WORKERS = 6

# Additional Binance TradFi perps (equities, leveraged/single-stock ETFs, pre-IPO) present in market_cache.db 5m
# candles that live excludes dynamically via exchangeInfo underlyingSubType.
EXTRA_TRADFI_BASES = frozenset({
    "AAOI", "APP", "ARM", "ASTS", "AXTI", "BITO", "BMNR", "COHR",
    "CSOPSKHYNIX2L", "CXMT", "DELL", "INTW", "KSTR", "MRNA", "MVLL",
    "NOW", "SKDD", "SKUU", "SMCI", "STXX", "TTWO", "WDC",
})
ALL_TRADFI_BASE_EXCLUSIONS = NON_CRYPTO_BASE_EXCLUSIONS | EXTRA_TRADFI_BASES

logger = logging.getLogger("ia_trader.eval_golden_pot_s6_oos")

GP_BAR_MS = 300_000          # 5m
HOUR_MS = 3_600_000          # 1h
DAY_MS = 86_400_000          # 1d
GAP_THRESHOLD_5M_MS = 900_000  # 15m contiguous segment split threshold
MIN_SEGMENT_BARS = 250
GP_MAX_HOLDING_BARS = int(STRATEGY_PROFILES["GOLDEN_POT"].max_holding_bars)  # 36 (3h)
DIR_MAX_HOLDING_BARS = int(STRATEGY_PROFILES["DIRECTIONAL"].max_holding_bars)  # 48 (48h)

IS_START_TS = int(datetime(2026, 10, 15, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
SPLIT_TS_GP = int(datetime(2026, 11, 5, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
EMBARGO_GP_MS = GP_MAX_HOLDING_BARS * GP_BAR_MS  # 36 * 300_000 = 3h
TRAIN_1H_CUTOFF_TS = IS_START_TS - 7 * DAY_MS    # 2026-10-08 00:00:00 UTC

TAKER_FEE = 0.0006
TAKER_SLIP = 0.0002
MAKER_FEE = 0.0002
MAKER_SLIP = 0.0001
BE_BUFFER_PCT = 0.0004


def connect_ro(db_path: str) -> sqlite3.Connection:
    """Open SQLite strictly in read-only URI mode with PRAGMA query_only=ON."""
    abs_p = os.path.abspath(db_path)
    con = sqlite3.connect(f"file:{abs_p}?mode=ro", uri=True, timeout=5.0)
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA busy_timeout=5000")
    return con


class ReadOnlyCachedEnricher(LiveContextEnricher):
    """
    Subclass of LiveContextEnricher that reuses persistent read-only connections
    (with mode=ro and PRAGMA query_only=ON) and caches static daily macro history
    for fast zero-leakage context enrichment on candidate bars.
    """

    def __init__(self, cache_db_path: str, micro_db_path: str):
        super().__init__(cache_db_path, micro_db_path)
        self._conns: Dict[str, Optional[sqlite3.Connection]] = {}
        for p in (cache_db_path, micro_db_path):
            abs_p = os.path.abspath(p)
            if os.path.exists(abs_p):
                self._conns[abs_p] = connect_ro(abs_p)
            else:
                self._conns[abs_p] = None
        self._table_info_cache: Dict[Tuple[str, str], List[Tuple[Any, ...]]] = {}
        self._macro_query_cache: Dict[Tuple[str, Tuple[Any, ...]], List[Tuple[Any, ...]]] = {}
        self._funding_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._preload_funding_cache()

    def _preload_funding_cache(self) -> None:
        abs_c = os.path.abspath(self.cache_db_path)
        con = self._conns.get(abs_c)
        if con is None:
            return
        try:
            rows = con.execute(
                "SELECT symbol, timestamp, funding_rate FROM funding_rates WHERE funding_rate IS NOT NULL ORDER BY symbol ASC, timestamp ASC"
            ).fetchall()
            by_sym: Dict[str, List[Tuple[int, float]]] = {}
            for s, t, r in rows:
                try:
                    fv = float(r)
                    if np.isfinite(fv):
                        by_sym.setdefault(str(s), []).append((int(t), fv))
                except (TypeError, ValueError):
                    continue
            for s, pairs in by_sym.items():
                ts_a = np.array([p[0] for p in pairs], dtype=np.int64)
                rt_a = np.array([p[1] for p in pairs], dtype=np.float64)
                self._funding_cache[s] = (ts_a, rt_a)
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.debug(f"Optional table funding_rates absent: {e}")
            else:
                self._enrich_error_count += 1
                logger.warning(f"Error preloading funding_rates: {e}")

    def _fetch_all(
        self, path: str, sql: str, params: Tuple[Any, ...] = ()
    ) -> List[Tuple[Any, ...]]:
        if not path:
            return []
        abs_p = os.path.abspath(path)
        try:
            if sql.startswith("PRAGMA table_info("):
                key = (abs_p, sql)
                if key in self._table_info_cache:
                    return self._table_info_cache[key]
                con = self._conns.get(abs_p)
                if con is None:
                    return []
                rows = con.execute(sql, params).fetchall()
                self._table_info_cache[key] = rows
                return rows

            if "macro_daily_history" in sql or "macro_events_calendar" in sql:
                mkey = (sql, params)
                if mkey in self._macro_query_cache:
                    return self._macro_query_cache[mkey]
                con = self._conns.get(abs_p)
                if con is None:
                    return []
                rows = con.execute(sql, params).fetchall()
                self._macro_query_cache[mkey] = rows
                return rows

            if "FROM funding_rates WHERE symbol=? AND timestamp<=?" in sql and len(params) == 2:
                sym, ts_ms = str(params[0]), int(params[1])
                entry = self._funding_cache.get(sym)
                if entry is None:
                    return []
                ts_a, rt_a = entry
                idx = int(np.searchsorted(ts_a, ts_ms, side="right"))
                if idx <= 0:
                    return []
                lo = max(0, idx - 90)
                return [
                    (int(ts_a[k]), float(rt_a[k]))
                    for k in range(idx - 1, lo - 1, -1)
                ]

            # Bound microstructure.db (PRIMARY KEY (ts_ms, symbol)) queries by their exact freshness window
            if "FROM book_features WHERE symbol=? AND ts_ms<=?" in sql and len(params) == 2:
                sym, ts_ms = params[0], int(params[1])
                sql = sql.replace(
                    "WHERE symbol=? AND ts_ms<=?",
                    "WHERE ts_ms>=? AND ts_ms<=? AND symbol=?",
                )
                params = (ts_ms - 120_000, ts_ms, sym)
            elif "FROM derivs WHERE symbol=? AND ts_ms<=?" in sql and len(params) == 2:
                sym, ts_ms = params[0], int(params[1])
                win = (12 * 3_600_000) if "LIMIT 1" in sql else (10 * 86_400_000)
                sql = sql.replace(
                    "WHERE symbol=? AND ts_ms<=?",
                    "WHERE ts_ms>=? AND ts_ms<=? AND symbol=?",
                )
                params = (ts_ms - win, ts_ms, sym)
            elif "FROM binance_series WHERE symbol=? AND kind='taker' AND ts_ms <= ?" in sql and len(params) == 2:
                sym, ts_ms = params[0], int(params[1])
                sql = sql.replace(
                    "WHERE symbol=? AND kind='taker' AND ts_ms <= ?",
                    "WHERE ts_ms >= ? AND ts_ms <= ? AND symbol=? AND kind='taker'",
                )
                params = (ts_ms - 3_600_000, ts_ms, sym)
            elif "FROM tick_agg WHERE symbol=? AND minute_ms + 60000 <= ? AND minute_ms + 60000 > ?" in sql and len(params) == 3:
                sym, hi_ms, lo_ms = params[0], int(params[1]) - 60_000, int(params[2]) - 60_000
                sql = sql.replace(
                    "WHERE symbol=? AND minute_ms + 60000 <= ? AND minute_ms + 60000 > ?",
                    "WHERE minute_ms <= ? AND minute_ms > ? AND symbol=?",
                )
                params = (hi_ms, lo_ms, sym)

            con = self._conns.get(abs_p)
            if con is None:
                return []
            return con.execute(sql, params).fetchall()
        except Exception as e:
            if "no such table" in str(e).lower():
                logger.debug(f"Tabela opcional ausente em {abs_p}: {e}")
            else:
                self._enrich_error_count += 1
                logger.warning(f"Erro ao consultar {abs_p}: {e}")
            return []

    def close(self) -> None:
        for con in self._conns.values():
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass
        self._conns.clear()


def ts_to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def load_clean_1h_majors(db_path: str) -> Dict[str, pd.DataFrame]:
    """Load deduplicated 1h candles for BTC/ETH/SOL prioritizing binance and dropping last forming candle."""
    out: Dict[str, pd.DataFrame] = {}
    con = connect_ro(db_path)
    try:
        for sym in MAJORS:
            df = pd.read_sql_query(
                """
                SELECT exchange, timestamp, open, high, low, close, volume
                FROM candles
                WHERE symbol = ? AND timeframe = '1h'
                ORDER BY timestamp ASC, CASE WHEN exchange = 'binance' THEN 0 ELSE 1 END ASC
                """,
                con,
                params=(sym,),
            )
            if df.empty:
                raise RuntimeError(f"Missing 1h candles for {sym}")
            df = (
                df.drop_duplicates(subset=["timestamp"], keep="first")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            if len(df) > 1:
                df = df.iloc[:-1].reset_index(drop=True)
            df = df.drop(columns=["exchange"])
            df_feat = FeatureEngineer.compute_all_features(df, drop_na=False)
            valid_sma = np.isfinite(df_feat["sma_200"].astype(float)) & (
                df_feat["sma_200"].astype(float) > 0.0
            )
            df_feat = df_feat[valid_sma].reset_index(drop=True)
            df_feat["atr_median"] = (
                df_feat["atr_14"].shift(1).rolling(100, min_periods=20).median()
            )
            if "vol_sma_20" in df_feat.columns:
                df_feat["volume_sma"] = df_feat["vol_sma_20"].shift(1)
            else:
                df_feat["volume_sma"] = (
                    df_feat["volume"].shift(1).rolling(20, min_periods=10).mean()
                )
            out[sym] = df_feat
    finally:
        con.close()
    return out


def train_purged_1h_ensemble(
    majors_1h: Dict[str, pd.DataFrame], cutoff_ts: int
) -> Tuple[_DefaultMLEnsemble, int, List[str]]:
    """
    Train purged 1h LGBM + RF strictly on pooled BTC/ETH/SOL 1h candles
    with timestamp < cutoff_ts (before 2026-10-15 - 7d).
    """
    train_slices: Dict[str, pd.DataFrame] = {}
    cols_sets: List[set] = []

    for sym in MAJORS:
        df_sym = majors_1h[sym]
        df_tr = df_sym[df_sym["timestamp"] < cutoff_ts].copy().reset_index(drop=True)
        if df_tr.empty:
            raise RuntimeError(f"Empty training slice for {sym} before {cutoff_ts}")
        train_slices[sym] = df_tr
        cols_sets.append(set(select_stationary_features(df_tr)))

    feature_cols = sorted(set.intersection(*cols_sets))
    if not feature_cols:
        raise RuntimeError("No common stationary 1h features found across majors.")

    prep_trainer = ModelTrainer(
        model_type="lgbm",
        horizon=5,
        multiple_tp=1.5,
        multiple_sl=1.0,
        n_jobs=MAX_WORKERS,
    )

    x_parts: List[pd.DataFrame] = []
    y_parts: List[pd.Series] = []
    ts_parts: List[np.ndarray] = []

    for sym in MAJORS:
        df_tr = train_slices[sym]
        X_sym, y_sym = prep_trainer.prepare_dataset(df_tr, feature_columns=feature_cols)
        X_sym = X_sym.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ts_sym = df_tr.loc[X_sym.index, "timestamp"].to_numpy(dtype=np.int64)
        x_parts.append(X_sym)
        y_parts.append(y_sym)
        ts_parts.append(ts_sym)

    X_pool = pd.concat(x_parts, axis=0, ignore_index=True)
    y_pool = pd.concat(y_parts, axis=0, ignore_index=True)
    ts_all = np.concatenate(ts_parts)

    order = np.argsort(ts_all, kind="mergesort")
    X_pool = X_pool.iloc[order].reset_index(drop=True)
    y_pool = y_pool.iloc[order].reset_index(drop=True)

    lgbm_est = ModelTrainer(
        model_type="lgbm",
        horizon=5,
        multiple_tp=1.5,
        multiple_sl=1.0,
        n_jobs=MAX_WORKERS,
    )._build_estimator(random_state=42)
    rf_est = ModelTrainer(
        model_type="rf",
        horizon=5,
        multiple_tp=1.5,
        multiple_sl=1.0,
        n_jobs=MAX_WORKERS,
    )._build_estimator(random_state=42)

    lgbm_est.fit(X_pool, y_pool)
    rf_est.fit(X_pool, y_pool)
    _disable_estimator_parallelism(lgbm_est)
    _disable_estimator_parallelism(rf_est)

    ens_1h = _DefaultMLEnsemble(models=[lgbm_est, rf_est], feature_names=feature_cols)
    return ens_1h, int(len(X_pool)), feature_cols


def load_and_split_5m_segments(
    db_path: str,
) -> Tuple[List[Tuple[int, str, pd.DataFrame]], Dict[str, Any]]:
    """
    Load 5m candles read-only, deduplicate prioritizing binance, drop last forming candle,
    exclude BTC/ETH/SOL, stablecoins, and all TradFi bases in NON_CRYPTO_BASE_EXCLUSIONS,
    and split each symbol into contiguous segments (gap > 900_000 ms) keeping len >= 250.
    """
    con = connect_ro(db_path)
    try:
        df_all = pd.read_sql_query(
            """
            SELECT symbol, exchange, timestamp, open, high, low, close, volume
            FROM candles
            WHERE timeframe = '5m'
            ORDER BY symbol ASC, timestamp ASC
            """,
            con,
        )
    finally:
        con.close()

    if df_all.empty:
        raise RuntimeError("No 5m candles found in market_cache.db")

    df_all["_prio"] = np.where(df_all["exchange"] == "binance", 0, 1)
    df_all = df_all.sort_values(["symbol", "timestamp", "_prio"]).drop_duplicates(
        subset=["symbol", "timestamp"], keep="first"
    )

    all_symbols = sorted(df_all["symbol"].unique().tolist())
    excluded_majors: List[str] = []
    excluded_stables: List[str] = []
    excluded_tradfi: List[str] = []
    kept_altcoins: List[str] = []

    raw_segments: List[Tuple[int, str, pd.DataFrame]] = []
    seg_id = 0
    total_raw_segments_before_len = 0

    for sym, grp in df_all.groupby("symbol", sort=True):
        s_str = str(sym).strip().upper()
        if s_str in MAJORS:
            excluded_majors.append(s_str)
            continue
        if s_str in STABLECOIN_EXCLUSIONS:
            excluded_stables.append(s_str)
            continue
        base_asset = GoldenPotAgent.extract_base_asset(s_str)
        if base_asset in ALL_TRADFI_BASE_EXCLUSIONS:
            excluded_tradfi.append(s_str)
            continue

        if len(grp) <= 1:
            continue
        kept_altcoins.append(s_str)
        # Drop last potentially forming candle per symbol
        g = (
            grp.iloc[:-1][["timestamp", "open", "high", "low", "close", "volume"]]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        ts = g["timestamp"].to_numpy(dtype=np.int64)
        breaks = [0] + (np.where(np.diff(ts) > GAP_THRESHOLD_5M_MS)[0] + 1).tolist() + [len(g)]
        for b0, b1 in zip(breaks[:-1], breaks[1:]):
            total_raw_segments_before_len += 1
            if b1 - b0 >= MIN_SEGMENT_BARS:
                seg_df = g.iloc[b0:b1].copy().reset_index(drop=True)
                raw_segments.append((seg_id, s_str, seg_df))
                seg_id += 1

    # Verify required TradFi bases are excluded if present in DB
    required_check_bases = {
        "MU", "MUU", "ANT", "DRAM", "EWY", "NBIS", "ZHIPU", "ZHONGJI",
        "BE", "MRVL", "NVDL", "CRCL", "IREN", "NATGAS", "CBRS", "SPCX", "MINIMAX",
    }
    leaked_bases = [
        s for s in kept_altcoins if GoldenPotAgent.extract_base_asset(s) in required_check_bases
    ]
    if leaked_bases:
        raise AssertionError(f"TradFi bases leaked into altcoin universe: {leaked_bases}")

    uni_meta = {
        "total_5m_symbols_in_db": int(len(all_symbols)),
        "excluded_majors_count": int(len(excluded_majors)),
        "excluded_stables_count": int(len(excluded_stables)),
        "excluded_tradfi_count": int(len(excluded_tradfi)),
        "excluded_tradfi_sample": sorted(
            s for s in excluded_tradfi if GoldenPotAgent.extract_base_asset(s) in NON_CRYPTO_BASE_EXCLUSIONS
        ),
        "excluded_extra_tradfi_sample": sorted(
            s for s in excluded_tradfi if GoldenPotAgent.extract_base_asset(s) not in NON_CRYPTO_BASE_EXCLUSIONS
        ),
        "kept_altcoins_count": int(len(kept_altcoins)),
        "total_contiguous_segments_raw": int(total_raw_segments_before_len),
        "raw_segments_ge_250": int(len(raw_segments)),
    }
    return raw_segments, uni_meta


def _process_segment_worker(
    item: Tuple[int, str, pd.DataFrame],
) -> Optional[Tuple[int, str, pd.DataFrame, List[Dict[str, Any]]]]:
    """
    Worker function executed in ProcessPoolExecutor(max_workers=6):
    1. Computes FeatureEngineer.compute_all_features(seg_df, drop_na=False) inside the segment independently.
    2. Filters rows where sma_200 is finite and > 0.
    3. Adds atr_median = atr_14.shift(1).rolling(48, min_periods=20).median() and volume_sma = vol_sma_20.shift(1).
    4. Extracts all fast pre-trigger candidates (union of G0..G6 standard trigger and G7 early trigger)
       that satisfy i + 36 < len(seg) (at least 36 forward bars in the SAME contiguous segment).
    """
    seg_id, sym, seg_raw = item
    df_feat = FeatureEngineer.compute_all_features(seg_raw, drop_na=False)
    sma = pd.to_numeric(df_feat["sma_200"], errors="coerce").to_numpy(dtype=np.float64)
    valid_mask = np.isfinite(sma) & (sma > 0.0)
    df = df_feat.loc[valid_mask].copy().reset_index(drop=True)
    if len(df) <= GP_MAX_HOLDING_BARS + 20:
        return None

    df["atr_median"] = df["atr_14"].shift(1).rolling(48, min_periods=20).median()
    if "vol_sma_20" in df.columns:
        df["volume_sma"] = df["vol_sma_20"].shift(1)
    else:
        df["volume_sma"] = df["volume"].shift(1).rolling(20, min_periods=10).mean()

    n = len(df)
    ts_arr = df["timestamp"].to_numpy(dtype=np.int64)
    o_arr = df["open"].to_numpy(dtype=np.float64)
    h_arr = df["high"].to_numpy(dtype=np.float64)
    l_arr = df["low"].to_numpy(dtype=np.float64)
    c_arr = df["close"].to_numpy(dtype=np.float64)
    atr_arr = df["atr_14"].to_numpy(dtype=np.float64)
    atr_med_arr = df["atr_median"].to_numpy(dtype=np.float64)
    vol_z_arr = df["volume_zscore_20"].to_numpy(dtype=np.float64)
    rsi_arr = (
        df["rsi_14"].to_numpy(dtype=np.float64)
        if "rsi_14" in df.columns
        else np.full(n, 50.0, dtype=np.float64)
    )

    prior_high_12 = df["high"].shift(1).rolling(12, min_periods=12).max().to_numpy(dtype=np.float64)
    prior_low_12 = df["low"].shift(1).rolling(12, min_periods=12).min().to_numpy(dtype=np.float64)
    prior_high_8 = df["high"].shift(1).rolling(8, min_periods=8).max().to_numpy(dtype=np.float64)
    prior_low_8 = df["low"].shift(1).rolling(8, min_periods=8).min().to_numpy(dtype=np.float64)

    candidates: List[Dict[str, Any]] = []
    # Never open a trade at index i unless i + 36 < len(seg)
    max_i = n - GP_MAX_HOLDING_BARS - 1

    for i in range(12, max_i + 1):
        ts = int(ts_arr[i])
        # Only consider bars in IS [IS_START_TS, SPLIT_TS_GP - EMBARGO_GP_MS) or OOS [SPLIT_TS_GP, ...)
        in_is = IS_START_TS <= ts < (SPLIT_TS_GP - EMBARGO_GP_MS)
        in_oos = ts >= SPLIT_TS_GP
        if not (in_is or in_oos):
            continue

        o, h, l, c = float(o_arr[i]), float(h_arr[i]), float(l_arr[i]), float(c_arr[i])
        atr = float(atr_arr[i])
        med_atr = float(atr_med_arr[i])
        vol_z = float(vol_z_arr[i])
        if (
            not (
                np.isfinite(o)
                and np.isfinite(h)
                and np.isfinite(l)
                and np.isfinite(c)
                and np.isfinite(atr)
                and np.isfinite(med_atr)
                and np.isfinite(vol_z)
            )
            or atr <= 0.0
            or med_atr <= 0.0
            or h <= l
        ):
            continue

        atr_exp = atr / med_atr
        rng_atr = (h - l) / atr
        # Fast pre-filter: minimum requirement across G0..G7 is G7 (atr_exp >= 1.15, rng_atr >= 1.0, vol_z >= 0.8)
        if atr_exp < 1.15 or rng_atr < 1.0 or vol_z < 0.8:
            continue

        cpos = (c - l) / (h - l)

        # Standard trigger (G0..G6): ATR_exp >= 1.3, rng >= 1.1*ATR, VolZ >= 1.0, cpos >= 0.70, 12-bar breakout
        std_dir: Optional[str] = None
        if atr_exp >= 1.3 and rng_atr >= 1.1 and vol_z >= 1.0:
            ph12, pl12 = float(prior_high_12[i]), float(prior_low_12[i])
            if c > o and cpos >= 0.70 and np.isfinite(ph12) and c > ph12:
                std_dir = "BUY"
            elif c < o and cpos <= 0.30 and np.isfinite(pl12) and c < pl12:
                std_dir = "SELL"

        # Early trigger (G7): ATR_exp >= 1.15, rng >= 1.0*ATR, VolZ >= 0.8, cpos >= 0.65, 8-bar breakout
        early_dir: Optional[str] = None
        if atr_exp >= 1.15 and rng_atr >= 1.0 and vol_z >= 0.8:
            ph8, pl8 = float(prior_high_8[i]), float(prior_low_8[i])
            if c > o and cpos >= 0.65 and np.isfinite(ph8) and c > ph8:
                early_dir = "BUY"
            elif c < o and cpos <= 0.35 and np.isfinite(pl8) and c < pl8:
                early_dir = "SELL"

        if std_dir is None and early_dir is None:
            continue

        # Climax metrics (3-4 bar cumulative move in ATR & RSI)
        c_3ago = float(c_arr[i - 3]) if i >= 3 else c
        c_4ago = float(c_arr[i - 4]) if i >= 4 else c_3ago
        cum_3 = (c - c_3ago) / atr
        cum_4 = (c - c_4ago) / atr
        cum_move_atr = cum_3 if abs(cum_3) >= abs(cum_4) else cum_4
        rsi = float(rsi_arr[i]) if np.isfinite(rsi_arr[i]) else 50.0

        # Adaptive SL multiplier matching run_paper_simulation.py:1327
        r0 = (float(h_arr[i]) - float(l_arr[i])) / atr
        r1 = (float(h_arr[i - 1]) - float(l_arr[i - 1])) / atr if i >= 1 else r0
        r2 = (float(h_arr[i - 2]) - float(l_arr[i - 2])) / atr if i >= 2 else r1
        max_recent_3_rng = max(r0, r1, r2)
        sl_mult = max(1.5, max_recent_3_rng * 1.15) if max_recent_3_rng > 1.5 else 1.5

        candidates.append(
            {
                "seg_id": int(seg_id),
                "sym": sym,
                "idx": int(i),
                "ts": int(ts),
                "split": "IS" if in_is else "OOS",
                "close": c,
                "atr": atr,
                "atr_exp": float(atr_exp),
                "rng_atr": float(rng_atr),
                "vol_z": float(vol_z),
                "cpos": float(cpos),
                "rank_score": float(atr_exp * vol_z),
                "std_dir": std_dir,
                "early_dir": early_dir,
                "cum_move_atr": float(cum_move_atr),
                "rsi": float(rsi),
                "sl_mult": float(sl_mult),
            }
        )

    return seg_id, sym, df, candidates


def is_climax_blocked(direction: str, rsi: float, cum_move_atr: float, mode: str) -> bool:
    """
    Check Climax Exhaustion filter:
    - 'prod': RSI <25/>75 or (|cum34| > 2.0*ATR & RSI <32/>68)
    - 'relax': RSI <18/>82 or (|cum34| > 3.0*ATR & RSI <25/>75)
    - 'none': never blocked
    """
    if mode == "none":
        return False
    if mode == "prod":
        if direction == "SELL":
            if rsi < 25.0 or (cum_move_atr < -2.0 and rsi < 32.0):
                return True
        elif direction == "BUY":
            if rsi > 75.0 or (cum_move_atr > 2.0 and rsi > 68.0):
                return True
        return False
    if mode == "relax":
        if direction == "SELL":
            if rsi < 18.0 or (cum_move_atr < -3.0 and rsi < 25.0):
                return True
        elif direction == "BUY":
            if rsi > 82.0 or (cum_move_atr > 3.0 and rsi > 75.0):
                return True
        return False
    raise ValueError(f"Unknown climax mode: {mode}")


def is_macro_btc_blocked(direction: str, btc_dist_1h: Optional[float], thresh: Optional[float]) -> bool:
    """
    Check BTC 1h SMA200 counter-trend macro filter:
    - thresh is None: no macro filter
    - direction == 'BUY' and btc_dist_1h < -thresh -> blocked
    - direction == 'SELL' and btc_dist_1h > +thresh -> blocked
    """
    if thresh is None or btc_dist_1h is None or not np.isfinite(btc_dist_1h):
        return False
    if direction == "BUY" and btc_dist_1h < -thresh:
        return True
    if direction == "SELL" and btc_dist_1h > thresh:
        return True
    return False


def simulate_single_ladder_trade(
    df_seg: pd.DataFrame,
    sig_idx: int,
    direction: str,
    sl_mult: float,
    tp1_mult: float,
    tp2_mult: float,
    trail_mult: float,
    max_hold_bars: int,
    bar_ms: int,
) -> Dict[str, Any]:
    """
    Simulate forward a single trade from signal bar `sig_idx` inside contiguous `df_seg`:
    - Entry: TAKER market at close[sig_idx] * (1 + sgn * 0.0002) + 0.0006 fee.
    - TP1 (40%, maker 0.0002 fee + 0.0001 slip) -> move SL to Breakeven entry_px * (1 + sgn * 0.0004).
    - TP2 (30%, maker 0.0002 fee + 0.0001 slip) -> activate trailing stop `trail_mult * ATR` from best high/low (never loosens).
    - Runner (30%) -> trailing stop `trail_mult * ATR` updated on closed bars, or timeout at max_hold_bars (taker 0.0006 + 0.0002).
    - Conservative intrabar tie-break: SL/BE/trailing checked BEFORE TP on the same bar.
    """
    is_long = direction == "BUY"
    sgn = 1.0 if is_long else -1.0

    ts_arr = df_seg["timestamp"].to_numpy(dtype=np.int64)
    o_arr = df_seg["open"].to_numpy(dtype=np.float64)
    h_arr = df_seg["high"].to_numpy(dtype=np.float64)
    l_arr = df_seg["low"].to_numpy(dtype=np.float64)
    c_arr = df_seg["close"].to_numpy(dtype=np.float64)
    atr_arr = df_seg["atr_14"].to_numpy(dtype=np.float64)

    sig_close = float(c_arr[sig_idx])
    atr = float(atr_arr[sig_idx])
    if not (np.isfinite(atr) and atr > 0.0):
        atr = sig_close * 0.008

    entry_px = sig_close * (1.0 + sgn * TAKER_SLIP)
    entry_ts = int(ts_arr[sig_idx]) + bar_ms

    # Unit initial quantity qty0 = 1.0 (R-multiple is scale-invariant)
    qty_rem = 1.0
    init_sl_dist = sl_mult * atr
    init_risk = init_sl_dist * 1.0
    sl = entry_px - sgn * init_sl_dist
    tp1 = entry_px + sgn * (tp1_mult * atr)
    tp2 = entry_px + sgn * (tp2_mult * atr)

    gross_pnl = 0.0
    total_fees = entry_px * 1.0 * TAKER_FEE

    tp1_hit = False
    tp2_hit = False
    trail_active = False
    trail_ref = 0.0
    trail_stop = 0.0
    pending_closed_bar: Optional[Tuple[float, float]] = None

    def close_leg(level_px: float, leg_qty: float, is_maker: bool) -> None:
        nonlocal qty_rem, gross_pnl, total_fees
        if leg_qty <= 0.0:
            return
        slip = MAKER_SLIP if is_maker else TAKER_SLIP
        fee_rate = MAKER_FEE if is_maker else TAKER_FEE
        exec_exit = level_px * (1.0 - sgn * slip)
        fee = leg_qty * exec_exit * fee_rate
        gross_pnl += sgn * (exec_exit - entry_px) * leg_qty
        total_fees += fee
        qty_rem = max(0.0, qty_rem - leg_qty)

    def ratchet_trail(bh: float, bl: float) -> None:
        nonlocal sl, trail_ref, trail_stop
        dist = trail_mult * atr
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

    exit_idx = min(len(df_seg) - 1, sig_idx + max_hold_bars)
    exit_ts = int(ts_arr[exit_idx]) + bar_ms
    exit_reason = "HORIZON_TIMEOUT"

    for k in range(sig_idx + 1, min(len(df_seg), sig_idx + max_hold_bars + 1)):
        o, h, l, c = float(o_arr[k]), float(h_arr[k]), float(l_arr[k]), float(c_arr[k])
        bar_ts = int(ts_arr[k])

        # 1. Update trailing stop from the PREVIOUS closed bar before checking current bar
        if trail_active and pending_closed_bar is not None:
            ratchet_trail(pending_closed_bar[0], pending_closed_bar[1])
            pending_closed_bar = None

        # 2. Conservative intrabar tie-break: check SL / BE / Trailing BEFORE TP on the same bar
        stop_hit = (l <= sl) if is_long else (h >= sl)
        if stop_hit:
            fill_px = min(sl, h, o) if is_long else max(sl, l, o)
            exit_reason = (
                "STOP_LOSS"
                if not tp1_hit
                else ("TRAILING_STOP" if (trail_active and sl == trail_stop) else "BREAKEVEN_STOP")
            )
            close_leg(fill_px, qty_rem, is_maker=False)
            exit_idx = k
            exit_ts = bar_ts + bar_ms
            break

        # 3. Check TP1 (40% maker -> move SL to Breakeven)
        if not tp1_hit and ((h >= tp1) if is_long else (l <= tp1)):
            tp1_hit = True
            close_leg(tp1, min(0.40, qty_rem), is_maker=True)
            be_px = entry_px * (1.0 + sgn * BE_BUFFER_PCT)
            sl = max(sl, be_px) if is_long else min(sl, be_px)
            # Same-bar breakeven check if bar didn't close safely beyond BE
            closed_safe = (c > sl) if is_long else (c < sl)
            if ((l <= sl) if is_long else (h >= sl)) and not closed_safe:
                fill_px = min(sl, h) if is_long else max(sl, l)
                close_leg(fill_px, qty_rem, is_maker=False)
                exit_reason = "BREAKEVEN_STOP"
                exit_idx = k
                exit_ts = bar_ts + bar_ms
                break

        # 4. Check TP2 (30% maker -> activate trailing stop)
        if tp1_hit and not tp2_hit and ((h >= tp2) if is_long else (l <= tp2)):
            tp2_hit = True
            close_leg(tp2, min(0.30, qty_rem), is_maker=True)
            if trail_mult > 0.0:
                trail_active = True
                trail_ref = tp2
                ratchet_trail(tp2, tp2)

        # 5. Queue closed bar extremes for trailing stop update on next bar
        if trail_active:
            pending_closed_bar = (h, l)

        # 6. Check max_holding_bars timeout at the end of the holding horizon
        if (k - sig_idx) >= max_hold_bars and qty_rem > 1e-9:
            close_leg(c, qty_rem, is_maker=False)
            exit_reason = "HORIZON_TIMEOUT"
            exit_idx = k
            exit_ts = bar_ts + bar_ms
            break

    if qty_rem > 1e-9:
        close_leg(float(c_arr[exit_idx]), qty_rem, is_maker=False)
        exit_reason = "HORIZON_TIMEOUT"

    net_pnl = gross_pnl - total_fees
    r_mult = float(net_pnl / init_risk) if init_risk > 0.0 else 0.0
    return {
        "entry_ts": int(entry_ts),
        "exit_ts": int(exit_ts),
        "exit_idx": int(exit_idx),
        "r_mult": float(r_mult),
        "exit_reason": exit_reason,
        "tp1_hit": bool(tp1_hit),
        "tp2_hit": bool(tp2_hit),
    }


def simulate_gp_portfolio_split(
    signals: List[Dict[str, Any]],
    split_days: float,
    cooldown_bars: int = 12,
    max_concurrent: int = 6,
    daily_quota: int = 3,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Simulate GOLDEN_POT portfolio execution on a chronological list of post-committee signals:
    - max_entries_per_bar = 1 across all altcoins per 5m bar (tie-break: highest atr_expansion * vol_z, then symbol).
    - cooldown_bars = 12 (1h) per symbol after position close (and no concurrent position on same symbol).
    - max_concurrent_positions = 6.
    - daily_quota = 3 per UTC day.
    Also computes unconstrained_signals_per_day (count of valid bar-winner signals before daily_quota cap / split_days).
    """
    if not signals:
        return [], 0.0

    # Group signals by decision bar timestamp `ts`
    by_ts: Dict[int, List[Dict[str, Any]]] = {}
    for sig in signals:
        by_ts.setdefault(int(sig["ts"]), []).append(sig)

    open_positions: List[Dict[str, Any]] = []
    last_close_ts_by_sym: Dict[str, int] = {}
    daily_counts: Dict[str, int] = {}
    executed_trades: List[Dict[str, Any]] = []
    unconstrained_count = 0
    cooldown_ms = cooldown_bars * GP_BAR_MS

    for ts in sorted(by_ts.keys()):
        decision_close_ts = ts + GP_BAR_MS
        # 1. Close positions that have exited on or before decision_close_ts
        still_open: List[Dict[str, Any]] = []
        for pos in open_positions:
            if int(pos["exit_ts"]) <= decision_close_ts:
                sym_p = str(pos["sym"])
                prev_c = last_close_ts_by_sym.get(sym_p, -10**18)
                last_close_ts_by_sym[sym_p] = max(prev_c, int(pos["exit_ts"]))
            else:
                still_open.append(pos)
        open_positions = still_open

        open_syms = {str(p["sym"]) for p in open_positions}

        # 2. Sort candidates at this 5m bar by highest (atr_expansion * vol_z)
        cands = sorted(
            by_ts[ts],
            key=lambda x: (-float(x["rank_score"]), str(x["sym"])),
        )

        # 3. Pick top eligible candidate (symbol not open and not in cooldown) -> max_entries_per_bar = 1
        chosen: Optional[Dict[str, Any]] = None
        for c in cands:
            sym = str(c["sym"])
            if sym in open_syms:
                continue
            last_exit = last_close_ts_by_sym.get(sym)
            if last_exit is not None and (decision_close_ts - last_exit) < cooldown_ms:
                continue
            chosen = c
            break

        if chosen is None:
            continue

        # Count signal before daily_quota cap
        if len(open_positions) < max_concurrent:
            unconstrained_count += 1

        # 4. Enforce max_concurrent_positions = 6 and daily_quota = 3 per UTC day
        if len(open_positions) >= max_concurrent:
            continue

        utc_day = datetime.fromtimestamp(decision_close_ts / 1000.0, tz=timezone.utc).strftime(
            "%Y-%m-%d"
        )
        if daily_counts.get(utc_day, 0) >= daily_quota:
            continue

        daily_counts[utc_day] = daily_counts.get(utc_day, 0) + 1
        open_positions.append(chosen)
        executed_trades.append(chosen)

    unconstrained_per_day = (
        round(float(unconstrained_count / split_days), 4) if split_days > 0 else 0.0
    )
    return executed_trades, unconstrained_per_day


def compute_split_metrics(
    trades: List[Dict[str, Any]],
    split_days: float,
    n_trials: int,
    variance_of_trials: float,
) -> Dict[str, Any]:
    """
    Compute required split metrics:
    n, trades_per_day, wr_pct, pf, pf_ex_top1, sum_r, avg_r, max_dd_r,
    sharpe_per_trade, dsr, bootstrap_ci95_avg_r.
    """
    n = len(trades)
    if n == 0:
        return {
            "n": 0,
            "trades_per_day": 0.0,
            "wr_pct": 0.0,
            "pf": 0.0,
            "pf_ex_top1": 0.0,
            "sum_r": 0.0,
            "avg_r": 0.0,
            "max_dd_r": 0.0,
            "sharpe_per_trade": 0.0,
            "dsr": 0.0,
            "bootstrap_ci95_avg_r": [0.0, 0.0],
        }

    r_arr = np.array([float(t["r_mult"]) for t in trades], dtype=np.float64)
    trades_per_day = round(float(n / split_days), 4) if split_days > 0 else 0.0
    wr_pct = round(float(np.mean(r_arr > 0.0) * 100.0), 2)

    wins_sum = float(np.sum(r_arr[r_arr > 0.0]))
    losses_sum = float(np.abs(np.sum(r_arr[r_arr < 0.0])))
    if losses_sum > 1e-12:
        pf = round(wins_sum / losses_sum, 4)
    else:
        pf = 99.0 if wins_sum > 0.0 else 0.0

    if n > 1:
        top1_idx = int(np.argmax(r_arr))
        r_ex = np.delete(r_arr, top1_idx)
        w_ex = float(np.sum(r_ex[r_ex > 0.0]))
        l_ex = float(np.abs(np.sum(r_ex[r_ex < 0.0])))
        if l_ex > 1e-12:
            pf_ex_top1 = round(w_ex / l_ex, 4)
        else:
            pf_ex_top1 = 99.0 if w_ex > 0.0 else 0.0
    else:
        pf_ex_top1 = 0.0

    sum_r = round(float(np.sum(r_arr)), 4)
    avg_r = round(float(np.mean(r_arr)), 4)

    cum_r = np.concatenate([[0.0], np.cumsum(r_arr)])
    peaks = np.maximum.accumulate(cum_r)
    max_dd_r = round(float(np.max(peaks - cum_r)), 4)

    sharpe_pt = round(float(calculate_per_period_sharpe(r_arr)), 4)
    dsr_res = deflated_sharpe_ratio(
        observed_sr_ann=sharpe_pt,
        returns=r_arr,
        n_trials=max(2, int(n_trials)),
        variance_of_trials=max(float(variance_of_trials), 1e-6),
        periods_per_year=1.0,
    )
    dsr_val = round(float(dsr_res["dsr"]), 4)

    boot_res = bootstrap_performance_metrics(
        returns=r_arr,
        n_iterations=1000,
        periods_per_year=1.0,
        seed=42,
    )
    ci_lo, ci_hi = boot_res["ann_ret_ci"]
    if n < 10 and n >= 2:
        # bootstrap_performance_metrics returns (0.0, 0.0) when n < 10; compute fallback bootstrap CI on avg_r
        rng = np.random.default_rng(42)
        samples = rng.choice(r_arr, size=(1000, n), replace=True).mean(axis=1)
        ci_lo = float(np.percentile(samples, 2.5))
        ci_hi = float(np.percentile(samples, 97.5))
    elif n == 1:
        ci_lo = ci_hi = float(r_arr[0])

    return {
        "n": int(n),
        "trades_per_day": trades_per_day,
        "wr_pct": wr_pct,
        "pf": pf,
        "pf_ex_top1": pf_ex_top1,
        "sum_r": sum_r,
        "avg_r": avg_r,
        "max_dd_r": max_dd_r,
        "sharpe_per_trade": sharpe_pt,
        "dsr": dsr_val,
        "bootstrap_ci95_avg_r": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
    }


def simulate_directional_1h_oos_20d(
    majors_1h: Dict[str, pd.DataFrame],
    core_full: TradingCore,
    enricher: ReadOnlyCachedEnricher,
    split_ts: int,
    oos_days: float,
) -> Dict[str, Any]:
    """
    Simulate DIRECTIONAL 1h over 2026-11-05 .. 2026-11-25 with the purged 1h TradingCore
    (conf >= 0.70, MAX_SAME_DIR_MAJORS = 2, STRATEGY_PROFILES['DIRECTIONAL'] stress costs).
    """
    prof = STRATEGY_PROFILES["DIRECTIONAL"]
    candidates_by_ts: Dict[int, List[Dict[str, Any]]] = {}

    for sym in MAJORS:
        df = majors_1h[sym]
        ts_arr = df["timestamp"].to_numpy(dtype=np.int64)
        c_arr = df["close"].to_numpy(dtype=np.float64)
        atr_arr = df["atr_14"].to_numpy(dtype=np.float64)

        start_i = int(np.searchsorted(ts_arr, split_ts, side="left"))
        for i in range(start_i, len(df) - 1):
            ts = int(ts_arr[i])
            close = float(c_arr[i])
            atr = float(atr_arr[i])
            if not (np.isfinite(close) and close > 0.0 and np.isfinite(atr) and atr > 0.0):
                continue
            row = df.iloc[i]
            sub_df = df.iloc[max(0, i - 60) : i + 1]
            ctx = enricher.enrich(sym, row, sub_df, ts + HOUR_MS)
            sig = core_full.evaluate_market_deterministic(
                context=ctx,
                symbol=sym,
                current_price=close,
                atr=atr,
                timestamp=ts,
            )
            if sig.is_vetoed or sig.direction not in ("BUY", "SELL") or sig.confidence < 0.70:
                continue

            sim = simulate_single_ladder_trade(
                df_seg=df,
                sig_idx=i,
                direction=sig.direction,
                sl_mult=float(prof.sl_atr_mult),
                tp1_mult=float(prof.tp1_atr_mult),
                tp2_mult=float(prof.tp2_atr_mult),
                trail_mult=float(prof.trail_atr_mult),
                max_hold_bars=DIR_MAX_HOLDING_BARS,
                bar_ms=HOUR_MS,
            )
            candidates_by_ts.setdefault(ts, []).append(
                {
                    "sym": sym,
                    "ts": ts,
                    "direction": sig.direction,
                    "confidence": float(sig.confidence),
                    **sim,
                }
            )

    open_positions: List[Dict[str, Any]] = []
    last_close_ts: Dict[str, int] = {}
    stop_streak: Dict[str, int] = {}
    executed: List[Dict[str, Any]] = []

    for ts in sorted(candidates_by_ts.keys()):
        decision_close_ts = ts + HOUR_MS
        still_open: List[Dict[str, Any]] = []
        for pos in open_positions:
            if int(pos["exit_ts"]) <= decision_close_ts:
                s_p = str(pos["sym"])
                last_close_ts[s_p] = max(last_close_ts.get(s_p, -10**18), int(pos["exit_ts"]))
                if pos["exit_reason"] == "STOP_LOSS":
                    stop_streak[s_p] = stop_streak.get(s_p, 0) + 1
                else:
                    stop_streak[s_p] = 0
            else:
                still_open.append(pos)
        open_positions = still_open

        cands = sorted(
            candidates_by_ts[ts],
            key=lambda x: (-float(x["confidence"]), str(x["sym"])),
        )
        for c in cands:
            sym = str(c["sym"])
            if any(p["sym"] == sym for p in open_positions):
                continue
            cd_bars = 2 if stop_streak.get(sym, 0) >= 2 else 1
            if sym in last_close_ts and (decision_close_ts - last_close_ts[sym]) < cd_bars * HOUR_MS:
                continue
            same_dir_count = sum(
                1 for p in open_positions if p["direction"] == c["direction"]
            )
            if same_dir_count >= 2:  # MAX_SAME_DIR_MAJORS = 2
                continue
            open_positions.append(c)
            executed.append(c)

    m = compute_split_metrics(executed, oos_days, n_trials=2, variance_of_trials=0.05)
    return {
        "n": m["n"],
        "trades_per_day": m["trades_per_day"],
        "wr_pct": m["wr_pct"],
        "pf": m["pf"],
        "pf_ex_top1": m["pf_ex_top1"],
        "sum_r": m["sum_r"],
        "avg_r": m["avg_r"],
        "max_dd_r": m["max_dd_r"],
        "sharpe_per_trade": m["sharpe_per_trade"],
    }


def main() -> int:
    t0 = time.time()
    print("=== S6 Zero-Leakage OOS Ablation: GOLDEN_POT 5m (G0..G7) vs DIRECTIONAL 1h ===", flush=True)

    prior_trials = get_prior_trials_count()
    n_trials_total = prior_trials + 8

    # 1. Load 1h Majors and train purged 1h ensemble before 2026-10-15 - 7d
    print("  [1/5] Loading 1h majors and training purged 1h LGBM+RF ensemble...", flush=True)
    majors_1h = load_clean_1h_majors(CACHE_DB)
    ens_1h, purged_1h_train_n, cols_1h = train_purged_1h_ensemble(
        majors_1h, TRAIN_1H_CUTOFF_TS
    )
    print(
        f"        Purged 1h ensemble trained: N={purged_1h_train_n} samples (< {ts_to_iso(TRAIN_1H_CUTOFF_TS)}), "
        f"{len(cols_1h)} stationary features.",
        flush=True,
    )

    # Setup Full Committee (6 agents) and Intraday Committee (RiskSentinel + TechnicalAgent + OrderFlowAgent)
    ta_full = TechnicalAgent(model=ens_1h)
    core_full = TradingCore(min_consensus_threshold=0.30, technical_agent=ta_full)
    core_full.learning_tracker = None

    ta_intra = TechnicalAgent(model=ens_1h)
    core_intraday = TradingCore(min_consensus_threshold=0.30, technical_agent=ta_intra)
    core_intraday.agents = [
        core_intraday.risk_sentinel,
        core_intraday.technical_agent,
        core_intraday.orderflow_agent,
    ]
    core_intraday.learning_tracker = None

    # Build causal BTC 1h dist_sma200 lookup (closed 1h candles: h1_ts + 3_600_000 <= gp_ts + 300_000)
    btc_1h_df = majors_1h["BTCUSDT"]
    btc_1h_close_ts = btc_1h_df["timestamp"].to_numpy(dtype=np.int64) + HOUR_MS
    btc_1h_dist = (
        (btc_1h_df["close"].to_numpy(dtype=np.float64) - btc_1h_df["sma_200"].to_numpy(dtype=np.float64))
        / btc_1h_df["sma_200"].to_numpy(dtype=np.float64)
    )

    def get_causal_btc_dist_1h(decision_close_ts_ms: int) -> Optional[float]:
        idx = int(np.searchsorted(btc_1h_close_ts, decision_close_ts_ms, side="right")) - 1
        if 0 <= idx < len(btc_1h_dist):
            val = float(btc_1h_dist[idx])
            return val if np.isfinite(val) else None
        return None

    # 2. Load 5m universe, exclude TradFi, split contiguous segments, compute features per segment
    print("  [2/5] Loading 5m universe, applying TradFi exclusions & contiguous segment splitting...", flush=True)
    raw_segments, uni_meta = load_and_split_5m_segments(CACHE_DB)
    print(
        f"        5m DB symbols={uni_meta['total_5m_symbols_in_db']} | "
        f"Excluded TradFi={uni_meta['excluded_tradfi_count']} | "
        f"Kept Altcoins={uni_meta['kept_altcoins_count']} | "
        f"Segments >=250={uni_meta['raw_segments_ge_250']}",
        flush=True,
    )

    print(f"  [3/5] Computing segment-level features & pre-triggers (ProcessPoolExecutor, max_workers={MAX_WORKERS})...", flush=True)
    seg_dfs: Dict[int, pd.DataFrame] = {}
    all_candidates: List[Dict[str, Any]] = []
    usable_segments = 0
    usable_bars = 0
    min_eval_ts: Optional[int] = None
    max_eval_ts: Optional[int] = None

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for res in pool.map(_process_segment_worker, raw_segments, chunksize=12):
            if res is None:
                continue
            seg_id, sym, df_seg, cands = res
            usable_segments += 1
            usable_bars += len(df_seg)
            t_first = int(df_seg["timestamp"].iloc[0])
            t_last = int(df_seg["timestamp"].iloc[-1])
            min_eval_ts = t_first if min_eval_ts is None else min(min_eval_ts, t_first)
            max_eval_ts = t_last if max_eval_ts is None else max(max_eval_ts, t_last)
            if cands:
                seg_dfs[seg_id] = df_seg
                all_candidates.extend(cands)

    print(
        f"        Usable segments={usable_segments} | Usable post-SMA200 bars={usable_bars} | "
        f"Union pre-trigger candidate bars={len(all_candidates)}",
        flush=True,
    )

    # Calculate exact IS and OOS window durations in days
    is_start_eff = max(IS_START_TS, min_eval_ts or IS_START_TS)
    oos_end_eff = (max_eval_ts or (SPLIT_TS_GP + 20 * DAY_MS)) + GP_BAR_MS
    is_days = max(1.0, (SPLIT_TS_GP - EMBARGO_GP_MS - is_start_eff) / DAY_MS)
    oos_days = max(1.0, (oos_end_eff - SPLIT_TS_GP) / DAY_MS)

    # Compute OOS bear vs non-bear duration in days from BTC 1h candles in [SPLIT_TS_GP, oos_end_eff]
    oos_btc_mask = (btc_1h_close_ts >= SPLIT_TS_GP) & (btc_1h_close_ts <= oos_end_eff)
    oos_btc_dists = btc_1h_dist[oos_btc_mask]
    if len(oos_btc_dists) > 0:
        bear_frac = float(np.mean(oos_btc_dists < -0.015))
    else:
        bear_frac = 0.5
    oos_bear_days = max(0.5, oos_days * bear_frac)
    oos_non_bear_days = max(0.5, oos_days * (1.0 - bear_frac))

    # 3. Enrich candidate bars & evaluate committees + forward ladder trade outcome once per candidate
    print("  [4/5] Enriching candidate bars & evaluating 6-agent / 3-agent committees...", flush=True)
    enricher = ReadOnlyCachedEnricher(CACHE_DB, MICRO_DB)
    gp_prof = STRATEGY_PROFILES["GOLDEN_POT"]

    try:
        for cand in all_candidates:
            seg_id = int(cand["seg_id"])
            i = int(cand["idx"])
            ts = int(cand["ts"])
            sym = str(cand["sym"])
            close = float(cand["close"])
            atr = float(cand["atr"])
            df_seg = seg_dfs[seg_id]

            decision_close_ts = ts + GP_BAR_MS
            btc_dist = get_causal_btc_dist_1h(decision_close_ts)
            cand["btc_dist_1h"] = btc_dist
            cand["oos_regime"] = (
                "OOS_BEAR"
                if (btc_dist is not None and btc_dist < -0.015)
                else "OOS_NON_BEAR"
            )

            row = df_seg.iloc[i]
            sub_df = df_seg.iloc[max(0, i - 60) : i + 1]
            ctx = enricher.enrich(sym, row, sub_df, decision_close_ts)

            sig_full = core_full.evaluate_market_deterministic(
                context=ctx,
                symbol=sym,
                current_price=close,
                atr=atr,
                timestamp=ts,
            )
            cand["full_vetoed"] = bool(sig_full.is_vetoed)
            cand["full_dir"] = str(sig_full.direction)
            cand["full_conf"] = float(sig_full.confidence)

            sig_intra = core_intraday.evaluate_market_deterministic(
                context=ctx,
                symbol=sym,
                current_price=close,
                atr=atr,
                timestamp=ts,
            )
            cand["intra_vetoed"] = bool(sig_intra.is_vetoed)
            cand["intra_dir"] = str(sig_intra.direction)
            cand["intra_conf"] = float(sig_intra.confidence)

            # Precompute forward trade outcome for standard direction and early direction
            for d_key in ("std_dir", "early_dir"):
                d_val = cand[d_key]
                if d_val in ("BUY", "SELL") and f"sim_{d_val}" not in cand:
                    cand[f"sim_{d_val}"] = simulate_single_ladder_trade(
                        df_seg=df_seg,
                        sig_idx=i,
                        direction=d_val,
                        sl_mult=float(cand["sl_mult"]),
                        tp1_mult=float(gp_prof.tp1_atr_mult),
                        tp2_mult=float(gp_prof.tp2_atr_mult),
                        trail_mult=float(gp_prof.trail_atr_mult),
                        max_hold_bars=GP_MAX_HOLDING_BARS,
                        bar_ms=GP_BAR_MS,
                    )

        # 4. Simulate DIRECTIONAL 1h 20d OOS baseline
        print("  [5/5] Simulating DIRECTIONAL 1h 20d OOS baseline and GOLDEN_POT G0..G7 variants...", flush=True)
        directional_1h_oos_20d = simulate_directional_1h_oos_20d(
            majors_1h=majors_1h,
            core_full=core_full,
            enricher=enricher,
            split_ts=SPLIT_TS_GP,
            oos_days=oos_days,
        )
    finally:
        enricher.close()

    # Define the 8 Pre-Registered Variants (G0..G7)
    variant_specs = [
        {
            "name": "G0_PROD",
            "trigger": "std",
            "macro_thresh": 0.015,
            "climax_mode": "prod",
            "committee": "full",
            "min_conf": 0.70,
        },
        {
            "name": "G1_RELAX_CLIMAX",
            "trigger": "std",
            "macro_thresh": 0.015,
            "climax_mode": "relax",
            "committee": "full",
            "min_conf": 0.70,
        },
        {
            "name": "G2_NO_CLIMAX",
            "trigger": "std",
            "macro_thresh": 0.015,
            "climax_mode": "none",
            "committee": "full",
            "min_conf": 0.70,
        },
        {
            "name": "G3_MACRO_BTC_35",
            "trigger": "std",
            "macro_thresh": 0.035,
            "climax_mode": "prod",
            "committee": "full",
            "min_conf": 0.70,
        },
        {
            "name": "G4_NO_MACRO_BTC",
            "trigger": "std",
            "macro_thresh": None,
            "climax_mode": "prod",
            "committee": "full",
            "min_conf": 0.70,
        },
        {
            "name": "G5_RELAX_CLIMAX_MACRO35",
            "trigger": "std",
            "macro_thresh": 0.035,
            "climax_mode": "relax",
            "committee": "full",
            "min_conf": 0.70,
        },
        {
            "name": "G6_INTRADAY_COMMITTEE",
            "trigger": "std",
            "macro_thresh": 0.015,
            "climax_mode": "relax",
            "committee": "intraday",
            "min_conf": 0.65,
        },
        {
            "name": "G7_EARLY_TRIGGER",
            "trigger": "early",
            "macro_thresh": 0.015,
            "climax_mode": "prod",
            "committee": "full",
            "min_conf": 0.70,
        },
    ]

    raw_variant_runs: Dict[str, Dict[str, Any]] = {}
    for spec in variant_specs:
        vname = spec["name"]
        t_key = "std_dir" if spec["trigger"] == "std" else "early_dir"
        macro_thresh = spec["macro_thresh"]
        climax_mode = spec["climax_mode"]
        comm_type = spec["committee"]
        min_conf = float(spec["min_conf"])

        raw_triggers = 0
        post_macro = 0
        post_climax = 0
        post_committee = 0

        is_signals: List[Dict[str, Any]] = []
        oos_signals: List[Dict[str, Any]] = []

        for cand in all_candidates:
            direction = cand[t_key]
            if direction is None:
                continue
            raw_triggers += 1

            if is_macro_btc_blocked(direction, cand["btc_dist_1h"], macro_thresh):
                continue
            post_macro += 1

            if is_climax_blocked(direction, cand["rsi"], cand["cum_move_atr"], climax_mode):
                continue
            post_climax += 1

            if comm_type == "full":
                vetoed = cand["full_vetoed"]
                c_dir = cand["full_dir"]
                c_conf = cand["full_conf"]
            else:
                vetoed = cand["intra_vetoed"]
                c_dir = cand["intra_dir"]
                c_conf = cand["intra_conf"]

            if vetoed or c_dir != direction or c_conf < min_conf:
                continue
            post_committee += 1

            sim_out = cand[f"sim_{direction}"]
            sig_record = {
                "sym": cand["sym"],
                "seg_id": cand["seg_id"],
                "idx": cand["idx"],
                "ts": cand["ts"],
                "direction": direction,
                "confidence": c_conf,
                "rank_score": cand["rank_score"],
                "btc_dist_1h": cand["btc_dist_1h"],
                "oos_regime": cand["oos_regime"],
                **sim_out,
            }
            if cand["split"] == "IS":
                is_signals.append(sig_record)
            else:
                oos_signals.append(sig_record)

        is_trades, is_unconstrained_spd = simulate_gp_portfolio_split(
            is_signals, is_days, cooldown_bars=12, max_concurrent=6, daily_quota=3
        )
        oos_trades, oos_unconstrained_spd = simulate_gp_portfolio_split(
            oos_signals, oos_days, cooldown_bars=12, max_concurrent=6, daily_quota=3
        )
        oos_bear_trades = [t for t in oos_trades if t["oos_regime"] == "OOS_BEAR"]
        oos_non_bear_trades = [t for t in oos_trades if t["oos_regime"] == "OOS_NON_BEAR"]

        raw_variant_runs[vname] = {
            "funnel_counts": {
                "raw_triggers": int(raw_triggers),
                "post_macro": int(post_macro),
                "post_climax": int(post_climax),
                "post_committee": int(post_committee),
            },
            "is_trades": is_trades,
            "oos_trades": oos_trades,
            "oos_bear_trades": oos_bear_trades,
            "oos_non_bear_trades": oos_non_bear_trades,
            "unconstrained_signals_per_day": oos_unconstrained_spd,
            "unconstrained_is_signals_per_day": is_unconstrained_spd,
        }

    # Compute cross-variant Sharpe variance for Deflated Sharpe Ratio (DSR)
    is_sharpes = [
        calculate_per_period_sharpe(
            np.array([t["r_mult"] for t in r["is_trades"]], dtype=np.float64)
        )
        if len(r["is_trades"]) >= 2
        else 0.0
        for r in raw_variant_runs.values()
    ]
    oos_sharpes = [
        calculate_per_period_sharpe(
            np.array([t["r_mult"] for t in r["oos_trades"]], dtype=np.float64)
        )
        if len(r["oos_trades"]) >= 2
        else 0.0
        for r in raw_variant_runs.values()
    ]
    var_is_sharpes = max(float(np.var(is_sharpes, ddof=1)), 1e-4)
    var_oos_sharpes = max(float(np.var(oos_sharpes, ddof=1)), 1e-4)

    variants_out: Dict[str, Any] = {}
    for spec in variant_specs:
        vname = spec["name"]
        r = raw_variant_runs[vname]
        is_m = compute_split_metrics(
            r["is_trades"], is_days, n_trials=n_trials_total, variance_of_trials=var_is_sharpes
        )
        oos_m = compute_split_metrics(
            r["oos_trades"], oos_days, n_trials=n_trials_total, variance_of_trials=var_oos_sharpes
        )
        oos_bear_m = compute_split_metrics(
            r["oos_bear_trades"],
            oos_bear_days,
            n_trials=n_trials_total,
            variance_of_trials=var_oos_sharpes,
        )
        oos_non_bear_m = compute_split_metrics(
            r["oos_non_bear_trades"],
            oos_non_bear_days,
            n_trials=n_trials_total,
            variance_of_trials=var_oos_sharpes,
        )
        variants_out[vname] = {
            "funnel_counts": r["funnel_counts"],
            "is": is_m,
            "oos": oos_m,
            "oos_bear": oos_bear_m,
            "oos_non_bear": oos_non_bear_m,
            "unconstrained_signals_per_day": r["unconstrained_signals_per_day"],
            "verdict": "NO-GO",  # resolved after G0_PROD baseline is set
        }

    g0_oos_tpd = float(variants_out["G0_PROD"]["oos"]["trades_per_day"])
    for vname, vdata in variants_out.items():
        oos_m = vdata["oos"]
        is_m = vdata["is"]
        is_go = (
            oos_m["n"] >= 15
            and oos_m["pf"] >= 1.25
            and oos_m["pf_ex_top1"] >= 1.10
            and oos_m["avg_r"] > 0.0
            and is_m["pf"] >= 1.00
            and oos_m["trades_per_day"] > g0_oos_tpd
        )
        vdata["verdict"] = "GO" if is_go else "NO-GO"

    report = {
        "metadata": {
            **uni_meta,
            "usable_segments": int(usable_segments),
            "usable_bars": int(usable_bars),
            "split_ts": int(SPLIT_TS_GP),
            "split_ts_iso": ts_to_iso(SPLIT_TS_GP),
            "embargo_ms": int(EMBARGO_GP_MS),
            "is_window_days": round(float(is_days), 3),
            "oos_window_days": round(float(oos_days), 3),
            "oos_bear_days": round(float(oos_bear_days), 3),
            "oos_non_bear_days": round(float(oos_non_bear_days), 3),
            "prior_trials": int(prior_trials),
            "n_trials_dsr": int(n_trials_total),
            "purged_1h_train_n": int(purged_1h_train_n),
            "purged_1h_train_cutoff_ts": int(TRAIN_1H_CUTOFF_TS),
            "purged_1h_train_cutoff_iso": ts_to_iso(TRAIN_1H_CUTOFF_TS),
            "elapsed_seconds": round(float(time.time() - t0), 2),
            "notes": [
                "5m altcoin candles in market_cache.db exist as ~400-bar chunks fetched when the live scanner selected each symbol (pre-selection lookback bias; NO-GO verdicts remain conservative, no GO claim should rely on pre-selection windows).",
                "Backtest scans all 220 non-TradFi altcoins on every bar without the live Top-6 24h scanner cap (GP_MAX_CANDIDATES=6) or live TapeReadingAgent _flow_gate.",
                "TradFi exclusions combine canonical NON_CRYPTO_BASE_EXCLUSIONS (85 bases) + EXTRA_TRADFI_BASES (22 Binance TradFi/ETF/equity bases observed in market_cache.db).",
            ],
        },
        "directional_1h_oos_20d": directional_1h_oos_20d,
        "variants": variants_out,
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  [DIRECTIONAL 1h OOS 20d] n={directional_1h_oos_20d['n']} | trades/d={directional_1h_oos_20d['trades_per_day']:.3f} | WR={directional_1h_oos_20d['wr_pct']:.1f}% | PF={directional_1h_oos_20d['pf']:.3f} | sum_R={directional_1h_oos_20d['sum_r']:+.3f} | avg_R={directional_1h_oos_20d['avg_r']:+.4f}", flush=True)
    print("\n  [GOLDEN_POT 5m Ablation Summary]", flush=True)
    for vname, v in variants_out.items():
        fc = v["funnel_counts"]
        ism = v["is"]
        oosm = v["oos"]
        print(
            f"    {vname:<24} | Funnel {fc['raw_triggers']}->{fc['post_macro']}->{fc['post_climax']}->{fc['post_committee']} "
            f"| IS: n={ism['n']:>2} PF={ism['pf']:.2f} avgR={ism['avg_r']:+.3f} "
            f"| OOS: n={oosm['n']:>2} ({oosm['trades_per_day']:.2f}/d, unc={v['unconstrained_signals_per_day']:.2f}/d) "
            f"WR={oosm['wr_pct']:>4.1f}% PF={oosm['pf']:.2f} (ex1={oosm['pf_ex_top1']:.2f}) avgR={oosm['avg_r']:+.3f} DD={oosm['max_dd_r']:.2f}R "
            f"| {v['verdict']}",
            flush=True,
        )
    print(f"\n[OK] Saved report to {OUTPUT_JSON} in {time.time() - t0:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
