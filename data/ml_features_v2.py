"""Causal Feature Pipeline v2 & Live Serving Parity (`data/ml_features_v2.py`).
Reuses `FeatureEngineer` (v1 untouched), guards multi-exchange volume seams, and computes
causal 8h funding carry, strict D-1 (`UTC midnight + 1d`) macro/on-chain, closed `4h`/`1d`
MTF, closed `1h` BTC cross-asset, and seasonality with `<15ms` live `<= 1e-9` parity."""

from __future__ import annotations
import math, os, sqlite3, threading, time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np, pandas as pd, ta
from db_conn import connect as _db_connect
from data.feature_engineering import FeatureEngineer

TIMEFRAME_MS: Dict[str, int] = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}

_TECH_STATIONARY_COLS: List[str] = [
    "dist_sma_20", "dist_sma_50", "dist_sma_200", "dist_ema_9", "dist_ema_21", "macd_line", "macd_signal", "macd_hist",
    "rsi_14", "stoch_k", "stoch_d", "natr_14", "bb_width", "bb_pband", "vol_sma_ratio", "volume_zscore_20",
    "dist_vwap_pct", "log_returns", "volatility_20", "hl_spread", "dist_resistance_pct", "dist_support_pct",
]

V2_FEATURE_COLUMNS: List[str] = [
    *_TECH_STATIONARY_COLS, "garman_klass_vol_24", "vol_regime_ratio",
    "funding_rate_bps", "funding_7d_ma_bps", "funding_zscore_30d",
    "macro_dxy_ret_5d", "macro_vix_level", "macro_vix_zscore_20d", "macro_sp500_ret_5d", "macro_fng_norm", "macro_stablecoin_growth_30d", "macro_risk_off",
    "mtf_4h_dist_ema21", "mtf_4h_rsi14", "mtf_4h_natr14", "mtf_1d_dist_sma50", "mtf_1d_dist_sma200", "mtf_1d_ret_5d",
    "btc_ret_6h", "btc_ret_24h", "btc_vol_24h", "rel_btc_ret_6h", "rel_btc_ret_24h", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

_NEUTRAL_DEFAULTS: Dict[str, float] = {
    **{c: 0.0 for c in V2_FEATURE_COLUMNS}, "rsi_14": 50.0, "stoch_k": 50.0, "stoch_d": 50.0,
    "vol_sma_ratio": 1.0, "vol_regime_ratio": 1.0, "macro_vix_level": 18.0, "mtf_4h_rsi14": 50.0, "hour_cos": 1.0, "dow_cos": 1.0,
}
_V2_SIG_COLS = frozenset({"garman_klass_vol_24", "vol_regime_ratio", "funding_rate_bps", "macro_vix_zscore_20d", "mtf_4h_dist_ema21", "btc_ret_6h"})

_EPOCH_UTC, _ONE_MS = pd.Timestamp("1970-01-01", tz="UTC"), pd.Timedelta("1ms")
_DAY_MS, _HOUR_MS, _FUNDING_8H_MS = 86_400_000, 3_600_000, 28_800_000
_HTF_WARMUP_MS, _FUNDING_WARMUP_MS, _MACRO_WARMUP_MS = 220 * _DAY_MS, 90 * _DAY_MS, 90 * _DAY_MS
_GK_COEF, _LIVE_CACHE_TTL_S, _LIVE_CACHE_MAX = 2.0 * math.log(2.0) - 1.0, 60.0, 128
_CACHE_LOCK = threading.Lock()
_MACRO_CACHE: Dict[Any, Tuple[float, pd.DataFrame]] = {}
_FUNDING_CACHE: Dict[Any, Tuple[float, pd.DataFrame]] = {}
_H1_CACHE: Dict[Any, Tuple[float, Any]] = {}


def clear_v2_cache() -> None:
    """Clear all in-memory live serving caches."""
    with _CACHE_LOCK:
        for c in (_MACRO_CACHE, _FUNDING_CACHE, _H1_CACHE):
            c.clear()


def is_v2_feature_bundle(bundle: Any) -> bool:
    """Return True if a model bundle dict or fitted estimator uses the v2 feature schema."""
    if bundle is None:
        return False
    if isinstance(bundle, dict):
        sv = str(bundle.get("schema_version", "")).strip().lower()
        if bundle.get("feature_pipeline") == "ml_features_v2" or sv.startswith("2.") or sv in ("2", "v2"):
            return True
        fn = bundle.get("feature_names")
        return bool(isinstance(fn, (list, tuple, set)) and (set(fn) == set(V2_FEATURE_COLUMNS) or (set(fn) & _V2_SIG_COLS)))
    for attr in ("feature_names", "feature_names_in_", "feature_name_"):
        fn = getattr(bundle, attr, None)
        if fn is not None and isinstance(fn, (list, tuple, set, np.ndarray, pd.Index)) and (set(fn) & _V2_SIG_COLS):
            return True
    return False


def _safe_float(v: Any, default: float) -> float:
    try:
        fv = float(v)
        return fv if math.isfinite(fv) else default
    except (TypeError, ValueError):
        return default

def _db_sig(path: Optional[str]) -> Tuple[str, int, int, int, int]:
    if not path or not os.path.exists(path):
        return ("", 0, 0, 0, 0)
    ap = os.path.abspath(path)
    try:
        st = os.stat(ap)
        wst = os.stat(ap + "-wal") if os.path.exists(ap + "-wal") else None
        return (ap, st.st_mtime_ns, st.st_size, wst.st_mtime_ns if wst else 0, wst.st_size if wst else 0)
    except OSError:
        return (ap, 0, 0, 0, 0)

def _query_ro(path: Optional[str], sql: str, params: Tuple[Any, ...] = ()) -> List[Tuple[Any, ...]]:
    if not path or not os.path.exists(path):
        return []
    con: Optional[sqlite3.Connection] = None
    try:
        con = _db_connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True, timeout=5.0)
        con.execute("PRAGMA busy_timeout=5000")
        return con.execute(sql, params).fetchall()
    except sqlite3.Error:
        return []
    finally:
        if con is not None:
            con.close()

def _cached_call(store: Dict[Any, Tuple[float, Any]], key: Any, fn: Callable[[], Any]) -> Any:
    with _CACHE_LOCK:
        hit = store.get(key)
        if hit is not None and (time.monotonic() - hit[0]) <= _LIVE_CACHE_TTL_S:
            return hit[1]
    val = fn()
    with _CACHE_LOCK:
        if len(store) >= _LIVE_CACHE_MAX:
            store.pop(next(iter(store)), None)
        store[key] = (time.monotonic(), val)
    return val

def _apply_exchange_seam_volume_guard(feat: pd.DataFrame) -> None:
    if "exchange" not in feat.columns or feat.empty:
        return
    ex = feat["exchange"].astype(str).str.strip().str.lower()
    if ex.nunique(dropna=True) <= 1:
        return
    seg_id = (ex != ex.shift(1)).cumsum()
    first_seg = int(seg_id.iloc[0])
    for sid, grp in feat.groupby(seg_id, sort=False):
        c, v = pd.to_numeric(grp["close"], errors="coerce"), pd.to_numeric(grp["volume"], errors="coerce")
        vsr = (v / (v.rolling(20, min_periods=20).mean() + 1e-8)).fillna(1.0)
        vz = ((v - v.rolling(20, min_periods=2).mean()) / (v.rolling(20, min_periods=2).std() + 1e-8)).fillna(0.0)
        vwap = (c * v).rolling(24, min_periods=2).sum() / (v.rolling(24, min_periods=2).sum() + 1e-8)
        dvwap = (((c - vwap) / (vwap + 1e-8)) * 100.0).fillna(0.0)
        if int(sid) > first_seg:
            k = min(19, len(grp))
            vsr.iloc[:k], vz.iloc[:k], dvwap.iloc[:k] = 1.0, 0.0, 0.0
        feat.loc[grp.index, ["vol_sma_ratio", "volume_zscore_20", "dist_vwap_pct"]] = np.column_stack([vsr, vz, dvwap])

def _gk_per_bar(df: pd.DataFrame) -> pd.Series:
    o = pd.to_numeric(df["open"], errors="coerce").where(lambda s: s > 0, np.nan)
    h = pd.to_numeric(df["high"], errors="coerce").where(lambda s: s > 0, np.nan)
    l = pd.to_numeric(df["low"], errors="coerce").where(lambda s: s > 0, np.nan)
    c = pd.to_numeric(df["close"], errors="coerce").where(lambda s: s > 0, np.nan)
    return 0.5 * (np.log(h / l) ** 2) - _GK_COEF * (np.log(c / o) ** 2)

def _add_range_and_regime_vol(feat: pd.DataFrame) -> None:
    gk = np.sqrt(_gk_per_bar(feat).rolling(24, min_periods=1).mean().clip(lower=0.0))
    feat["garman_klass_vol_24"] = gk.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    natr_pos = pd.to_numeric(feat["natr_14"], errors="coerce").where(lambda s: s > 0, np.nan)
    feat["vol_regime_ratio"] = (natr_pos / (natr_pos.rolling(50, min_periods=10).mean() + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

def _compute_funding_table(raw_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    cols = ["funding_ts", "funding_rate_bps", "funding_7d_ma_bps", "funding_zscore_30d"]
    tc = "timestamp" if raw_df is not None and "timestamp" in raw_df.columns else ("ts_ms" if raw_df is not None and "ts_ms" in raw_df.columns else None)
    if raw_df is None or raw_df.empty or tc is None or "funding_rate" not in raw_df.columns:
        return pd.DataFrame(columns=cols)
    df = raw_df.assign(funding_ts=pd.to_numeric(raw_df[tc], errors="coerce"), funding_rate=pd.to_numeric(raw_df["funding_rate"], errors="coerce")).dropna(subset=["funding_ts", "funding_rate"])
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["funding_ts"] = df["funding_ts"].astype("int64")
    df = df.assign(_p=df["exchange"].astype(str).str.lower() != "binance").sort_values(["funding_ts", "_p"]).drop_duplicates("funding_ts", keep="first") if "exchange" in df.columns else df.sort_values("funding_ts").drop_duplicates("funding_ts", keep="last")
    bps = df["funding_rate"].reset_index(drop=True).astype(float) * 1e4
    m30, s30 = bps.rolling(90, min_periods=10).mean(), bps.rolling(90, min_periods=10).std()
    return pd.DataFrame({
        "funding_ts": df["funding_ts"].to_numpy("int64"),
        "funding_rate_bps": bps.to_numpy("float64"),
        "funding_7d_ma_bps": bps.rolling(21, min_periods=3).mean().fillna(0.0).to_numpy("float64"),
        "funding_zscore_30d": np.asarray(np.where((m30.notna()) & (s30 > 1e-8), (bps - m30) / s30, 0.0), dtype="float64"),
    })

def _load_funding_table(symbol: str, cache_db_path: Optional[str], min_ts_ms: int, max_ts_ms: int, funding_df: Optional[pd.DataFrame] = None, micro_db_path: Optional[str] = None) -> pd.DataFrame:
    if funding_df is not None:
        fdf = funding_df[funding_df["symbol"] == symbol] if "symbol" in funding_df.columns and (funding_df["symbol"] == symbol).any() else funding_df
        tc = "timestamp" if "timestamp" in fdf.columns else "ts_ms"
        return _compute_funding_table(fdf[pd.to_numeric(fdf[tc], errors="coerce") <= max_ts_ms] if tc in fdf.columns else fdf)
    rows = _query_ro(cache_db_path, "SELECT timestamp, funding_rate, exchange FROM funding_rates WHERE symbol=? AND timestamp>=? AND timestamp<=? ORDER BY timestamp ASC", (symbol, int(min_ts_ms - _FUNDING_WARMUP_MS), int(max_ts_ms)))
    raw = pd.DataFrame(rows, columns=["timestamp", "funding_rate", "exchange"]) if rows else pd.DataFrame()
    if micro_db_path and (max_ts_ms - (int(raw["timestamp"].max()) if not raw.empty else 0)) > 12 * _HOUR_MS:
        last_ts = int(raw["timestamp"].max()) if not raw.empty else 0
        d_rows = _query_ro(micro_db_path, "SELECT ts_ms, funding_rate FROM derivs WHERE symbol=? AND ts_ms>=? AND ts_ms<=? ORDER BY ts_ms ASC", (symbol, max(last_ts + 1, int(min_ts_ms - _FUNDING_WARMUP_MS)), int(max_ts_ms)))
        if d_rows:
            ddf = pd.DataFrame(d_rows, columns=["timestamp", "funding_rate"]).dropna()
            if not ddf.empty:
                ddf = ddf.assign(_slot=ddf["timestamp"].astype("int64") // _FUNDING_8H_MS).drop_duplicates("_slot", keep="last").drop(columns=["_slot"]).assign(exchange="binance")
                raw = pd.concat([raw, ddf], ignore_index=True) if not raw.empty else ddf
    return _compute_funding_table(raw)

def _compute_macro_avail_table(raw_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    mcols = ["avail_ts", "macro_dxy_ret_5d", "macro_vix_level", "macro_vix_zscore_20d", "macro_sp500_ret_5d", "macro_fng_norm", "macro_stablecoin_growth_30d", "macro_risk_off"]
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=mcols)
    if {"series_name", "date_str", "value"}.issubset(raw_df.columns):
        df = raw_df[raw_df["series_name"] != "DEFI_TVL"].copy()
    else:
        dcol = next((c for c in ("date_str", "date", "observation_date") if c in raw_df.columns), None)
        recs = [pd.DataFrame({"series_name": c, "date_str": raw_df[dcol], "value": raw_df[c]}) for c in ("DXY", "VIX", "SP500", "FEAR_AND_GREED", "STABLECOINS_CIRCULATING_USD", "STABLECOIN_MCAP") if dcol and c in raw_df.columns]
        if not recs:
            return pd.DataFrame(columns=mcols)
        df = pd.concat(recs, ignore_index=True)

    dt = pd.to_datetime(df["date_str"], utc=True, errors="coerce").dt.floor("D")
    df = df.assign(avail_ts=((dt - _EPOCH_UTC) // _ONE_MS) + _DAY_MS, value=pd.to_numeric(df["value"], errors="coerce")).dropna(subset=["avail_ts", "value"])
    if df.empty:
        return pd.DataFrame(columns=mcols)
    df["avail_ts"] = df["avail_ts"].astype("int64")
    def _s(*names: str) -> pd.DataFrame:
        sub = df[df["series_name"].isin(names)]
        return pd.DataFrame(columns=["avail_ts", "value"]) if sub.empty else sub.assign(_p=sub["series_name"] != names[0]).sort_values(["avail_ts", "_p"]).drop_duplicates("avail_ts", keep="first")[["avail_ts", "value"]].reset_index(drop=True)

    parts: List[pd.DataFrame] = []
    for sn, col, lag in (("DXY", "macro_dxy_ret_5d", 5), ("SP500", "macro_sp500_ret_5d", 5), ("STABLECOINS_CIRCULATING_USD", "macro_stablecoin_growth_30d", 30)):
        s = _s(sn, "STABLECOIN_MCAP") if sn.startswith("STABLE") else _s(sn)
        if not s.empty:
            parts.append(pd.DataFrame({"avail_ts": s["avail_ts"], col: s["value"].where(s["value"] > 0, np.nan).pct_change(lag, fill_method=None)}))
    vix, fng = _s("VIX"), _s("FEAR_AND_GREED")
    if not vix.empty:
        v = vix["value"].where(vix["value"] > 0, np.nan)
        vm, vs = v.rolling(20, min_periods=5).mean(), v.rolling(20, min_periods=5).std()
        parts.append(pd.DataFrame({"avail_ts": vix["avail_ts"], "macro_vix_level": v, "macro_vix_zscore_20d": np.where((vm.notna()) & (vs > 1e-8), (v - vm) / vs, 0.0)}))
    if not fng.empty:
        parts.append(pd.DataFrame({"avail_ts": fng["avail_ts"], "macro_fng_norm": (fng["value"] - 50.0) / 50.0}))
    if not parts:
        return pd.DataFrame(columns=mcols)

    grid = pd.DataFrame({"avail_ts": np.array(sorted(set().union(*(p["avail_ts"].tolist() for p in parts))), dtype="int64")})
    for p in parts:
        grid = grid.merge(p, on="avail_ts", how="left")
    grid = grid.sort_values("avail_ts").ffill().reset_index(drop=True)
    for col in mcols[1:-1]:
        grid[col] = grid[col].replace([np.inf, -np.inf], np.nan).fillna(_NEUTRAL_DEFAULTS[col]) if col in grid.columns else _NEUTRAL_DEFAULTS[col]
    grid["macro_risk_off"] = ((grid["macro_vix_level"] > 25.0) | (grid["macro_dxy_ret_5d"] > 0.015)).astype(float)
    return grid[mcols]

def _load_macro_table(cache_db_path: Optional[str], min_ts_ms: int = 0, macro_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if macro_df is not None:
        return _compute_macro_avail_table(macro_df)
    ym = pd.Timestamp(max(0, int(min_ts_ms - _MACRO_WARMUP_MS)), unit="ms", tz="UTC").strftime("%Y-%m") if min_ts_ms > _MACRO_WARMUP_MS else "1970-01"
    sig = (_db_sig(cache_db_path), ym)
    def _fetch() -> pd.DataFrame:
        rows = _query_ro(cache_db_path, "SELECT series_name, date_str, value FROM macro_daily_history WHERE series_name IN ('DXY','VIX','SP500','FEAR_AND_GREED','STABLECOINS_CIRCULATING_USD','STABLECOIN_MCAP') AND date_str >= ? ORDER BY date_str ASC", (f"{ym}-01",))
        return _compute_macro_avail_table(pd.DataFrame(rows, columns=["series_name", "date_str", "value"]) if rows else None)
    return _cached_call(_MACRO_CACHE, sig, _fetch) if sig[0][0] else _fetch()

def _resample_closed_bars(df_src: Optional[pd.DataFrame], src_ms: int, target_ms: int) -> pd.DataFrame:
    base_cols, req = ["timestamp", "htf_close_ts", "open", "high", "low", "close", "volume"], ["timestamp", "open", "high", "low", "close", "volume"]
    if df_src is None or df_src.empty or src_ms <= 0 or target_ms <= src_ms or any(c not in df_src.columns for c in req):
        return pd.DataFrame(columns=base_cols)
    src = df_src.copy()
    for c in req:
        src[c] = pd.to_numeric(src[c], errors="coerce")
    src = src.dropna(subset=req)
    if src.empty:
        return pd.DataFrame(columns=base_cols)
    src["timestamp"] = src["timestamp"].astype("int64")
    src = src.assign(_p=src["exchange"].astype(str).str.lower() != "binance").sort_values(["timestamp", "_p"]).drop_duplicates("timestamp", keep="first") if "exchange" in src.columns else src.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    agg = src.set_index(pd.to_datetime(src["timestamp"], unit="ms", utc=True)).resample(f"{int(target_ms) // 60_000}min", origin="epoch", closed="left", label="left").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "timestamp": "max"}).dropna(subset=["open", "high", "low", "close", "timestamp"])
    if agg.empty:
        return pd.DataFrame(columns=base_cols)
    htf_open = ((agg.index - _EPOCH_UTC) // _ONE_MS).astype("int64").to_numpy()
    mask = (agg["timestamp"].astype("int64").to_numpy() + int(src_ms)) == (htf_open + int(target_ms))
    if not np.any(mask):
        return pd.DataFrame(columns=base_cols)
    agg, open_kept = agg.iloc[mask], htf_open[mask]
    return pd.DataFrame({"timestamp": open_kept, "htf_close_ts": open_kept + int(target_ms), "open": agg["open"].to_numpy("float64"), "high": agg["high"].to_numpy("float64"), "low": agg["low"].to_numpy("float64"), "close": agg["close"].to_numpy("float64"), "volume": agg["volume"].to_numpy("float64")})


def resample_1h_to_htf_closed(df_1h: Optional[pd.DataFrame], htf_ms: int) -> pd.DataFrame:
    """Resample closed `1h` bars using `origin="epoch"` and drop incomplete trailing HTF buckets."""
    return _resample_closed_bars(df_1h, _HOUR_MS, int(htf_ms))

def _load_closed_1h(symbol: str, cache_db_path: Optional[str], min_open_ts: int, max_close_ts: int) -> pd.DataFrame:
    max_open, cols = int(max_close_ts) - _HOUR_MS, ["timestamp", "open", "high", "low", "close", "volume"]
    rows = _query_ro(cache_db_path, "SELECT timestamp, open, high, low, close, volume FROM candles WHERE exchange='binance' AND symbol=? AND timeframe='1h' AND timestamp>=? AND timestamp<=? ORDER BY timestamp ASC", (symbol, int(min_open_ts), max_open)) or _query_ro(cache_db_path, "SELECT timestamp, open, high, low, close, volume FROM candles WHERE symbol=? AND timeframe='1h' AND timestamp>=? AND timestamp<=? ORDER BY timestamp ASC", (symbol, int(min_open_ts), max_open))
    return pd.DataFrame(rows, columns=cols).drop_duplicates("timestamp", keep="first") if rows else pd.DataFrame(columns=cols)

def _compute_h1_returns_table(df_1h: Optional[pd.DataFrame], prefix: str, with_vol: bool = False) -> pd.DataFrame:
    cols = ["h1_close_ts", f"{prefix}_ret_6h", f"{prefix}_ret_24h", *([f"{prefix}_vol_24h"] if with_vol else [])]
    if df_1h is None or df_1h.empty or not {"timestamp", "close"}.issubset(df_1h.columns):
        return pd.DataFrame(columns=cols)
    s = df_1h[["timestamp", "close"]].apply(pd.to_numeric, errors="coerce").dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if s.empty:
        return pd.DataFrame(columns=cols)
    c = s["close"]
    out: Dict[str, Any] = {"h1_close_ts": s["timestamp"].to_numpy("int64") + _HOUR_MS, f"{prefix}_ret_6h": c.pct_change(6, fill_method=None).to_numpy("float64"), f"{prefix}_ret_24h": c.pct_change(24, fill_method=None).to_numpy("float64")}
    if with_vol:
        out[f"{prefix}_vol_24h"] = c.pct_change(fill_method=None).rolling(24, min_periods=2).std().clip(lower=0.0).fillna(0.0).to_numpy("float64")
    return pd.DataFrame(out)

def _compute_sym_mtf_and_h1(df_1h_sym: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d4, d1 = resample_1h_to_htf_closed(df_1h_sym, TIMEFRAME_MS["4h"]), resample_1h_to_htf_closed(df_1h_sym, TIMEFRAME_MS["1d"])
    if not d4.empty:
        c4, h4, l4 = d4["close"], d4["high"], d4["low"]
        ema21, rsi14 = ta.trend.EMAIndicator(close=c4, window=21, fillna=False).ema_indicator(), ta.momentum.RSIIndicator(close=c4, window=14, fillna=False).rsi().fillna(50.0)
        atr14 = ta.volatility.AverageTrueRange(high=h4, low=l4, close=c4, window=14, fillna=False).average_true_range() if len(d4) >= 14 else pd.concat([h4 - l4, (h4 - c4.shift(1)).abs(), (l4 - c4.shift(1)).abs()], axis=1).max(axis=1).expanding(1).mean()
        f4h = pd.DataFrame({"htf_close_ts": d4["htf_close_ts"].to_numpy("int64"), "mtf_4h_dist_ema21": ((c4 - ema21) / (ema21 + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy("float64"), "mtf_4h_rsi14": rsi14.to_numpy("float64"), "mtf_4h_natr14": ((atr14 / (c4 + 1e-8)) * 100.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy("float64")})
    else:
        f4h = pd.DataFrame(columns=["htf_close_ts", "mtf_4h_dist_ema21", "mtf_4h_rsi14", "mtf_4h_natr14"])
    if not d1.empty:
        c1 = d1["close"]
        s50, s200 = ta.trend.SMAIndicator(close=c1, window=50, fillna=False).sma_indicator(), ta.trend.SMAIndicator(close=c1, window=200, fillna=False).sma_indicator()
        f1d = pd.DataFrame({"htf_close_ts": d1["htf_close_ts"].to_numpy("int64"), "mtf_1d_dist_sma50": ((c1 - s50) / (s50 + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy("float64"), "mtf_1d_dist_sma200": ((c1 - s200) / (s200 + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy("float64"), "mtf_1d_ret_5d": c1.pct_change(5, fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy("float64")})
    else:
        f1d = pd.DataFrame(columns=["htf_close_ts", "mtf_1d_dist_sma50", "mtf_1d_dist_sma200", "mtf_1d_ret_5d"])
    return f4h, f1d, _compute_h1_returns_table(df_1h_sym, "sym", with_vol=False)

def _resolve_sym_1h(df_src: pd.DataFrame, symbol: str, timeframe: str, tf_ms: int, cache_db_path: Optional[str], min_open_ts: int, max_close_ts: int, df_1h_sym: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    req = ["timestamp", "open", "high", "low", "close", "volume"]
    if df_1h_sym is not None:
        return df_1h_sym[pd.to_numeric(df_1h_sym["timestamp"], errors="coerce") + _HOUR_MS <= max_close_ts].copy()
    db_h1 = _load_closed_1h(symbol, cache_db_path, min_open_ts - _HTF_WARMUP_MS, max_close_ts)
    if timeframe == "1h" and all(c in df_src.columns for c in req):
        cur = df_src[req][pd.to_numeric(df_src["timestamp"], errors="coerce") + _HOUR_MS <= max_close_ts].copy()
        return pd.concat([db_h1[db_h1["timestamp"] < int(cur["timestamp"].min())] if not cur.empty else db_h1, cur], ignore_index=True) if not db_h1.empty else cur
    return db_h1 if not db_h1.empty else _resample_closed_bars(df_src, tf_ms, _HOUR_MS)[req]

def _asof_lookup(close_ts: np.ndarray, right: pd.DataFrame, rkey: str, cols: List[str]) -> Dict[str, np.ndarray]:
    if right.empty:
        return {c: np.full(len(close_ts), np.nan, dtype="float64") for c in cols}
    idx = np.searchsorted(right[rkey].to_numpy("int64"), close_ts, side="right") - 1
    valid, sidx = idx >= 0, np.clip(idx, 0, len(right) - 1)
    return {c: np.where(valid, right[c].to_numpy("float64")[sidx], np.nan) for c in cols}

def _external_arrays(close_ts: np.ndarray, symbol: str, ftbl: pd.DataFrame, mtbl: pd.DataFrame, f4h: pd.DataFrame, f1d: pd.DataFrame, fsym: pd.DataFrame, fbtc: pd.DataFrame) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for tbl, key, cols in (
        (ftbl, "funding_ts", ["funding_rate_bps", "funding_7d_ma_bps", "funding_zscore_30d"]),
        (mtbl, "avail_ts", ["macro_dxy_ret_5d", "macro_vix_level", "macro_vix_zscore_20d", "macro_sp500_ret_5d", "macro_fng_norm", "macro_stablecoin_growth_30d", "macro_risk_off"]),
        (f4h, "htf_close_ts", ["mtf_4h_dist_ema21", "mtf_4h_rsi14", "mtf_4h_natr14"]),
        (f1d, "htf_close_ts", ["mtf_1d_dist_sma50", "mtf_1d_dist_sma200", "mtf_1d_ret_5d"]),
    ):
        for c, v in _asof_lookup(close_ts, tbl, key, cols).items():
            out[c] = np.where(np.isfinite(v), v, _NEUTRAL_DEFAULTS[c])
    ab = _asof_lookup(close_ts, fbtc, "h1_close_ts", ["btc_ret_6h", "btc_ret_24h", "btc_vol_24h"])
    br6, br24, bvol = ab["btc_ret_6h"], ab["btc_ret_24h"], ab["btc_vol_24h"]
    out["btc_ret_6h"], out["btc_ret_24h"], out["btc_vol_24h"] = np.where(np.isfinite(br6), br6, 0.0), np.where(np.isfinite(br24), br24, 0.0), np.where(np.isfinite(bvol), bvol, 0.0)
    if str(symbol).upper() == "BTCUSDT":
        z = np.zeros(len(close_ts), dtype="float64")
        out["rel_btc_ret_6h"], out["rel_btc_ret_24h"] = z, z
    else:
        asym = _asof_lookup(close_ts, fsym, "h1_close_ts", ["sym_ret_6h", "sym_ret_24h"])
        sr6, sr24 = asym["sym_ret_6h"], asym["sym_ret_24h"]
        out["rel_btc_ret_6h"] = np.where(np.isfinite(sr6) & np.isfinite(br6), sr6 - br6, 0.0)
        out["rel_btc_ret_24h"] = np.where(np.isfinite(sr24) & np.isfinite(br24), sr24 - br24, 0.0)
    return out


def build_v2_features_batch(
    df_candles: pd.DataFrame, symbol: str, timeframe: str = "1h",
    cache_db_path: Optional[str] = None, funding_df: Optional[pd.DataFrame] = None,
    macro_df: Optional[pd.DataFrame] = None, df_1h_sym: Optional[pd.DataFrame] = None,
    df_1h_btc: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build causal v2 feature DataFrame retaining base OHLCV/ATR/SMA cols + finite float64 v2 cols."""
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    empty_cols = ["timestamp", "open", "high", "low", "close", "volume", *V2_FEATURE_COLUMNS]
    if df_candles is None or len(df_candles) == 0:
        return pd.DataFrame(columns=empty_cols)
    tf_ms, feat = TIMEFRAME_MS[timeframe], FeatureEngineer.compute_all_features(df_candles, drop_na=False)
    if feat.empty:
        return pd.DataFrame(columns=empty_cols)
    ts = pd.to_numeric(feat["timestamp"], errors="coerce").to_numpy("int64")
    feat["timestamp"] = ts
    for c in ("dist_resistance_pct", "dist_support_pct"):
        if c not in feat.columns:
            feat[c] = 0.0
    _apply_exchange_seam_volume_guard(feat)
    _add_range_and_regime_vol(feat)

    h_frac, d_frac = ((ts % _DAY_MS) // 60_000) / 1440.0, (((ts // _DAY_MS) + 3) % 7) / 7.0
    feat["hour_sin"], feat["hour_cos"] = np.sin(2.0 * np.pi * h_frac), np.cos(2.0 * np.pi * h_frac)
    feat["dow_sin"], feat["dow_cos"] = np.sin(2.0 * np.pi * d_frac), np.cos(2.0 * np.pi * d_frac)

    close_ts = ts + tf_ms
    min_open_ts, max_close_ts = int(ts[0]), int(close_ts[-1])
    ftbl = _load_funding_table(symbol, cache_db_path, min_open_ts, max_close_ts, funding_df=funding_df)
    mtbl = _load_macro_table(cache_db_path, min_ts_ms=min_open_ts, macro_df=macro_df)
    sym_1h = _resolve_sym_1h(feat, symbol, timeframe, tf_ms, cache_db_path, min_open_ts, max_close_ts, df_1h_sym)
    f4h, f1d, fsym = _compute_sym_mtf_and_h1(sym_1h)
    btc_1h = df_1h_btc[pd.to_numeric(df_1h_btc["timestamp"], errors="coerce") + _HOUR_MS <= max_close_ts].copy() if df_1h_btc is not None else (sym_1h if str(symbol).upper() == "BTCUSDT" else _load_closed_1h("BTCUSDT", cache_db_path, min_open_ts - _HTF_WARMUP_MS, max_close_ts))
    fbtc = _compute_h1_returns_table(btc_1h, "btc", with_vol=True)
    for c, arr in _external_arrays(close_ts, symbol, ftbl, mtbl, f4h, f1d, fsym, fbtc).items():
        feat[c] = arr
    for col in V2_FEATURE_COLUMNS:
        v = pd.to_numeric(feat[col], errors="coerce").to_numpy("float64")
        feat[col] = np.where(np.isfinite(v), v, _NEUTRAL_DEFAULTS[col])
    return feat


def enrich_v2_features_live(
    symbol: str, df_feat: Optional[pd.DataFrame], timeframe: str,
    cache_db_path: Optional[str], decision_ts_ms: int, micro_db_path: Optional[str] = None,
) -> Dict[str, float]:
    """Compute `<15ms` live v2 feature dict at `decision_ts_ms` with `<= 1e-9` batch parity."""
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    if df_feat is None or df_feat.empty or "timestamp" not in df_feat.columns:
        return dict(_NEUTRAL_DEFAULTS)

    tf_ms, dec_ts = TIMEFRAME_MS[timeframe], int(decision_ts_ms)
    ts_arr = pd.to_numeric(df_feat["timestamp"], errors="coerce").to_numpy("float64")
    closed = np.isfinite(ts_arr) & ((ts_arr + tf_ms) <= dec_ts)
    if not closed.any():
        return dict(_NEUTRAL_DEFAULTS)

    work = df_feat if closed.all() else df_feat.loc[closed]
    if all(c in work.columns for c in _TECH_STATIONARY_COLS):
        feat = work if work["timestamp"].is_monotonic_increasing else work.sort_values("timestamp")
        if "exchange" in feat.columns and feat["exchange"].astype(str).str.strip().str.lower().nunique(dropna=True) > 1:
            feat = feat.reset_index(drop=True).copy()
            _apply_exchange_seam_volume_guard(feat)
    else:
        feat = FeatureEngineer.compute_all_features(work, drop_na=False)
        if feat.empty:
            return dict(_NEUTRAL_DEFAULTS)
        for c in ("dist_resistance_pct", "dist_support_pct"):
            if c not in feat.columns:
                feat[c] = 0.0
        _apply_exchange_seam_volume_guard(feat)

    last_row = feat.iloc[-1]
    out: Dict[str, float] = {c: _safe_float(last_row[c], _NEUTRAL_DEFAULTS[c]) for c in _TECH_STATIONARY_COLS}
    if "garman_klass_vol_24" in feat.columns and "vol_regime_ratio" in feat.columns and len(feat) < 50:
        out["garman_klass_vol_24"] = max(0.0, _safe_float(last_row["garman_klass_vol_24"], 0.0))
        out["vol_regime_ratio"] = _safe_float(last_row["vol_regime_ratio"], 1.0)
    else:
        gk_arr = _gk_per_bar(feat.iloc[-24:]).to_numpy("float64")
        gk_fin = gk_arr[np.isfinite(gk_arr)]
        out["garman_klass_vol_24"] = math.sqrt(max(0.0, float(np.mean(gk_fin)))) if len(gk_fin) > 0 else 0.0
        natr_w = pd.to_numeric(feat["natr_14"].iloc[-50:], errors="coerce").to_numpy("float64")
        pos = natr_w[np.isfinite(natr_w) & (natr_w > 0.0)]
        out["vol_regime_ratio"] = (out["natr_14"] / (float(np.mean(pos)) + 1e-8)) if (out["natr_14"] > 0.0 and len(pos) >= 10) else 1.0

    last_open_ts = int(last_row["timestamp"])
    h_frac, d_frac = ((last_open_ts % _DAY_MS) // 60_000) / 1440.0, (((last_open_ts // _DAY_MS) + 3) % 7) / 7.0
    out["hour_sin"], out["hour_cos"] = math.sin(2.0 * math.pi * h_frac), math.cos(2.0 * math.pi * h_frac)
    out["dow_sin"], out["dow_cos"] = math.sin(2.0 * math.pi * d_frac), math.cos(2.0 * np.pi * d_frac)

    csig, msig = _db_sig(cache_db_path), _db_sig(micro_db_path)
    ftbl = _cached_call(_FUNDING_CACHE, (csig, msig, symbol, dec_ts // 60_000, dec_ts), lambda: _load_funding_table(symbol, cache_db_path, dec_ts, dec_ts, micro_db_path=micro_db_path))
    mtbl = _load_macro_table(cache_db_path, min_ts_ms=dec_ts)
    h_slot = (dec_ts // _HOUR_MS) * _HOUR_MS
    skey = ("SYM", csig, symbol, timeframe, h_slot, last_open_ts, out["log_returns"], len(feat))
    (f4h, f1d, fsym), sym_1h_ref = _cached_call(_H1_CACHE, skey, lambda: (lambda s1: (_compute_sym_mtf_and_h1(s1), s1))(_resolve_sym_1h(feat, symbol, timeframe, tf_ms, cache_db_path, dec_ts, dec_ts)))
    bkey: Tuple[Any, ...] = ("BTC_SELF", skey) if str(symbol).upper() == "BTCUSDT" else ("BTC", csig, h_slot)
    fbtc = _cached_call(_H1_CACHE, bkey, lambda: _compute_h1_returns_table(sym_1h_ref if str(symbol).upper() == "BTCUSDT" else _load_closed_1h("BTCUSDT", cache_db_path, dec_ts - _HTF_WARMUP_MS, dec_ts), "btc", with_vol=True))

    for c, arr in _external_arrays(np.array([dec_ts], dtype="int64"), symbol, ftbl, mtbl, f4h, f1d, fsym, fbtc).items():
        out[c] = _safe_float(arr[0], _NEUTRAL_DEFAULTS[c])
    return {col: float(out[col]) for col in V2_FEATURE_COLUMNS}
