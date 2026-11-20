"""tests/test_wave_p2_ml_and_hardening.py
Wave P2 Independent QA Test Suite: ML v2, deduplication, hot-reloading & hardening.
"""
import asyncio, json, os, sqlite3, time
from unittest.mock import MagicMock, patch
import joblib, numpy as np, pandas as pd, pytest

from scripts.audit_and_dedup_candles import audit_databases, deduplicate_candles
from data.ml_features_v2 import (
    V2_FEATURE_COLUMNS, build_v2_features_batch, enrich_v2_features_live, clear_v2_cache
)
from model_trainer import PurgedTimeSeriesSplit, ModelTrainer
from agents.technical_agent import _DefaultMLEnsemble, TechnicalAgent
from learning_tracker import ModelEvolutionManager
from scripts.run_paper_simulation import PaperTradingDaemon
from scripts.binance_ws_collector import BinanceFuturesWSCollector
from data.live_context_enricher import LiveContextEnricher
from agents.tape_reading_agent import TapeReadingAgent
from scripts.backtest_scalp_15m_calibration import simulate_symbol, PreparedSymbolData
import scripts.eval_and_train_ml_v2 as ev


class _DummyClassifier:
    def __init__(self, cols=None):
        self.cols, self.classes_ = cols, np.array([0, 1, 2])

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame) and self.cols is not None:
            assert list(X.columns) == self.cols
        return np.array([[0.1, 0.7, 0.2]] * (len(X) if hasattr(X, "__len__") else 1))


def test_audit_and_deduplicate_candles(tmp_path):
    cache_db, micro_db, bkp_dir = str(tmp_path / "cache.db"), str(tmp_path / "micro.db"), str(tmp_path / "backups")
    con = sqlite3.connect(cache_db)
    con.executescript("""
        CREATE TABLE candles (symbol TEXT, timeframe TEXT, exchange TEXT, timestamp INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL);
        CREATE TABLE funding_rates (symbol TEXT, timestamp INTEGER, funding_rate REAL);
        CREATE TABLE macro_daily_history (series_name TEXT, date_str TEXT, value REAL);
    """)
    ts_bin = 1700006400000
    rows = [
        ("ETHUSDT", "4h", "binance", 1000, 10, 11, 9, 10, 100),
        ("ETHUSDT", "4h", "bitget", 1000, 10, 11, 9, 10, 50),
        ("BTCUSDT", "1d", "binance", ts_bin, 30000, 31000, 29000, 30500, 200),
        ("BTCUSDT", "1d", "bitget", ts_bin + 16 * 3600 * 1000, 30000, 31000, 29000, 30500, 100),
        ("SOLUSDT", "15m", "bitget", 10000, 20, 21, 19, 20, 10),
        ("SOLUSDT", "15m", "binance", 20000, 20, 21, 19, 20, 500),
    ]
    con.executemany("INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.execute("INSERT INTO funding_rates VALUES ('BTCUSDT', 1000, 0.0001), ('BTCUSDT', 1000, 0.0001)")
    con.execute("INSERT INTO macro_daily_history VALUES ('VIX', '2023-11-15', 15.0), ('VIX', '2023-11-15', 15.0)")
    con.commit(); con.close()

    m_con = sqlite3.connect(micro_db)
    m_con.executescript("CREATE TABLE binance_series (ts_ms INTEGER PRIMARY KEY, symbol TEXT); INSERT INTO binance_series VALUES (1000, 'BTCUSDT');")
    m_con.close()

    rep = audit_databases(cache_db, micro_db)
    assert rep["exact_duplicates"]["total_rowids"] == 1 and "ETHUSDT:4h" in rep["exact_duplicates"]["grouped_counts"]
    assert rep["offset_1d_duplicates"]["total_rowids"] == 1 and "BTCUSDT" in rep["offset_1d_duplicates"]["counts_by_symbol"]
    assert len(rep["non_overlapping_seams"]) == 1 and rep["non_overlapping_seams"][0]["symbol"] == "SOLUSDT"
    assert rep["funding_rates_duplicates"] == 1 and rep["macro_daily_duplicates"] == 1
    assert rep["microstructure"]["binance_series"]["pk_duplicates"] == 0

    dedup = deduplicate_candles(cache_db, bkp_dir)
    assert dedup["status"] == "DEDUPLICATED" and dedup["total_deleted"] == 2
    assert dedup["integrity_check"].lower() == "ok" and os.path.exists(dedup["backup_file"])

    c_con = sqlite3.connect(cache_db)
    rem_rows = c_con.execute("SELECT symbol, timeframe, exchange FROM candles").fetchall()
    c_con.close()
    assert len(rem_rows) == 4
    exchanges = {(r[0], r[1], r[2]) for r in rem_rows}
    assert ("SOLUSDT", "15m", "bitget") in exchanges and ("SOLUSDT", "15m", "binance") in exchanges
    assert ("ETHUSDT", "4h", "binance") in exchanges and ("BTCUSDT", "1d", "binance") in exchanges

    dedup2 = deduplicate_candles(cache_db, bkp_dir)
    assert dedup2["status"] == "CLEAN" and dedup2["total_deleted"] == 0


def test_ml_features_v2_anti_lookahead_and_perturbation():
    n, start = 80, 1700006400000
    ts = np.arange(start, start + n * 3600_000, 3600_000, dtype=np.int64)
    df1 = pd.DataFrame({
        "timestamp": ts, "open": 100.0 + np.sin(np.arange(n)), "high": 105.0 + np.sin(np.arange(n)),
        "low": 95.0 + np.sin(np.arange(n)), "close": 101.0 + np.sin(np.arange(n)), "volume": 1000.0 + np.arange(n)
    })
    k, cutoff_ts = 40, ts[40] + 3600_000
    f_ts = np.arange(start, start + n * 3600_000, 8 * 3600_000, dtype=np.int64)
    funding1 = pd.DataFrame({"timestamp": f_ts, "symbol": "BTCUSDT", "funding_rate": 0.0001 * (1 + (f_ts % 3))})
    funding2 = funding1.copy(); funding2.loc[funding2["timestamp"] > cutoff_ts, "funding_rate"] = 0.099

    dates = pd.date_range("2023-11-15", periods=10, freq="D").strftime("%Y-%m-%d")
    macro1 = pd.concat([pd.DataFrame({"series_name": name, "date_str": dates, "value": 10.0 + np.arange(10)})
                        for name in ["DXY", "VIX", "SP500", "FEAR_AND_GREED", "STABLECOIN_MCAP"]], ignore_index=True)
    macro2 = macro1.copy(); macro2.loc[macro2["date_str"] > "2023-11-17", "value"] *= 50.0

    df2 = df1.copy(); df2.loc[k + 1:, "close"] *= 2.0; df2.loc[k + 1:, "volume"] *= 5.0
    f1 = build_v2_features_batch(df1, "BTCUSDT", "1h", funding_df=funding1, macro_df=macro1)
    f2 = build_v2_features_batch(df2, "BTCUSDT", "1h", funding_df=funding2, macro_df=macro2)

    for col in V2_FEATURE_COLUMNS:
        diff = np.max(np.abs(f1.iloc[:k + 1][col].to_numpy() - f2.iloc[:k + 1][col].to_numpy()))
        assert diff == 0.0, f"Anti-lookahead leakage in column {col}: {diff}"


def test_ml_features_v2_seam_and_funding_zscore():
    n, start = 60, 1700006400000
    ts = np.arange(start, start + n * 900_000, 900_000, dtype=np.int64)
    df = pd.DataFrame({
        "timestamp": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": [10.0] * 30 + [10000.0] * 30, "exchange": ["bitget"] * 30 + ["binance"] * 30
    })
    feat = build_v2_features_batch(df, "SOLUSDT", "15m")
    switch_bar = feat.iloc[30]
    assert switch_bar["vol_sma_ratio"] == 1.0 and switch_bar["volume_zscore_20"] == 0.0

    f_ts = np.arange(start - 100 * 8 * 3600_000, start + n * 900_000, 8 * 3600_000, dtype=np.int64)
    funding = pd.DataFrame({"timestamp": f_ts, "symbol": "SOLUSDT", "funding_rate": 0.0001 * np.sin(np.linspace(0, 10, len(f_ts)))})
    f_res = build_v2_features_batch(df, "SOLUSDT", "15m", funding_df=funding)
    z, ma = f_res["funding_zscore_30d"].to_numpy(), f_res["funding_7d_ma_bps"].to_numpy()
    assert np.all(np.isfinite(z)) and np.std(z) > 0.05
    assert np.all(np.isfinite(ma)) and np.std(ma) > 0.01


def test_ml_features_v2_batch_vs_live_parity(tmp_path):
    clear_v2_cache()
    db = str(tmp_path / "cache.db")
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE candles (symbol TEXT, timeframe TEXT, exchange TEXT, timestamp INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL);
        CREATE TABLE funding_rates (symbol TEXT, timestamp INTEGER, funding_rate REAL);
        CREATE TABLE macro_daily_history (series_name TEXT, date_str TEXT, value REAL);
    """)
    n, start = 60, 1700006400000
    ts = np.arange(start, start + n * 3600_000, 3600_000, dtype=np.int64)
    rows = [("BTCUSDT", "1h", "binance", int(t), 100.0 + i, 105.0 + i, 95.0 + i, 101.0 + i, 500.0 + i) for i, t in enumerate(ts)]
    con.executemany("INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    f_ts = np.arange(start - 20 * 8 * 3600_000, start + n * 3600_000, 8 * 3600_000, dtype=np.int64)
    con.executemany("INSERT INTO funding_rates VALUES (?, ?, ?)", [("BTCUSDT", int(t), 0.0001) for t in f_ts])
    dates = pd.date_range("2023-11-01", periods=30, freq="D").strftime("%Y-%m-%d")
    for name in ["DXY", "VIX", "SP500", "FEAR_AND_GREED", "STABLECOIN_MCAP"]:
        con.executemany("INSERT INTO macro_daily_history VALUES (?, ?, ?)", [(name, d, 20.0) for d in dates])
    con.commit(); con.close()

    df_candles = pd.read_sql_query("SELECT * FROM candles WHERE symbol='BTCUSDT' AND timeframe='1h' ORDER BY timestamp", sqlite3.connect(db))
    batch = build_v2_features_batch(df_candles, "BTCUSDT", "1h", cache_db_path=db)
    last_row = batch.iloc[-1]
    live = enrich_v2_features_live("BTCUSDT", df_candles, "1h", db, int(last_row["timestamp"]) + 3600_000)

    for col in V2_FEATURE_COLUMNS:
        bv, lv = float(last_row[col]), float(live[col])
        assert np.isfinite(bv) and np.isfinite(lv) and abs(bv - lv) <= 1e-9, f"Parity mismatch in {col}: batch={bv}, live={lv}"


def test_purged_time_series_split_and_trainer_metadata(tmp_path):
    bars_ts = np.arange(1000, 1000 + 40 * 100, 100, dtype=np.int64)
    ts = np.repeat(bars_ts, 3)
    n_samples = len(ts)
    X = pd.DataFrame({"f1": np.random.randn(n_samples), "f2": np.random.randn(n_samples)})
    y = pd.Series(np.random.randint(0, 3, n_samples))

    splitter = PurgedTimeSeriesSplit(n_splits=3, horizon_embargo=2, min_train_ratio=0.4)
    splits = splitter.split(X, y, timestamps=ts, embargo_ms=200)
    assert len(splits) == 3
    for tr, te in splits:
        assert len(set(tr).intersection(set(te))) == 0 and np.min(ts[te]) >= np.max(ts[tr]) + 200

    legacy_splits = splitter.split(X, y)
    assert len(legacy_splits) == 3
    for tr, te in legacy_splits:
        assert np.min(te) >= np.max(tr) + splitter.horizon_embargo

    trainer = ModelTrainer(model_type="rf", models_dir=str(tmp_path))
    trainer.train_walk_forward = lambda *a, **kw: {"acc": 0.8}
    trainer.train_full_model(X, y, save_name="m.joblib", extra_metadata={"schema_version": "2.0.0", "feature_pipeline": "ml_features_v2"})
    loaded = joblib.load(str(tmp_path / "m.joblib"))
    assert loaded["schema_version"] == "2.0.0" and loaded["feature_pipeline"] == "ml_features_v2"
    assert not any(fn.startswith(".tmp_") for fn in os.listdir(tmp_path))


def test_heterogeneous_ml_ensemble_and_technical_agent_reload(tmp_path, monkeypatch):
    m1, m2 = _DummyClassifier(["a", "b"]), _DummyClassifier(["x", "y"])
    ens = _DefaultMLEnsemble(models_with_features=[(m1, ["a", "b"]), (m2, ["x", "y"])])
    p_df = ens.predict_proba(pd.DataFrame({"a": [1.0], "b": [2.0], "x": [3.0], "y": [4.0]}))
    assert p_df.shape == (1, 3) and np.isclose(np.sum(p_df), 1.0)
    p_dict = ens.predict_proba({"a": 1.0, "x": 2.0})
    assert p_dict.shape == (1, 3) and np.isclose(np.sum(p_dict), 1.0)

    p1, p2 = str(tmp_path / "lgbm.joblib"), str(tmp_path / "rf.joblib")
    joblib.dump({"model": m1, "feature_names": ["a", "b"]}, p1)
    joblib.dump({"model": m2, "feature_names": ["x", "y"]}, p2)

    TechnicalAgent.invalidate_default_model_cache()
    monkeypatch.setattr(TechnicalAgent, "_get_model_dir_paths", classmethod(lambda cls: [p1, p2]))
    try:
        loaded = TechnicalAgent._get_default_model()
        assert loaded is not None and len(loaded.models) == 2

        time.sleep(0.02)
        joblib.dump({"model": m1, "feature_names": ["a", "b"], "v": 2}, p1)
        reloaded = TechnicalAgent._get_default_model()
        assert reloaded is not None and reloaded is not loaded
    finally:
        TechnicalAgent.invalidate_default_model_cache()
    assert TechnicalAgent._cached_default_model is None


def test_model_evolution_manager_and_paper_daemon_v2(tmp_path, monkeypatch):
    champ_path = str(tmp_path / "lgbm_model_v1.joblib")
    joblib.dump({"model": "fake", "feature_pipeline": "ml_features_v2", "schema_version": "2.0.0", "features": ["a", "b"]}, champ_path)
    mtime_before = os.path.getmtime(champ_path)

    mem = ModelEvolutionManager(model_dir=str(tmp_path))
    res = mem.run_retraining_cycle(symbol="BTCUSDT", timeframe="1h", model_type="lgbm")
    assert res.get("status") == "SKIPPED_V2_GUARD" and os.path.getmtime(champ_path) == mtime_before

    p15 = str(tmp_path / "lgbm_15m_v1.joblib")
    joblib.dump({"model": _DummyClassifier(["f1"]), "feature_names": ["f1"], "feature_pipeline": "ml_features_v2"}, p15)

    daemon = PaperTradingDaemon.__new__(PaperTradingDaemon)
    daemon._scalp_ml15_signature = None
    daemon._scalp_ml15_models = []
    daemon.enable_live_context_enrichment = True
    daemon.tape_agent = MagicMock()
    daemon.engine = MagicMock(db_path=str(tmp_path / "cache.db"))

    monkeypatch.setattr(PaperTradingDaemon, "_scalp_ml15_paths", staticmethod(lambda: [p15]))
    daemon.invalidate_scalp_ml15_cache()
    assert len(daemon._scalp_ml15_models) == 1

    time.sleep(0.02)
    joblib.dump({"model": _DummyClassifier(["f1", "f2"]), "feature_names": ["f1", "f2"], "feature_pipeline": "ml_features_v2"}, p15)
    df_feat = pd.DataFrame({"timestamp": [1000], "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [50.0], "f1": [1.0], "f2": [2.0]})
    probs = daemon._predict_scalp_ml15_probs(df_feat.iloc[0], df_feat=df_feat, sym="BTCUSDT", decision_ts_ms=1000 + 900_000)
    assert probs is not None and probs[0] == 0.7 and probs[1] == 0.2


def test_p2d_binance_ws_and_live_context_enricher(tmp_path):
    received = []
    def on_trade(sym, ts_ms, p, q, m):
        time.sleep(0.002); received.append((sym, ts_ms, p, q, m))

    collector = BinanceFuturesWSCollector(symbols=["BTCUSDT"], on_trade=on_trade)
    assert collector.ws_base_url.endswith("/market/stream")

    async def _test_worker():
        q = asyncio.Queue()
        task = asyncio.create_task(collector._dispatch_worker(q))
        for i in range(4): await q.put(("BTCUSDT", 1000 + i, 50000.0, 1.0, True))
        await asyncio.sleep(0.02)
        collector.stop_event.set()
        await task

    asyncio.run(_test_worker())
    assert len(received) == 4

    items = [{"date": "2026-11-19", "timestamp": 1000, "close": 100.0}, {"date": "2026-11-20", "timestamp": 2000, "close": 101.0}]
    parsed = LiveContextEnricher._parse_yf(items, utc_today="2026-11-20")
    assert len(parsed) == 1 and parsed[0][0] == "2026-11-19"

    db = str(tmp_path / "micro.db")
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE binance_series (ts_ms INTEGER, symbol TEXT, kind TEXT, v1 REAL, v2 REAL, v3 REAL);
        INSERT INTO binance_series VALUES (1000000, 'BTCUSDT', 'taker', 10, 100, 50);
        INSERT INTO binance_series VALUES (1900000, 'BTCUSDT', 'taker', 10, 120, 60);
    """)
    con.close()

    enricher = LiveContextEnricher(cache_db_path=db, micro_db_path=db)
    assert enricher._fetch_all(db, "SELECT * FROM missing_tbl") == [] and enricher._enrich_error_count == 0
    assert enricher._fetch_all(db, "SELECT BAD_SYNTAX") == [] and enricher._enrich_error_count == 1

    ctx = enricher.enrich("BTCUSDT", {"timestamp": 2500000}, None, 2500000)
    assert ctx["enrich_errors"] == 1 and abs(ctx.get("micro_taker_imb_5m", 0.0) - (1.0 / 3.0)) < 1e-4


def test_p2d_tape_reading_and_backtest_maker_fill(tmp_path):
    db = str(tmp_path / "micro.db")
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE binance_series (ts_ms INTEGER, symbol TEXT, kind TEXT, v1 REAL, v2 REAL, v3 REAL);
        INSERT INTO binance_series VALUES (1000000, 'BTCUSDT', 'taker', 10, 100, 50);
        INSERT INTO binance_series VALUES (1900000, 'BTCUSDT', 'taker', 10, 120, 60);
        CREATE TABLE active_symbols (foo INTEGER);
    """)

    agent = TapeReadingAgent(db_path=db)
    res = agent._fetch_binance_series_taker(con, "BTCUSDT", 2500000)
    assert res is not None and abs(res[0] - (1.0 / 3.0)) < 1e-4

    con.execute("DELETE FROM binance_series WHERE ts_ms = 1900000"); con.commit()
    assert agent._fetch_binance_series_taker(con, "BTCUSDT", 1100000) is not None
    agent.refresh_covered_symbols_from_db()
    con.close()

    n = 10
    df = pd.DataFrame({
        "timestamp": np.arange(1000, 1000 + n * 100, 100, dtype=np.int64),
        "open": [100.0] * n, "high": [110.0] * n, "low": [90.0] * n, "close": [101.0] * n,
        "atr_14": [1.0] * n, "volume": [100.0] * n
    })
    p_buy, in_sqz = np.zeros(n), np.zeros(n, dtype=bool)
    p_buy[3], in_sqz[1], in_sqz[2] = 0.8, True, True

    df_skip = df.copy(); df_skip.loc[4, "low"] = 102.0
    d_skip = PreparedSymbolData("BTCUSDT", df_skip, p_buy, np.zeros(n), np.array([50.0]*n), np.array([2.0]*n), in_sqz, np.array([100.0]*n), np.array([90.0]*n), np.array([0.01]*n))
    assert len(simulate_symbol(d_skip, 0.0, -999.0, 0.5, maker_fill_mode="realistic")) == 0

    df_fill = df.copy(); df_fill.loc[4, "low"] = 100.0
    d_fill = PreparedSymbolData("BTCUSDT", df_fill, p_buy, np.zeros(n), np.array([50.0]*n), np.array([2.0]*n), in_sqz, np.array([100.0]*n), np.array([90.0]*n), np.array([0.01]*n))
    assert len(simulate_symbol(d_fill, 0.0, -999.0, 0.5, maker_fill_mode="realistic")) == 1


def test_eval_and_train_ml_v2_gates_and_dry_run(tmp_path):
    def _cand(name, ver, n_trades, pf, er, zero_l=0, zero_r=0, dsr=0.96):
        return {
            "name": name, "version": ver,
            "oos": {"n_trades": n_trades, "profit_factor": pf, "expectancy_r": er, "sharpe": 1.5},
            "zero_imp_lgbm": zero_l, "zero_imp_rf": zero_r, "dsr": {"dsr": dsr}
        }

    v1 = _cand("v1", "v1", 50, 1.2, 0.05)
    assert ev.classify_suite_gate([v1, _cand("v2", "v2", 50, 1.5, 0.1, dsr=0.96)], 0.10)["decision"] == "PASS_ALL"
    assert ev.classify_suite_gate([v1, _cand("v2", "v2", 50, 1.5, 0.1, dsr=0.90)], 0.10)["decision"] == "PARTIAL_DSR_BLOCK"
    assert ev.classify_suite_gate([v1, _cand("v2", "v2", 50, 1.1, 0.02, dsr=0.96)], 0.10)["decision"] == "FAIL_REFUTED"
    assert ev.classify_suite_gate([v1, _cand("v2", "v2", 15, 2.0, 0.5, dsr=0.99)], 0.05)["decision"] == "FAIL_REFUTED"

    models_dir = str(tmp_path / "models")
    rep_path = str(tmp_path / "report.json")
    os.makedirs(models_dir)

    mock_df = pd.DataFrame({"timestamp": [1000, 2000], "open": [100, 101], "high": [102, 103], "low": [99, 100], "close": [101, 102], "volume": [10, 20], "sma_200": [100.0, 100.0]})
    for c in ev.V2_FEATURE_COLUMNS: mock_df[c] = 0.0

    def mock_eval(spec, *args, **kwargs):
        return {
            "name": spec.name, "version": spec.version, "timeframe": spec.timeframe, "train_symbols": list(spec.train_symbols),
            "horizon": spec.horizon, "multiple_tp": spec.multiple_tp, "multiple_sl": spec.multiple_sl, "decision_threshold": spec.decision_threshold,
            "embargo_ms": 100, "dev_samples": 50, "n_features": 10, "zero_imp_lgbm": 0, "zero_imp_rf": 0, "zero_imp_both": 0,
            "feature_health": {}, "walk_forward_dev": {},
            "oos": {"n_trades": 25, "profit_factor": 1.6 if spec.version == "v2" else 1.2, "expectancy_r": 0.15, "sharpe": 1.8, "win_rate_pct": 55.0, "total_net_r": 3.75, "total_return_pct": 5.0, "max_dd": 0.02},
            "_daily_rets": np.array([0.01, -0.005, 0.02, 0.001, -0.002, 0.015, -0.001, 0.008, 0.004, -0.003])
        }

    with patch.object(ev, "load_candles_ro", return_value=mock_df), \
         patch.object(ev, "load_suite_features", return_value=({s: mock_df for s in ev.MAJORS}, {s: mock_df for s in ev.MAJORS})), \
         patch.object(ev, "_evaluate_candidate", side_effect=mock_eval), \
         patch.object(ev, "run_micro_ablation_15m", return_value={"status": "OK", "oos_delta": {"profit_factor_delta": 0.1, "expectancy_r_delta": 0.02, "sharpe_delta": 0.1}, "window_start_utc": "2026-01-01", "window_end_utc": "2026-03-01"}):

        ev.run_benchmark("cache.db", "micro.db", models_dir, rep_path, log_trials=True, promote_1h=True, promote_15m=True, dry=True)
        assert os.path.exists(rep_path) and len(os.listdir(models_dir)) == 0
        with open(rep_path) as f:
            data = json.load(f)
        assert data["suites"]["1h"]["decision"] in ("PASS_ALL", "PARTIAL_DSR_BLOCK", "FAIL_REFUTED")

    sanitized = ev._sanitize_feature_frame(pd.DataFrame({"macro_vix_level": [np.nan], "rsi_14": [np.inf]}), ["macro_vix_level", "rsi_14", "vol_sma_ratio"])
    assert sanitized["macro_vix_level"].iloc[0] == 18.0 and sanitized["rsi_14"].iloc[0] == 50.0 and sanitized["vol_sma_ratio"].iloc[0] == 1.0
