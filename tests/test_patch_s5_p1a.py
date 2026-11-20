import os, sqlite3, datetime
import pandas as pd
import pytest

from data.live_context_enricher import LiveContextEnricher
from agents.orderflow_agent import OrderFlowAgent
from agents.macro_agent import MacroAgent
from agents.sentiment_agent import SentimentAgent
from agents.onchain_agent import OnChainAgent
from agents.risk_sentinel_agent import RiskSentinelAgent
from agents.base_agent import AgentSignal
from consensus_judge import ConsensusJudge


def _init_dbs(tmp_path):
    c_path, m_path = str(tmp_path / "cache.db"), str(tmp_path / "micro.db")
    with sqlite3.connect(c_path) as c:
        c.execute("CREATE TABLE macro_daily_history (series_name TEXT, date_str TEXT, timestamp_ms INTEGER, value REAL, extra_info TEXT, PRIMARY KEY (series_name, date_str))")
        c.execute("CREATE TABLE macro_events_calendar (event_name TEXT, date_str TEXT, timestamp_ms INTEGER, PRIMARY KEY (event_name, date_str))")
        c.execute("CREATE TABLE funding_rates (exchange TEXT, symbol TEXT, timestamp INTEGER, funding_rate REAL, mark_price REAL)")
    with sqlite3.connect(m_path) as m:
        m.execute("CREATE TABLE book_features (ts_ms INTEGER, symbol TEXT, spread_bps REAL, bid_n10 REAL, ask_n10 REAL, bid_n5 REAL, ask_n5 REAL)")
    return c_path, m_path


def test_d1_anti_lookahead_and_contextual_voting(tmp_path):
    cache_db, micro_db = _init_dbs(tmp_path)
    start_dt = datetime.datetime(2026, 10, 17, 0, 0, tzinfo=datetime.timezone.utc)
    with sqlite3.connect(cache_db) as con:
        # 32 daily rows: 2026-10-17 through 2026-11-17 (D-1)
        for i in range(32):
            cur = start_dt + datetime.timedelta(days=i)
            ds, ts = cur.strftime("%Y-%m-%d"), int(cur.timestamp() * 1000)
            dxy = 100.0 if i < 26 else 100.0 + (i - 25) * 1.0  # 5d ret > +0.005, > SMA20
            vix = 26.0 if i < 31 else 30.0                     # elevated D-1 VIX
            sp = 5000.0 if i < 26 else 5000.0 - (i - 25) * 40.0 # 5d ret < -0.01
            fng = 28.0 if i < 31 else 20.0                     # extreme fear D-1
            stbl = 100e9 + (i / 31.0) * 2e9                    # 30d growth > +0.5%
            tvl = 50e9 + (i / 31.0) * 3e9                      # 7d growth > +1.0%
            con.executemany("INSERT INTO macro_daily_history VALUES (?, ?, ?, ?, ?)", [
                ("DXY", ds, ts, dxy, ""), ("VIX", ds, ts, vix, ""),
                ("SP500", ds, ts, sp, ""), ("FEAR_AND_GREED", ds, ts, fng, ""),
                ("STABLECOINS_CIRCULATING_USD", ds, ts, stbl, ""), ("DEFI_TVL", ds, ts, tvl, ""),
            ])
        # Poisoned Day-D (2026-11-18 00:00 UTC) and D+1 (2026-11-19)
        d_ts = int((start_dt + datetime.timedelta(days=32)).timestamp() * 1000)
        dp1_ts = int((start_dt + datetime.timedelta(days=33)).timestamp() * 1000)
        con.executemany("INSERT INTO macro_daily_history VALUES (?, ?, ?, ?, ?)", [
            ("FEAR_AND_GREED", "2026-11-18", d_ts, 95.0, "p"), ("VIX", "2026-11-18", d_ts, 12.0, "p"),
            ("FEAR_AND_GREED", "2026-11-19", dp1_ts, 99.0, "p"), ("VIX", "2026-11-19", dp1_ts, 10.0, "p"),
        ])

    enricher = LiveContextEnricher(cache_db_path=cache_db, micro_db_path=micro_db)
    q_ts = int(datetime.datetime(2026, 11, 18, 15, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    ctx = enricher.enrich("BTCUSDT", {}, None, q_ts)

    assert ctx["vix_value"] == 30.0 and ctx["fear_and_greed"] == 20.0
    for k in ("dxy_ret_5d", "dxy_above_sma20", "vix_value", "sp500_ret_5d",
              "fear_and_greed", "stablecoin_growth_pct_30d", "tvl_growth_pct_7d", "tvl_change_24h"):
        assert k in ctx, f"Missing {k}"

    assert MacroAgent().evaluate(ctx).direction == "SELL"
    assert SentimentAgent().evaluate(ctx).direction == "BUY"
    assert OnChainAgent().evaluate(ctx).direction == "BUY"


def test_stale_macro_guard(tmp_path):
    cache_db, micro_db = _init_dbs(tmp_path)
    ts_ms = int(datetime.datetime(2026, 11, 18, 15, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    stale_ts = ts_ms - (15 * 86_400_000)
    with sqlite3.connect(cache_db) as con:
        con.executemany("INSERT INTO macro_daily_history VALUES (?, ?, ?, ?, ?)", [
            ("DXY", "2026-11-03", stale_ts, 102.0, ""), ("VIX", "2026-11-03", stale_ts, 22.0, ""),
            ("SP500", "2026-11-03", stale_ts, 4900.0, ""), ("FEAR_AND_GREED", "2026-11-03", stale_ts, 50.0, ""),
            ("STABLECOINS_CIRCULATING_USD", "2026-11-03", stale_ts, 100e9, ""), ("DEFI_TVL", "2026-11-03", stale_ts, 50e9, "")
        ])
    ctx = LiveContextEnricher(cache_db_path=cache_db, micro_db_path=micro_db).enrich("BTCUSDT", {}, None, ts_ms)
    for k in ("dxy_ret_5d", "dxy_above_sma20", "vix_value", "sp500_ret_5d",
              "fear_and_greed", "stablecoin_growth_pct_30d", "tvl_growth_pct_7d", "tvl_change_24h"):
        assert k not in ctx, f"Stale {k} was populated"


def test_macro_events_calendar_and_risksentinel(tmp_path):
    cache_db, micro_db = _init_dbs(tmp_path)
    cpi_ms = int(datetime.datetime(2026, 11, 12, 0, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    fomc_ms = int(datetime.datetime(2026, 11, 4, 0, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    with sqlite3.connect(cache_db) as con:
        con.execute("INSERT INTO macro_events_calendar VALUES (?, ?, ?)", ("US_CPI_RELEASE", "2026-11-12", cpi_ms))
        con.execute("INSERT INTO macro_events_calendar VALUES (?, ?, ?)", ("FOMC_DECISION", "2026-11-04", fomc_ms))

    enricher = LiveContextEnricher(cache_db_path=cache_db, micro_db_path=micro_db)
    sentinel = RiskSentinelAgent()

    t1 = int(datetime.datetime(2026, 11, 12, 13, 45, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    ctx1 = enricher.enrich("BTCUSDT", {}, None, t1)
    assert ctx1.get("is_cpi_fomc_window") is True
    assert sentinel.evaluate(ctx1).metrics.get("veto_type") == "SCHEDULED_MACRO_EVENT"

    t2 = int(datetime.datetime(2026, 11, 12, 10, 0, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    ctx2 = enricher.enrich("BTCUSDT", {}, None, t2)
    assert ctx2.get("is_cpi_fomc_window") is False
    assert sentinel.evaluate(ctx2).is_veto is False

    t3 = int(datetime.datetime(2026, 11, 4, 19, 15, tzinfo=datetime.timezone.utc).timestamp() * 1000)
    ctx3 = enricher.enrich("BTCUSDT", {}, None, t3)
    assert ctx3.get("is_cpi_fomc_window") is True
    assert sentinel.evaluate(ctx3).metrics.get("veto_type") == "SCHEDULED_MACRO_EVENT"


def test_live_funding_zscore_and_sentinel(tmp_path):
    cache_db, micro_db = _init_dbs(tmp_path)
    ts_ms = 1_700_000_000_000
    with sqlite3.connect(cache_db) as con:
        # Latest outlier 1h before ts_ms
        con.execute("INSERT INTO funding_rates VALUES (?, ?, ?, ?, ?)", ("binance", "BTCUSDT", ts_ms - 3_600_000, 0.0050, 50000.0))
        # 14 baseline rows at 8h intervals (0.0001 +/- 1e-5)
        for i in range(14):
            t = ts_ms - 3_600_000 - (i + 1) * 8 * 3_600_000
            con.execute("INSERT INTO funding_rates VALUES (?, ?, ?, ?, ?)", ("binance", "BTCUSDT", t, 0.0001 + (i % 3 - 1) * 1e-5, 50000.0))

    enricher = LiveContextEnricher(cache_db_path=cache_db, micro_db_path=micro_db)
    ctx = enricher.enrich("BTCUSDT", {}, None, ts_ms)
    assert ctx["funding_zscore"] > 3.5

    sentinel = RiskSentinelAgent()
    # RiskSentinelAgent enforces |funding_zscore| > 4.0; verify dislocation veto
    sig_ext = sentinel.evaluate({"funding_zscore": 4.5})
    assert sig_ext.is_veto is True and sig_ext.metrics.get("veto_type") == "EXTREME_FUNDING_DISLOCATION"

    # Also verify live trigger when expanding history samples beyond 18 rows (|z| > 4.0)
    with sqlite3.connect(cache_db) as con:
        for j in range(14, 22):
            con.execute("INSERT INTO funding_rates VALUES (?, ?, ?, ?, ?)",
                        ("binance", "BTCUSDT", ts_ms - 3_600_000 - (j + 1) * 8 * 3_600_000, 0.0001, 50000.0))
    ctx20 = enricher.enrich("BTCUSDT", {}, None, ts_ms)
    assert ctx20["funding_zscore"] > 4.0
    sig20 = sentinel.evaluate(ctx20)
    assert sig20.is_veto is True and sig20.metrics.get("veto_type") == "EXTREME_FUNDING_DISLOCATION"


def test_ohlcv_and_book_vetoes(tmp_path):
    cache_db, micro_db = _init_dbs(tmp_path)
    ts_ms = 1_000_000
    with sqlite3.connect(micro_db) as con:
        con.execute("INSERT INTO book_features VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ts_ms - 1000, "BTCUSDT", 25.0, None, None, 60.0, 40.0))

    enricher = LiveContextEnricher(cache_db_path=cache_db, micro_db_path=micro_db)
    sentinel = RiskSentinelAgent()

    # Illiquidity veto: vol_sma_20=1000, volume=100
    sig_illiq = sentinel.evaluate(enricher.enrich("BTCUSDT", {"volume": 100, "vol_sma_20": 1000}, None, ts_ms))
    assert sig_illiq.is_veto is True and sig_illiq.metrics.get("veto_type") == "ILLIQUIDITY"

    # Volatility spike: 50 rows (49 bars atr=10, last bar atr=30)
    df_feat = pd.DataFrame({"atr_14": [10.0] * 49 + [30.0], "volume": [1000] * 50})
    sig_vol = sentinel.evaluate(enricher.enrich("BTCUSDT", {"atr_14": 30.0}, df_feat, ts_ms))
    assert sig_vol.is_veto is True and sig_vol.metrics.get("veto_type") == "VOLATILITY_SPIKE"

    # Wide spread + bid_n5/ask_n5 fallback (bid_n10 is NULL)
    ctx_book = enricher.enrich("BTCUSDT", {}, None, ts_ms)
    assert ctx_book.get("micro_book_imb_10") == pytest.approx(0.20, abs=1e-4)
    sig_book = sentinel.evaluate(ctx_book)
    assert sig_book.is_veto is True and sig_book.metrics.get("veto_type") == "WIDE_SPREAD"


def test_fd_leak_check(tmp_path):
    cache_db, micro_db = _init_dbs(tmp_path)
    enricher = LiveContextEnricher(cache_db_path=cache_db, micro_db_path=micro_db)
    enricher.enrich("BTCUSDT", {}, None, 1_000_000)
    fds_before = len(os.listdir("/proc/self/fd"))
    for _ in range(30):
        enricher.enrich("BTCUSDT", {}, None, 1_000_000)
    fds_after = len(os.listdir("/proc/self/fd"))
    assert abs(fds_after - fds_before) <= 2


def test_orderflow_agent_microstructure_and_candles():
    agent = OrderFlowAgent()
    base = {"open": 100.0, "high": 100.7, "low": 99.5, "close": 100.0}

    # Real microstructure: BUY and SELL
    sig_buy = agent.evaluate({**base, "micro_taker_imb_5m": 0.35, "micro_book_imb_10": 0.25})
    assert sig_buy.direction == "BUY" and sig_buy.metrics["micro_score"] > 0.18

    sig_sell = agent.evaluate({**base, "micro_taker_imb_5m": -0.35, "micro_book_imb_10": -0.25})
    assert sig_sell.direction == "SELL"

    # Neutral tape suppresses weak/medium candle
    sig_neut = agent.evaluate({**base, "close": 100.35, "micro_taker_imb_5m": 0.05, "micro_book_imb_10": 0.02})
    assert sig_neut.direction == "HOLD"

    # Without microstructure: weak -> HOLD, strong -> BUY, volume-confirmed -> BUY
    assert agent.evaluate({**base, "close": 100.3, "volume": 100, "vol_sma_20": 100}).direction == "HOLD"
    assert agent.evaluate({**base, "close": 100.6, "volume": 100, "vol_sma_20": 100}).direction == "BUY"
    assert agent.evaluate({**base, "close": 100.3, "volume": 150, "vol_sma_20": 100}).direction == "BUY"


def test_consensus_judge_cross_regime_dampening():
    judge = ConsensusJudge()

    # 1. BEAR_TREND dampened
    sig1 = AgentSignal("A1", "BUY", 0.80, 1.0, "test", metrics={"accuracy": 0.333, "ece": 0.431, "total_samples": 12})
    res1 = judge.judge_deterministic([sig1], regime="BEAR_TREND").agent_signals[0]
    assert res1.confidence <= 0.55 and res1.metrics.get("calibrated_dampened") is True

    # 2. BULL_TREND undampened
    sig2 = AgentSignal("A2", "BUY", 0.80, 1.0, "test", metrics={"accuracy": 0.40, "ece": 0.20, "total_samples": 12})
    res2 = judge.judge_deterministic([sig2], regime="BULL_TREND").agent_signals[0]
    assert res2.confidence == 0.80 and not res2.metrics.get("calibrated_dampened")

    # 3. Small sample guard: total_samples=3 stays 0.80
    sig3 = AgentSignal("A3", "BUY", 0.80, 1.0, "test", metrics={"accuracy": 0.333, "ece": 0.431, "total_samples": 3})
    res3 = judge.judge_deterministic([sig3], regime="BEAR_TREND").agent_signals[0]
    assert res3.confidence == 0.80 and not res3.metrics.get("calibrated_dampened")
