"""
test_patch_s6_onda1.py - Independent QA Behavioral Test Suite for Patch S6 Onda 1.
Covers S6-2: Dynamic + Static Non-Crypto / TradFi Filtering in GOLDEN_POT.

Uses fake/stub HTTP transport and tmp_path DBs; zero live network calls and zero
modifications to production DBs.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
scripts_dir = os.path.join(ROOT, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from agents.golden_pot_agent import (
    GoldenPotAgent,
    NON_CRYPTO_BASE_EXCLUSIONS,
    NON_CRYPTO_SUBTYPES,
    GP_QUOTE_SUFFIXES,
)
from run_paper_simulation import GP_EXCLUDED_BASE_ASSETS, PaperTradingDaemon
from smart_trade import PaperTradingEngine

# ---------------------------------------------------------------------------
# Stub HTTP and Exchange Connectors for Hermetic Testing
# ---------------------------------------------------------------------------

class StubHttpClient:
    """Fake AsyncHttpClient returning canned responses and tracking call counts."""

    def __init__(self, routes: Optional[Dict[str, Any]] = None):
        self.routes = routes or {}
        self.call_counts: Dict[str, int] = {}

    async def get_json(self, url: str) -> Any:
        self.call_counts[url] = self.call_counts.get(url, 0) + 1
        resp = self.routes.get(url)
        if isinstance(resp, Exception):
            raise resp
        if callable(resp):
            return await resp() if asyncio.iscoroutinefunction(resp) else resp()
        return resp

    async def close(self) -> None:
        pass

class StubBinanceConnector:
    def __init__(self, http: StubHttpClient):
        self.http = http
        self.FUTURES_URL = "https://fapi.binance.com"

class StubBitgetConnector:
    def __init__(self, http: StubHttpClient):
        self.http = http
        self.BASE_URL = "https://api.bitget.com"

@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_market_cache.db"
    return str(db_file)

# ===========================================================================
# Contract 1: Exchange Info Classification
# ===========================================================================

def test_classify_non_crypto_underlying_types_non_coin():
    """Flags symbols whose underlyingType is non-COIN (EQUITY, COMMODITY, INDEX, etc.)."""
    payload = {
        "symbols": [
            {"symbol": "TSLAUSDT", "underlyingType": "EQUITY", "underlyingSubType": []},
            {"symbol": "KR005930USDT", "underlyingType": "KR_EQUITY", "underlyingSubType": []},
            {"symbol": "CN600519USDT", "underlyingType": "CN_EQUITY", "underlyingSubType": []},
            {"symbol": "HK0700USDT", "underlyingType": "HK_EQUITY", "underlyingSubType": []},
            {"symbol": "GOLDUSDT", "underlyingType": "COMMODITY", "underlyingSubType": []},
            {"symbol": "STRIPESTOCKUSDT", "underlyingType": "PREMARKET", "underlyingSubType": []},
            {"symbol": "SPXINDEXUSDT", "underlyingType": "INDEX", "underlyingSubType": []},
            # Case-insensitivity check
            {"symbol": "LOWERCASEEQUSDT", "underlyingType": "equity", "underlyingSubType": []},
        ]
    }
    flagged = GoldenPotAgent.classify_non_crypto_from_exchange_info(payload)
    expected = {
        "TSLAUSDT", "KR005930USDT", "CN600519USDT", "HK0700USDT",
        "GOLDUSDT", "STRIPESTOCKUSDT", "SPXINDEXUSDT", "LOWERCASEEQUSDT"
    }
    assert flagged == expected

def test_classify_non_crypto_underlying_subtypes_intersection():
    """Flags symbols whose underlyingSubType intersects NON_CRYPTO_SUBTYPES."""
    payload = {
        "symbols": [
            {"symbol": "TRADFI1USDT", "underlyingType": "COIN", "underlyingSubType": ["TRADFI"]},
            {"symbol": "PREIPO1USDT", "underlyingType": "COIN", "underlyingSubType": ["PRE-IPO"]},
            {"symbol": "PREMKT1USDT", "underlyingType": "COIN", "underlyingSubType": ["PRE-MARKET"]},
            {"symbol": "STOCK1USDT", "underlyingType": "COIN", "underlyingSubType": ["STOCK"]},
            {"symbol": "ETF1USDT", "underlyingType": "COIN", "underlyingSubType": ["ETF"]},
            {"symbol": "EQ1USDT", "underlyingType": "COIN", "underlyingSubType": ["EQUITY"]},
            {"symbol": "COMM1USDT", "underlyingType": "COIN", "underlyingSubType": ["COMMODITY"]},
            # Single string subtype instead of list
            {"symbol": "STRINGSUBSUSDT", "underlyingType": "COIN", "underlyingSubType": "TRADFI"},
            # Mixed lowercase subtype
            {"symbol": "LOWERSUBSUSDT", "underlyingType": "COIN", "underlyingSubType": ["stock", "ai"]},
        ]
    }
    flagged = GoldenPotAgent.classify_non_crypto_from_exchange_info(payload)
    expected = {
        "TRADFI1USDT", "PREIPO1USDT", "PREMKT1USDT", "STOCK1USDT",
        "ETF1USDT", "EQ1USDT", "COMM1USDT", "STRINGSUBSUSDT", "LOWERSUBSUSDT"
    }
    assert flagged == expected

def test_classify_non_crypto_preserves_coin_crypto_and_rwa():
    """Must preserve underlyingType=='COIN' crypto/RWA tokens without false positives."""
    payload = {
        "symbols": [
            {"symbol": "BTCUSDT", "underlyingType": "COIN", "underlyingSubType": []},
            {"symbol": "ETHUSDT", "underlyingType": "COIN", "underlyingSubType": []},
            {"symbol": "SOLUSDT", "underlyingType": "COIN", "underlyingSubType": []},
            {"symbol": "PAXGUSDT", "underlyingType": "COIN", "underlyingSubType": ["RWA"]},
            {"symbol": "XAUTUSDT", "underlyingType": "COIN", "underlyingSubType": ["RWA"]},
            {"symbol": "MANTRAUSDT", "underlyingType": "COIN", "underlyingSubType": ["LAYER1"]},
            {"symbol": "AIGENSYNUSDT", "underlyingType": "COIN", "underlyingSubType": ["AI"]},
        ]
    }
    flagged = GoldenPotAgent.classify_non_crypto_from_exchange_info(payload)
    assert flagged == frozenset(), f"Preserved crypto/RWA tokens were wrongly flagged: {flagged}"

def test_classify_non_crypto_handles_malformed_payloads_cleanly():
    """Handles None, empty dict, missing/non-list symbols, and bad items without raising."""
    bad_payloads = [
        None,
        {},
        {"symbols": None},
        {"symbols": "not_a_list"},
        {"symbols": 12345},
        {"other_key": [1, 2, 3]},
        "string_payload",
        [1, 2, 3],
        {
            "symbols": [
                None,
                {},
                "not_a_dict",
                {"symbol": None, "underlyingType": "EQUITY"},
                {"symbol": "", "underlyingType": "EQUITY"},
                {"symbol": "VALIDUSDT", "underlyingType": "COIN", "underlyingSubType": None},
            ]
        },
    ]
    for p in bad_payloads:
        res = GoldenPotAgent.classify_non_crypto_from_exchange_info(p)
        assert isinstance(res, frozenset)
        assert res == frozenset()

# ===========================================================================
# Contract 2: Dynamic Filtering, TTL Caching, and Bitget Fallback
# ===========================================================================

@pytest.mark.asyncio
async def test_fetch_market_universe_tickers_dynamic_tradfi_exclusion(temp_db):
    """fetch_market_universe_tickers dynamically excludes TradFi discovered from exchangeInfo."""
    exchange_info_url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    binance_tickers_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"

    http = StubHttpClient({
        exchange_info_url: {
            "symbols": [
                {"symbol": "BTCUSDT", "underlyingType": "COIN", "underlyingSubType": []},
                {"symbol": "XYZSTOCKUSDT", "underlyingType": "EQUITY", "underlyingSubType": []},
            ]
        },
        binance_tickers_url: [
            {
                "symbol": "BTCUSDT", "lastPrice": "65000.0", "weightedAvgPrice": "65000.0",
                "highPrice": "66000.0", "lowPrice": "64000.0", "quoteVolume": "100000000.0",
                "priceChangePercent": "2.5"
            },
            {
                "symbol": "XYZSTOCKUSDT", "lastPrice": "150.0", "weightedAvgPrice": "150.0",
                "highPrice": "155.0", "lowPrice": "145.0", "quoteVolume": "80000000.0",
                "priceChangePercent": "1.2"
            },
        ],
    })

    agent = GoldenPotAgent(
        min_volume_usd=20_000_000.0,
        db_path=temp_db,
        binance_connector=StubBinanceConnector(http),
        bitget_connector=StubBitgetConnector(http),
    )

    try:
        # Verify XYZSTOCKUSDT is NOT in static exclusions
        assert "XYZSTOCK" not in NON_CRYPTO_BASE_EXCLUSIONS

        tickers = await agent.fetch_market_universe_tickers()
        symbols = [t["symbol"] for t in tickers]
        assert "BTCUSDT" in symbols
        assert "XYZSTOCKUSDT" not in symbols, "Dynamic TradFi symbol must be excluded"
        assert agent.is_non_crypto_symbol("XYZSTOCKUSDT") is True
    finally:
        await agent.close()

@pytest.mark.asyncio
async def test_fetch_market_universe_tickers_ttl_caching_and_force(temp_db):
    """Caches exchangeInfo within 6h TTL; force=True forces refresh."""
    exchange_info_url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    binance_tickers_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"

    http = StubHttpClient({
        exchange_info_url: {
            "symbols": [
                {"symbol": "BTCUSDT", "underlyingType": "COIN", "underlyingSubType": []},
                {"symbol": "XYZSTOCKUSDT", "underlyingType": "EQUITY", "underlyingSubType": []},
            ]
        },
        binance_tickers_url: [
            {
                "symbol": "BTCUSDT", "lastPrice": "65000.0", "weightedAvgPrice": "65000.0",
                "highPrice": "66000.0", "lowPrice": "64000.0", "quoteVolume": "100000000.0",
                "priceChangePercent": "2.5"
            }
        ],
    })

    agent = GoldenPotAgent(
        min_volume_usd=20_000_000.0,
        db_path=temp_db,
        binance_connector=StubBinanceConnector(http),
        bitget_connector=StubBitgetConnector(http),
    )

    try:
        # First call: exchangeInfo must be queried once
        await agent.fetch_market_universe_tickers()
        assert http.call_counts[exchange_info_url] == 1

        # Second call immediately after: TTL cache must be used (NO new call to exchangeInfo)
        await agent.fetch_market_universe_tickers()
        assert http.call_counts[exchange_info_url] == 1

        # Force refresh: MUST re-request exchangeInfo
        await agent.refresh_non_crypto_symbols(force=True)
        assert http.call_counts[exchange_info_url] == 2
    finally:
        await agent.close()

@pytest.mark.asyncio
async def test_fetch_market_universe_tickers_bitget_fallback_applies_filters(temp_db):
    """When Binance 24hr tickers fail/empty, Bitget fallback filters both dynamic and static TradFi."""
    exchange_info_url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    binance_tickers_url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    bitget_tickers_url = "https://api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES"

    http = StubHttpClient({
        exchange_info_url: {
            "symbols": [
                {"symbol": "BTCUSDT", "underlyingType": "COIN", "underlyingSubType": []},
                {"symbol": "DYNAMICSTOCKUSDT", "underlyingType": "EQUITY", "underlyingSubType": []},
            ]
        },
        # Binance ticker fails with exception
        binance_tickers_url: RuntimeError("Binance 502 Bad Gateway"),
        # Bitget provides fallback universe
        bitget_tickers_url: {
            "data": [
                {"symbol": "BTCUSDT", "lastPr": "65000.0", "high24h": "66000.0", "low24h": "64000.0", "usdtVolume": "100000000.0", "change24h": "0.025"},
                {"symbol": "DYNAMICSTOCKUSDT", "lastPr": "200.0", "high24h": "205.0", "low24h": "195.0", "usdtVolume": "50000000.0", "change24h": "0.01"},
                {"symbol": "MUUSDT", "lastPr": "110.0", "high24h": "112.0", "low24h": "108.0", "usdtVolume": "60000000.0", "change24h": "0.03"},
                {"symbol": "USDCUSDT", "lastPr": "1.0", "high24h": "1.0", "low24h": "1.0", "usdtVolume": "50000000.0", "change24h": "0.0"},
            ]
        },
    })

    agent = GoldenPotAgent(
        min_volume_usd=20_000_000.0,
        db_path=temp_db,
        binance_connector=StubBinanceConnector(http),
        bitget_connector=StubBitgetConnector(http),
    )

    try:
        tickers = await agent.fetch_market_universe_tickers()
        symbols = [t["symbol"] for t in tickers]
        assert "BTCUSDT" in symbols
        assert "DYNAMICSTOCKUSDT" not in symbols, "Dynamic TradFi must be filtered out in Bitget loop"
        assert "MUUSDT" not in symbols, "Static TradFi must be filtered out in Bitget loop"
        assert "USDCUSDT" not in symbols, "Stablecoin must be filtered out in Bitget loop"
    finally:
        await agent.close()

# ===========================================================================
# Contract 3: Resilience on Exchange Info Failure
# ===========================================================================

@pytest.mark.asyncio
async def test_exchange_info_failure_logs_warning_and_preserves_dynamic_cache(temp_db):
    """When /fapi/v1/exchangeInfo raises an exception, logs warning and preserves cached dynamic set."""
    exchange_info_url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    http = StubHttpClient({
        exchange_info_url: RuntimeError("Binance network timeout")
    })

    agent = GoldenPotAgent(
        db_path=temp_db,
        binance_connector=StubBinanceConnector(http),
        bitget_connector=StubBitgetConnector(http),
    )

    try:
        # Pre-seed cache to simulate previous successful load
        agent._non_crypto_symbols = frozenset({"PREVIOUSDYNAMICUSDT"})
        agent._non_crypto_loaded_at = time.monotonic() - 30000.0  # expired TTL to trigger fetch

        # Should NOT raise
        res = await agent.refresh_non_crypto_symbols(force=True)
        assert "PREVIOUSDYNAMICUSDT" in res
        assert agent._non_crypto_symbols == frozenset({"PREVIOUSDYNAMICUSDT"})
        # Retry backoff set to ~300s in the future
        assert agent._non_crypto_retry_until > time.monotonic()
    finally:
        await agent.close()

def test_static_non_crypto_base_exclusions_coverage(temp_db):
    """Clean fallback to static NON_CRYPTO_BASE_EXCLUSIONS blocks required TradFi assets."""
    agent = GoldenPotAgent(db_path=temp_db)

    # Required symbols from contract 3
    required_exclusions = [
        "MUUSDT", "MUUUSDT", "ZHONGJIUSDT", "ZHIPUUSDT", "NATGASUSDT",
        "DRAMUSDT", "NBISUSDT", "EWYUSDT", "NVDAUSDT"
    ]
    for sym in required_exclusions:
        assert agent.is_non_crypto_symbol(sym) is True, f"{sym} should be excluded by static rules"

    # Crypto / RWA tokens must remain allowed
    allowed_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "PAXGUSDT", "XAUTUSDT", "MANTRAUSDT"]
    for sym in allowed_symbols:
        assert agent.is_non_crypto_symbol(sym) is False, f"{sym} should NOT be excluded"

@pytest.mark.asyncio
async def test_exchange_info_retry_backoff_avoids_spamming(temp_db):
    """Respects retry backoff so immediate subsequent scans do not spam /fapi/v1/exchangeInfo."""
    exchange_info_url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    http = StubHttpClient({
        exchange_info_url: RuntimeError("Binance 429 Too Many Requests")
    })

    agent = GoldenPotAgent(
        db_path=temp_db,
        binance_connector=StubBinanceConnector(http),
        bitget_connector=StubBitgetConnector(http),
    )

    try:
        # First call fails and sets retry backoff
        await agent.refresh_non_crypto_symbols(force=True)
        assert http.call_counts[exchange_info_url] == 1
        assert agent._non_crypto_retry_until > time.monotonic()

        # Immediate second call without force MUST NOT call HTTP
        await agent.refresh_non_crypto_symbols(force=False)
        assert http.call_counts[exchange_info_url] == 1

        # Invalid payload (not a dict or missing symbols list) also sets retry backoff
        http.routes[exchange_info_url] = {"symbols": "invalid"}
        agent._non_crypto_retry_until = 0.0  # reset backoff to allow one fetch
        await agent.refresh_non_crypto_symbols(force=False)
        assert http.call_counts[exchange_info_url] == 2
        assert agent._non_crypto_retry_until > time.monotonic()
    finally:
        await agent.close()

# ===========================================================================
# Contract 4: Daemon Integration and Exports
# ===========================================================================

def test_daemon_gp_excluded_base_assets_export_and_is_gp_excluded_symbol():
    """GP_EXCLUDED_BASE_ASSETS exports expanded static set; is_gp_excluded_symbol blocks TradFi."""
    # Verify export
    assert GP_EXCLUDED_BASE_ASSETS == NON_CRYPTO_BASE_EXCLUSIONS

    # Blocked symbols from contract
    blocked = ["MUUSDT", "ZHONGJIUSDT", "NATGASUSDT", "ANTUSDT", "EWYUSDT"]
    for sym in blocked:
        assert PaperTradingDaemon.is_gp_excluded_symbol(sym) is True, f"{sym} should be blocked"

    # Allowed symbols from contract
    allowed = ["BTCUSDT", "SOLUSDT", "ZECUSDT", "PAXGUSDT"]
    for sym in allowed:
        assert PaperTradingDaemon.is_gp_excluded_symbol(sym) is False, f"{sym} should be allowed"

def test_daemon_is_gp_symbol_excluded_combines_static_and_dynamic(temp_db):
    """_is_gp_symbol_excluded blocks both static TradFi and arbitrary dynamically discovered symbols."""
    engine = PaperTradingEngine(initial_capital=10000.0, db_path=temp_db)

    class FakeScannerWithDynamicFilter:
        def __init__(self, dynamic_symbols):
            self._dynamic = set(dynamic_symbols)

        def is_non_crypto_symbol(self, sym: str) -> bool:
            return sym in self._dynamic

        async def scan_market(self, force_refresh=False):
            return {"top_scalps": []}

    scanner = FakeScannerWithDynamicFilter(["ARBITRARYSTOCKUSDT", "NEWTECHUSDT"])
    daemon = PaperTradingDaemon(engine=engine, golden_pot_agent=scanner)

    # 1. Blocks static TradFi symbols (e.g. MUUSDT, ZHONGJIUSDT)
    assert daemon._is_gp_symbol_excluded("MUUSDT") is True
    assert daemon._is_gp_symbol_excluded("ZHONGJIUSDT") is True

    # 2. Blocks dynamically discovered symbols from scanner
    assert daemon._is_gp_symbol_excluded("ARBITRARYSTOCKUSDT") is True
    assert daemon._is_gp_symbol_excluded("NEWTECHUSDT") is True

    # 3. Allows clean crypto symbols
    assert daemon._is_gp_symbol_excluded("BTCUSDT") is False
    assert daemon._is_gp_symbol_excluded("ETHUSDT") is False
    assert daemon._is_gp_symbol_excluded("SOLUSDT") is False

    # 4. Graceful fallback if scanner has no is_non_crypto_symbol attribute
    daemon_no_method = PaperTradingDaemon(engine=engine, golden_pot_agent=object())
    assert daemon_no_method._is_gp_symbol_excluded("MUUSDT") is True
    assert daemon_no_method._is_gp_symbol_excluded("BTCUSDT") is False

@pytest.mark.asyncio
async def test_evaluate_golden_pot_opportunities_funnel_rejects_excluded_symbols(temp_db):
    """evaluate_golden_pot_opportunities skips excluded symbols with funnel reject 'excluded_symbol'."""
    engine = PaperTradingEngine(initial_capital=10000.0, db_path=temp_db)

    class MockScanner:
        def __init__(self, items, dynamic_exclusions):
            self.items = items
            self.dynamic = set(dynamic_exclusions)

        def is_non_crypto_symbol(self, sym: str) -> bool:
            return sym in self.dynamic

        async def scan_market(self, force_refresh=False):
            return {
                "top_swings": [self.items[0]],
                "top_scalps": self.items[1:],
            }

    pool_items = [
        {"symbol": "MUUSDT", "volume_24h_usd": 50_000_000.0},          # static TradFi
        {"symbol": "DYNAMICSTOCKUSDT", "volume_24h_usd": 40_000_000.0}, # dynamic TradFi
        {"symbol": "CRYPTOOKUSDT", "volume_24h_usd": 60_000_000.0},     # legitimate crypto
    ]

    scanner = MockScanner(pool_items, dynamic_exclusions=["DYNAMICSTOCKUSDT"])
    daemon = PaperTradingDaemon(engine=engine, golden_pot_agent=scanner)

    # Intercept _fr calls to verify funnel rejects
    funnel_calls = []
    original_fr = daemon._fr

    def spy_fr(strategy, sym, bar_ts, reason):
        funnel_calls.append((strategy, sym, reason))
        return original_fr(strategy, sym, bar_ts, reason)

    daemon._fr = spy_fr

    # Stub candle fetcher and feature computation so CRYPTOOKUSDT does not fail downstream
    async def fake_fetch(sym, now_ts):
        import pandas as pd
        return pd.DataFrame({"timestamp": [now_ts]})

    daemon._gp_fetch_closed_candles = fake_fetch
    daemon._compute_swing_features = lambda df: None  # will cause candidate to abort gracefully

    await daemon.evaluate_golden_pot_opportunities()

    # Verify both static (MUUSDT) and dynamic (DYNAMICSTOCKUSDT) received 'excluded_symbol' funnel rejects
    rejected_symbols = [sym for strat, sym, reason in funnel_calls if reason == "excluded_symbol"]
    assert "MUUSDT" in rejected_symbols, "MUUSDT should have been rejected with excluded_symbol"
    assert "DYNAMICSTOCKUSDT" in rejected_symbols, "DYNAMICSTOCKUSDT should have been rejected with excluded_symbol"
    assert "CRYPTOOKUSDT" not in rejected_symbols, "CRYPTOOKUSDT should not have been rejected with excluded_symbol"
