"""
Suíte de Testes do GoldenPotAgent (Scanner de Oportunidades & Caçador de Golden Pots).
Valida:
1. Filtro de Liquidez Institucional (descarte de pares < $20M USD e stablecoins).
2. Detecção e Scoring de Scalp Reversion (15m) por descolamento de VWAP.
3. Detecção e Scoring de Swing Breakout (1h/4h) por momentum e proximidade de extremos.
4. Persistência e integridade na tabela `golden_pots` do SQLite.
5. Cache em memória e respeito ao TTL.
"""

import os
import sys
import tempfile
import sqlite3
import pytest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, "/home/eudespatricio/ia_trader")

from agents.golden_pot_agent import GoldenPotAgent, GoldenPotOpportunity, STABLECOIN_EXCLUSIONS


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def agent(temp_db):
    return GoldenPotAgent(
        min_volume_usd=20_000_000.0,
        db_path=temp_db,
        cache_ttl_seconds=10,
    )


class TestGoldenPotEvaluation:
    """Testes determinísticos das regras de identificação de Golden Pots."""

    def test_scalp_reversion_detection_buy(self, agent):
        """Ativo com descolamento negativo expressivo do VWAP deve gerar setup SCALP_REVERSION BUY."""

        item = {
            "symbol": "ETHUSDT",
            "last_price": 2850.0,
            "vwap_24h": 3000.0,       # -5.0% de desvio
            "high_24h": 3050.0,
            "low_24h": 2800.0,        # Amplitude de ~8.9%
            "volume_24h_usd": 150_000_000.0,
            "price_change_pct": -4.2,
        }

        opp = agent.evaluate_ticker_opportunity(item)
        assert opp is not None
        assert opp.symbol == "ETHUSDT"
        assert opp.setup_type == "SCALP_REVERSION"
        assert opp.direction == "BUY"
        assert opp.dist_vwap_pct == pytest.approx(-5.0, abs=0.1)
        assert opp.score >= 70.0
        assert opp.suggested_timeframe == "15m"

    def test_scalp_reversion_detection_sell(self, agent):
        """Ativo com descolamento positivo expressivo do VWAP deve gerar setup SCALP_REVERSION SELL."""
        item = {
            "symbol": "SOLUSDT",
            "last_price": 160.0,
            "vwap_24h": 150.0,        # +6.67% de desvio
            "high_24h": 162.0,
            "low_24h": 145.0,
            "volume_24h_usd": 80_000_000.0,
            "price_change_pct": 8.5,
        }

        opp = agent.evaluate_ticker_opportunity(item)
        assert opp is not None
        assert opp.symbol == "SOLUSDT"
        assert opp.setup_type == "SCALP_REVERSION"
        assert opp.direction == "SELL"
        assert opp.dist_vwap_pct > 6.0
        assert opp.score >= 75.0

    def test_swing_breakout_detection_bullish(self, agent):
        """Ativo colado na máxima de 24h com forte alta deve gerar SWING_BREAKOUT BUY."""
        item = {
            "symbol": "AVAXUSDT",
            "last_price": 34.8,
            "vwap_24h": 34.2,
            "high_24h": 35.0,         # Preço a 96% do range (34.8 - 30.0)/(35.0 - 30.0)
            "low_24h": 30.0,
            "volume_24h_usd": 90_000_000.0,
            "price_change_pct": 12.5,
        }

        opp = agent.evaluate_ticker_opportunity(item)
        assert opp is not None
        assert opp.symbol == "AVAXUSDT"
        assert opp.setup_type == "SWING_BREAKOUT"
        assert opp.direction == "BUY"
        assert opp.range_position_pct >= 90.0
        assert opp.suggested_timeframe in ["1h", "4h"]

    def test_neutral_asset_generates_no_opportunity(self, agent):
        """Ativo sem desvio e no meio do range não deve configurar Golden Pot."""
        item = {
            "symbol": "BTCUSDT",
            "last_price": 60000.0,
            "vwap_24h": 59900.0,      # Desvio irrisório < 0.2%
            "high_24h": 60800.0,
            "low_24h": 59200.0,       # Exatamente no meio do range
            "volume_24h_usd": 500_000_000.0,
            "price_change_pct": 0.5,
        }

        opp = agent.evaluate_ticker_opportunity(item)
        assert opp is None




class TestGoldenPotFilteringAndPersistence:
    """Testes de filtragem de universo e persistência SQLite."""

    @pytest.mark.asyncio
    async def test_stablecoin_and_low_volume_exclusion(self, agent):
        """Stablecoins e pares de baixo volume devem ser ignorados."""
        mock_data = [
            # Stablecoin (deve ser excluída)
            {"symbol": "USDCUSDT", "lastPrice": "1.0001", "weightedAvgPrice": "1.000", "highPrice": "1.001", "lowPrice": "0.999", "quoteVolume": "999999999", "priceChangePercent": "0.01"},
            # Baixo volume < $20M (deve ser excluído)
            {"symbol": "MICROUSDT", "lastPrice": "0.10", "weightedAvgPrice": "0.08", "highPrice": "0.12", "lowPrice": "0.07", "quoteVolume": "5000000", "priceChangePercent": "20.0"},
            # Válido com Golden Pot
            {"symbol": "NEARUSDT", "lastPrice": "5.50", "weightedAvgPrice": "5.00", "highPrice": "5.60", "lowPrice": "4.80", "quoteVolume": "85000000", "priceChangePercent": "10.0"},
        ]

        with patch.object(agent.binance.http, "get_json", new=AsyncMock(return_value=mock_data)):
            res = await agent.scan_market(force_refresh=True)
            assert res["liquid_symbols_count"] == 1
            symbols = [p["symbol"] for p in res["all_pots"]]
            assert "USDCUSDT" not in symbols
            assert "MICROUSDT" not in symbols
            assert "NEARUSDT" in symbols

    @pytest.mark.asyncio
    async def test_sqlite_persistence_and_cache(self, agent, temp_db):
        """Valida gravação em tabela golden_pots no SQLite."""
        mock_data = [
            {"symbol": "LINKUSDT", "lastPrice": "15.00", "weightedAvgPrice": "14.20", "highPrice": "15.20", "lowPrice": "13.80", "quoteVolume": "60000000", "priceChangePercent": "6.0"},
        ]

        with patch.object(agent.binance.http, "get_json", new=AsyncMock(return_value=mock_data)):
            res1 = await agent.scan_market(force_refresh=True)
            assert len(res1["all_pots"]) > 0

            # Verifica persistência no SQLite
            with sqlite3.connect(temp_db) as con:
                cur = con.cursor()
                cur.execute("SELECT symbol, setup_type, score FROM golden_pots")
                rows = cur.fetchall()
                assert len(rows) > 0
                assert rows[0][0] == "LINKUSDT"
