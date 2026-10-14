"""
Testes unitários e de integração da Fase 1: Camada de Dados e Cache Persistente.
Valida:
1. Contratos de dados (CandleData, OrderBookDepth, MacroDataPoint, etc.)
2. Prevenção rigorosa de Lookahead Bias (release_date vs observation_date)
3. Conectores e mecanismos de fallback
4. DataLoader com cache SQLite local (/datasets/market_cache.db)
5. Sanitização, desduplicação e ordenação cronológica ascendente
6. Multi-cadência (alta freq vs baixa freq com TTL)
"""

import os
import sys
import asyncio
import tempfile
import unittest
from datetime import datetime

# Adiciona o diretório raiz ao sys.path para importação dos módulos
sys.path.insert(0, "/home/eudespatricio/ia_trader")

from data.data_integrations import (
    CandleData,
    OrderBookDepth,
    MacroDataPoint,
    SentimentPost,
    BitgetConnector,
    CoinGlassConnector,
    FredConnector,
    YahooFinanceConnector,
    CryptoPanicConnector,
    DefiLlamaConnector,
    CoinMarketCalConnector,
    DataIntegrationManager,
)
from data.data_loader import DataLoader


class TestDataIntegrations(unittest.TestCase):
    """Validação dos contratos e lógica pura dos conectores."""

    def test_candle_data_contract(self):
        c = CandleData(
            timestamp=1700000000000,
            open=50000.0,
            high=51000.0,
            low=49500.0,
            close=50800.0,
            volume=120.5,
            quote_volume=6121400.0,
            symbol="BTCUSDT",
            timeframe="1h",
        )
        d = c.to_dict()
        self.assertEqual(d["symbol"], "BTCUSDT")
        self.assertEqual(d["close"], 50800.0)
        self.assertEqual(d["timestamp"], 1700000000000)

    def test_order_book_spread_and_imbalance(self):
        bids = [(50000.0, 10.0), (49990.0, 5.0)]
        asks = [(50010.0, 5.0), (50020.0, 5.0)]
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        spread = round(best_ask - best_bid, 2)
        total_bid_vol = sum(b[1] for b in bids)  # 15.0
        total_ask_vol = sum(a[1] for a in asks)  # 10.0
        imbalance = round((total_bid_vol - total_ask_vol) / (total_bid_vol + total_ask_vol), 4)

        depth = OrderBookDepth(
            symbol="BTCUSDT",
            timestamp=1700000000000,
            bids=bids,
            asks=asks,
            spread=spread,
            imbalance=imbalance,
        )

        self.assertEqual(depth.spread, 10.0)
        self.assertEqual(depth.imbalance, 0.2)  # (15 - 10) / 25 = 5 / 25 = 0.2

    def test_fred_anti_lookahead_release_date(self):
        """Garante que a release_date calculada é posterior à observation_date."""
        fred = FredConnector()
        # CPI de Janeiro de 2024
        obs_date = "2024-01-01"
        rel_date = fred._calculate_effective_release_date("CPIAUCSL", obs_date)
        
        self.assertTrue(rel_date > obs_date, f"release_date {rel_date} deve ser estritamente posterior a {obs_date}")
        self.assertEqual(rel_date, "2024-02-14", "CPI de Jan deve ser divulgado em meados de Fev")

        # Taxa FEDFUNDS de 2024-03-01
        ff_obs = "2024-03-01"
        ff_rel = fred._calculate_effective_release_date("FEDFUNDS", ff_obs)
        self.assertTrue(ff_rel > ff_obs)
        self.assertEqual(ff_rel, "2024-03-02", "Taxa diária possui divulgação D+1")

    def test_sentiment_score_logic(self):
        """Validação do algoritmo de cálculo de sentimento da CryptoPanic."""
        p_bullish = SentimentPost("1", "BTC sobe forte", "2024-01-01", "CoinDesk", "bullish", 0.75)
        p_bearish = SentimentPost("2", "Queda brusca", "2024-01-01", "Bloomberg", "bearish", -0.50)
        self.assertGreater(p_bullish.score, 0)
        self.assertLess(p_bearish.score, 0)


class TestDataLoaderAndCache(unittest.TestCase):
    """Validação da camada orquestradora DataLoader com SQLite."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.loader = DataLoader(db_path=self.temp_db.name)

    def tearDown(self):
        asyncio.run(self.loader.close())
        if os.path.exists(self.temp_db.name):
            os.remove(self.temp_db.name)

    def test_database_schema_initialization(self):
        """Verifica se tabelas essenciais foram criadas no banco de cache."""
        with self.loader._get_connection() as conn:
            tables = [row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()]
        self.assertIn("candles", tables)
        self.assertIn("order_books", tables)
        self.assertIn("macro_series", tables)
        self.assertIn("low_freq_cache", tables)

    def test_candle_persistence_sanitization_and_sorting(self):
        """Testa inserção, desduplicação, correção de integridade e ordenação estrita ascendente."""
        raw_candles = [
            CandleData(1700000000000 + 3600000, 50100.0, 50500.0, 50000.0, 50400.0, 10.0, 504000.0, "BTCUSDT", "1h"),
            CandleData(1700000000000, 50000.0, 50200.0, 49900.0, 50100.0, 15.0, 751500.0, "BTCUSDT", "1h"),
            # Anômalo (preço zero, deve ser descartado pela sanitização)
            CandleData(1700000000000 + 7200000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "BTCUSDT", "1h"),
        ]

        self.loader._persist_candles(raw_candles)

        # Carrega do banco e sanitiza
        stored = self.loader._load_candles_from_db("BTCUSDT", "1h", limit=10)
        sanitized = self.loader._sanitize_candles(stored)

        # Deve ter descartado o candle com preço zerado
        self.assertEqual(len(sanitized), 2)
        # O primeiro elemento deve ser o mais antigo (1700000000000)
        self.assertEqual(sanitized[0]["timestamp"], 1700000000000)
        self.assertEqual(sanitized[1]["timestamp"], 1700000000000 + 3600000)
        self.assertLess(sanitized[0]["timestamp"], sanitized[1]["timestamp"], "Ordenação temporal deve ser estritamente ascendente")

    def test_ttl_cache_mechanism(self):
        """Valida que o cache de baixa frequência expira corretamente após o TTL."""
        category = "sentiment"
        key = "BTC_score"
        data = {"score": 0.65, "label": "bullish"}

        # Salva com TTL de 2 segundos
        self.loader._save_to_cache(category, key, data, ttl_seconds=2)
        cached = self.loader._get_from_cache(category, key)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["score"], 0.65)

        # Simula expiração alterando o timestamp do banco para o passado
        with self.loader._get_connection() as conn:
            conn.execute("UPDATE low_freq_cache SET expires_at = 0 WHERE category = ? AND cache_key = ?", (category, key))
            conn.commit()

        expired = self.loader._get_from_cache(category, key)
        self.assertIsNone(expired, "Registro expirado deve retornar None")

    def test_macro_anti_lookahead_filtering(self):
        """
        Valida que o DataLoader NÃO vaza informações futuras (release_date > data do trade).
        """
        with self.loader._get_connection() as conn:
            # Dado 1: CPI de Dezembro/2023, divulgado em 2024-01-12 (valor: 306.7)
            conn.execute("""
                INSERT OR REPLACE INTO macro_series (series_id, observation_date, release_date, value)
                VALUES ('CPIAUCSL', '2023-12-01', '2024-01-12', 306.7)
            """)
            # Dado 2: CPI de Janeiro/2024, divulgado em 2024-02-13 (valor: 308.4)
            conn.execute("""
                INSERT OR REPLACE INTO macro_series (series_id, observation_date, release_date, value)
                VALUES ('CPIAUCSL', '2024-01-01', '2024-02-13', 308.4)
            """)
            conn.commit()

        # Simula um trade realizado em 2024-01-20 (antes da divulgação do CPI de Jan em 2024-02-13)
        cpi_jan20 = self.loader._get_latest_macro_val("CPIAUCSL", max_release_date="2024-01-20")
        self.assertEqual(cpi_jan20, 306.7, "O modelo em 20/01/2024 deve enxergar apenas o CPI divulgado em 12/01/2024")

        # Simula um trade após 2024-02-13
        cpi_mar01 = self.loader._get_latest_macro_val("CPIAUCSL", max_release_date="2024-03-01")
        self.assertEqual(cpi_mar01, 308.4, "O modelo após 13/02/2024 pode enxergar o novo valor divulgado")


if __name__ == "__main__":
    unittest.main()
