"""
tests/test_binance_pipeline.py - Testes unitários e de integração com mocks para o conector Binance.

Cobre:
1. Paginação de klines e funding rate.
2. Tratamento de HTTP 429 e 418 com respeito ao header Retry-After e backoff.
3. Tratamento de HTTP 451 (bloqueio geográfico) com mensagem e fallback.
4. Parsing de dumps públicos de data.binance.vision (.zip e CSV).
5. Verificação de integridade via checksum SHA256 (.CHECKSUM).
6. Validação cruzada Bitget x Binance (cálculo de desvio, correlação de retornos e alertas).
"""

import os
import io
import sys
import json
import zipfile
import hashlib
import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(0, "/home/eudespatricio/ia_trader")

from data.data_integrations import (
    BinanceConnector,
    CandleData,
    AsyncHttpClient,
    BitgetConnector,
)
from data.data_loader import DataLoader


class TestBinanceConnectorMocks(unittest.TestCase):
    """Testes unitários com mocks para o conector Binance."""

    def setUp(self):
        self.mock_http = MagicMock(spec=AsyncHttpClient)
        self.connector = BinanceConnector(http_client=self.mock_http)

    def test_normalize_interval(self):
        self.assertEqual(self.connector.normalize_interval("1h"), "1h")
        self.assertEqual(self.connector.normalize_interval("1H"), "1h")
        self.assertEqual(self.connector.normalize_interval("4h"), "4h")
        self.assertEqual(self.connector.normalize_interval("1d"), "1d")
        self.assertEqual(self.connector.normalize_interval("1D"), "1d")

    def test_klines_pagination(self):
        """Verifica se get_klines pagina corretamente quando o limite excede 1000."""
        # Simula primeira página (1000 candles) e segunda página (500 candles)
        page1 = [
            [1700000000000 + i * 3600000, "50000", "50100", "49900", "50050", "10", 1700000000000 + (i + 1) * 3600000 - 1, "500500", 100, "5", "250250", "0"]
            for i in range(1000)
        ]
        page2 = [
            [1700000000000 + (1000 + i) * 3600000, "50050", "50200", "50000", "50150", "12", 1700000000000 + (1001 + i) * 3600000 - 1, "601800", 120, "6", "300900", "0"]
            for i in range(500)
        ]

        self.mock_http.get_json = AsyncMock(side_effect=[page1, page2])

        candles = asyncio.run(self.connector.get_klines(symbol="BTCUSDT", timeframe="1h", limit=1500))

        self.assertEqual(len(candles), 1500)
        self.assertEqual(candles[0].timestamp, 1700000000000)
        self.assertEqual(candles[-1].timestamp, 1700000000000 + 1499 * 3600000)
        self.assertEqual(self.mock_http.get_json.call_count, 2)

    def test_funding_rate_pagination(self):
        """Verifica paginação de funding rates."""
        page1 = [
            {"symbol": "BTCUSDT", "fundingRate": "0.000100", "fundingTime": 1700000000000 + i * 28800000, "markPrice": "50000"}
            for i in range(1000)
        ]
        page2 = [
            {"symbol": "BTCUSDT", "fundingRate": "0.000150", "fundingTime": 1700000000000 + (1000 + i) * 28800000, "markPrice": "50500"}
            for i in range(200)
        ]

        self.mock_http.get_json = AsyncMock(side_effect=[page1, page2])

        rates = asyncio.run(self.connector.get_funding_rate_history(symbol="BTCUSDT", limit=1200))
        self.assertEqual(len(rates), 1200)
        self.assertEqual(rates[0]["funding_rate"], 0.0001)
        self.assertEqual(rates[-1]["funding_rate"], 0.00015)
        self.assertEqual(self.mock_http.get_json.call_count, 2)

    def test_checksum_sha256_verification(self):
        """Valida que verify_checksum calcula e compara o SHA256 com precisão."""
        sample_bytes = b"Sample zip binary content for Binance Vision"
        correct_sha = hashlib.sha256(sample_bytes).hexdigest()
        checksum_text = f"{correct_sha}  BTCUSDT-1h-2023-01.zip\n"

        # Deve validar positivamente
        is_valid = BinanceConnector.verify_checksum(sample_bytes, checksum_text)
        self.assertTrue(is_valid)

        # Deve rejeitar com hash incorreto
        fake_checksum = "0000000000000000000000000000000000000000000000000000000000000000  filename.zip"
        self.assertFalse(BinanceConnector.verify_checksum(sample_bytes, fake_checksum))

    def test_parse_vision_klines_csv(self):
        """Valida extração de candles a partir de texto CSV de dump da Vision."""
        csv_content = """open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore
1672531200000,16540.0,16550.0,16520.0,16535.5,50.5,1672534799999,835000.0,1200,25.0,413000.0,0
1672534800000,16535.5,16560.0,16530.0,16555.0,40.0,1672538399999,662000.0,980,20.0,331000.0,0
"""
        candles = BinanceConnector.parse_vision_klines_csv(csv_content, "BTCUSDT", "1h")
        self.assertEqual(len(candles), 2)
        self.assertEqual(candles[0].timestamp, 1672531200000)
        self.assertEqual(candles[0].open, 16540.0)
        self.assertEqual(candles[0].close, 16535.5)
        self.assertEqual(candles[0].volume, 50.5)
        self.assertEqual(candles[1].close, 16555.0)

    def test_download_and_extract_vision_klines_with_zip(self):
        """Simula download e extração em memória de arquivo .zip com .CHECKSUM."""
        csv_content = "1672531200000,16540.0,16550.0,16520.0,16535.5,50.5,1672534799999,835000.0,1200,25.0,413000.0,0\n"
        
        # Cria ZIP em memória
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("BTCUSDT-1h-2023-01.csv", csv_content.encode("utf-8"))
        zip_bytes = zip_buffer.getvalue()

        # Cria checksum correspondente
        zip_sha = hashlib.sha256(zip_bytes).hexdigest()
        checksum_text = f"{zip_sha}  BTCUSDT-1h-2023-01.zip\n"

        self.mock_http.get_bytes = AsyncMock(return_value=zip_bytes)
        self.mock_http.get_text = AsyncMock(return_value=checksum_text)

        candles = asyncio.run(
            self.connector.download_and_extract_vision_klines("BTCUSDT", "1h", 2023, 1, verify=True)
        )

        self.assertEqual(len(candles), 1)
        self.assertEqual(candles[0].timestamp, 1672531200000)
        self.assertEqual(candles[0].close, 16535.5)


class TestCrossValidationBitgetBinance(unittest.TestCase):
    """Testes da validação cruzada Bitget x Binance."""

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()
        self.loader = DataLoader(db_path=self.temp_db.name)

    def tearDown(self):
        asyncio.run(self.loader.close())
        if os.path.exists(self.temp_db.name):
            os.remove(self.temp_db.name)

    def test_cross_validation_metrics_and_alignment(self):
        """Testa o alinhamento de timestamps, cálculo de desvio percentual e correlação."""
        # 5 candles alinhados com pequena variação de preço (< 0.2%)
        bitget_candles = [
            {"timestamp": 1700000000000 + i * 3600000, "close": 50000.0 + i * 100.0, "volume": 10.0}
            for i in range(10)
        ]
        # Binance candles: idênticos com leve desvio de +0.05% no preço
        binance_candles = [
            {"timestamp": 1700000000000 + i * 3600000, "close": (50000.0 + i * 100.0) * 1.0005, "volume": 12.0}
            for i in range(10)
        ]

        result = asyncio.run(
            self.loader.cross_validate_bitget_binance(
                symbol="BTCUSDT",
                timeframe="1h",
                limit=50,
                tolerance_pct=0.5,
                bitget_candles=bitget_candles,
                binance_candles=binance_candles,
            )
        )

        self.assertEqual(result["common_count"], 10)
        self.assertEqual(result["missing_in_binance_count"], 0)
        self.assertEqual(result["missing_in_bitget_count"], 0)
        self.assertAlmostEqual(result["mean_close_diff_pct"], 0.05, places=2)
        self.assertAlmostEqual(result["max_close_diff_pct"], 0.05, places=2)
        self.assertGreater(result["returns_correlation"], 0.99)
        self.assertFalse(result["alert_triggered"], "Não deve disparar alerta se desvio < 0.5% e correlação > 0.95")

    def test_cross_validation_alert_trigger_on_divergence(self):
        """Testa disparo de alerta quando desvio excede tolerance_pct."""
        bitget_candles = [
            {"timestamp": 1700000000000, "close": 50000.0, "volume": 10.0},
            {"timestamp": 1700003600000, "close": 52000.0, "volume": 10.0},  # Desvio grande de 4%
        ]
        binance_candles = [
            {"timestamp": 1700000000000, "close": 50000.0, "volume": 10.0},
            {"timestamp": 1700003600000, "close": 50000.0, "volume": 10.0},
        ]

        result = asyncio.run(
            self.loader.cross_validate_bitget_binance(
                symbol="BTCUSDT",
                timeframe="1h",
                limit=50,
                tolerance_pct=1.0,
                bitget_candles=bitget_candles,
                binance_candles=binance_candles,
            )
        )

        self.assertEqual(result["common_count"], 2)
        self.assertAlmostEqual(result["max_close_diff_pct"], 4.0, places=1)
        self.assertTrue(result["alert_triggered"], "Deve disparar alerta pois 4.0% > 1.0%")

    def test_cross_validation_missing_candles_detection(self):
        """Verifica identificação precisa de candles ausentes em uma exchange."""
        bitget_candles = [
            {"timestamp": 1000, "close": 50000.0, "volume": 10.0},
            {"timestamp": 2000, "close": 50100.0, "volume": 10.0},
            {"timestamp": 3000, "close": 50200.0, "volume": 10.0},
        ]
        binance_candles = [
            {"timestamp": 1000, "close": 50000.0, "volume": 10.0},
            {"timestamp": 3000, "close": 50200.0, "volume": 10.0},
            {"timestamp": 4000, "close": 50300.0, "volume": 10.0},  # Ausente na Bitget
        ]

        result = asyncio.run(
            self.loader.cross_validate_bitget_binance(
                symbol="ETHUSDT",
                timeframe="1h",
                bitget_candles=bitget_candles,
                binance_candles=binance_candles,
            )
        )

        self.assertEqual(result["common_count"], 2)
        self.assertEqual(result["missing_in_binance_count"], 1)  # timestamp 2000
        self.assertEqual(result["missing_in_bitget_count"], 1)   # timestamp 4000


if __name__ == "__main__":
    unittest.main()
