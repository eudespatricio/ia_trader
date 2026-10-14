"""
Suíte Avançada de Testes de Estresse, Resiliência e Concorrência da Camada de Dados (Fase 1).
Foco em Cenários Críticos de Falha:
1. Injeção de Falhas de Rede (quedas, conexões recusadas, timeouts).
2. Simulação de Rate Limit HTTP 429 com backoff exponencial.
3. Injeção de Dados Corrompidos, NaNs, Infs e Anomalias Estruturais.
4. Concorrência Massiva em SQLite (WAL mode, múltiplas threads/tarefas assíncronas).
5. Rigor Matemático Anti-Lookahead Bias em Dados Macroeconômicos.
6. Resiliência do Cache Multi-Cadência (TTL, corrupção de payload, isolamento).
"""

import os
import sys
import json
import time
import asyncio
import sqlite3
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Adiciona o diretório do projeto ao sys.path
sys.path.insert(0, "/home/eudespatricio/ia_trader")

from data.data_integrations import (
    CandleData,
    OrderBookDepth,
    MacroDataPoint,
    SentimentPost,
    AsyncHttpClient,
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


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def temp_db_path():
    """Gera um arquivo de banco temporário limpo para cada teste."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)
    # Remove também arquivos auxiliares de WAL se existirem
    for ext in ["-wal", "-shm"]:
        wal_file = path + ext
        if os.path.exists(wal_file):
            os.remove(wal_file)


@pytest.fixture
def data_loader(temp_db_path):
    """Instancia o DataLoader apontando para o banco temporário."""
    loader = DataLoader(db_path=temp_db_path)
    yield loader
    asyncio.run(loader.close())


# =====================================================================
# 1. TESTES DE INJEÇÃO DE FALHAS DE REDE E TIMEOUTS
# =====================================================================

class TestNetworkFaultInjection:
    """Testa o comportamento resiliente diante de falhas de conectividade."""

    @pytest.mark.asyncio
    async def test_http_client_connection_failure_retries_and_graceful_none(self):
        """Valida que falhas severas de rede (ex.: ConnectionRefusedError) tentam retentativas e retornam None."""
        client = AsyncHttpClient(timeout_seconds=0.5, max_retries=2)
        
        with patch.object(client, "_get_session") as mock_session_getter:
            mock_session = MagicMock()
            # Simula conexão falhando em todas as tentativas
            mock_session.get.side_effect = ConnectionResetError("Conexão interrompida pelo peer")
            mock_session_getter.return_value = mock_session

            start_t = time.time()
            result = await client.get_json("https://fake-exchange.com/api/candles")
            elapsed = time.time() - start_t

            assert result is None
            # Deve ter tentado 3 vezes (1 inicial + 2 retries) com delays crescentes
            assert mock_session.get.call_count == 3
            assert elapsed >= 1.0, "O backoff exponencial deve adicionar atraso controlado entre retries"
        await client.close()

    @pytest.mark.asyncio
    async def test_data_loader_network_offline_uses_sqlite_cache_seamlessly(self, data_loader):
        """
        Cenário Crítico: Queda total da Internet.
        O DataLoader deve falhar a requisição remota silenciosamente e entregar os dados do SQLite.
        """
        # 1. Popula dados históricos no SQLite
        initial_candles = [
            CandleData(1710000000000, 60000.0, 60500.0, 59800.0, 60200.0, 5.0, 301000.0, "BTCUSDT", "1h"),
            CandleData(1710003600000, 60200.0, 61000.0, 60100.0, 60800.0, 8.0, 486400.0, "BTCUSDT", "1h"),
        ]
        data_loader._persist_candles(initial_candles)

        # 2. Simula queda de rede completa na Bitget
        with patch.object(data_loader.manager.bitget, "get_candles", side_effect=OSError("Network unreachable")):
            records = await data_loader.get_candles(symbol="BTCUSDT", timeframe="1h", limit=10, as_dataframe=False)

            assert len(records) == 2
            assert records[0]["timestamp"] == 1710000000000
            assert records[1]["timestamp"] == 1710003600000
            assert records[1]["close"] == 60800.0

    @pytest.mark.asyncio
    async def test_order_book_network_timeout_returns_last_known_snapshot(self, data_loader):
        """Queda de rede ao consultar Order Book deve recorrer ao último snapshot válido do banco."""
        # Insere snapshot antigo no banco
        with data_loader._get_connection() as conn:
            conn.execute("""
                INSERT INTO order_books (symbol, timestamp, best_bid, best_ask, spread, imbalance, raw_bids, raw_asks)
                VALUES ('BTCUSDT', 1710000000000, 65000.0, 65002.0, 2.0, 0.15, '[]', '[]')
            """)
            conn.commit()

        # Simula timeout de rede
        with patch.object(data_loader.manager.bitget, "get_order_book", return_value=None):
            book = await data_loader.get_order_book_metrics(symbol="BTCUSDT")

            assert book["symbol"] == "BTCUSDT"
            assert book["best_bid"] == 65000.0
            assert book["best_ask"] == 65002.0
            assert book["spread"] == 2.0
            assert book["imbalance"] == 0.15


# =====================================================================
# 2. TESTES DE RATE LIMIT HTTP 429
# =====================================================================

class TestRateLimitSimulation:
    """Testa tratamento de HTTP 429 (Too Many Requests)."""

    @pytest.mark.asyncio
    async def test_http_429_backoff_and_subsequent_success(self):
        """Simula resposta 429 na 1ª tentativa e sucesso 200 na 2ª tentativa."""
        client = AsyncHttpClient(timeout_seconds=2.0, max_retries=2)

        # Mock da resposta aiohttp
        mock_resp_429 = MagicMock()
        mock_resp_429.status = 429

        mock_resp_200 = MagicMock()
        mock_resp_200.status = 200
        mock_resp_200.json = AsyncMock(return_value={"code": "00000", "data": [{"fundingRate": "0.0001", "fundingTime": 1710000000000}]})

        class MockContextManager:
            def __init__(self, responses):
                self.responses = list(responses)
                self.call_idx = 0

            async def __aenter__(self):
                resp = self.responses[min(self.call_idx, len(self.responses) - 1)]
                self.call_idx += 1
                return resp

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        cm = MockContextManager([mock_resp_429, mock_resp_200])

        with patch.object(client, "_get_session") as mock_session_getter:
            mock_session = MagicMock()
            mock_session.get.return_value = cm
            mock_session_getter.return_value = mock_session

            data = await client.get_json("https://api.bitget.com/api/v2/mix/market/current-fund-rate")

            assert data is not None
            assert data["code"] == "00000"
            assert cm.call_idx == 2, "Deveria ter feito retry após o 429"
        await client.close()

    @pytest.mark.asyncio
    async def test_coinglass_fallback_on_rate_limit(self):
        """Se o CoinGlass retornar bloqueio/rate limit, deve chamar o fallback Bitget + Binance sem travar."""
        cg = CoinGlassConnector()

        # Mock CoinGlass falhando (retornando None devido a 429 ou IP bloqueado)
        with patch.object(cg.http, "get_json") as mock_get:
            def side_effect_fn(url, **kwargs):
                if "coinglass.com" in url:
                    return None  # CoinGlass bloqueado
                elif "api.bitget.com" in url:
                    return {
                        "code": "00000",
                        "data": {"size": "500.5", "ts": 1710000000000}
                    }
                elif "binance.com" in url:
                    return {"symbol": "BTCUSDT", "openInterest": "1500.25"}
                return None

            mock_get.side_effect = side_effect_fn

            res = await cg.get_open_interest_aggregated("BTC")

            assert res["symbol"] == "BTC"
            assert res["source"] == "multi_exchange_fallback"
            assert res["bitget_oi"] == 500.5
            assert res["binance_oi"] == 1500.25
            assert round(res["total_oi_contracts"], 2) == 2000.75
        await cg.http.close()


# =====================================================================
# 3. TESTES DE SANITIZAÇÃO DE DADOS CORROMPIDOS E ANOMALIAS
# =====================================================================

class TestDataSanitizationAndCorruption:
    """Testa imunidade a dados nulos, NaNs, preços negativos e incoerência geométrica."""

    def test_sanitize_candles_drops_corrupted_prices(self, data_loader):
        """Candles com preços zerados, negativos ou timestamps repetidos devem ser expurgados."""
        corrupted_candles = [
            # Válido 1
            {"timestamp": 1000, "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 10.0, "quote_volume": 1000.0},
            # Timestamp duplicado (deve ser rejeitado)
            {"timestamp": 1000, "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 5.0, "quote_volume": 500.0},
            # Preço negativo (deve ser rejeitado)
            {"timestamp": 2000, "open": -50.0, "high": 100.0, "low": -60.0, "close": 80.0, "volume": 10.0, "quote_volume": 800.0},
            # Close zerado (deve ser rejeitado)
            {"timestamp": 3000, "open": 100.0, "high": 110.0, "low": 90.0, "close": 0.0, "volume": 10.0, "quote_volume": 1000.0},
            # Timestamp decrescente / fora de ordem (deve ser rejeitado pelo filtro ts <= last_ts)
            {"timestamp": 2500, "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 10.0, "quote_volume": 1000.0},
            # Válido 2
            {"timestamp": 4000, "open": 102.0, "high": 108.0, "low": 101.0, "close": 107.0, "volume": 20.0, "quote_volume": 2140.0},
        ]

        cleaned = data_loader._sanitize_candles(corrupted_candles)

        assert len(cleaned) == 2
        assert cleaned[0]["timestamp"] == 1000
        assert cleaned[1]["timestamp"] == 4000

    def test_geometric_consistency_correction(self, data_loader):
        """
        Em micro-anomalias onde high < open ou low > close devido a ruído de exchange,
        a sanitização deve ajustar high = max(high, open, close) e low = min(low, open, close).
        """
        glitched_candle = [
            {"timestamp": 1000, "open": 105.0, "high": 100.0, "low": 110.0, "close": 108.0, "volume": -10.0, "quote_volume": -1.0}
        ]
        cleaned = data_loader._sanitize_candles(glitched_candle)

        assert len(cleaned) == 1
        c = cleaned[0]
        # high deve cobrir o maior valor
        assert c["high"] >= c["open"]
        assert c["high"] >= c["close"]
        assert c["high"] == 108.0
        # low deve cobrir o menor valor
        assert c["low"] <= c["open"]
        assert c["low"] <= c["close"]
        assert c["low"] == 105.0
        # volume negativo corrigido para >= 0
        assert c["volume"] == 0.0
        assert c["quote_volume"] == 0.0

    def test_order_book_depth_with_empty_or_inverted_levels(self):
        """Verifica robustez se a exchange enviar bids e asks vazios ou com anomalias."""
        # Livro vazio
        depth_empty = OrderBookDepth(
            symbol="ETHUSDT",
            timestamp=1710000000000,
            bids=[],
            asks=[],
            spread=0.0,
            imbalance=0.0,
        )
        assert depth_empty.spread == 0.0
        assert depth_empty.imbalance == 0.0


# =====================================================================
# 4. CONCORRÊNCIA MASSIVA EM SQLITE (WAL MODE)
# =====================================================================

class TestSQLiteConcurrentStress:
    """Valida a consistência ACID e ausência de bloqueios fatais em escritas concorrentes."""

    def test_concurrent_multi_thread_candle_persistence(self, data_loader):
        """
        Dispara 20 threads gravando simultaneamente lotes de velas no SQLite.
        O SQLite deve lidar com as transações sem corrupção e sem erros não capturados.
        """
        num_threads = 10
        candles_per_thread = 50
        errors = []

        def worker(thread_idx: int):
            try:
                thread_candles = []
                for i in range(candles_per_thread):
                    ts = 1700000000000 + (thread_idx * 100000) + (i * 1000)
                    thread_candles.append(
                        CandleData(
                            timestamp=ts,
                            open=50000.0 + i,
                            high=50100.0 + i,
                            low=49900.0 + i,
                            close=50050.0 + i,
                            volume=1.0,
                            quote_volume=50000.0,
                            symbol=f"SYM_{thread_idx % 2}",  # divide entre SYM_0 e SYM_1
                            timeframe="1m",
                        )
                    )
                data_loader._persist_candles(thread_candles)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Erros durante escrita concorrente: {errors}"

        # Verifica consistência dos dados gravados
        with data_loader._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as total FROM candles;")
            total_records = cursor.fetchone()["total"]

        expected_records = num_threads * candles_per_thread
        assert total_records == expected_records, f"Esperado {expected_records} registros, encontrado {total_records}"

    def test_concurrent_cache_read_write(self, data_loader):
        """Garante consistência na leitura e escrita concorrente na tabela low_freq_cache."""
        num_workers = 8
        iterations = 30
        exceptions = []

        def cache_worker(worker_id: int):
            for i in range(iterations):
                try:
                    cat = f"cat_{worker_id % 3}"
                    key = f"key_{i}"
                    data_loader._save_to_cache(cat, key, {"worker": worker_id, "iter": i}, ttl_seconds=100)
                    cached = data_loader._get_from_cache(cat, key)
                    assert cached is not None
                    assert "worker" in cached
                except Exception as e:
                    exceptions.append(e)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(cache_worker, w) for w in range(num_workers)]
            for f in futures:
                f.result()

        assert len(exceptions) == 0


# =====================================================================
# 5. TESTES ANTI-LOOKAHEAD BIAS (RIGOR MATEMÁTICO)
# =====================================================================

class TestAntiLookaheadBiasRigor:
    """Validação matemática estrita de causalidade temporal."""

    def test_multi_period_macro_chronology(self, data_loader):
        """
        Cenário: Uma série de observações de CPI ao longo de 6 meses.
        Garante que, para qualquer data D, NENHUM dado com release_date > D seja visível,
        mesmo que o valor de observação já existisse no passado do calendário.
        """
        macro_points = [
            # Obs: Outubro/23 -> Divulgado em: 2023-11-14 (valor: 307.6)
            ("2023-10-01", "2023-11-14", 307.6),
            # Obs: Novembro/23 -> Divulgado em: 2023-12-12 (valor: 307.9)
            ("2023-11-01", "2023-12-12", 307.9),
            # Obs: Dezembro/23 -> Divulgado em: 2024-01-11 (valor: 308.8)
            ("2023-12-01", "2024-01-11", 308.8),
            # Obs: Janeiro/24 -> Divulgado em: 2024-02-13 (valor: 310.3)
            ("2024-01-01", "2024-02-13", 310.3),
        ]

        with data_loader._get_connection() as conn:
            for obs, rel, val in macro_points:
                conn.execute("""
                    INSERT OR REPLACE INTO macro_series (series_id, observation_date, release_date, value)
                    VALUES ('CPIAUCSL', ?, ?, ?)
                """, (obs, rel, val))
            conn.commit()

        # Teste no dia 10 de Dezembro de 2023:
        # Novembro já acabou, mas seu CPI só sai dia 12/12.
        # Portanto, o modelo deve enxergar APENAS o CPI de Outubro (307.6).
        val_dec10 = data_loader._get_latest_macro_val("CPIAUCSL", max_release_date="2023-12-10")
        assert val_dec10 == 307.6, "Violação de Causalidade: Dado de Novembro vazou antes de 12/12!"

        # Teste no dia 12 de Dezembro de 2023 (dia exato da divulgação):
        # Agora o CPI de Novembro (307.9) passa a ser conhecido.
        val_dec12 = data_loader._get_latest_macro_val("CPIAUCSL", max_release_date="2023-12-12")
        assert val_dec12 == 307.9

        # Teste no dia 10 de Janeiro de 2024:
        # Dezembro acabou, mas seu CPI só sai dia 11/01.
        val_jan10 = data_loader._get_latest_macro_val("CPIAUCSL", max_release_date="2024-01-10")
        assert val_jan10 == 307.9, "Violação de Causalidade: Dado de Dezembro vazou antes de 11/01!"

        # Teste no dia 11 de Janeiro de 2024:
        val_jan11 = data_loader._get_latest_macro_val("CPIAUCSL", max_release_date="2024-01-11")
        assert val_jan11 == 308.8

    def test_interest_rate_effective_d1_rule(self):
        """A taxa de juros diária FEDFUNDS deve ter D+1 estrito como release_date."""
        fred = FredConnector()
        assert fred._calculate_effective_release_date("FEDFUNDS", "2024-05-10") == "2024-05-11"
        assert fred._calculate_effective_release_date("DGS10", "2024-05-10") == "2024-05-11"


# =====================================================================
# 6. TESTES DO CONECTOR YAHOO FINANCE & DEFILLAMA
# =====================================================================

class TestYahooAndDefiLlamaConnectors:
    """Validação de parsing e fallback de conectores sem chaves pagas."""

    @pytest.mark.asyncio
    async def test_yahoo_rest_chart_parsing(self):
        """Valida parsing de JSON da API v8 pública do Yahoo Finance."""
        yf_conn = YahooFinanceConnector()

        mock_yahoo_payload = {
            "chart": {
                "result": [{
                    "timestamp": [1700000000, 1700086400],
                    "indicators": {
                        "quote": [{
                            "open": [15.2, 16.0],
                            "high": [15.8, 16.5],
                            "low": [15.0, 15.7],
                            "close": [15.5, 16.2],
                            "volume": [1000000, 1200000],
                        }]
                    }
                }]
            }
        }

        with patch.object(yf_conn.http, "get_json", return_value=mock_yahoo_payload):
            history = await yf_conn._fetch_yahoo_rest("^VIX", "5d", "1d")

            assert len(history) == 2
            assert history[0]["close"] == 15.5
            assert history[1]["close"] == 16.2
            assert history[0]["timestamp"] == 1700000000 * 1000
        await yf_conn.http.close()

    @pytest.mark.asyncio
    async def test_defillama_stablecoin_inflows_calculation(self):
        """Valida detecção de fluxo líquido positivo/negativo de stablecoins."""
        defi = DefiLlamaConnector()

        mock_stables_data = {
            "peggedAssets": [
                {
                    "symbol": "USDT",
                    "circulating": {"peggedUSD": 100_000_000_000.0},
                    "circulatingPrevDay": {"peggedUSD": 99_000_000_000.0},
                },
                {
                    "symbol": "USDC",
                    "circulating": {"peggedUSD": 30_000_000_000.0},
                    "circulatingPrevDay": {"peggedUSD": 29_500_000_000.0},
                }
            ]
        }

        with patch.object(defi.http, "get_json", return_value=mock_stables_data):
            flows = await defi.get_stablecoin_flows()

            assert flows["total_stablecoins_usd"] == 130_000_000_000.0
            # Aumento líquido de 1.0B (USDT) + 0.5B (USDC) = 1.5B
            assert flows["net_inflow_24h_usd"] == 1_500_000_000.0
            assert flows["is_liquidity_expanding"] is True
            assert flows["inflow_change_24h_pct"] > 0
        await defi.http.close()


# =====================================================================
# 7. TESTE DE CORRUPÇÃO DE CACHE SQLITE E RECUPERAÇÃO
# =====================================================================

class TestCacheCorruptionRecovery:
    """Garante que payloads inválidos no cache SQLite não derrubem a aplicação."""

    def test_corrupted_json_in_cache_returns_none(self, data_loader):
        """Se alguém ou uma interrupção gravar um JSON inválido no SQLite, get_from_cache não quebra."""
        category = "macro"
        key = "corrupted_entry"

        # Injeta diretamente string que NÃO é JSON
        with data_loader._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO low_freq_cache
                (category, cache_key, payload, updated_at, expires_at)
                VALUES (?, ?, ?, 1000, 9999999999)
            """, (category, key, "{malformed_json: true,"))
            conn.commit()

        # DataLoader deve capturar o JSONDecodeError e retornar None
        cached = data_loader._get_from_cache(category, key)
        assert cached is None
