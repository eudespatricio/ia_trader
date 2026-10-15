"""
Módulo de Integrações de Dados do IA Trader (Fase 1).
Fornece conectores assíncronos e independentes para fontes de dados 100% gratuitas:
1. Bitget API Pública (OHLCV Futuros/Spot, Order Book Depth e Funding Rate)
2. CoinGlass (Scraping/Endpoint Público de Open Interest e Funding Rate)
3. FRED API / CSV (Federal Reserve - CPI, Taxa de Juros, sem Lookahead Bias)
4. Yahoo Finance (DXY, VIX, S&P 500 via yfinance ou REST fallback)
5. CryptoPanic API Pública (Notícias e sentimento ponderado)
6. DefiLlama API Pública (TVL e Fluxo de Stablecoins)
7. CoinMarketCal API / Eventos Públicos de Mercado

Desenvolvido com tipagem estrita (Python 3.10+), suporte nativo a aiohttp com
fallback assíncrono para urllib/requests, controle de rate limit e sanitização.
"""

from __future__ import annotations

import os
import sys
import json
import time
import asyncio
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union

# Fallback gracioso para aiohttp
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# Fallback gracioso para yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# Configuração de Logger resiliente
logger = logging.getLogger("ia_trader.data_integrations")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [data_integrations] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =====================================================================
# DATA CONTRACTS & SCHEMAS
# =====================================================================

@dataclass
class CandleData:
    """Contrato padronizado para velas OHLCV."""
    timestamp: int           # Timestamp Unix em milissegundos
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    symbol: str
    timeframe: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrderBookDepth:
    """Contrato padronizado para profundidade do livro de ofertas."""
    symbol: str
    timestamp: int
    bids: List[Tuple[float, float]]  # [(preço, quantidade), ...] ordenado decrescente
    asks: List[Tuple[float, float]]  # [(preço, quantidade), ...] ordenado crescente
    spread: float
    imbalance: float                 # Razão de desbalanceamento de volume (-1.0 a +1.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MacroDataPoint:
    """
    Contrato para dados macroeconômicos do FRED.
    CRÍTICO: release_date representa a data efetiva de divulgação pública para
    prevenir rigorosamente qualquer vazamento temporal (lookahead bias).
    """
    series_id: str
    observation_date: str            # YYYY-MM-DD (período de referência econômico)
    release_date: str                # YYYY-MM-DD (data de publicação oficial)
    value: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SentimentPost:
    """Contrato padronizado para notícias e sentimento."""
    id: str
    title: str
    published_at: str
    source: str
    sentiment_label: str             # 'bullish', 'bearish', 'neutral'
    score: float                     # Pontuação entre -1.0 (extremo pânico) e +1.0 (extrema euforia)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OnChainMetric:
    """Contrato para fluxos on-chain e DeFi (DefiLlama)."""
    metric_type: str                 # 'tvl', 'stablecoin_mcap', 'stablecoin_flow'
    value: float
    timestamp: int
    chain: str = "all"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MarketEvent:
    """Contrato para eventos de calendário econômico e cripto."""
    title: str
    date: str                        # YYYY-MM-DD
    coin: str
    category: str
    impact: str                      # 'high', 'medium', 'low'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =====================================================================
# ASYNC HTTP CLIENT COM FALLBACK UNIVERSAL & RATE LIMITER
# =====================================================================

class AsyncHttpClient:
    """
    Cliente HTTP assíncrono resiliente.
    Utiliza aiohttp quando instalado, com fallback automático para urllib.request
    executado em threads assíncronas caso aiohttp não esteja presente.
    Possui controle de timeout, backoff exponencial e limitação de concorrência.
    """

    def __init__(self, timeout_seconds: float = 10.0, max_retries: int = 2):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._session: Optional[Any] = None
        self._semaphore = asyncio.Semaphore(10)  # Limite de 10 conexões simultâneas
        self.default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
        }

    async def _get_session(self):
        if HAS_AIOHTTP:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
                self._session = aiohttp.ClientSession(timeout=timeout, headers=self.default_headers)
            return self._session
        return None

    async def get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        """Executa requisição GET assíncrona retornando JSON decodificado."""
        async with self._semaphore:
            custom_headers = self.default_headers.copy()
            if headers:
                custom_headers.update(headers)

            for attempt in range(self.max_retries + 1):
                try:
                    if HAS_AIOHTTP:
                        session = await self._get_session()
                        async with session.get(url, params=params, headers=custom_headers) as resp:
                            if resp.status == 200:
                                return await resp.json(content_type=None)
                            elif resp.status in (429, 418):
                                retry_hdr = resp.headers.get("Retry-After")
                                delay = float(retry_hdr) if (retry_hdr and retry_hdr.isdigit()) else (2.0 * (2 ** attempt))
                                logger.warning(f"Rate limit / Ban temporário em {url} (HTTP {resp.status}). Aguardando {delay:.1f}s (Tentativa {attempt + 1}).")
                                await asyncio.sleep(delay)
                                continue
                            elif resp.status == 451:
                                logger.warning(f"Bloqueio geográfico HTTP 451 em {url}. Ativando fallback para data.binance.vision / cache local.")
                                return None
                            else:
                                text = await resp.text()
                                logger.warning(f"Erro HTTP {resp.status} em {url}: {text[:150]}")
                                if attempt == self.max_retries:
                                    return None
                    else:
                        # Fallback assíncrono com urllib.request
                        return await asyncio.to_thread(self._urllib_get_json, url, params, custom_headers)
                except Exception as exc:
                    logger.debug(f"Exceção na requisição GET para {url} (tentativa {attempt + 1}): {exc}")
                    if attempt < self.max_retries:
                        await asyncio.sleep(1.0 * (2 ** attempt))
                    else:
                        logger.error(f"Falha definitiva ao consultar {url}: {exc}")
                        return None
        return None

    def _urllib_get_json(self, url: str, params: Optional[Dict[str, Any]], headers: Dict[str, str]) -> Any:
        """Execução síncrona de requisição GET via urllib para fallback."""
        import urllib.request
        import urllib.parse
        import urllib.error

        final_url = url
        if params:
            query = urllib.parse.urlencode(params)
            final_url = f"{url}?{query}" if "?" not in url else f"{url}&{query}"

        req = urllib.request.Request(final_url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                content = response.read().decode("utf-8")
                return json.loads(content)
        except urllib.error.HTTPError as he:
            if he.code in (429, 418):
                retry_hdr = he.headers.get("Retry-After") if hasattr(he, "headers") else None
                delay = float(retry_hdr) if (retry_hdr and retry_hdr.isdigit()) else 2.0
                logger.warning(f"Rate limit HTTP {he.code} via urllib para {final_url}. Retry-After: {delay}s.")
                time.sleep(delay)
            elif he.code == 451:
                logger.warning(f"Bloqueio geográfico HTTP 451 via urllib para {final_url}. Ativando fallback para data.binance.vision.")
            else:
                logger.warning(f"HTTPError no fallback urllib para {final_url}: {he.code}")
            return None
        except Exception as e:
            logger.debug(f"Erro no fallback urllib para {final_url}: {e}")
            return None

    async def get_text(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Executa requisição GET assíncrona retornando texto puro (ex.: CSV ou Scraping)."""
        async with self._semaphore:
            custom_headers = self.default_headers.copy()
            if headers:
                custom_headers.update(headers)

            for attempt in range(self.max_retries + 1):
                try:
                    if HAS_AIOHTTP:
                        session = await self._get_session()
                        async with session.get(url, params=params, headers=custom_headers) as resp:
                            if resp.status == 200:
                                return await resp.text()
                            elif resp.status in (429, 418):
                                retry_hdr = resp.headers.get("Retry-After")
                                delay = float(retry_hdr) if (retry_hdr and retry_hdr.isdigit()) else (2.0 * (2 ** attempt))
                                await asyncio.sleep(delay)
                                continue
                            elif resp.status == 451:
                                logger.warning(f"Bloqueio geográfico HTTP 451 em {url} (get_text).")
                                return None
                            return None
                    else:
                        import urllib.request
                        import urllib.parse
                        final_url = url
                        if params:
                            final_url = f"{url}?{urllib.parse.urlencode(params)}"
                        req = urllib.request.Request(final_url, headers=custom_headers)
                        return await asyncio.to_thread(self._urllib_get_text, req)
                except Exception as exc:
                    logger.debug(f"Falha ao obter texto de {url}: {exc}")
                    if attempt < self.max_retries:
                        await asyncio.sleep(1.0)
                    else:
                        return None
        return None

    def _urllib_get_text(self, req: Any) -> Optional[str]:
        import urllib.request
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                return response.read().decode("utf-8")
        except Exception:
            return None

    async def get_bytes(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Optional[bytes]:
        """Executa requisição GET assíncrona retornando bytes brutos (ex.: arquivos ZIP de data.binance.vision)."""
        async with self._semaphore:
            custom_headers = self.default_headers.copy()
            if headers:
                custom_headers.update(headers)

            for attempt in range(self.max_retries + 1):
                try:
                    if HAS_AIOHTTP:
                        session = await self._get_session()
                        async with session.get(url, params=params, headers=custom_headers) as resp:
                            if resp.status == 200:
                                return await resp.read()
                            elif resp.status in (429, 418):
                                retry_hdr = resp.headers.get("Retry-After")
                                delay = float(retry_hdr) if (retry_hdr and retry_hdr.isdigit()) else (2.0 * (2 ** attempt))
                                await asyncio.sleep(delay)
                                continue
                            elif resp.status == 451:
                                logger.warning(f"Bloqueio geográfico HTTP 451 em {url} (get_bytes).")
                                return None
                            return None
                    else:
                        import urllib.request
                        import urllib.parse
                        final_url = url
                        if params:
                            final_url = f"{url}?{urllib.parse.urlencode(params)}"
                        req = urllib.request.Request(final_url, headers=custom_headers)
                        return await asyncio.to_thread(self._urllib_get_bytes, req)
                except Exception as exc:
                    logger.debug(f"Falha ao obter bytes de {url}: {exc}")
                    if attempt < self.max_retries:
                        await asyncio.sleep(1.0)
                    else:
                        return None
        return None

    def _urllib_get_bytes(self, req: Any) -> Optional[bytes]:
        import urllib.request
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                return response.read()
        except Exception:
            return None

    async def close(self):
        """Encerra recursos de conexão."""
        if HAS_AIOHTTP and self._session and not self._session.closed:
            await self._session.close()


_TF_MS: Dict[str, int] = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
    "1w": 604_800_000,
}


def timeframe_to_ms(timeframe: str) -> int:
    """Duração do timeframe em ms (0 se desconhecido)."""
    return _TF_MS.get(str(timeframe).lower(), 0)


def drop_open_candles(candles: List[CandleData], timeframe: str, now_ms: Optional[int] = None) -> List[CandleData]:
    """
    V7 (anti-lookahead): descarta a vela em formação (timestamp + intervalo > agora).
    Uma vela só é consumível quando fechada; sua OHLCV parcial mudaria depois.
    """
    tf_ms = timeframe_to_ms(timeframe)
    if tf_ms <= 0:
        return candles
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    return [c for c in candles if c.timestamp + tf_ms <= now]


# =====================================================================
# 1. BITGET API CONNECTOR (FUTUROS & SPOT PÚBLICO)
# =====================================================================

class BitgetConnector:
    """
    Conector oficial para a Bitget API v2 pública.
    Suporta velas OHLCV de futuros (USDT-FUTURES) e spot, livro de ofertas e taxa de financiamento (Funding Rate).
    Nenhuma chave de API é necessária para leitura de dados de mercado.
    """

    BASE_URL = "https://api.bitget.com"

    def __init__(self, http_client: Optional[AsyncHttpClient] = None):
        self.http = http_client or AsyncHttpClient()

    async def get_candles(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        limit: int = 100,
        product_type: str = "USDT-FUTURES",
    ) -> List[CandleData]:
        """
        Busca velas históricas recentes da Bitget.
        Timeframes válidos: '1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d', '1w'.
        """
        # Mapeamento de normalização de timeframes para a Bitget API v2
        granularity_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1H", "4h": "4H", "12h": "12H", "1d": "1D", "1w": "1W",
            "1H": "1H", "4H": "4H", "1D": "1D", "1W": "1W",
        }
        granularity = granularity_map.get(timeframe, "1H")

        if product_type.upper() == "SPOT":
            endpoint = f"{self.BASE_URL}/api/v2/spot/market/candles"
            params = {
                "symbol": symbol.upper(),
                "granularity": granularity,
                "limit": str(min(limit, 500)),
            }
        else:
            endpoint = f"{self.BASE_URL}/api/v2/mix/market/candles"
            params = {
                "symbol": symbol.upper(),
                "productType": product_type,
                "granularity": granularity,
                "limit": str(min(limit, 500)),
            }

        data = await self.http.get_json(endpoint, params=params)
        candles: List[CandleData] = []

        if data and isinstance(data, dict) and data.get("code") == "00000":
            raw_candles = data.get("data", [])
            for c in raw_candles:
                try:
                    # Bitget v2 candle schema: [ts, open, high, low, close, volume, quoteVolume]
                    candle = CandleData(
                        timestamp=int(c[0]),
                        open=float(c[1]),
                        high=float(c[2]),
                        low=float(c[3]),
                        close=float(c[4]),
                        volume=float(c[5]),
                        quote_volume=float(c[6]) if len(c) > 6 else float(c[5]) * float(c[4]),
                        symbol=symbol.upper(),
                        timeframe=timeframe.lower(),
                    )
                    candles.append(candle)
                except (ValueError, IndexError) as err:
                    logger.debug(f"Erro ao converter vela da Bitget: {err}")
                    continue

        # Ordenação temporal estrita ascendente (mais antigo para mais recente)
        candles.sort(key=lambda x: x.timestamp)
        return drop_open_candles(candles, timeframe)

    async def get_order_book(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 50,
        product_type: str = "USDT-FUTURES",
    ) -> Optional[OrderBookDepth]:
        """
        Retorna profundidade do livro de ofertas (Bids e Asks) e calcula spread e imbalance.
        """
        endpoint = f"{self.BASE_URL}/api/v2/mix/market/merge-depth"
        params = {
            "symbol": symbol.upper(),
            "productType": product_type,
            "limit": str(min(limit, 100)),
        }

        data = await self.http.get_json(endpoint, params=params)
        if not data or data.get("code") != "00000" or not data.get("data"):
            return None

        book_data = data["data"]
        raw_bids = book_data.get("bids", [])
        raw_asks = book_data.get("asks", [])

        bids: List[Tuple[float, float]] = []
        asks: List[Tuple[float, float]] = []

        for b in raw_bids:
            try:
                bids.append((float(b[0]), float(b[1])))
            except (ValueError, IndexError):
                continue

        for a in raw_asks:
            try:
                asks.append((float(a[0]), float(a[1])))
            except (ValueError, IndexError):
                continue

        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])

        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        spread = round(best_ask - best_bid, 4) if (best_ask > 0 and best_bid > 0) else 0.0

        bid_vol = sum(b[1] for b in bids)
        ask_vol = sum(a[1] for a in asks)
        total_vol = bid_vol + ask_vol
        imbalance = round((bid_vol - ask_vol) / total_vol, 4) if total_vol > 0 else 0.0

        ts = int(book_data.get("ts", time.time() * 1000))

        return OrderBookDepth(
            symbol=symbol.upper(),
            timestamp=ts,
            bids=bids,
            asks=asks,
            spread=spread,
            imbalance=imbalance,
        )

    async def get_funding_rate(
        self,
        symbol: str = "BTCUSDT",
        product_type: str = "USDT-FUTURES",
    ) -> Dict[str, Any]:
        """Retorna taxa de financiamento (Funding Rate) atual e previsão."""
        endpoint = f"{self.BASE_URL}/api/v2/mix/market/current-fund-rate"
        params = {
            "symbol": symbol.upper(),
            "productType": product_type,
        }
        data = await self.http.get_json(endpoint, params=params)
        if data and data.get("code") == "00000" and data.get("data"):
            items = data["data"]
            if isinstance(items, list) and items:
                item = items[0]
                return {
                    "symbol": symbol.upper(),
                    "funding_rate": float(item.get("fundingRate", 0.0)),
                    "funding_time": int(item.get("fundingTime", 0)),
                    "next_funding_time": int(item.get("nextFundingTime", 0)),
                }
        return {"symbol": symbol.upper(), "funding_rate": 0.0, "funding_time": 0, "next_funding_time": 0}

    async def get_open_interest(
        self,
        symbol: str = "BTCUSDT",
        product_type: str = "USDT-FUTURES",
    ) -> Dict[str, Any]:
        """Retorna Open Interest (interesse em aberto) de contratos futuros."""
        endpoint = f"{self.BASE_URL}/api/v2/mix/market/open-interest"
        params = {
            "symbol": symbol.upper(),
            "productType": product_type,
        }
        data = await self.http.get_json(endpoint, params=params)
        if data and data.get("code") == "00000" and data.get("data"):
            info = data["data"]
            # Extrai open interest em moedas e em valor nocional
            amount = float(info.get("openInterestList", [{}])[0].get("size", 0.0)) if "openInterestList" in info else float(info.get("size", 0.0))
            return {
                "symbol": symbol.upper(),
                "open_interest": amount,
                "timestamp": int(info.get("ts", time.time() * 1000)),
            }
        return {"symbol": symbol.upper(), "open_interest": 0.0, "timestamp": int(time.time() * 1000)}


# =====================================================================
# 1.1 BINANCE CONNECTOR (SPOT, FUTUROS USDT-M & DATA.BINANCE.VISION)
# =====================================================================

class BinanceConnector:
    """
    Conector oficial e independente para a Binance API pública (Spot e Futuros USDT-M)
    e dumps históricos do data.binance.vision.
    100% público e gratuito, sem necessidade de chaves de API para leitura.
    Suporta:
    - Klines Spot e Futuros USDT-M com paginação cronológica automática.
    - Funding rate histórico paginado (/fapi/v1/fundingRate).
    - Open interest em tempo real e série temporal recente (/futures/data/openInterestHist).
    - Taker buy/sell volume e long/short account ratio.
    - ExchangeInfo e 24hr Ticker.
    - Dumps públicos em ZIP de data.binance.vision com validação de checksum SHA256.
    - Mitigação de rate limit (429/418 com Retry-After) e bloqueio regional (451).
    """

    SPOT_URL = "https://api.binance.com"
    FUTURES_URL = "https://fapi.binance.com"
    DATA_VISION_URL = "https://data.binance.vision"

    # Mapeamento e normalização de granularidade/timeframe
    INTERVAL_MAP = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
        "1d": "1d", "3d": "3d", "1w": "1w", "1M": "1M",
        "1H": "1h", "2H": "2h", "4H": "4h", "6H": "6h", "8H": "8h", "12H": "12h",
        "1D": "1d", "3D": "3d", "1W": "1w",
    }

    def __init__(self, http_client: Optional[AsyncHttpClient] = None):
        self.http = http_client or AsyncHttpClient()

    def normalize_interval(self, timeframe: str) -> str:
        return self.INTERVAL_MAP.get(timeframe, self.INTERVAL_MAP.get(timeframe.lower(), "1h"))

    async def get_klines(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        market_type: str = "futures",
    ) -> List[CandleData]:
        """
        Busca velas OHLCV da Binance (Spot ou Futuros USDT-M) com paginação automática.
        Se limit > 1000 ou intervalo longo com start_time/end_time, pagina em lotes de até 1000.
        """
        sym = symbol.upper()
        interval = self.normalize_interval(timeframe)
        is_spot = (market_type.lower() == "spot")
        base_url = f"{self.SPOT_URL}/api/v3/klines" if is_spot else f"{self.FUTURES_URL}/fapi/v1/klines"

        candles: List[CandleData] = []
        target_limit = limit if limit > 0 else 1000
        current_start = start_time
        max_batch_size = 1000

        # Loop de paginação
        while True:
            remaining = target_limit - len(candles)
            if remaining <= 0:
                break
            batch_limit = min(remaining, max_batch_size)

            params: Dict[str, Any] = {
                "symbol": sym,
                "interval": interval,
                "limit": str(batch_limit),
            }
            if current_start is not None:
                params["startTime"] = str(current_start)
            if end_time is not None:
                params["endTime"] = str(end_time)

            data = await self.http.get_json(base_url, params=params)

            if not data or not isinstance(data, list) or len(data) == 0:
                break

            batch_candles: List[CandleData] = []
            for k in data:
                try:
                    # Binance kline schema:
                    # [0: open_time, 1: open, 2: high, 3: low, 4: close, 5: volume, 6: close_time, 7: quote_volume, ...]
                    c = CandleData(
                        timestamp=int(k[0]),
                        open=float(k[1]),
                        high=float(k[2]),
                        low=float(k[3]),
                        close=float(k[4]),
                        volume=float(k[5]),
                        quote_volume=float(k[7]) if len(k) > 7 else float(k[5]) * float(k[4]),
                        symbol=sym,
                        timeframe=timeframe.lower(),
                    )
                    batch_candles.append(c)
                except (ValueError, IndexError, TypeError) as err:
                    logger.debug(f"Erro ao parsear kline da Binance: {err}")
                    continue

            if not batch_candles:
                break

            candles.extend(batch_candles)

            # Se pegou menos que o solicitado ou alcançou o final
            if len(batch_candles) < batch_limit:
                break

            last_ts = batch_candles[-1].timestamp
            current_start = last_ts + 1
            if end_time is not None and current_start >= end_time:
                break

            # Pequeno intervalo para respeitar taxa de requisição
            if len(candles) < target_limit:
                await asyncio.sleep(0.05)

        # Ordenação temporal estrita ascendente
        candles.sort(key=lambda x: x.timestamp)
        return drop_open_candles(candles, timeframe)

    async def get_funding_rate_history(
        self,
        symbol: str = "BTCUSDT",
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retorna histórico de taxas de financiamento (Funding Rate) paginado.
        Endpoint: /fapi/v1/fundingRate
        """
        sym = symbol.upper()
        url = f"{self.FUTURES_URL}/fapi/v1/fundingRate"
        results: List[Dict[str, Any]] = []
        target_limit = limit if limit > 0 else 1000
        current_start = start_time
        max_batch_size = 1000

        while True:
            remaining = target_limit - len(results)
            if remaining <= 0:
                break
            batch_limit = min(remaining, max_batch_size)

            params: Dict[str, Any] = {
                "symbol": sym,
                "limit": str(batch_limit),
            }
            if current_start is not None:
                params["startTime"] = str(current_start)
            if end_time is not None:
                params["endTime"] = str(end_time)

            data = await self.http.get_json(url, params=params)
            if not data or not isinstance(data, list) or len(data) == 0:
                break

            batch_rates = []
            for item in data:
                try:
                    batch_rates.append({
                        "symbol": sym,
                        "funding_rate": float(item.get("fundingRate", 0.0)),
                        "funding_time": int(item.get("fundingTime", 0)),
                        "mark_price": float(item.get("markPrice", 0.0)) if item.get("markPrice") else None,
                    })
                except (ValueError, TypeError):
                    continue

            if not batch_rates:
                break

            results.extend(batch_rates)
            if len(batch_rates) < batch_limit:
                break

            last_time = batch_rates[-1]["funding_time"]
            current_start = last_time + 1
            if end_time is not None and current_start >= end_time:
                break

            await asyncio.sleep(0.05)

        results.sort(key=lambda x: x["funding_time"])
        return results

    async def get_open_interest(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Retorna Open Interest em tempo real de futuros USDT-M."""
        sym = symbol.upper()
        url = f"{self.FUTURES_URL}/fapi/v1/openInterest"
        data = await self.http.get_json(url, params={"symbol": sym})
        if data and isinstance(data, dict):
            return {
                "symbol": sym,
                "open_interest": float(data.get("openInterest", 0.0)),
                "timestamp": int(data.get("time", time.time() * 1000)),
            }
        return {"symbol": sym, "open_interest": 0.0, "timestamp": int(time.time() * 1000)}

    async def get_open_interest_history(
        self,
        symbol: str = "BTCUSDT",
        period: str = "5m",
        limit: int = 30,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retorna histórico de Open Interest (até ~30 dias).
        Endpoint: /futures/data/openInterestHist
        """
        sym = symbol.upper()
        url = f"{self.FUTURES_URL}/futures/data/openInterestHist"
        params: Dict[str, Any] = {
            "symbol": sym,
            "period": period,
            "limit": str(min(limit, 500)),
        }
        if start_time:
            params["startTime"] = str(start_time)
        if end_time:
            params["endTime"] = str(end_time)

        data = await self.http.get_json(url, params=params)
        results: List[Dict[str, Any]] = []
        if data and isinstance(data, list):
            for item in data:
                try:
                    results.append({
                        "symbol": sym,
                        "sum_open_interest": float(item.get("sumOpenInterest", 0.0)),
                        "sum_open_interest_value": float(item.get("sumOpenInterestValue", 0.0)),
                        "timestamp": int(item.get("timestamp", 0)),
                    })
                except (ValueError, TypeError):
                    continue
        results.sort(key=lambda x: x["timestamp"])
        return results

    async def get_taker_volume_ratio(
        self,
        symbol: str = "BTCUSDT",
        period: str = "5m",
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Retorna proporção de volume agressor comprador vs vendedor (Taker Buy/Sell Ratio).
        Endpoint: /futures/data/takerlongshortRatio
        """
        sym = symbol.upper()
        url = f"{self.FUTURES_URL}/futures/data/takerlongshortRatio"
        params = {"symbol": sym, "period": period, "limit": str(min(limit, 500))}
        data = await self.http.get_json(url, params=params)
        results: List[Dict[str, Any]] = []
        if data and isinstance(data, list):
            for item in data:
                try:
                    results.append({
                        "symbol": sym,
                        "buy_sell_ratio": float(item.get("buySellRatio", 1.0)),
                        "buy_vol": float(item.get("buyVol", 0.0)),
                        "sell_vol": float(item.get("sellVol", 0.0)),
                        "timestamp": int(item.get("timestamp", 0)),
                    })
                except (ValueError, TypeError):
                    continue
        results.sort(key=lambda x: x["timestamp"])
        return results

    async def get_long_short_ratio(
        self,
        symbol: str = "BTCUSDT",
        period: str = "5m",
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Retorna proporção global de contas compradas vs vendidas (Long/Short Ratio).
        Endpoint: /futures/data/globalLongShortAccountRatio
        """
        sym = symbol.upper()
        url = f"{self.FUTURES_URL}/futures/data/globalLongShortAccountRatio"
        params = {"symbol": sym, "period": period, "limit": str(min(limit, 500))}
        data = await self.http.get_json(url, params=params)
        results: List[Dict[str, Any]] = []
        if data and isinstance(data, list):
            for item in data:
                try:
                    results.append({
                        "symbol": sym,
                        "long_short_ratio": float(item.get("longShortRatio", 1.0)),
                        "long_account": float(item.get("longAccount", 0.5)),
                        "short_account": float(item.get("shortAccount", 0.5)),
                        "timestamp": int(item.get("timestamp", 0)),
                    })
                except (ValueError, TypeError):
                    continue
        results.sort(key=lambda x: x["timestamp"])
        return results

    async def get_exchange_info(self, market_type: str = "futures") -> Dict[str, Any]:
        """Consulta metadados e pares disponíveis no mercado Spot ou Futuros."""
        is_spot = (market_type.lower() == "spot")
        url = f"{self.SPOT_URL}/api/v3/exchangeInfo" if is_spot else f"{self.FUTURES_URL}/fapi/v1/exchangeInfo"
        data = await self.http.get_json(url)
        return data if isinstance(data, dict) else {}

    async def get_ticker_24hr(self, symbol: str = "BTCUSDT", market_type: str = "futures") -> Dict[str, Any]:
        """Retorna resumo de variação de preço, volume e extremos em 24 horas."""
        sym = symbol.upper()
        is_spot = (market_type.lower() == "spot")
        url = f"{self.SPOT_URL}/api/v3/ticker/24hr" if is_spot else f"{self.FUTURES_URL}/fapi/v1/ticker/24hr"
        data = await self.http.get_json(url, params={"symbol": sym})
        if data and isinstance(data, dict):
            return {
                "symbol": sym,
                "last_price": float(data.get("lastPrice", 0.0)),
                "price_change_pct": float(data.get("priceChangePercent", 0.0)),
                "high_24h": float(data.get("highPrice", 0.0)),
                "low_24h": float(data.get("lowPrice", 0.0)),
                "volume_24h": float(data.get("volume", 0.0)),
                "quote_volume_24h": float(data.get("quoteVolume", 0.0)),
                "timestamp": int(data.get("closeTime", time.time() * 1000)),
            }
        return {"symbol": sym, "last_price": 0.0, "price_change_pct": 0.0, "timestamp": int(time.time() * 1000)}

    # -----------------------------------------------------------------
    # LEITOR DE DUMPS PÚBLICOS DATA.BINANCE.VISION (.ZIP COM STREAM E CHECKSUM)
    # -----------------------------------------------------------------

    def get_vision_url(
        self,
        symbol: str,
        timeframe: str,
        year: int,
        month: int,
        day: Optional[int] = None,
        data_type: str = "klines",
        market_type: str = "futures",
    ) -> str:
        """Gera URL oficial para download de dump público da data.binance.vision."""
        sym = symbol.upper()
        interval = self.normalize_interval(timeframe)
        mkt_path = "futures/um" if market_type.lower() == "futures" else "spot"

        if data_type == "fundingRate":
            # Dumps de funding rate são mensais
            return f"{self.DATA_VISION_URL}/data/{mkt_path}/monthly/fundingRate/{sym}/{sym}-fundingRate-{year}-{month:02d}.zip"

        if day is not None:
            # Diário
            return f"{self.DATA_VISION_URL}/data/{mkt_path}/daily/{data_type}/{sym}/{interval}/{sym}-{interval}-{year}-{month:02d}-{day:02d}.zip"
        else:
            # Mensal
            return f"{self.DATA_VISION_URL}/data/{mkt_path}/monthly/{data_type}/{sym}/{interval}/{sym}-{interval}-{year}-{month:02d}.zip"

    @staticmethod
    def verify_checksum(data_bytes: bytes, checksum_text: str) -> bool:
        """
        Verifica a integridade de dados contra o hash SHA256 do arquivo .CHECKSUM da Binance.
        O arquivo checksum costuma ter formato: '<hash_hex>  <filename>' ou apenas '<hash_hex>'.
        """
        import hashlib
        if not data_bytes or not checksum_text:
            return False
        expected_hash = checksum_text.strip().split()[0].lower()
        actual_hash = hashlib.sha256(data_bytes).hexdigest().lower()
        return actual_hash == expected_hash

    @staticmethod
    def parse_vision_klines_csv(csv_text: str, symbol: str, timeframe: str) -> List[CandleData]:
        """
        Interpreta CSV extraído de dumps de klines da data.binance.vision.
        Suporta com ou sem linha de cabeçalho.
        """
        candles: List[CandleData] = []
        if not csv_text:
            return candles

        lines = csv_text.strip().splitlines()
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            # Ignora cabeçalho se houver texto
            if parts[0].isalpha() or "open" in parts[0].lower():
                continue
            try:
                ts = int(parts[0])
                op = float(parts[1])
                hi = float(parts[2])
                lo = float(parts[3])
                cl = float(parts[4])
                vol = float(parts[5])
                qvol = float(parts[7]) if len(parts) > 7 else vol * cl

                candles.append(CandleData(
                    timestamp=ts,
                    open=op,
                    high=hi,
                    low=lo,
                    close=cl,
                    volume=vol,
                    quote_volume=qvol,
                    symbol=symbol.upper(),
                    timeframe=timeframe.lower(),
                ))
            except (ValueError, IndexError):
                continue

        candles.sort(key=lambda x: x.timestamp)
        return candles

    @staticmethod
    def parse_vision_funding_csv(csv_text: str, symbol: str) -> List[Dict[str, Any]]:
        """Interpreta CSV extraído de dumps de fundingRate da data.binance.vision."""
        results: List[Dict[str, Any]] = []
        if not csv_text:
            return results

        lines = csv_text.strip().splitlines()
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            if parts[0].isalpha() or "calc" in parts[0].lower() or "symbol" in parts[0].lower():
                continue
            try:
                # Formato: symbol, fundingRate, fundingTime, markPrice
                if len(parts) >= 4 and not parts[0].isdigit():
                    results.append({
                        "symbol": parts[0].upper(),
                        "funding_rate": float(parts[1]),
                        "funding_time": int(parts[2]),
                        "mark_price": float(parts[3]) if parts[3] else None,
                    })
                else:
                    results.append({
                        "symbol": symbol.upper(),
                        "funding_rate": float(parts[2]),
                        "funding_time": int(parts[0]),
                        "mark_price": None,
                    })
            except (ValueError, IndexError):
                continue

        results.sort(key=lambda x: x["funding_time"])
        return results

    async def download_and_extract_vision_klines(
        self,
        symbol: str,
        timeframe: str,
        year: int,
        month: int,
        day: Optional[int] = None,
        verify: bool = True,
        market_type: str = "futures",
    ) -> List[CandleData]:
        """
        Baixa o dump zip público de data.binance.vision, verifica checksum se disponível,
        e descompacta em memória retornando List[CandleData].
        """
        import zipfile
        import io

        url = self.get_vision_url(symbol, timeframe, year, month, day=day, data_type="klines", market_type=market_type)
        zip_bytes = await self.http.get_bytes(url)
        if not zip_bytes:
            logger.debug(f"Dump da Binance Vision não encontrado ou erro de rede para {url}")
            return []

        # Validação opcional de checksum
        if verify:
            checksum_url = f"{url}.CHECKSUM"
            checksum_text = await self.http.get_text(checksum_url)
            if checksum_text:
                if not self.verify_checksum(zip_bytes, checksum_text):
                    logger.warning(f"Checksum mismatch para dump da Binance Vision: {url}")
                    return []
                else:
                    logger.debug(f"Checksum SHA256 verificado com sucesso para {url}")

        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                if not csv_files:
                    return []
                with z.open(csv_files[0]) as f:
                    csv_text = f.read().decode("utf-8", errors="ignore")
                    return self.parse_vision_klines_csv(csv_text, symbol, timeframe)
        except Exception as e:
            logger.warning(f"Erro ao descompactar dump da Binance Vision {url}: {e}")
            return []

    async def download_and_extract_vision_funding(
        self,
        symbol: str,
        year: int,
        month: int,
        verify: bool = True,
    ) -> List[Dict[str, Any]]:
        """Baixa e descompacta dump de funding rate da data.binance.vision."""
        import zipfile
        import io

        url = self.get_vision_url(symbol, "8h", year, month, data_type="fundingRate", market_type="futures")
        zip_bytes = await self.http.get_bytes(url)
        if not zip_bytes:
            logger.debug(f"Dump de funding da Binance Vision não encontrado ou erro de rede para {url}")
            return []

        if verify:
            checksum_url = f"{url}.CHECKSUM"
            checksum_text = await self.http.get_text(checksum_url)
            if checksum_text and not self.verify_checksum(zip_bytes, checksum_text):
                logger.warning(f"Checksum mismatch para dump de funding da Binance Vision: {url}")
                return []

        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                if not csv_files:
                    return []
                with z.open(csv_files[0]) as f:
                    csv_text = f.read().decode("utf-8", errors="ignore")
                    return self.parse_vision_funding_csv(csv_text, symbol)
        except Exception as e:
            logger.warning(f"Erro ao descompactar funding dump da Binance Vision {url}: {e}")
            return []


# =====================================================================
# 2. COINGLASS CONNECTOR (OPEN INTEREST & FUNDING RATE SCRAPING/PUBLIC)
# =====================================================================

class CoinGlassConnector:
    """
    Conector resiliente para agregação de Open Interest e Funding Rate.
    Utiliza endpoints públicos abertos do ecossistema cripto com fallback automático
    para Bitget e Binance pública caso restrições de IP ocorram no CoinGlass.
    """

    def __init__(self, http_client: Optional[AsyncHttpClient] = None):
        self.http = http_client or AsyncHttpClient()
        self.bitget = BitgetConnector(self.http)

    async def get_open_interest_aggregated(self, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Retorna Open Interest agregado do mercado para o ativo base.
        Tenta fonte pública CoinGlass; se falhar ou houver rate limit, aciona fallback direto.
        """
        clean_sym = symbol.upper().replace("USDT", "")
        # Tentativa de endpoint público aberto CoinGlass
        cg_url = f"https://capi.coinglass.com/api/player/v1/openInterest/detail?symbol={clean_sym}"
        data = await self.http.get_json(cg_url)

        if data and isinstance(data, dict) and data.get("code") == "0":
            payload = data.get("data", {})
            return {
                "symbol": clean_sym,
                "total_oi_usd": float(payload.get("totalOpenInterestUsd", 0.0)),
                "oi_change_24h": float(payload.get("h24Change", 0.0)),
                "source": "coinglass",
                "timestamp": int(time.time() * 1000),
            }

        # Fallback de Alta Confiabilidade: Consulta Bitget + Binance Pública
        logger.debug(f"CoinGlass indisponível ou bloqueado. Acionando fallback multi-exchange para {clean_sym}...")
        bitget_oi = await self.bitget.get_open_interest(f"{clean_sym}USDT")

        # Fallback secundário na Binance pública (sem necessidade de autenticação)
        binance_url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={clean_sym}USDT"
        binance_data = await self.http.get_json(binance_url)
        binance_oi = float(binance_data.get("openInterest", 0.0)) if binance_data else 0.0

        total_oi = bitget_oi["open_interest"] + binance_oi
        return {
            "symbol": clean_sym,
            "total_oi_contracts": total_oi,
            "bitget_oi": bitget_oi["open_interest"],
            "binance_oi": binance_oi,
            "source": "multi_exchange_fallback",
            "timestamp": int(time.time() * 1000),
        }


# =====================================================================
# 3. FRED API CONNECTOR (MACROECONOMIA SEM LOOKAHEAD BIAS)
# =====================================================================

class FredConnector:
    """
    Conector para dados macroeconômicos do Federal Reserve (FRED).
    Séries suportadas:
    - CPIAUCSL: Índice de Preços ao Consumidor (Inflação)
    - FEDFUNDS: Taxa de Juros Efetiva do Federal Reserve
    - DGS10: Taxa dos Títulos do Tesouro Americano de 10 Anos
    - M2SL: Oferta Monetária M2

    RIGOR METODOLÓGICO:
    Garante ausência total de lookahead bias atribuindo estritamente a data
    efetiva de publicação oficial (release_date) aos pontos de dados.
    Funciona 100% gratuito mesmo sem API key via download direto de CSV oficial.
    """

    FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def __init__(self, api_key: Optional[str] = None, http_client: Optional[AsyncHttpClient] = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY", "")
        self.http = http_client or AsyncHttpClient()

    async def get_series(
        self,
        series_id: str = "CPIAUCSL",
        limit: int = 24,
    ) -> List[MacroDataPoint]:
        """
        Obtém os dados históricos de uma série do FRED com controle rigoroso de publicação.
        Se FRED_API_KEY estiver vazia, utiliza download gratuito de CSV oficial.
        """
        if self.api_key:
            return await self._get_series_api(series_id, limit)
        return await self._get_series_csv(series_id, limit)

    async def _get_series_api(self, series_id: str, limit: int) -> List[MacroDataPoint]:
        """Consulta via API oficial do FRED."""
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": str(limit),
        }
        data = await self.http.get_json(self.FRED_BASE_URL, params=params)
        results: List[MacroDataPoint] = []

        if data and "observations" in data:
            for obs in data["observations"]:
                obs_date = obs.get("date", "")
                val_str = obs.get("value", ".")
                if val_str == "." or not obs_date:
                    continue
                try:
                    val = float(val_str)
                    # Prevenção de lookahead: se não vier release_date explícito na API simples,
                    # aplica a regra regulamentar padrão (CPI divulgado ~15 dias após o término do mês de referência)
                    release_date = self._calculate_effective_release_date(series_id, obs_date)
                    results.append(MacroDataPoint(
                        series_id=series_id,
                        observation_date=obs_date,
                        release_date=release_date,
                        value=val,
                    ))
                except ValueError:
                    continue

        results.sort(key=lambda x: x.release_date)
        return results

    async def _get_series_csv(self, series_id: str, limit: int) -> List[MacroDataPoint]:
        """Fallback 100% gratuito sem chave via download direto de CSV oficial do FRED."""
        params = {"id": series_id}
        csv_text = await self.http.get_text(self.FRED_CSV_URL, params=params)
        results: List[MacroDataPoint] = []

        if not csv_text:
            return results

        lines = csv_text.strip().splitlines()
        if len(lines) <= 1:
            return results

        # Cabeçalho: DATE, CPIAUCSL
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 2:
                obs_date = parts[0].strip()
                val_str = parts[1].strip()
                if val_str in (".", ""):
                    continue
                try:
                    val = float(val_str)
                    release_date = self._calculate_effective_release_date(series_id, obs_date)
                    results.append(MacroDataPoint(
                        series_id=series_id,
                        observation_date=obs_date,
                        release_date=release_date,
                        value=val,
                    ))
                except ValueError:
                    continue

        results.sort(key=lambda x: x.release_date)
        return results[-limit:] if limit > 0 else results

    def _calculate_effective_release_date(self, series_id: str, observation_date_str: str) -> str:
        """
        Calcula a data mínima de disponibilidade pública do dado macroeconômico
        para evitar viés de antecipação (lookahead bias) em backtests.
        """
        try:
            obs_dt = datetime.strptime(observation_date_str, "%Y-%m-%d")
            # Para CPI (mensal): divulgado tipicamente por volta do dia 12-15 do mês subsequente
            if series_id.upper().startswith("CPI"):
                # V1: margem conservadora -> dia 20 do mês seguinte (BLS publica ~10-15; revisões/atrasos)
                next_month = (obs_dt.replace(day=1) + timedelta(days=32)).replace(day=20)
                return next_month.strftime("%Y-%m-%d")
            # V1: FEDFUNDS é MÉDIA MENSAL (observation_date = 1º dia do mês): só é conhecida
            # após o fim do mês -> release = último dia do mês + 1 dia (= 1º dia do mês seguinte).
            elif series_id.upper() == "FEDFUNDS":
                return (obs_dt.replace(day=1) + timedelta(days=32)).replace(day=1).strftime("%Y-%m-%d")
            # Séries diárias (DGS10, T10Y2Y): divulgadas no dia seguinte (+1 dia)
            elif series_id.upper() in ("DGS10", "T10Y2Y"):
                return (obs_dt + timedelta(days=1)).strftime("%Y-%m-%d")
            # Regra padrão para séries mensais gerais: +30 dias
            return (obs_dt + timedelta(days=30)).strftime("%Y-%m-%d")
        except Exception:
            return observation_date_str


# =====================================================================
# 4. YAHOO FINANCE CONNECTOR (DXY, VIX, S&P 500)
# =====================================================================

class YahooFinanceConnector:
    """
    Conector para indicadores de intermercado e macro globais:
    - DXY (Índice Dólar): 'DX-Y.NYB' ou ETF 'UUP'
    - VIX (Índice de Volatilidade): '^VIX'
    - S&P 500: '^GSPC'
    Utiliza biblioteca yfinance com fallback transparente via endpoint REST público do Yahoo Finance.
    """

    YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

    def __init__(self, http_client: Optional[AsyncHttpClient] = None):
        self.http = http_client or AsyncHttpClient()

    async def get_history(
        self,
        symbol: str = "^VIX",
        range_str: str = "1mo",
        interval: str = "1d",
    ) -> List[Dict[str, Any]]:
        """
        Busca histórico de preços de índices intermercado.
        Retorna lista de dicionários ordenados cronologicamente.
        """
        if HAS_YFINANCE:
            try:
                loop = asyncio.get_running_loop()
                data = await loop.run_in_executor(None, self._fetch_yfinance_sync, symbol, range_str, interval)
                if data:
                    return data
            except Exception as e:
                logger.debug(f"Falha na biblioteca yfinance para {symbol}: {e}. Tentando fallback REST...")

        return await self._fetch_yahoo_rest(symbol, range_str, interval)

    def _fetch_yfinance_sync(self, symbol: str, period: str, interval: str) -> List[Dict[str, Any]]:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        results: List[Dict[str, Any]] = []
        if df.empty:
            return results

        for idx, row in df.iterrows():
            ts = int(idx.timestamp() * 1000) if hasattr(idx, "timestamp") else 0
            results.append({
                "timestamp": ts,
                "date": str(idx)[:10],
                "open": float(row.get("Open", 0.0)),
                "high": float(row.get("High", 0.0)),
                "low": float(row.get("Low", 0.0)),
                "close": float(row.get("Close", 0.0)),
                "volume": float(row.get("Volume", 0.0)),
                "symbol": symbol,
            })
        return results

    async def _fetch_yahoo_rest(self, symbol: str, range_str: str, interval: str) -> List[Dict[str, Any]]:
        """Fallback REST direto na API v8 do Yahoo Finance."""
        url = self.YAHOO_CHART_URL.format(symbol=symbol)
        params = {"range": range_str, "interval": interval}
        data = await self.http.get_json(url, params=params)
        results: List[Dict[str, Any]] = []

        if not data or not isinstance(data, dict):
            return results

        chart = data.get("chart", {})
        result_list = chart.get("result", [])
        if not result_list:
            return results

        chart_data = result_list[0]
        timestamps = chart_data.get("timestamp", [])
        indicators = chart_data.get("indicators", {}).get("quote", [{}])[0]

        opens = indicators.get("open", [])
        highs = indicators.get("high", [])
        lows = indicators.get("low", [])
        closes = indicators.get("close", [])
        volumes = indicators.get("volume", [])

        for i, ts in enumerate(timestamps):
            try:
                c = closes[i] if i < len(closes) else None
                if c is None:
                    continue
                results.append({
                    "timestamp": int(ts) * 1000,
                    "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                    "open": float(opens[i] or c),
                    "high": float(highs[i] or c),
                    "low": float(lows[i] or c),
                    "close": float(c),
                    "volume": float(volumes[i] or 0.0),
                    "symbol": symbol,
                })
            except (ValueError, TypeError, IndexError):
                continue

        results.sort(key=lambda x: x["timestamp"])
        return results


# =====================================================================
# 5. CRYPTOPANIC API CONNECTOR (SENTIMENTO & NOTÍCIAS)
# =====================================================================

class CryptoPanicConnector:
    """
    Conector para agregação de notícias de sentimento do ecossistema cripto.
    Calcula pontuação normalizada de sentimento com base nos votos de comunidade (bullish/bearish).
    Suporta chave de API pública do CryptoPanic ou fallback aberto.
    """

    BASE_URL = "https://cryptopanic.com/api/v1/posts/"

    def __init__(self, auth_token: Optional[str] = None, http_client: Optional[AsyncHttpClient] = None):
        self.auth_token = auth_token or os.getenv("CRYPTOPANIC_API_KEY", "")
        self.http = http_client or AsyncHttpClient()

    async def get_recent_news(self, currency: str = "BTC", limit: int = 20) -> List[SentimentPost]:
        """
        Retorna notícias recentes com tags e pontuação de sentimento calculada.
        Score: -1.0 (máximo pessimismo) a +1.0 (máxima euforia).
        """
        params: Dict[str, Any] = {
            "currencies": currency.upper(),
            "public": "true",
        }
        if self.auth_token:
            params["auth_token"] = self.auth_token

        data = await self.http.get_json(self.BASE_URL, params=params)
        posts: List[SentimentPost] = []

        if data and isinstance(data, dict) and "results" in data:
            for item in data.get("results", [])[:limit]:
                title = item.get("title", "")
                votes = item.get("votes", {})
                positive = votes.get("positive", 0) + votes.get("liked", 0)
                negative = votes.get("negative", 0) + votes.get("disliked", 0) + votes.get("toxic", 0)

                total_votes = positive + negative
                if total_votes > 0:
                    score = (positive - negative) / total_votes
                else:
                    score = 0.0

                if score > 0.2:
                    label = "bullish"
                elif score < -0.2:
                    label = "bearish"
                else:
                    label = "neutral"

                posts.append(SentimentPost(
                    id=str(item.get("id", "")),
                    title=title,
                    published_at=item.get("published_at", datetime.now(timezone.utc).isoformat()),
                    source=item.get("source", {}).get("title", "CryptoPanic"),
                    sentiment_label=label,
                    score=round(score, 3),
                ))

        return posts

    async def get_consolidated_sentiment_score(self, currency: str = "BTC") -> Dict[str, Any]:
        """Retorna indicador sintético consolidado de sentimento do mercado (-1.0 a +1.0)."""
        posts = await self.get_recent_news(currency=currency, limit=30)
        if not posts:
            return {
                "currency": currency,
                "score": 0.0,
                "sentiment": "neutral",
                "sample_size": 0,
                "bullish_ratio": 0.0,
                "bearish_ratio": 0.0,
            }

        scores = [p.score for p in posts]
        avg_score = sum(scores) / len(scores)
        bullish_count = sum(1 for p in posts if p.sentiment_label == "bullish")
        bearish_count = sum(1 for p in posts if p.sentiment_label == "bearish")

        return {
            "currency": currency,
            "score": round(avg_score, 3),
            "sentiment": "bullish" if avg_score > 0.15 else ("bearish" if avg_score < -0.15 else "neutral"),
            "sample_size": len(posts),
            "bullish_ratio": round(bullish_count / len(posts), 3),
            "bearish_ratio": round(bearish_count / len(posts), 3),
        }


# =====================================================================
# 6. DEFILLAMA CONNECTOR (TVL & FLUXO DE STABLECOINS)
# =====================================================================

class DefiLlamaConnector:
    """
    Conector 100% gratuito e sem necessidade de chave de API para dados on-chain:
    - TVL Total e por blockchains (Ethereum, Solana, Arbitrum, etc.)
    - Capitalização de mercado e influxos líquidos de Stablecoins (USDT, USDC, DAI)
    """

    BASE_URL = "https://api.llama.fi"
    STABLECOINS_URL = "https://stablecoins.llama.fi"

    def __init__(self, http_client: Optional[AsyncHttpClient] = None):
        self.http = http_client or AsyncHttpClient()

    async def get_total_tvl(self) -> Dict[str, Any]:
        """Obtém o Total Value Locked (TVL) agregado no ecossistema DeFi."""
        url = f"{self.BASE_URL}/charts"
        data = await self.http.get_json(url)

        if data and isinstance(data, list) and data:
            latest = data[-1]
            prev = data[-2] if len(data) > 1 else latest
            tvl_now = float(latest.get("totalLiquidityUSD", 0.0))
            tvl_prev = float(prev.get("totalLiquidityUSD", tvl_now))
            pct_change_24h = ((tvl_now - tvl_prev) / tvl_prev * 100.0) if tvl_prev > 0 else 0.0

            return {
                "tvl_usd": tvl_now,
                "change_24h_pct": round(pct_change_24h, 2),
                "timestamp": int(latest.get("date", time.time())),
            }
        return {"tvl_usd": 0.0, "change_24h_pct": 0.0, "timestamp": int(time.time())}

    async def get_stablecoin_flows(self) -> Dict[str, Any]:
        """
        Retorna fluxo de liquidez fresca via market cap agregado de stablecoins.
        Um aumento no supply de stablecoins sinaliza injeção de liquidez compradora.
        """
        url = f"{self.STABLECOINS_URL}/stablecoins?includePrices=true"
        data = await self.http.get_json(url)

        if data and isinstance(data, dict) and "peggedAssets" in data:
            assets = data["peggedAssets"]
            total_mcap = 0.0
            total_prev_day = 0.0

            for asset in assets:
                circulating = asset.get("circulating", {})
                current_pegged = float(circulating.get("peggedUSD", 0.0)) if isinstance(circulating, dict) else 0.0
                total_mcap += current_pegged

                # Mudança diária
                prev_day = asset.get("circulatingPrevDay", {})
                prev_pegged = float(prev_day.get("peggedUSD", current_pegged)) if isinstance(prev_day, dict) else current_pegged
                total_prev_day += prev_pegged

            net_flow_24h = total_mcap - total_prev_day
            flow_pct = ((net_flow_24h / total_prev_day) * 100.0) if total_prev_day > 0 else 0.0

            return {
                "total_stablecoins_usd": round(total_mcap, 2),
                "net_inflow_24h_usd": round(net_flow_24h, 2),
                "inflow_change_24h_pct": round(flow_pct, 4),
                "is_liquidity_expanding": net_flow_24h > 0,
                "timestamp": int(time.time()),
            }

        return {
            "total_stablecoins_usd": 0.0,
            "net_inflow_24h_usd": 0.0,
            "inflow_change_24h_pct": 0.0,
            "is_liquidity_expanding": False,
            "timestamp": int(time.time()),
        }


# =====================================================================
# 7. COINMARKETCAL CONNECTOR (EVENTOS DE MERCADO & FORKS)
# =====================================================================

class CoinMarketCalConnector:
    """
    Conector para eventos do calendário cripto e riscos de deslistagem/hard forks.
    Suporta API Key se presente, ou fallback gratuito via feeds de mercado.
    """

    BASE_URL = "https://developers.coinmarketcal.com/v1/events"

    def __init__(self, api_key: Optional[str] = None, http_client: Optional[AsyncHttpClient] = None):
        self.api_key = api_key or os.getenv("COINMARKETCAL_API_KEY", "")
        self.http = http_client or AsyncHttpClient()

    async def get_upcoming_events(self, coin: str = "BTC", days_ahead: int = 7) -> List[MarketEvent]:
        """Busca eventos relevantes com janela configurável de antecedência."""
        events: List[MarketEvent] = []
        if not self.api_key:
            logger.debug("COINMARKETCAL_API_KEY não informada. Conector operando em modo fallback.")
            return events

        headers = {"x-api-key": self.api_key}
        params = {"coins": coin.lower(), "max": "10"}
        data = await self.http.get_json(self.BASE_URL, params=params, headers=headers)

        if data and isinstance(data, dict) and "body" in data:
            for item in data.get("body", []):
                events.append(MarketEvent(
                    title=item.get("title", {}).get("en", "Crypto Event"),
                    date=item.get("date_event", "")[:10],
                    coin=coin.upper(),
                    category=item.get("categories", [{}])[0].get("name", "General"),
                    impact="high" if item.get("is_hot", False) else "medium",
                ))

        return events


# =====================================================================
# FAÇADA GERENCIADORA UNIFICADA (DATA INTEGRATION MANAGER)
# =====================================================================

class DataIntegrationManager:
    """
    Gerenciador unificado de todas as integrações de dados externas.
    Proporciona um ponto centralizado de acesso resiliente para o DataLoader.
    """

    def __init__(self):
        self.http = AsyncHttpClient()
        self.bitget = BitgetConnector(self.http)
        self.binance = BinanceConnector(self.http)
        self.coinglass = CoinGlassConnector(self.http)
        self.fred = FredConnector(http_client=self.http)
        self.yfinance = YahooFinanceConnector(self.http)
        self.cryptopanic = CryptoPanicConnector(http_client=self.http)
        self.defillama = DefiLlamaConnector(self.http)
        self.coinmarketcal = CoinMarketCalConnector(http_client=self.http)

    async def fetch_all_macro_snapshots(self) -> Dict[str, Any]:
        """Consulta paralela de todos os indicadores macroeconômicos e intermercado."""
        tasks = [
            self.fred.get_series("CPIAUCSL", limit=5),
            self.fred.get_series("FEDFUNDS", limit=10),
            self.yfinance.get_history("^VIX", range_str="5d", interval="1d"),
            self.yfinance.get_history("DX-Y.NYB", range_str="5d", interval="1d"),
            self.yfinance.get_history("^GSPC", range_str="5d", interval="1d"),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "cpi": results[0] if not isinstance(results[0], Exception) else [],
            "fedfunds": results[1] if not isinstance(results[1], Exception) else [],
            "vix": results[2] if not isinstance(results[2], Exception) else [],
            "dxy": results[3] if not isinstance(results[3], Exception) else [],
            "sp500": results[4] if not isinstance(results[4], Exception) else [],
            "timestamp": int(time.time() * 1000),
        }

    async def close(self):
        """Libera conexões e recursos HTTP."""
        await self.http.close()
