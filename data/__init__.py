"""
Módulo de Dados do IA Trader.
Camada responsável por ingestão de dados 100% gratuitos, cache local persistente SQLite
e preparação limpa para o Comitê de Agentes e Modelos de Machine Learning.
"""

from .data_integrations import (
    CandleData,
    OrderBookDepth,
    MacroDataPoint,
    SentimentPost,
    OnChainMetric,
    MarketEvent,
    BitgetConnector,
    BinanceConnector,
    CoinGlassConnector,
    FredConnector,
    YahooFinanceConnector,
    CryptoPanicConnector,
    DefiLlamaConnector,
    CoinMarketCalConnector,
    DataIntegrationManager,
)

from .data_loader import DataLoader
try:
    from .feature_engineering import FeatureEngineer
except ImportError:
    FeatureEngineer = None

__all__ = [
    "CandleData",
    "OrderBookDepth",
    "MacroDataPoint",
    "SentimentPost",
    "OnChainMetric",
    "MarketEvent",
    "BitgetConnector",
    "BinanceConnector",
    "CoinGlassConnector",
    "FredConnector",
    "YahooFinanceConnector",
    "CryptoPanicConnector",
    "DefiLlamaConnector",
    "CoinMarketCalConnector",
    "DataIntegrationManager",
    "DataLoader",
    "FeatureEngineer",
]
