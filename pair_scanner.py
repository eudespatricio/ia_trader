"""
pair_scanner.py - Varredura Estratégica de Pares, Volatilidade Anômala e Risco de Delistagem.
Fase 9 do Projeto IA Trader (Páginas 3 e 4 da Documentação Oficial).

Objetivo:
1. Varredura contínua e periódica de pares na Bitget (e Binance) para identificar:
   - Picos incomuns de volatilidade (ATR relativo > 2.5x ou expansão anormal de volume).
   - Pares com descolamento de momentum e fluxo comprador/vendedor agressivo.
   - Detecção de risco de anomalia ou delistagem iminente (quedas bruscas sem volume ou spreads anormais).
2. Dimensionamento de Risco Específico:
   - Reserva de capital de risco dedicada e isolada (ex: máximo 5% do capital total) para evitar contaminar a carteira principal de Scalps e Golden Pots.
3. Classificação e Alertas em Tempo Real para o Dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ia_trader.pair_scanner")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [pair_scanner] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class VolatilityOpportunity:
    """Oportunidade identificada na varredura de mercado."""
    symbol: str
    opportunity_type: str         # "VOLATILITY_BREAKOUT", "PUMP_DETECTED", "DELISTING_RISK", "VOLUME_SURGE"
    relative_volatility: float    # Ratio ATR atual / ATR médio
    relative_volume: float        # Ratio Volume atual / Volume médio
    change_24h_pct: float
    current_price: float
    risk_level: str               # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    allocated_capital_pct: float  # Ex: 2.0% para operações especiais
    rationale: str
    timestamp: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PairScanner:
    """
    Scanner autônomo de mercado para pares voláteis e alertas especiais.
    Trabalha com listas de pares da exchange e dados em cache.
    """

    DEFAULT_WATCHLIST = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "ENAUSDT", "BTWUSDT",
        "ZECUSDT", "BATUSDT", "OPUSDT", "STRKUSDT", "KORUUSDT"
    ]

    def __init__(
        self,
        db_path: str = "datasets/market_cache.db",
        volatility_threshold: float = 2.0,  # ATR 2x acima da média
        volume_surge_threshold: float = 2.5, # Volume 2.5x acima da média
        max_isolated_capital_pct: float = 5.0, # Teto de 5% de capital para pares especiais
    ):
        self.db_path = db_path
        self.volatility_threshold = float(volatility_threshold)
        self.volume_surge_threshold = float(volume_surge_threshold)
        self.max_isolated_capital_pct = float(max_isolated_capital_pct)
        self.cached_opportunities: List[VolatilityOpportunity] = []
        self.last_scan_time: float = 0.0

    def analyze_pair_candles(self, symbol: str, df_candles: pd.DataFrame) -> Optional[VolatilityOpportunity]:
        """
        Analisa a série recente de velas de um par e detecta anomalias.
        """
        if df_candles.empty or len(df_candles) < 25:
            return None

        closes = df_candles["close"].astype(float).values
        highs = df_candles["high"].astype(float).values
        lows = df_candles["low"].astype(float).values
        volumes = df_candles["volume"].astype(float).values
        now_ms = int(time.time() * 1000)

        # Cálculo de True Range e ATR
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
        )
        atr_14 = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(tr[-1])
        atr_baseline = float(np.mean(tr)) if len(tr) > 0 else atr_14

        vol_ratio = (atr_14 / (atr_baseline + 1e-12))
        curr_vol = float(volumes[-1])
        mean_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else curr_vol
        volume_ratio = (curr_vol / (mean_vol + 1e-12))

        curr_price = float(closes[-1])
        price_24h_ago = float(closes[0])
        change_pct = ((curr_price - price_24h_ago) / price_24h_ago) * 100.0

        # 1. Detecção de Risco de Colapso / Delistagem Iminente
        # Queda abrupta superior a 40% com volume anômalo ou colapso de liquidez
        if change_pct <= -35.0 or (change_pct <= -20.0 and volume_ratio < 0.20):
            return VolatilityOpportunity(
                symbol=symbol,
                opportunity_type="DELISTING_RISK",
                relative_volatility=round(vol_ratio, 2),
                relative_volume=round(volume_ratio, 2),
                change_24h_pct=round(change_pct, 2),
                current_price=curr_price,
                risk_level="CRITICAL",
                allocated_capital_pct=0.0, # Zero capital para pares sob suspeita de delistagem
                rationale=f"ALERTA CRÍTICO: Queda extrema de {change_pct:.1f}% com anomalia de liquidez. Risco severo de delistagem ou colapso.",
                timestamp=now_ms
            )

        # 2. Explosão de Volume com Rompimento de Volatilidade (Pump / Volatility Surge)
        if vol_ratio >= self.volatility_threshold and volume_ratio >= self.volume_surge_threshold:
            opp_type = "PUMP_DETECTED" if change_pct > 10.0 else "VOLATILITY_BREAKOUT"
            risk_level = "HIGH" if change_pct > 15.0 else "MEDIUM"
            alloc = min(self.max_isolated_capital_pct, 2.5)

            return VolatilityOpportunity(
                symbol=symbol,
                opportunity_type=opp_type,
                relative_volatility=round(vol_ratio, 2),
                relative_volume=round(volume_ratio, 2),
                change_24h_pct=round(change_pct, 2),
                current_price=curr_price,
                risk_level=risk_level,
                allocated_capital_pct=alloc,
                rationale=f"Expansão atípica: Volatilidade {vol_ratio:.1f}x da média e Volume {volume_ratio:.1f}x da linha de base.",
                timestamp=now_ms
            )

        # 3. Volatilidade Acima da Média com Volume Regular
        if vol_ratio >= self.volatility_threshold:
            return VolatilityOpportunity(
                symbol=symbol,
                opportunity_type="VOLATILITY_BREAKOUT",
                relative_volatility=round(vol_ratio, 2),
                relative_volume=round(volume_ratio, 2),
                change_24h_pct=round(change_pct, 2),
                current_price=curr_price,
                risk_level="MEDIUM",
                allocated_capital_pct=1.5,
                rationale=f"Volatilidade elevada ({vol_ratio:.1f}x acima da média). Oportunidade para scalping de momentum.",
                timestamp=now_ms
            )

        return None

    def scan_watchlist(self, symbols_or_candles: Any = None) -> List[VolatilityOpportunity]:
        """
        Executa a varredura em um lote de pares e retorna as oportunidades ranqueadas.
        Aceita tanto um dicionário {symbol: DataFrame} quanto uma lista de símbolos.
        """
        opps: List[VolatilityOpportunity] = []

        if isinstance(symbols_or_candles, dict):
            for symbol, df in symbols_or_candles.items():
                opp = self.analyze_pair_candles(symbol, df)
                if opp is not None:
                    opps.append(opp)
        elif isinstance(symbols_or_candles, list):
            from data.data_loader import DataLoader
            loader = DataLoader(db_path=self.db_path)
            for symbol in symbols_or_candles:
                try:
                    df = loader.get_cached_candles(symbol, "15m", limit=50)
                    if not df.empty:
                        opp = self.analyze_pair_candles(symbol, df)
                        if opp is not None:
                            opps.append(opp)
                except Exception:
                    continue
        elif symbols_or_candles is None:
            return self.scan_watchlist(self.DEFAULT_WATCHLIST)

        # Ordena por volatilidade relativa decrescente
        opps.sort(key=lambda x: x.relative_volatility, reverse=True)
        self.cached_opportunities = opps
        self.last_scan_time = time.time()
        return opps

