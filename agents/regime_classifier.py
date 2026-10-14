"""
regime_classifier.py - Classificador de Regime Macroeconômico de Mercado do IA Trader.
Fase 4 & Governança Macroeconômica.

Missão:
Classificar de forma 100% autônoma, matemática e sem lookahead bias o regime atual do mercado cripto:
1. BULL_TREND: Tendência de alta consolidada (Preço > SMA50 > SMA200, RSI > 50, MACD > 0).
   -> Favorece Swing Long e Breakouts de Momentum.
2. BEAR_TREND: Tendência de baixa consolidada (Preço < SMA50 < SMA200, RSI < 50, MACD < 0).
   -> Favorece Scalps na ponta vendedora e operações defensivas.
3. SIDEWAYS_RANGE: Consolidação e mercado lateral (Preço oscilando próximo à SMA50, Bollinger Width baixo).
   -> Regime ótimo para Scalper Cirúrgico (15m) e Reversões de Golden Pots.
4. HIGH_VOLATILITY_PANIC: Expansão anômala de volatilidade (ATR > 2x média de 30 períodos ou quedas bruscas).
   -> Alerta de Risco Máximo: Redução automática de lote e stops defensivos.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

logger = logging.getLogger("ia_trader.regime_classifier")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [regime] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class RegimeClassification:
    regime: str                     # "BULL_TREND", "BEAR_TREND", "SIDEWAYS_RANGE", "HIGH_VOLATILITY_PANIC"
    bias: str                       # "BULLISH", "BEARISH", "NEUTRAL", "RISK_OFF"
    confidence: float               # [0.0, 1.0]
    recommended_strategies: List[str]
    rationale: str
    metrics: Dict[str, float]
    timestamp: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MarketRegimeClassifier:
    """
    Classificador Quantitativo de Regimes de Mercado.
    Avalia a estrutura do BTC e ETH para ditar o contexto macro para todos os agentes.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "market_cache.db"
        )
        self._last_classification: Optional[RegimeClassification] = None
        self._last_classified_time: float = 0.0
        self._init_database_table()

    def _init_database_table(self) -> None:
        """Cria tabela SQLite para histórico de regimes."""
        if not self.db_path:
            return
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as con:
                con.execute("PRAGMA journal_mode=WAL")
                con.execute("""
                    CREATE TABLE IF NOT EXISTS market_regime (
                        timestamp INTEGER PRIMARY KEY,
                        regime TEXT NOT NULL,
                        bias TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        recommended_strategies TEXT NOT NULL,
                        rationale TEXT NOT NULL,
                        btc_price REAL,
                        sma_50 REAL,
                        sma_200 REAL,
                        atr_ratio REAL,
                        rsi_14 REAL
                    )
                """)
                con.commit()
        except Exception as e:
            logger.warning(f"Erro ao inicializar tabela market_regime: {e}")

    def classify_from_df(self, df: pd.DataFrame, symbol: str = "BTCUSDT") -> RegimeClassification:
        """
        Classifica o regime a partir de um DataFrame de velas (mínimo 50 períodos, ideal 100+).
        """
        now_ts = int(time.time() * 1000)
        if df.empty or len(df) < 20:
            return RegimeClassification(
                regime="SIDEWAYS_RANGE",
                bias="NEUTRAL",
                confidence=0.50,
                recommended_strategies=["SCALP_REVERSION"],
                rationale="Dados insuficientes para classificação macro detalhada; assumindo consolidação neutra.",
                metrics={},
                timestamp=now_ts,
            )

        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        last_price = float(close.iloc[-1])
        sma20 = float(close.rolling(20, min_periods=5).mean().iloc[-1])
        sma50 = float(close.rolling(50, min_periods=10).mean().iloc[-1])
        sma200 = float(close.rolling(200, min_periods=20).mean().iloc[-1])

        # RSI 14
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=5).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = float(100 - (100 / (1 + rs)).iloc[-1])
        if np.isnan(rsi):
            rsi = 50.0

        # ATR 14 e ATR ratio de volatilidade
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14, min_periods=5).mean()
        atr_baseline = tr.rolling(50, min_periods=10).mean()

        curr_atr = float(atr14.iloc[-1]) if not atr14.empty else (last_price * 0.01)
        base_atr = float(atr_baseline.iloc[-1]) if not atr_baseline.empty and atr_baseline.iloc[-1] > 0 else curr_atr
        atr_ratio = curr_atr / base_atr if base_atr > 0 else 1.0

        # Retorno de 24 períodos
        ret_24p = float(((close.iloc[-1] - close.iloc[-min(24, len(close))]) / close.iloc[-min(24, len(close))]) * 100.0)

        metrics = {
            "last_price": last_price,
            "sma_20": sma20,
            "sma_50": sma50,
            "sma_200": sma200,
            "rsi_14": rsi,
            "atr_14": curr_atr,
            "atr_ratio": atr_ratio,
            "return_24p_pct": ret_24p,
        }

        # 1. Regra de Pânico / Alta Volatilidade Anômala
        if atr_ratio >= 2.2 or ret_24p <= -12.0:
            regime = "HIGH_VOLATILITY_PANIC"
            bias = "RISK_OFF"
            confidence = min(0.95, 0.60 + (atr_ratio - 2.0) * 0.2)
            recs = ["RISK_VETO", "DEFENSIVE_SCALP"]
            rationale = f"Volatilidade extrema detectada (ATR ratio {atr_ratio:.2f}x). Pânico ou choque de liquidez no mercado."

        # 2. Tendência de Alta Consolidada (Bull Trend)
        elif last_price > sma50 and (last_price > sma200 or sma20 > sma50) and rsi >= 50.0 and ret_24p >= 1.5:
            regime = "BULL_TREND"
            bias = "BULLISH"
            confidence = 0.85 if (last_price > sma200 and sma50 > sma200) else 0.70
            recs = ["SWING_BREAKOUT", "MOMENTUM_TREND", "SCALP_LONG"]
            rationale = f"Estrutura altista dominante: Preço acima das médias (SMA50: ${sma50:,.2f}), RSI em {rsi:.1f} e retorno de {ret_24p:+.2f}%."

        # 3. Tendência de Baixa Consolidada (Bear Trend)
        elif last_price < sma50 and (last_price < sma200 or sma20 < sma50) and rsi <= 50.0 and ret_24p <= -1.5:
            regime = "BEAR_TREND"
            bias = "BEARISH"
            confidence = 0.85 if (last_price < sma200 and sma50 < sma200) else 0.70
            recs = ["SWING_SHORT", "SCALP_SHORT", "DEFENSIVE_REDUCE_SIZE"]
            rationale = f"Pressão vendedora macro: Preço abaixo da SMA50 (${sma50:,.2f}), RSI fraco ({rsi:.1f}) e retorno de {ret_24p:+.2f}%."

        # 4. Mercado Lateral / Consolidação (Sideways Range)
        else:
            regime = "SIDEWAYS_RANGE"
            bias = "NEUTRAL"
            confidence = 0.75
            recs = ["SCALP_REVERSION", "GOLDEN_POT_REVERSION"]
            rationale = f"Mercado oscilando em faixa lateral (RSI {rsi:.1f}, ret 24p {ret_24p:+.2f}%). Regime excelente para reversão à média e scalps cirúrgicos."

        classification = RegimeClassification(
            regime=regime,
            bias=bias,
            confidence=round(confidence, 2),
            recommended_strategies=recs,
            rationale=rationale,
            metrics=metrics,
            timestamp=now_ts,
        )

        self._last_classification = classification
        self._last_classified_time = time.time()
        self._save_regime(classification)

        return classification

    def _save_regime(self, reg: RegimeClassification) -> None:
        """Salva snapshot do regime no SQLite."""
        if not self.db_path:
            return
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as con:
                con.execute("""
                    INSERT OR REPLACE INTO market_regime (
                        timestamp, regime, bias, confidence, recommended_strategies,
                        rationale, btc_price, sma_50, sma_200, atr_ratio, rsi_14
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    reg.timestamp,
                    reg.regime,
                    reg.bias,
                    reg.confidence,
                    ",".join(reg.recommended_strategies),
                    reg.rationale,
                    reg.metrics.get("last_price", 0.0),
                    reg.metrics.get("sma_50", 0.0),
                    reg.metrics.get("sma_200", 0.0),
                    reg.metrics.get("atr_ratio", 1.0),
                    reg.metrics.get("rsi_14", 50.0),
                ))
                con.commit()
        except Exception as e:
            logger.debug(f"Aviso ao persistir market_regime: {e}")
