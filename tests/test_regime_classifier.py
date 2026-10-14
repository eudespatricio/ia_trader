"""
test_regime_classifier.py - Testes Unitários do Classificador de Regime Macroeconômico.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from agents.regime_classifier import MarketRegimeClassifier, RegimeClassification


class TestMarketRegimeClassifier:

    def test_empty_df_fallback(self):
        classifier = MarketRegimeClassifier(db_path=None)
        res = classifier.classify_from_df(pd.DataFrame())
        assert res.regime == "SIDEWAYS_RANGE"
        assert res.bias == "NEUTRAL"
        assert res.confidence == 0.50

    def test_bull_trend_classification(self):
        """Preço em forte alta consistente acima das médias."""
        prices = [50000.0 + i * 200.0 for i in range(100)] # Sobe de 50k a 70k
        df = pd.DataFrame({
            "timestamp": [1700000000000 + i * 3600000 for i in range(100)],
            "open": prices,
            "high": [p + 50.0 for p in prices],
            "low": [p - 50.0 for p in prices],
            "close": [p + 20.0 for p in prices],
            "volume": [1000.0 for _ in range(100)],
        })
        classifier = MarketRegimeClassifier(db_path=None)
        res = classifier.classify_from_df(df)
        assert res.regime == "BULL_TREND"
        assert res.bias == "BULLISH"
        assert res.confidence >= 0.70
        assert "SWING_BREAKOUT" in res.recommended_strategies

    def test_panic_high_volatility_classification(self):
        """Queda abrupta de mais de 15% em 24 períodos com explosão de ATR."""
        prices = [60000.0] * 50
        # Queda repentina de crash
        for i in range(25):
            prices.append(prices[-1] - 800.0) # Despenca $20k

        df = pd.DataFrame({
            "timestamp": [1700000000000 + i * 3600000 for i in range(len(prices))],
            "open": prices,
            "high": [p + 200.0 for p in prices],
            "low": [p - 400.0 for p in prices],
            "close": [p - 100.0 for p in prices],
            "volume": [5000.0 for _ in range(len(prices))],
        })
        classifier = MarketRegimeClassifier(db_path=None)
        res = classifier.classify_from_df(df)
        assert res.regime == "HIGH_VOLATILITY_PANIC"
        assert res.bias == "RISK_OFF"
        assert "RISK_VETO" in res.recommended_strategies
