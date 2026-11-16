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


def test_rsi_all_gains_is_100_not_neutral_50(tmp_path):
    import numpy as np
    import pandas as pd
    from agents.regime_classifier import MarketRegimeClassifier
    n = 60
    close = pd.Series(100.0 + np.arange(n) * 1.0)
    df = pd.DataFrame({"close": close, "high": close + 0.5, "low": close - 0.5})
    res = MarketRegimeClassifier(db_path=str(tmp_path / "r.db")).classify_from_df(df)
    assert res.metrics["rsi_14"] == 100.0


def test_consolidated_bear_trend_with_pause_in_24h():
    """Tendência de baixa consolidada (< SMA50 < SMA200, -4% da SMA200) permanece BEAR_TREND com pausa 24p."""
    # 225 barras caindo de 80k para 70k, depois 25 barras oscilando com ret_24p ~= -0.5% e RSI < 50
    closes = [80000.0 - i * 44.0 for i in range(225)]
    base = closes[-1]
    for i in range(24):
        closes.append(base - (5.0 if i % 2 == 0 else -3.0) - i * 1.5)
    closes.append(closes[-24] * 0.995)  # ret_24p == -0.5%
    n = len(closes)
    df = pd.DataFrame({
        "timestamp": [1700000000000 + i * 3600000 for i in range(n)],
        "open": closes,
        "high": [c + 20.0 for c in closes],
        "low": [c - 20.0 for c in closes],
        "close": closes,
        "volume": [1000.0 for _ in range(n)],
    })
    res = MarketRegimeClassifier(db_path=None).classify_from_df(df)
    assert res.metrics["has_real_sma200"] is True
    assert res.regime == "BEAR_TREND"
    assert res.bias == "BEARISH"

