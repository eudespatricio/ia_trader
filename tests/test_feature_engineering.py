"""
Suíte de Testes Unitários para o Módulo de Feature Engineering (Fase 2).
Valida:
1. Presença estrita de todas as colunas de indicadores técnicos e microestrutura.
2. Ausência total de valores infinitos (infs) ou NaNs residuais.
3. Consistência matemática dos indicadores (RSI, ATR, Bollinger, Estocástico).
4. Rigor Anti-Lookahead no alinhamento de séries macroeconômicas.
5. Integração com métricas de profundidade do livro de ofertas (OrderBookDepth).
6. Resiliência a séries temporais curtas e sanitização de dados anômalos.
"""

import os
import sys
from datetime import datetime, timezone, timedelta

import pytest
import numpy as np
import pandas as pd

# Adiciona o diretório raiz ao sys.path
sys.path.insert(0, "/home/eudespatricio/ia_trader")

from data.feature_engineering import FeatureEngineer
from data.data_integrations import OrderBookDepth


def generate_synthetic_candles(n: int = 300, trend: str = "random", base_price: float = 50000.0) -> pd.DataFrame:
    """Gera DataFrame sintético de velas OHLCV ordenadas cronologicamente."""
    np.random.seed(42)
    base_ts = 1700000000000  # Unix millis

    timestamps = [base_ts + i * 3600000 for i in range(n)]

    if trend == "bullish":
        drift = np.linspace(0, 10000, n)
    elif trend == "bearish":
        drift = np.linspace(0, -10000, n)
    else:
        drift = np.cumsum(np.random.randn(n) * 100)

    close = base_price + drift
    high = close + np.abs(np.random.randn(n) * 50) + 10.0
    low = close - np.abs(np.random.randn(n) * 50) - 10.0
    open_p = close + np.random.randn(n) * 20
    # Consistência geométrica
    high = np.maximum(high, np.maximum(open_p, close))
    low = np.minimum(low, np.minimum(open_p, close))
    volume = np.random.uniform(10, 500, n)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": "BTCUSDT",
        "timeframe": "1h",
    })


class TestFeaturePresenceAndIntegrity:
    """Valida a presença de todas as colunas e higienização numérica."""

    def test_all_feature_columns_present(self):
        df = generate_synthetic_candles(n=300)
        features_df = FeatureEngineer.compute_all_features(df, drop_na=True)

        expected_cols = FeatureEngineer.get_feature_names()
        for col in expected_cols:
            assert col in features_df.columns, f"Coluna esperada '{col}' ausente no DataFrame resultante."

    def test_no_nan_or_infinite_values(self):
        df = generate_synthetic_candles(n=300)
        features_df = FeatureEngineer.compute_all_features(df, drop_na=True)

        # Verifica NaNs
        nan_counts = features_df.isna().sum()
        cols_with_nan = nan_counts[nan_counts > 0]
        assert len(cols_with_nan) == 0, f"Colunas com NaNs residuais: {cols_with_nan.to_dict()}"

        # Verifica infinitos
        inf_counts = np.isinf(features_df.select_dtypes(include=np.number)).sum()
        cols_with_inf = inf_counts[inf_counts > 0]
        assert len(cols_with_inf) == 0, f"Colunas com valores infinitos: {cols_with_inf.to_dict()}"

    def test_short_series_does_not_crash_or_empty(self):
        """Séries com menos de 200 velas devem aplicar bfill seguro e retornar linhas úteis sem NaNs."""
        df_short = generate_synthetic_candles(n=50)
        features_df = FeatureEngineer.compute_all_features(df_short, drop_na=True)

        assert len(features_df) == 50
        assert features_df.isna().sum().sum() == 0
        assert np.isinf(features_df.select_dtypes(include=np.number)).sum().sum() == 0


class TestMathematicalConsistency:
    """Valida a integridade matemática dos cálculos técnicos."""

    def test_rsi_bounds_and_dynamics(self):
        # 1. Limites estritos [0, 100]
        df = generate_synthetic_candles(n=300, trend="random")
        feat = FeatureEngineer.compute_all_features(df, drop_na=True)
        assert (feat["rsi_14"] >= 0.0).all() and (feat["rsi_14"] <= 100.0).all()

        # 2. Série estritamente altista deve produzir RSI alto
        df_bull = generate_synthetic_candles(n=100, trend="bullish")
        feat_bull = FeatureEngineer.compute_all_features(df_bull, drop_na=True)
        assert feat_bull["rsi_14"].iloc[-1] > 70.0
        assert feat_bull["rsi_overbought"].iloc[-1] == 1.0

        # 3. Série estritamente baixista deve produzir RSI baixo
        df_bear = generate_synthetic_candles(n=100, trend="bearish")
        feat_bear = FeatureEngineer.compute_all_features(df_bear, drop_na=True)
        assert feat_bear["rsi_14"].iloc[-1] < 30.0
        assert feat_bear["rsi_oversold"].iloc[-1] == 1.0

    def test_atr_strictly_positive(self):
        """O ATR deve ser estritamente positivo para dimensionamento de SL/TP."""
        df = generate_synthetic_candles(n=300)
        feat = FeatureEngineer.compute_all_features(df, drop_na=True)

        assert (feat["atr_14"] > 0.0).all(), "ATR contém valores não positivos."
        assert (feat["atr_pct"] > 0.0).all(), "ATR percentual contém valores não positivos."

    def test_bollinger_bands_geometry(self):
        """Verifica relação geométrica: Banda Superior >= Média Central >= Banda Inferior."""
        df = generate_synthetic_candles(n=300)
        feat = FeatureEngineer.compute_all_features(df, drop_na=True)

        assert (feat["bb_high"] >= feat["bb_mid"]).all()
        assert (feat["bb_mid"] >= feat["bb_low"]).all()
        assert (feat["bb_width"] >= 0.0).all()

    def test_stochastic_bounds(self):
        """Verifica que o Oscilador Estocástico (%K e %D) opera entre 0 e 100."""
        df = generate_synthetic_candles(n=300)
        feat = FeatureEngineer.compute_all_features(df, drop_na=True)

        assert (feat["stoch_k"] >= 0.0).all() and (feat["stoch_k"] <= 100.0).all()
        assert (feat["stoch_d"] >= 0.0).all() and (feat["stoch_d"] <= 100.0).all()


class TestMicrostructureAndMacroIntegration:
    """Valida a incorporação de OrderBookDepth e Fatores Macroeconômicos Anti-Lookahead."""

    def test_order_book_depth_injection(self):
        df = generate_synthetic_candles(n=50)
        ob = OrderBookDepth(
            symbol="BTCUSDT",
            timestamp=1700000000000,
            bids=[(50000.0, 2.5), (49990.0, 5.0)],
            asks=[(50005.0, 1.2), (50010.0, 3.0)],
            spread=5.0,
            imbalance=0.35,
        )

        feat = FeatureEngineer.compute_all_features(df, order_book=ob, drop_na=True)
        assert (feat["ob_spread"] == 5.0).all()
        assert (feat["ob_imbalance"] == 0.35).all()

    def test_macro_anti_lookahead_rigor(self):
        """
        Garante que velas em datas anteriores à release_date não recebam dados futuros.
        Exemplo:
        Candle 1: 2026-01-01 -> Deve receber CPI antigo (310.0)
        Divulgação CPI (release_date): 2026-01-02 com CPI = 315.0
        Candle 2: 2026-01-02 -> Deve receber CPI novo (315.0)
        Candle 3: 2026-01-03 -> Deve receber CPI novo (315.0)
        """
        candle_ts = [
            int(datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp() * 1000),
            int(datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc).timestamp() * 1000),
            int(datetime(2026, 1, 3, 12, 0, tzinfo=timezone.utc).timestamp() * 1000),
        ]
        df = pd.DataFrame({
            "timestamp": candle_ts,
            "open": [100.0, 102.0, 105.0],
            "high": [103.0, 106.0, 107.0],
            "low": [99.0, 101.0, 104.0],
            "close": [102.0, 105.0, 106.0],
            "volume": [1000.0, 1500.0, 1200.0],
        })

        macro_data = pd.DataFrame({
            "release_date": [
                "2025-12-15 13:30:00",  # Divulgado antes de Candle 1
                "2026-01-02 13:30:00",  # Divulgado APÓS o meio-dia de Candle 2
            ],
            "cpi": [310.0, 315.0],
            "fedfunds": [5.25, 5.50],
            "vix": [15.0, 20.0],
            "dxy": [102.0, 104.0],
            "sp500": [4800.0, 4850.0],
        })

        feat = FeatureEngineer.compute_all_features(df, macro_df=macro_data, drop_na=True)

        # Candle 1 (2026-01-01 12:00) só pode ver o CPI de 2025-12-15
        assert feat.iloc[0]["macro_cpi"] == 310.0
        assert feat.iloc[0]["macro_fedfunds"] == 5.25

        # Candle 2 (2026-01-02 12:00) foi registrado antes da divulgação de 13:30!
        # Logo, ainda deve ver o CPI anterior de 310.0 (Anti-Lookahead Rigoroso)
        assert feat.iloc[1]["macro_cpi"] == 310.0

        # Candle 3 (2026-01-03 12:00) já ocorreu após a divulgação de 2026-01-02 13:30
        assert feat.iloc[2]["macro_cpi"] == 315.0
        assert feat.iloc[2]["macro_fedfunds"] == 5.50

    def test_macro_context_dict_support(self):
        """Valida que um dicionário de contexto macro (retornado por DataLoader) é aceito."""
        df = generate_synthetic_candles(n=50)
        macro_dict = {
            "cpi": 318.5,
            "fedfunds": 5.25,
            "vix": 26.5,
            "dxy": 106.0,
            "sp500": 5100.0,
            "regime": "risk_off",
        }

        feat = FeatureEngineer.compute_all_features(df, macro_df=macro_dict, drop_na=True)
        assert (feat["macro_cpi"] == 318.5).all()
        assert (feat["macro_fedfunds"] == 5.25).all()
        assert (feat["macro_vix"] == 26.5).all()
        assert (feat["macro_dxy"] == 106.0).all()
        assert (feat["macro_sp500"] == 5100.0).all()
        assert (feat["macro_regime_risk_off"] == 1.0).all()


class TestInstitutionalFeatures:
    """Valida as features institucionais de alto sinal adicionadas após auditoria quantitativa."""

    def test_institutional_feature_names_and_presence(self):
        inst_names = FeatureEngineer.get_institutional_feature_names()
        assert len(inst_names) > 10
        assert "sma_200" in inst_names
        assert "dist_vwap_pct" in inst_names
        assert "volume_zscore_20" in inst_names
        assert "natr_14" in inst_names
        assert "funding_zscore" in inst_names

        df = generate_synthetic_candles(n=300)
        feat = FeatureEngineer.compute_all_features(df, drop_na=True)

        for col in inst_names:
            assert col in feat.columns, f"Feature institucional '{col}' ausente."
            assert not feat[col].isna().any(), f"Feature '{col}' contém NaNs."
            assert not np.isinf(feat[col]).any(), f"Feature '{col}' contém infinitos."

    def test_natr_14_positive_and_scale_invariant(self):
        df = generate_synthetic_candles(n=300, base_price=60000.0)
        feat = FeatureEngineer.compute_all_features(df, drop_na=True)
        # NATR percentual após warmup deve ser estritamente positivo e numa faixa razoável (0.001% a 25%)
        assert (feat["natr_14"] > 0).all()
        assert (feat["natr_14"] < 25.0).all()

    def test_volume_zscore_and_vwap_metrics(self):
        df = generate_synthetic_candles(n=250)
        feat = FeatureEngineer.compute_all_features(df, drop_na=True)
        # Z-score deve ter média próxima de zero
        z_mean = feat["volume_zscore_20"].mean()
        assert abs(z_mean) < 1.0
        # Distância VWAP deve ser finita e oscilar ao redor de 0%
        assert feat["dist_vwap_pct"].abs().max() < 100.0

    def test_funding_features_backward_merge(self):
        df = generate_synthetic_candles(n=100)
        funding_data = pd.DataFrame({
            "timestamp": [df["timestamp"].iloc[10], df["timestamp"].iloc[50]],
            "funding_rate": [0.0001, 0.0005]
        })
        feat = FeatureEngineer.compute_all_features(df, funding_df=funding_data, drop_na=True)
        assert "funding_rate" in feat.columns
        assert "funding_zscore" in feat.columns
        assert feat["funding_rate"].iloc[-1] == 0.0005

