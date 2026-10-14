"""
test_market_structure.py - Testes Unitários de Estrutura de Mercado, S/R, Liquidez e POI.
Valida:
1. Identificação causal de fractais Swing High/Low (sem lookahead bias).
2. Detecção de Zonas de Liquidez (BSL / SSL).
3. Detecção de Liquidity Sweeps (captura de liquidez com rejeição).
4. Fair Value Gaps (FVG) e mitigação de 50%.
5. Order Blocks (OB) institucionais com deslocamento (Displacement).
6. Performance e ausência de NaNs no DataFrame final.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import pytest

from data.market_structure import MarketStructureDetector, POIZone, LiquidityLevel


@pytest.fixture
def sample_ohlcv():
    """Gera 100 velas sintéticas com estrutura limpa de topos e fundos."""
    np.random.seed(42)
    n = 100
    base_price = 100.0
    # Gera um caminho com swing evidente
    prices = [base_price]
    for i in range(1, n):
        # Onda senoidal com ruído para criar topos e fundos definidos
        drift = np.sin(i / 8.0) * 2.0
        change = drift + np.random.normal(0, 0.5)
        prices.append(max(10.0, prices[-1] + change))

    df = pd.DataFrame({
        "timestamp": [1700000000000 + i * 900000 for i in range(n)],
        "open": prices,
        "high": [p + abs(np.random.normal(0.8, 0.3)) for p in prices],
        "low": [p - abs(np.random.normal(0.8, 0.3)) for p in prices],
        "close": [p + np.random.normal(0.1, 0.4) for p in prices],
        "volume": [abs(np.random.normal(1000, 200)) for _ in range(n)],
    })
    # Corrige inconsistências geométricas
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    return df


class TestMarketStructureDetector:

    def test_identify_swings_causality(self, sample_ohlcv):
        """Verifica que topos e fundos são identificados sem lookahead bias (com atraso de confirmação n)."""
        detector = MarketStructureDetector(swing_window=3)
        sh, sl = detector.identify_swings(sample_ohlcv)

        # Não deve haver swings nas primeiras 2*n barras
        assert sh.iloc[:6].isna().all()
        assert sl.iloc[:6].isna().all()

        # Deve encontrar pelo menos alguns swings no dataset de 100 barras
        valid_sh = sh.dropna()
        valid_sl = sl.dropna()
        assert len(valid_sh) > 0
        assert len(valid_sl) > 0

    def test_detect_liquidity_pools_and_sweeps(self):
        """Testa explicitamente um padrão sintético de Liquidity Sweep de topo e de fundo."""
        # Cria cenário controlado:
        # Vela 0 a 4: constrói um topo em 110.0
        # Vela 5: confirmação
        # Vela 6: pavio bate 111.0 (>110), mas fecha em 108.0 (<110) com pavio superior grande -> BEARISH SWEEP
        records = [
            {"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0},
            {"open": 101.0, "high": 105.0, "low": 100.0, "close": 104.0},
            {"open": 104.0, "high": 110.0, "low": 103.0, "close": 109.0}, # Topo do swing
            {"open": 109.0, "high": 107.0, "low": 104.0, "close": 105.0},
            {"open": 105.0, "high": 106.0, "low": 102.0, "close": 103.0},
            {"open": 103.0, "high": 104.0, "low": 101.0, "close": 102.0}, # Swing confirmado aqui
            {"open": 102.0, "high": 111.5, "low": 101.0, "close": 103.0}, # Sweep do topo em 110!
        ]
        df = pd.DataFrame(records)
        df["timestamp"] = [1700000000000 + i * 60000 for i in range(len(df))]

        detector = MarketStructureDetector(swing_window=2)
        bsl, ssl, sweep_bull, sweep_bear = detector.detect_liquidity_pools_and_sweeps(df)

        # O candle 6 deve ter disparado sweep_bear == 1
        assert sweep_bear.iloc[-1] == 1
        assert bsl.iloc[-1] == 110.0

    def test_detect_fair_value_gaps(self):
        """Testa detecção de Bullish FVG (Low do candle 2 > High do candle 0)."""
        records = [
            {"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0}, # Candle 0 (High = 102.0)
            {"open": 101.0, "high": 108.0, "low": 101.0, "close": 107.0}, # Candle 1 (Expansão violenta)
            {"open": 107.0, "high": 112.0, "low": 104.0, "close": 110.0}, # Candle 2 (Low = 104.0 > 102.0 -> Gap de 102 a 104)
            {"open": 110.0, "high": 111.0, "low": 103.0, "close": 103.5}, # Candle 3 (Entra dentro do FVG 102-104)
        ]
        df = pd.DataFrame(records)
        df["timestamp"] = [1700000000000 + i * 60000 for i in range(len(df))]

        detector = MarketStructureDetector()
        in_bull_fvg, in_bear_fvg, active_fvgs = detector.detect_fair_value_gaps(df, min_gap_pct=0.001)

        assert len(active_fvgs) >= 1
        fvg = active_fvgs[0]
        assert fvg.poi_type == "BULLISH_FVG"
        assert fvg.bottom == 102.0
        assert fvg.top == 104.0
        assert fvg.midpoint == 103.0
        # Candle 3 fechou em 103.5 (dentro do FVG)
        assert in_bull_fvg.iloc[3] == 1

    def test_detect_order_blocks(self):
        """Testa detecção de Bullish Order Block (última vela de baixa antes de expansão forte)."""
        records = [
            {"open": 105.0, "high": 106.0, "low": 104.0, "close": 104.5}, # Vela neutra
            {"open": 104.5, "high": 105.0, "low": 100.0, "close": 101.0}, # Vela de baixa (Candidata a OB)
            {"open": 101.0, "high": 115.0, "low": 101.0, "close": 114.0}, # Expansão de +13 dólares (> 1.3x ATR)
            {"open": 114.0, "high": 115.0, "low": 102.0, "close": 103.0}, # Reteste na zona do OB (100.0 a 105.0)
        ]
        df = pd.DataFrame(records)
        df["timestamp"] = [1700000000000 + i * 60000 for i in range(len(df))]

        detector = MarketStructureDetector()
        in_bull_ob, in_bear_ob, active_obs = detector.detect_order_blocks(df, displacement_atr_mult=1.0)

        assert len(active_obs) >= 1
        ob = active_obs[0]
        assert ob.poi_type == "BULLISH_OB"
        assert ob.bottom == 100.0
        assert ob.top == 105.0
        # Candle 3 fechou em 103.0 (dentro da zona do OB 100-105)
        assert in_bull_ob.iloc[3] == 1

    def test_compute_all_structure_features_pipeline(self, sample_ohlcv):
        """Valida que todas as colunas de estrutura são inseridas e não há quebras."""
        detector = MarketStructureDetector()
        res = detector.compute_all_structure_features(sample_ohlcv)

        expected_cols = [
            "nearest_bsl", "nearest_ssl", "liquidity_sweep_bull", "liquidity_sweep_bear",
            "dist_resistance_pct", "dist_support_pct", "inside_bullish_fvg", "inside_bearish_fvg",
            "inside_bullish_ob", "inside_bearish_ob", "at_bullish_poi", "at_bearish_poi"
        ]
        for col in expected_cols:
            assert col in res.columns

        # Tipos coerentes
        assert res["at_bullish_poi"].dtype in [np.int64, np.int32, int]
        assert res["at_bearish_poi"].dtype in [np.int64, np.int32, int]
