"""
tests/test_trend_following.py - Testes unitários para o motor de Trend Following (trend_following.py).
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trend_following import DailyTSMOMStrategy, evaluate_tsmom_benchmarks


def create_synthetic_candles(n_days: int = 300, trend: float = 0.001) -> pd.DataFrame:
    np.random.seed(42)
    prices = [100.0]
    for _ in range(n_days - 1):
        r = np.random.normal(loc=trend, scale=0.02)
        prices.append(prices[-1] * (1.0 + r))
    
    dates = [f"2023-01-01 + {i}d" for i in range(n_days)]
    return pd.DataFrame({
        "timestamp": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1000.0] * n_days,
    })


def test_tsmom_signal_anti_lookahead():
    """Verifica que active_weight[t] é estritamente baseado no sinal de t-1."""
    df = create_synthetic_candles(n_days=250, trend=0.002)
    strat = DailyTSMOMStrategy(lookbacks=[20, 50], vol_window=20)
    data = strat.compute_signals_and_weights(df)

    # active_weight deve ser igual ao target_weight deslocado em 1
    np.testing.assert_array_almost_equal(
        data["active_weight"].values[1:],
        data["target_weight"].values[:-1]
    )
    assert data["active_weight"].iloc[0] == 0.0


def test_tsmom_no_trade_band_reduces_turnover():
    """Verifica que a banda de no-trade reduz o número de mudanças de posição."""
    df = create_synthetic_candles(n_days=250, trend=0.0005)
    
    # Com banda 0.20
    strat_band = DailyTSMOMStrategy(lookbacks=[20, 50], no_trade_band=0.20)
    res_band = strat_band.run_backtest(df)

    # Sem banda (0.0)
    strat_no_band = DailyTSMOMStrategy(lookbacks=[20, 50], no_trade_band=0.0)
    res_no_band = strat_no_band.run_backtest(df)

    # A banda de no-trade deve ter menor ou igual turnover anual
    assert res_band.turnover_annual <= res_no_band.turnover_annual + 1e-4
    assert res_band.total_fees_paid <= res_no_band.total_fees_paid + 1e-4


def test_tsmom_backtest_costs_deducted():
    """Verifica que taxas e slippage são deduzidas corretamente."""
    df = create_synthetic_candles(n_days=250, trend=0.001)
    strat = DailyTSMOMStrategy(lookbacks=[20, 50], fee_rate=0.0006, slippage=0.0003)
    res = strat.run_backtest(df)

    assert res.total_fees_paid > 0.0
    # Net profit deve ser menor que o gross profit devido a taxas e funding
    assert res.net_profit < res.gross_profit


def test_tsmom_benchmarks_sanity():
    """Verifica cálculo correto de Buy & Hold e SMA 200."""
    df = create_synthetic_candles(n_days=250, trend=0.001)
    bm = evaluate_tsmom_benchmarks(df)

    assert "buy_and_hold" in bm
    assert "sma_200" in bm
    assert "sharpe_ratio" in bm["buy_and_hold"]
    assert "max_drawdown_pct" in bm["buy_and_hold"]
    assert "sharpe_ratio" in bm["sma_200"]
