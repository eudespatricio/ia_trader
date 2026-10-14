"""
tests/test_carry_simulator.py - Testes unitários para o simulador de funding carry (funding_carry.py).
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from funding_carry import FundingCarrySimulator, CarrySimulationResult


def create_synthetic_data(
    n_bars: int = 100,
    base_price: float = 50000.0,
    price_trend: float = 0.0,
    funding_rate: float = 0.0001,  # 0.01% por período
    start_ts: int = 1700000000000,
    interval_ms: int = 8 * 3600 * 1000,
) -> pd.DataFrame:
    timestamps = [start_ts + i * interval_ms for i in range(n_bars)]
    prices = [base_price * (1.0 + price_trend * (i / n_bars)) for i in range(n_bars)]
    rates = [funding_rate] * n_bars
    return pd.DataFrame({
        "timestamp": timestamps,
        "close": prices,
        "funding_rate": rates,
    })


def test_carry_delta_neutrality_sanity():
    """
    Sanity Check: Se o funding rate for ZERO, e o preço dobrar (+100%),
    o resultado bruto deve ser ZERO (delta-neutro perfeito), sofrendo apenas os custos de corretagem das 4 pernas.
    """
    df = create_synthetic_data(n_bars=60, base_price=50000.0, price_trend=1.0, funding_rate=0.0)
    # Força entrada com limiar negativo para garantir posição
    sim = FundingCarrySimulator(
        initial_capital=10000.0,
        spot_fee_rate=0.0006,
        spot_slippage=0.0002,
        perp_fee_rate=0.0006,
        perp_slippage=0.0003,
        entry_funding_threshold_ann=-1.0,  # Entra sempre
        exit_funding_threshold_ann=-2.0,   # Nunca sai antes do fim
    )
    res = sim.run(df)

    assert len(res.trades) == 1
    t = res.trades[0]
    # Funding bruto deve ser zero
    assert t.gross_funding_collected == pytest.approx(0.0, abs=1e-5)
    # PnL líquido deve ser exatamente o custo negativo das taxas de 4 pernas
    assert t.net_pnl < 0.0
    assert t.net_pnl == pytest.approx(-t.fees_paid_4_legs, rel=1e-3)
    # A perda percentual total deve ser em torno do custo de 4 pernas (~0.34% sobre o notional de 5000 = ~17 USD)
    assert 10.0 < t.fees_paid_4_legs < 40.0


def test_carry_positive_funding_accumulation():
    """
    Verifica que funding positivo constante (ex: 0.0002 = 0.02%/8h = ~21.9% a.a.)
    produz lucro líquido positivo superando os custos de rotação das 4 pernas em 100 períodos.
    """
    df = create_synthetic_data(n_bars=100, base_price=50000.0, price_trend=0.0, funding_rate=0.0002)
    sim = FundingCarrySimulator(
        initial_capital=10000.0,
        entry_funding_threshold_ann=0.05,
        exit_funding_threshold_ann=0.01,
    )
    res = sim.run(df)

    assert res.net_profit > 0.0
    assert res.total_funding_gross > res.total_fees_paid
    assert res.annualized_return > 5.0  # Retorno anualizado de dois dígitos
    assert res.max_drawdown < 1.0       # Drawdown mínimo na estratégia delta-neutra sem descasamento


def test_carry_negative_funding_detection():
    """
    Verifica que se o funding for negativo (-0.0001), o simulador registra e não entra
    se o limiar de entrada for positivo.
    """
    df = create_synthetic_data(n_bars=60, base_price=50000.0, price_trend=0.0, funding_rate=-0.0001)
    sim = FundingCarrySimulator(
        initial_capital=10000.0,
        entry_funding_threshold_ann=0.08,
        exit_funding_threshold_ann=0.02,
    )
    res = sim.run(df)

    # Não deve ter executado trades pois funding é negativo (< 8%)
    assert len(res.trades) == 0
    assert res.net_profit == 0.0
    assert res.pct_negative_funding_periods == 100.0


def test_stress_test_shock_30pct():
    """
    Verifica o stress test de salto de preço de +30%:
    Com 50% spot e 50% margem perp, a margem na perna short deve cair para ~53.8%,
    passando no teste de segurança (> 30% buffer).
    """
    sim = FundingCarrySimulator(initial_capital=10000.0, min_safe_margin_pct=0.30)
    stress_res = sim._evaluate_stress_shock_30pct(np.array([50000.0]), np.array([0.0001]))

    assert stress_res["passed"] is True
    # (5000 - 1500) / (5000 * 1.30) = 3500 / 6500 = 0.53846
    assert stress_res["margin_plus_30"] == pytest.approx(3500.0 / 6500.0, abs=1e-4)


def test_anti_lookahead_rolling_shift():
    """
    Verifica que o sinal de funding é estritamente defasado (shift 1).
    Um pico repentino de funding na vela `i` NÃO pode ativar um trade na mesma vela `i`.
    """
    n_bars = 40
    df = create_synthetic_data(n_bars=n_bars, base_price=50000.0, funding_rate=0.00001)
    # Adiciona pico repentino na vela 25
    df.loc[25, "funding_rate"] = 0.05  # pico massivo
    
    sim = FundingCarrySimulator(
        initial_capital=10000.0,
        entry_funding_threshold_ann=0.15,
        exit_funding_threshold_ann=0.05,
    )
    # A vela 25 não deve ter entrada na barra 25
    res = sim.run(df)
    for t in res.trades:
        # A entrada deve ser estritamente posterior à barra do pico (>= barra 26)
        assert t.timestamp_entry >= df.loc[26, "timestamp"]
