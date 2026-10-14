"""
tests/test_statistical_rigor.py - Testes unitários para o módulo statistical_rigor.py
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from statistical_rigor import (
    calculate_per_period_sharpe,
    calculate_annualized_sharpe,
    compute_expected_max_sharpe,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    bootstrap_performance_metrics,
    initialize_trial_log,
    log_trial,
    get_total_trials_count,
    TRIAL_LOG_PATH,
)


def test_sharpe_calculations():
    # Retornos com média 0.001 e std ~0.01
    np.random.seed(42)
    rets = np.random.normal(loc=0.001, scale=0.01, size=365)
    
    sr_period = calculate_per_period_sharpe(rets)
    assert 0.05 < sr_period < 0.15

    sr_ann = calculate_annualized_sharpe(rets, periods_per_year=365)
    assert sr_ann == pytest.approx(sr_period * np.sqrt(365), rel=1e-3)

    # Série degenerada ou vazia
    assert calculate_per_period_sharpe(np.array([])) == 0.0
    assert calculate_per_period_sharpe(np.array([0.01])) == 0.0
    assert calculate_per_period_sharpe(np.array([0.01, 0.01, 0.01])) == 0.0


def test_expected_max_sharpe_properties():
    # Com 1 tentativa, o valor esperado é a média
    assert compute_expected_max_sharpe(n_trials=1, variance_of_sharpes=0.25, mean_sharpe=0.5) == 0.5

    # Com mais tentativas, o valor esperado de max(SR) deve crescer monotonicamente
    e_max_10 = compute_expected_max_sharpe(n_trials=10, variance_of_sharpes=0.25, mean_sharpe=0.0)
    e_max_50 = compute_expected_max_sharpe(n_trials=50, variance_of_sharpes=0.25, mean_sharpe=0.0)
    e_max_200 = compute_expected_max_sharpe(n_trials=200, variance_of_sharpes=0.25, mean_sharpe=0.0)

    assert 0.0 < e_max_10 < e_max_50 < e_max_200


def test_deflated_sharpe_ratio():
    np.random.seed(42)
    # Estratégia genuinamente boa com Sharpe ~2.0
    rets_good = np.random.normal(loc=0.002, scale=0.015, size=730)
    sr_ann = calculate_annualized_sharpe(rets_good, periods_per_year=365)

    # 1 tentativa, variância zero -> DSR alto
    res_single = deflated_sharpe_ratio(
        observed_sr_ann=sr_ann,
        returns=rets_good,
        n_trials=1,
        variance_of_trials=0.0,
        periods_per_year=365,
    )
    assert res_single["dsr"] > 0.99
    assert res_single["passed_95pct"] is True

    # Com 500 tentativas e alta dispersão de Sharpes testados, DSR deve ser severamente deflacionado
    res_multiple = deflated_sharpe_ratio(
        observed_sr_ann=sr_ann,
        returns=rets_good,
        n_trials=500,
        variance_of_trials=1.0,
        periods_per_year=365,
    )
    assert res_multiple["dsr"] < res_single["dsr"]
    assert res_multiple["expected_max_sr"] > 2.0


def test_probability_of_backtest_overfitting_cscv():
    np.random.seed(42)
    t_obs = 500
    n_strats = 10

    # Cenário 1: Ruído puro (10 estratégias aleatórias com retornos i.i.d. de média 0)
    noise_matrix = np.random.normal(loc=0.0, scale=0.01, size=(t_obs, n_strats))
    pbo_noise = probability_of_backtest_overfitting(noise_matrix, n_splits=6, periods_per_year=365)

    assert 0.0 <= pbo_noise["pbo"] <= 1.0
    assert pbo_noise["n_combinations"] == 20  # comb(6, 3) = 20
    # Em ruído puro, PBO tende a ficar próximo ou acima de 0.3
    assert pbo_noise["mean_oos_rank"] < 0.8

    # Cenário 2: Uma estratégia é consistentemente muito superior (coluna 0 com drift positivo)
    signal_matrix = np.random.normal(loc=0.0, scale=0.01, size=(t_obs, n_strats))
    signal_matrix[:, 0] += 0.005  # sinal forte
    pbo_signal = probability_of_backtest_overfitting(signal_matrix, n_splits=6, periods_per_year=365)

    # Quando há alfa real e estável, o modelo ótimo IS mantém o topo no OOS (PBO baixo)
    assert pbo_signal["pbo"] < 0.25
    assert pbo_signal["mean_oos_rank"] > 0.70


def test_bootstrap_performance_metrics():
    np.random.seed(42)
    rets = np.random.normal(loc=0.001, scale=0.01, size=200)
    ci = bootstrap_performance_metrics(rets, n_iterations=300, periods_per_year=365)

    assert "sharpe_ci" in ci
    assert "ann_ret_ci" in ci
    assert "max_dd_ci" in ci

    assert ci["sharpe_ci"][0] <= ci["sharpe_ci"][1]
    assert ci["ann_ret_ci"][0] <= ci["ann_ret_ci"][1]
    assert ci["max_dd_ci"][0] <= ci["max_dd_ci"][1]


def test_trial_log_system(tmp_path, monkeypatch):
    test_csv = tmp_path / "trial_log.csv"
    monkeypatch.setattr("statistical_rigor.TRIAL_LOG_PATH", str(test_csv))

    initial_count = initialize_trial_log()
    assert initial_count == 56  # 16 Fase 3 + 40 Fase 4
    assert os.path.exists(test_csv)

    # Log new trial
    tid = log_trial(
        phase="Fase R",
        experiment="E1_TSMOM",
        strategy_name="TSMOM_BTC_Test",
        symbol="BTCUSDT",
        timeframe="1d",
        params={"lookbacks": [20, 50, 100, 200]},
        sample_period="2020-2025",
        is_oos=True,
        trades_count=120,
        net_pnl=150.0,
        net_pnl_pct=15.0,
        sharpe_ratio=0.85,
        max_drawdown_pct=-18.0,
        win_rate_pct=52.0,
        status_label="TESTE",
        notes="Teste unitário",
    )

    assert tid == f"TRIAL_{initial_count + 1:04d}"
    assert get_total_trials_count() == initial_count + 1

    df = pd.read_csv(test_csv)
    assert len(df) == initial_count + 1
    assert df.iloc[-1]["trial_id"] == tid
    assert df.iloc[-1]["experiment"] == "E1_TSMOM"
