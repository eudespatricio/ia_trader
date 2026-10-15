import numpy as np
import pandas as pd

from data.pattern_features import PATTERN_COLUMNS, _confirmed_pivots, _divergence, compute_pattern_features


def _walk(n=600, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1, n))
    o = c + rng.normal(0, 0.3, n)
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.5, n))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.5, n))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c})


def test_columns():
    out = compute_pattern_features(_walk())
    assert list(out.columns) == PATTERN_COLUMNS


def test_causality_truncation_and_future_perturbation():
    df = _walk()
    full = compute_pattern_features(df)
    for T in (150, 300, 450):
        part = compute_pattern_features(df.iloc[:T + 1].copy())
        np.testing.assert_allclose(part.iloc[:T + 1].to_numpy(), full.iloc[:T + 1].to_numpy(), equal_nan=True)
    d2 = df.copy()
    d2.loc[301:, ["open", "high", "low", "close"]] *= 1.37
    alt = compute_pattern_features(d2)
    np.testing.assert_allclose(alt.iloc[:301].to_numpy(), full.iloc[:301].to_numpy(), equal_nan=True)


def _bars(rows):
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def test_pin_bars():
    df = _bars([(10, 11, 9, 10)] * 3 + [(10.0, 10.1, 8.0, 10.1),      # pavio inferior 2.0 >= 2x corpo 0.1, fecha no topo
                                        (10.0, 12.0, 9.9, 9.9)])      # pavio superior, fecha embaixo
    o = compute_pattern_features(df)
    assert o["pat_pin_bull"].iloc[3] == 1 and o["pat_pin_bear"].iloc[3] == 0
    assert o["pat_pin_bear"].iloc[4] == 1 and o["pat_pin_bull"].iloc[4] == 0
    assert o["pat_pin_bull"].iloc[:3].sum() == 0


def test_engulf_and_inside():
    df = _bars([(10, 10.6, 9.4, 9.6),      # vermelha
                (9.5, 10.9, 9.4, 10.8),    # verde engolfa corpo anterior (o<=c1, c>=o1)
                (10.7, 10.8, 10.0, 10.1),  # inside? high 10.8<10.9 e low 10.0>9.4
                (10.2, 10.3, 9.5, 9.6)])
    o = compute_pattern_features(df)
    assert o["pat_engulf_bull"].iloc[1] == 1
    assert o["pat_inside"].iloc[2] == 1 and o["pat_inside"].iloc[1] == 0


def test_pivot_confirmation_delay():
    low = np.array([5, 4, 3, 2, 1, 2, 3, 4, 5, 6], float)   # pivot em i=4
    m = _confirmed_pivots(low, 3, "low")
    assert list(np.flatnonzero(m)) == [7]                    # confirmado só em 4+3


def test_divergence_logic():
    n = 3
    price = np.zeros(40)
    price[10], price[25] = 100.0, 95.0                       # preço faz fundo mais baixo
    rsi = np.full(40, 50.0)
    rsi[10], rsi[25] = 25.0, 35.0                            # RSI faz fundo mais alto
    piv = np.zeros(40, bool)
    piv[10 + n] = piv[25 + n] = True
    d = _divergence(price, rsi, piv, n, 30, True)
    assert list(np.flatnonzero(d)) == [28]
    assert _divergence(price, rsi, piv, n, 10, True).sum() == 0   # janela > 10 barras
