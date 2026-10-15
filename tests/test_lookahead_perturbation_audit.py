"""
Auditoria de Lookahead Bias via testes de perturbação (golden_rule_guardian).
Princípio: alterar dados FUTUROS (> t) não pode alterar nenhuma saída em t.
Não modifica código de produção.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.feature_engineering import FeatureEngineer  # noqa: E402
from model_trainer import (  # noqa: E402
    ModelTrainer, PurgedTimeSeriesSplit, create_triple_barrier_labels,
)
from backtesting import BacktestEngine  # noqa: E402

H = 3_600_000
T0 = 1_700_000_000_000 - (1_700_000_000_000 % H)


def make_candles(n=700, seed=1):
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.003, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.003, n))
    vol = rng.uniform(100, 1000, n)
    return pd.DataFrame({"timestamp": T0 + np.arange(n) * H, "open": open_, "high": high,
                         "low": low, "close": close, "volume": vol})


def perturb_future(df, t, seed=99):
    out = df.copy()
    rng = np.random.default_rng(seed)
    m = len(out) - (t + 1)
    c = 30000 * np.exp(np.cumsum(rng.normal(0, 0.02, m)))
    out.loc[t + 1:, "close"] = c
    out.loc[t + 1:, "open"] = c * 0.99
    out.loc[t + 1:, "high"] = c * 1.05
    out.loc[t + 1:, "low"] = c * 0.95
    out.loc[t + 1:, "volume"] = rng.uniform(1, 1e5, m)
    return out


def feats(df, **kw):
    return FeatureEngineer.compute_all_features(df, drop_na=False, **kw)


def _cmp(a, b, t, label):
    cols = [c for c in a.columns if c != "timestamp" and c in b.columns]
    diff = []
    for c in cols:
        x, y = a[c].iloc[:t + 1].values.astype(float), b[c].iloc[:t + 1].values.astype(float)
        if not np.allclose(x, y, rtol=1e-9, atol=1e-9, equal_nan=True):
            diff.append(c)
    assert not diff, f"[{label}] lookahead: colunas alteradas por dados futuros: {diff}"


def test_features_causal_no_macro():
    df = make_candles()
    t = 400
    _cmp(feats(df), feats(perturb_future(df, t)), t, "tecnicos")


def test_features_prefix_equals_full():
    df = make_candles()
    t = 400
    full = feats(df)
    pref = feats(df.iloc[:t + 1].copy())
    _cmp(full, pref, t, "prefixo")


def test_macro_release_alignment_and_future_perturbation():
    df = make_candles()
    t = 400
    ts_t = df["timestamp"].iloc[t]
    rel = pd.to_datetime(df["timestamp"].iloc[[50, 300, 500]], unit="ms", utc=True)
    macro = pd.DataFrame({"release_date": rel.values, "cpi": [300.0, 310.0, 999.0],
                          "vix": [15.0, 20.0, 80.0]})
    a = feats(df, macro_df=macro)
    # release em idx 500 (> t) não pode aparecer em t
    assert a["macro_cpi"].iloc[t] == 310.0
    assert a["macro_cpi"].iloc[:t + 1].max() < 999.0
    macro2 = macro.copy()
    macro2.loc[2, ["cpi", "vix"]] = [1.0, 1.0]
    b = feats(df, macro_df=macro2)
    _cmp(a, b, t, "macro-futuro")


def test_macro_leading_backfill_leak():
    """Candles ANTES da 1a release não podem receber o valor da 1a release (futuro)."""
    df = make_candles()
    rel = pd.to_datetime(df["timestamp"].iloc[[300]], unit="ms", utc=True)
    macro = pd.DataFrame({"release_date": rel.values, "cpi": [777.0], "vix": [33.0]})
    a = feats(df, macro_df=macro)
    assert a["macro_cpi"].iloc[0] != 777.0, "bfill propaga CPI futuro (idx 300) para idx 0"


def test_funding_leading_backfill_leak():
    df = make_candles()
    fdf = pd.DataFrame({"timestamp": df["timestamp"].iloc[[300, 308]].values,
                        "funding_rate": [0.05, 0.05]})
    a = feats(df, funding_df=fdf)
    assert a["funding_rate"].iloc[0] != 0.05, "bfill propaga funding futuro para o passado"


def test_clean_features_short_series_bfill_leak():
    df = make_candles(150)
    t = 10
    a = FeatureEngineer.compute_all_features(df, drop_na=True)
    b = FeatureEngineer.compute_all_features(perturb_future(df, t), drop_na=True)
    # linhas iniciais (aquecimento) preenchidas por bfill dependem do futuro
    assert np.isclose(a["sma_20"].iloc[0], b["sma_20"].iloc[0]), "bfill de warmup (len<=200) usa futuro"


def test_macro_dict_snapshot_is_broadcast_over_history():
    df = make_candles()
    a = feats(df, macro_df={"cpi": 400.0, "vix": 50.0, "dxy": 90.0, "sp500": 1.0, "fedfunds": 1.0})
    # V4 corrigido: snapshot atual só vale para a última barra; histórico neutro
    assert a["macro_cpi"].iloc[-1] == 400.0
    assert (a["macro_cpi"].iloc[:-1] != 400.0).all()


def test_triple_barrier_label_uses_only_window():
    df = make_candles(300)
    df["atr_14"] = df["close"] * 0.01
    h = 5
    base = create_triple_barrier_labels(df, horizon=h)
    for i in (50, 120, 200):
        p = perturb_future(df, i + h)
        p["atr_14"] = p["close"] * 0.01
        lab = create_triple_barrier_labels(p, horizon=h)
        assert lab.iloc[i] == base.iloc[i]
    assert base.iloc[-h:].isna().all()


def test_purged_split_embargo():
    X = pd.DataFrame({"a": np.arange(1000)})
    for h in (3, 5, 12):
        for tr, te in PurgedTimeSeriesSplit(4, h).split(X):
            assert te.min() - tr.max() > h - 1 + 0  # gap >= h
            assert te.min() >= tr.max() + h


def test_walk_forward_train_set_independent_of_test_future(tmp_path):
    df = make_candles(900)
    tr = ModelTrainer(model_type="rf", horizon=5, models_dir=str(tmp_path), n_jobs=2)
    X, y = tr.prepare_dataset(FeatureEngineer.compute_all_features(df, drop_na=True))
    tr_idx, te_idx = PurgedTimeSeriesSplit(4, 5).split(X)[0]
    fdf = FeatureEngineer.compute_all_features(df, drop_na=True)
    cut_ts = fdf["timestamp"].iloc[int(te_idx.min())]
    cut_raw = int(np.where(df["timestamp"].values == cut_ts)[0][0])
    df2 = perturb_future(df, cut_raw - 1)
    X2, y2 = tr.prepare_dataset(FeatureEngineer.compute_all_features(df2, drop_na=True))
    n = len(tr_idx)
    pd.testing.assert_frame_equal(X.iloc[:n].reset_index(drop=True), X2.iloc[:n].reset_index(drop=True))
    assert (y.iloc[:n].values == y2.iloc[:n].values).all()


def test_backtest_prefix_invariant_to_future():
    df = make_candles(500)
    df["atr_14"] = df["close"] * 0.01
    rng = np.random.default_rng(3)
    sig = rng.choice([0, 1, 2], size=len(df), p=[.8, .1, .1])
    eng = BacktestEngine(max_holding_bars=5)
    k = 250
    r1 = eng.run(df, sig)
    d2 = perturb_future(df, k)
    d2["atr_14"] = d2["close"] * 0.01
    s2 = sig.copy()
    s2[k + 1:] = rng.choice([0, 1, 2], size=len(s2) - k - 1)
    r2 = eng.run(d2, s2)
    assert r1.equity_curve[:k + 1] == r2.equity_curve[:k + 1]
    t1 = [t for t in r1.trades if t.exit_idx <= k]
    t2 = [t for t in r2.trades if t.exit_idx <= k]
    assert [t.to_dict() for t in t1] == [t.to_dict() for t in t2]


def test_backtest_trailing_stop_intrabar_ordering():
    """Trailing: sobe SL com a máxima da barra e testa mínima da MESMA barra (ordem intra-barra assumida)."""
    df = make_candles(60)
    df["atr_14"] = 100.0
    df.loc[5, ["open", "high", "low", "close"]] = [30000, 30400, 29960, 30300]
    df.loc[4, ["open", "high", "low", "close"]] = [30000, 30010, 29990, 30000]
    sig = np.zeros(60, dtype=int)
    sig[4] = 1
    eng = BacktestEngine(trailing_stop=True, trailing_atr_multiple=1.5, tp_atr_multiple=50, sl_atr_multiple=3.0,
                         slippage=0.0)
    res = eng.run(df, sig)
    tr = res.trades[0]
    # SL inicial = 29700; trailing recalculado com a máxima da PRÓPRIA barra (30400-150=30250) e
    # imediatamente testado contra a mínima da mesma barra => TRAILING_SL na barra de entrada.
    assert tr.exit_idx == 5 and tr.exit_reason == "TRAILING_SL"
    # documenta: o SL de 30250 só existe porque a máxima da barra é conhecida (ordem H->L assumida)
    assert tr.exit_reason == "TRAILING_SL"
