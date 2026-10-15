"""Testes do experimento A/B de consenso (scripts/ab_consensus_experiment.py). Produção intocada."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.macro_agent import MacroAgent  # noqa: E402
from agents.onchain_agent import OnChainAgent  # noqa: E402
from agents.orderflow_agent import OrderFlowAgent  # noqa: E402
from agents.risk_sentinel_agent import RiskSentinelAgent  # noqa: E402
from agents.sentiment_agent import SentimentAgent  # noqa: E402
from agents.technical_agent import TechnicalAgent  # noqa: E402
from consensus_judge import ConsensusJudge  # noqa: E402
from agents.base_agent import AgentSignal  # noqa: E402
from scripts import ab_consensus_experiment as ab  # noqa: E402

H = 3_600_000


def _tech_ctx(rsi, bull=True):
    return {"close": 110.0 if bull else 90.0, "sma_200": 100.0, "dist_sma_200": 0.1 if bull else -0.1,
            "rsi_14": rsi, "dist_vwap_pct": 1.0 if bull else -1.0}


def test_p5_caps_overbought_momentum_buy_and_symmetric_sell():
    base, p5 = TechnicalAgent(), ab.TechnicalAgentP5()
    assert base.evaluate(_tech_ctx(75.0)).direction == "BUY"
    assert p5.evaluate(_tech_ctx(75.0)).direction == "HOLD"
    assert base.evaluate(_tech_ctx(25.0, bull=False)).direction == "SELL"
    assert p5.evaluate(_tech_ctx(25.0, bull=False)).direction == "HOLD"
    # dentro da faixa: idêntico ao original
    for rsi, bull in [(55.0, True), (60.0, True), (45.0, False), (40.0, False)]:
        a, b = base.evaluate(_tech_ctx(rsi, bull)), p5.evaluate(_tech_ctx(rsi, bull))
        assert (a.direction, round(a.confidence, 9)) == (b.direction, round(b.confidence, 9))


def _flow_ctx(close=100.3, volume=100.0, **extra):
    c = {"open": 100.0, "high": 101.0, "low": 99.0, "close": close, "volume": volume, "funding_rate": 0.0,
         "dist_vwap_pct": 0.0, "volume_zscore_20": 0.0, "funding_zscore": 0.0}
    c.update(extra)
    return c


def test_p4_requires_strong_close_or_volume():
    base, p4 = OrderFlowAgent(), ab.OrderFlowAgentP4()
    weak = _flow_ctx(close=100.3)            # close_position = +0.3, sem volume_sma (vol_ratio = 1)
    assert base.evaluate(weak).direction == "BUY"
    assert p4.evaluate(weak).direction == "HOLD"
    strong = _flow_ctx(close=100.6)          # +0.6
    assert p4.evaluate(strong).direction == "BUY"
    vol_conf = _flow_ctx(close=100.3, volume_sma=50.0)   # vol_ratio = 2.0
    assert p4.evaluate(vol_conf).direction == "BUY"


def test_p6_confidence_is_monotonic_at_25_and_75_but_original_is_not():
    base, p6 = SentimentAgent(), ab.SentimentAgentP6()
    lo = base.evaluate({"fear_and_greed": 25.0}).confidence
    hi = base.evaluate({"fear_and_greed": 25.5}).confidence
    assert lo < hi  # defeito original: extremo menos confiante que moderado
    assert p6.evaluate({"fear_and_greed": 25.0}).confidence >= p6.evaluate({"fear_and_greed": 25.5}).confidence
    assert p6.evaluate({"fear_and_greed": 75.0}).confidence >= p6.evaluate({"fear_and_greed": 74.5}).confidence
    # monotônico em toda a faixa de medo
    confs = [p6.evaluate({"fear_and_greed": float(f)}).confidence for f in range(0, 43)]
    assert all(a >= b - 1e-12 for a, b in zip(confs, confs[1:]))
    # sem dados: inalterado
    assert p6.evaluate({}).confidence == base.evaluate({}).confidence


def test_p3_dxy_clamp_and_tvl_counted_once():
    ctx = {"dxy_ret_5d": -0.01, "dxy_above_sma20": False}
    assert MacroAgent().evaluate(ctx).confidence == pytest.approx(0.73)
    assert ab.MacroAgentP3().evaluate(ctx).confidence == pytest.approx(0.68)
    # sem excesso: idêntico
    ctx2 = {"dxy_ret_5d": -0.01, "dxy_above_sma20": True, "vix_value": 15.0}
    assert MacroAgent().evaluate(ctx2).confidence == ab.MacroAgentP3().evaluate(ctx2).confidence
    # TVL proxy: original conta 1.2 (stable proxy) + 0.8 (tvl) = 2.0 -> 0.78 ; P3 -> 1.2 -> 0.70
    oc = {"tvl_growth_pct_7d": 0.02}
    assert OnChainAgent().evaluate(oc).confidence == pytest.approx(0.78)
    assert ab.OnChainAgentP3().evaluate(oc).confidence == pytest.approx(0.70)
    # com stablecoin real, nada muda
    oc2 = {"tvl_growth_pct_7d": 0.02, "stablecoin_growth_pct_30d": 0.01}
    assert OnChainAgent().evaluate(oc2).confidence == ab.OnChainAgentP3().evaluate(oc2).confidence


def _signals(tech_dir, tech_conf, flow_dir="HOLD", flow_conf=0.5):
    s = RiskSentinelAgent().evaluate({})
    t = AgentSignal("TechnicalAgent", tech_dir, tech_conf, 1.0, "t")
    f = AgentSignal("OrderFlowAgent", flow_dir, flow_conf, 1.0, "f")
    return [s, t, f, SentimentAgent().evaluate({}), MacroAgent().evaluate({}), OnChainAgent().evaluate({})]


def test_p1a_excludes_no_data_agents_from_denominator_p1b_lowers_threshold():
    sigs = _signals("BUY", 0.8)
    assert ConsensusJudge(0.30).judge_deterministic(sigs).direction == "HOLD"        # 0.8/4.5 = 0.178
    assert ab.JudgeNoDataExcluded(0.30).judge_deterministic(sigs).direction == "BUY"  # 0.8/1.8 = 0.444
    sigs2 = _signals("BUY", 0.7, "BUY", 0.6)                                          # (0.7+0.48)/4.5 = 0.262
    assert ConsensusJudge(0.30).judge_deterministic(sigs2).direction == "HOLD"
    assert ConsensusJudge(0.22).judge_deterministic(sigs2).direction == "BUY"


def test_p1a_keeps_agents_with_data():
    ctx = {"fear_and_greed": 10.0, "dxy_ret_5d": -0.01, "stablecoin_growth_pct_30d": 0.01}
    assert not ab.JudgeNoDataExcluded.has_no_data(SentimentAgent().evaluate(ctx))
    assert not ab.JudgeNoDataExcluded.has_no_data(MacroAgent().evaluate(ctx))
    assert not ab.JudgeNoDataExcluded.has_no_data(OnChainAgent().evaluate(ctx))
    assert ab.JudgeNoDataExcluded.has_no_data(SentimentAgent().evaluate({}))
    assert ab.JudgeNoDataExcluded.has_no_data(MacroAgent().evaluate({}))
    assert ab.JudgeNoDataExcluded.has_no_data(OnChainAgent().evaluate({}))
    veto = RiskSentinelAgent().evaluate({"atr_14": 10.0, "atr_median": 1.0})
    assert veto.is_veto and not ab.JudgeNoDataExcluded.has_no_data(veto)


def test_p2_mapping_activates_vetoes():
    sentinel = RiskSentinelAgent()
    raw = {"atr_14": 3.0, "volume": 1.0, "vol_sma_20": 100.0}
    assert not sentinel.evaluate(raw).is_veto                      # baseline: chaves ausentes
    mapped = ab.map_p2_context(raw, atr_median=1.0)
    s = sentinel.evaluate(mapped)
    assert s.is_veto and s.metrics["veto_type"] == "VOLATILITY_SPIKE"
    calm = ab.map_p2_context({"atr_14": 1.0, "volume": 1.0, "vol_sma_20": 100.0}, atr_median=1.0)
    s2 = sentinel.evaluate(calm)
    assert s2.is_veto and s2.metrics["veto_type"] == "ILLIQUIDITY"


def _sym(n=40, price=100.0):
    ts = np.arange(n, dtype=np.int64) * H
    o = np.full(n, price)
    return {"symbol": "X", "ts": ts, "open": o.copy(), "high": o + 0.05, "low": o - 0.05, "close": o.copy(),
            "atr": np.full(n, 1.0)}


def _dec(n, at, d=1):
    direction = np.zeros(n, dtype=np.int8)
    direction[at] = d
    return {"direction": direction, "conf": np.full(n, 0.8)}


NO_FUND = (np.array([], dtype=np.int64), np.array([0.0]))


def test_trade_sim_enters_next_open_and_charges_full_costs():
    s = _sym()
    s["open"][6] = 100.0
    tr = ab.simulate_trades(s, _dec(40, 5), NO_FUND, max_hold=10)
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t["entry_ts"] == s["ts"][6]            # entrada na abertura de t+1, nunca em t
    assert t["reason"] == "TIMEOUT"
    # preço plano: perde ~ 2*(slippage + fee) = 0.16%
    assert t["ret_notional"] == pytest.approx(-2 * (ab.SLIPPAGE + ab.TAKER_FEE), abs=1e-4)


def test_trade_sim_stop_wins_tie_and_future_perturbation_does_not_change_past_trade():
    s = _sym()
    # barra de entrada (t+1=6) toca SL e TP ao mesmo tempo -> SL
    s["high"][6], s["low"][6] = 102.0, 98.0
    tr = ab.simulate_trades(s, _dec(40, 5), NO_FUND, max_hold=10)
    assert tr.iloc[0]["reason"] == "SL"
    # perturbar barras APÓS a saída não altera o trade
    s2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    s2["high"][30:] *= 1.5
    s2["low"][30:] *= 0.5
    tr2 = ab.simulate_trades(s2, _dec(40, 5), NO_FUND, max_hold=10)
    assert tr.iloc[0]["ret_notional"] == tr2.iloc[0]["ret_notional"]


def test_funding_sign_long_pays_short_receives():
    s = _sym()
    fund = (np.array([s["ts"][8]], dtype=np.int64), np.array([0.0, 0.001]))
    long_t = ab.simulate_trades(s, _dec(40, 5, 1), fund, max_hold=10).iloc[0]
    short_t = ab.simulate_trades(s, _dec(40, 5, -1), fund, max_hold=10).iloc[0]
    assert long_t["ret_notional"] < short_t["ret_notional"]
    assert (short_t["ret_notional"] - long_t["ret_notional"]) == pytest.approx(0.002, abs=2e-4)


def test_daily_context_has_no_lookahead():
    daily = pd.DataFrame({
        "fear_and_greed": [10.0, 90.0],
        "avail_ms": [10 * ab.DAY_MS + ab.DAY_MS, 11 * ab.DAY_MS + ab.DAY_MS],  # dado do dia D só no fim de D+1... (D+24h)
    })
    # barra do dia 10, 12:00 (fecha 13:00): só enxerga dado ainda não disponível -> NaN
    ts = np.array([10 * ab.DAY_MS + 12 * H, 11 * ab.DAY_MS + 12 * H, 12 * ab.DAY_MS + 12 * H], dtype=np.int64)
    out = ab.align_daily_context(ts, daily)
    assert np.isnan(out["fear_and_greed"].iloc[0])
    assert out["fear_and_greed"].iloc[1] == 10.0
    assert out["fear_and_greed"].iloc[2] == 90.0


def test_block_bootstrap_zero_diff_is_not_significant():
    r = ab.block_bootstrap_diff(np.zeros(100))
    assert r["p_le0"] == 1.0 and r["mean"] == 0.0


def test_block_bootstrap_detects_real_shift():
    rng = np.random.default_rng(0)
    d = rng.normal(0.01, 0.01, 400)
    r = ab.block_bootstrap_diff(d)
    assert r["lo"] > 0 and r["p_le0"] < 0.01


def test_decisions_are_causal_on_synthetic_candles():
    """Truncar o futuro não pode alterar decisões passadas (features + agentes + juiz)."""
    rng = np.random.default_rng(3)
    n = 800
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.004, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.004, n))
    df = pd.DataFrame({"timestamp": np.arange(n, dtype=np.int64) * H + 1_700_000_000_000 // H * H, "open": open_,
                       "high": high, "low": low, "close": close, "volume": rng.uniform(100, 1000, n)})

    def run(d):
        feat = ab.compute_features(d)
        feat = feat[(feat["sma_200"] > 0) & np.isfinite(feat["sma_200"])].reset_index(drop=True)
        med = feat["atr_14"].rolling(ab.ATR_MEDIAN_WINDOW, min_periods=ab.ATR_MEDIAN_WINDOW).median().to_numpy()
        eng = ab.DecisionEngine(feat[ab.CTX_COLS].to_dict("records"), med, "PROD")
        v = ab.Variant("T", p2=True, p4=True, p5=True, p1a=True)
        return feat["timestamp"].to_numpy(), eng.decisions(v)

    ts_full, d_full = run(df)
    ts_cut, d_cut = run(df.iloc[:600])
    m = len(ts_cut) - 5   # ignora as últimas barras do corte (bordas de janela do detector de estrutura)
    assert np.array_equal(ts_full[:m], ts_cut[:m])
    assert np.array_equal(d_full["direction"][:m], d_cut["direction"][:m])
    assert np.allclose(d_full["conf"][:m], d_cut["conf"][:m])


def test_build_variants_has_unique_names_and_two_baselines():
    vs = ab.build_variants()
    names = [v.name for v in vs]
    assert len(names) == len(set(names))
    assert sum(v.is_baseline() for v in vs) == 2
