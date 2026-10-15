"""
tests/parity/test_backtest_vs_paper_parity.py

Testes de PARIDADE backtest x piloto ("laranjas com laranjas").

Política
--------
* Cada teste afirma o estado DESEJADO (paridade). Divergências conhecidas estão marcadas `xfail` com motivo,
  `strict=False` (quando o outro agente / a correção chegar, o teste vira XPASS e o marcador deve ser removido).
* Divergências aceitas POR DESENHO NÃO são xfail: têm teste verde que documenta a aceitação
  (`scripts.parity_audit.ACCEPTED_BY_DESIGN`).
* Carry/funding-carry: STAND-BY por decisão do usuário -> fora de escopo (não há teste/xfail).
* Somente leitura do SQLite de mercado (mode=ro), DB temporário para o PaperTradingEngine, zero rede.
"""

from __future__ import annotations

import inspect
import os
import re
import sqlite3

import pytest

from scripts import parity_audit as pa

ROOT = pa.ROOT
pytestmark = pytest.mark.skipif(not os.path.exists(pa.DB_PATH), reason="cache SQLite de mercado ausente")


def _src(path: str) -> str:
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return f.read()


def _pilot_parity_backtest(feat, signals, test_start, **engine_kw):
    """BacktestEngine(pilot_parity=True) adaptado ao TradeView do harness (sig_idx = barra do sinal = entry_idx - 1:
    entrada em close(t) == open(t+1), registrada em ts(t+1), como no piloto). O harness em si não é alterado."""
    from backtesting import BacktestEngine
    df = feat.iloc[test_start:].reset_index(drop=True)
    res = BacktestEngine(initial_capital=pa.BT_PARAMS["initial_capital"], pilot_parity=True, risk_per_trade=None,
                         **engine_kw).run(df, signals[test_start:], atr_col="atr_14")
    return [pa.TradeView(
        sig_idx=test_start + t.entry_idx - 1, direction=t.direction, entry_ts=int(t.entry_timestamp),
        entry_price=t.entry_price, exit_ts=int(t.exit_timestamp), exit_price=t.exit_price,
        notional=t.entry_price * t.size, fees=t.fee_total, funding=t.funding_total, pnl_net=t.pnl_net,
        reasons=[t.exit_reason]) for t in res.trades]


def _parity_report(monkeypatch_target, **engine_kw):
    """Compara com o RollbackGuard do DIRECTIONAL desligado: ele usa RELÓGIO DE PAREDE (cooldown de 2h após perdas
    consecutivas) e portanto bloquearia entradas de um replay histórico acelerado; não é função das velas
    (não modelado no BacktestEngine por design)."""
    from unittest import mock
    from smart_trade import PaperTradingEngine
    orig = pa.run_backtest
    pa.run_backtest = lambda f, sg, ts: _pilot_parity_backtest(f, sg, ts, **engine_kw)
    try:
        with mock.patch.object(PaperTradingEngine, "can_execute_directional", lambda self, *a, **k: (True, "OK")):
            return pa.run_parity(symbols=["BTCUSDT"], bars=500, scenario="forced")
    finally:
        pa.run_backtest = orig


@pytest.fixture(scope="module")
def report():
    """BacktestEngine(pilot_parity=True) com defaults do piloto: timeout do perfil (48 barras) e funding UTC 8h
    (taxa de fallback 0.01%, pois o DB temporário do harness não tem `funding_rates`)."""
    return _parity_report(None, funding_rate_8h=0.0001)


def _count(rep, cat: str) -> int:
    return rep.counts().get(cat, 0)


# ----------------------------------------------------------------------------------------------
# Sanidade do próprio harness (devem passar sempre)
# ----------------------------------------------------------------------------------------------
def test_harness_reads_market_db_read_only():
    with pa._ro_conn() as c:
        with pytest.raises(sqlite3.OperationalError):
            c.execute("CREATE TABLE _parity_probe (x INTEGER)")


def test_harness_produces_comparable_trades(report):
    m = report.per_symbol["BTCUSDT"]
    assert m["n_signals"] > 10 and m["n_bt_trades"] > 10 and m["n_pp_trades"] > 10


def test_every_divergence_is_classified(report):
    assert set(report.counts()) <= set(pa.CATEGORIES)


def test_sizing_divergence_is_accepted_by_design(report):
    """Lote fixo em margem (piloto) vs 1% de risco (backtest): aceito; compara-se retorno % do notional."""
    assert "SIZING" in pa.ACCEPTED_BY_DESIGN
    assert "SIZING" not in report.relevant_counts()


# ----------------------------------------------------------------------------------------------
# Paridade de execução (mesmas velas, mesmos sinais)
# ----------------------------------------------------------------------------------------------
def test_entry_timing_parity(report):
    assert _count(report, "ENTRY_TIMING") == 0


def test_entry_price_parity(report):
    assert _count(report, "ENTRY_PRICE") == 0


def test_exit_reason_parity(report):
    assert _count(report, "EXIT_REASON") == 0


def test_exit_timing_parity(report):
    assert _count(report, "EXIT_TIMING") == 0


def test_fee_model_parity(report):
    assert _count(report, "EXIT_FEE_MODEL") == 0


def test_funding_applied_to_directional_parity(report):
    """Funding UTC 8h existe nos DOIS lados. O TradeView do harness registra funding=0 do lado piloto (categoria FUNDING é
    artefato dessa limitação), então a prova é o PnL líquido (que inclui o funding do ledger do piloto) bater."""
    assert _count(report, "FUNDING") > 0           # o backtest de fato aplica funding (teste não vacuoso)
    assert _count(report, "PNL_PCT") == 0


def test_net_return_pct_parity(report):
    assert _count(report, "PNL_PCT") == 0


def test_no_unmatched_trades(report):
    assert _count(report, "UNMATCHED_BT") + _count(report, "UNMATCHED_PAPER") == 0


# ----------------------------------------------------------------------------------------------
# Paridade estrutural (funcionalidades que existem/não existem em cada lado)
# ----------------------------------------------------------------------------------------------
def _paper_engine(tmp_path):
    from smart_trade import PaperTradingEngine
    return PaperTradingEngine(initial_capital=10_000.0, db_path=str(tmp_path / "p.db"), restore=False)


@pytest.mark.xfail(reason="Backtest/A-B/phase4 NUNCA passam volume_sma/atr_median; piloto idem: vetos VOLATILITY_SPIKE "
                          "e ILLIQUIDITY do RiskSentinel são inertes em AMBOS (docs/AB_CONSENSO_RESULTADOS.md). "
                          "Paridade existe, mas a FUNCIONALIDADE não: fonte de verdade = implementar no contexto de ambos.",
                   strict=False)
def test_pilot_context_feeds_sentinel_baselines():
    from data.feature_engineering import FeatureEngineer
    import pandas as pd
    df = pa.load_closed_candles("BTCUSDT", 900)
    feat = pa.build_features(df)
    ctx = feat.iloc[-1].to_dict()   # exatamente o que evaluate_swing_opportunities passa
    assert ctx.get("volume_sma") and ctx.get("atr_median")


@pytest.mark.xfail(reason="Sentiment/Macro/OnChain recebem contexto vazio no piloto (sem fear_and_greed/dxy/vix/stablecoin) "
                          "-> sempre HOLD 'sem dados'; backtest phase4 usa F&G/DXY/VIX/stablecoins reais (D-1).",
                   strict=False)
def test_pilot_context_feeds_macro_sentiment_onchain():
    feat = pa.build_features(pa.load_closed_candles("BTCUSDT", 900))
    ctx = feat.iloc[-1].to_dict()
    assert all(ctx.get(k) is not None for k in ("fear_and_greed", "dxy_ret_5d", "vix_value", "stablecoin_growth_pct_30d"))


@pytest.mark.xfail(reason="TechnicalAgent do piloto roda sem modelo (core.py: TechnicalAgent()); phase4 usa probas LGBM "
                          "walk-forward (proba_buy/proba_sell) com confidence_threshold=0.55. models/*.joblib nunca são carregados.",
                   strict=False)
def test_pilot_technical_agent_uses_ml_model():
    from core import TradingCore
    assert TradingCore().technical_agent.model is not None


def test_pilot_and_ab_use_same_judge_weights_and_threshold():
    """A/B e piloto: ConsensusJudge(0.30) com pesos default (0.8/0.6/0.6/0.5). (phase4 usa pesos 1.0: ver matriz.)"""
    from core import TradingCore
    j = TradingCore().consensus_judge
    assert j.min_consensus_threshold == pytest.approx(0.30)
    assert j.default_weights["OrderFlowAgent"] == 0.8 and j.default_weights["SentimentAgent"] == 0.6


@pytest.mark.xfail(reason="phase4 (run_phase4_evaluation) usa pesos 1.0 uniformes por passo de ablação; A/B e piloto usam "
                          "pesos default (0.8/0.6/0.6/0.5). Resultados de phase4 não são do comitê que o piloto roda.",
                   strict=False)
def test_phase4_weights_equal_pilot_weights():
    src = _src("scripts/run_phase4_evaluation.py")
    assert "step_weights = {name: 1.0" not in src


@pytest.mark.xfail(reason="phase4 não aplica gate confidence>=0.70 (A/B e piloto sim); atr_median=atr_14 (veto inerte) "
                          "e sentinel max_atr_volatility_multiplier=2.5 explícito.", strict=False)
def test_phase4_applies_pilot_confidence_gate():
    src = _src("scripts/run_phase4_evaluation.py")
    assert re.search(r"confidence\s*>=\s*0\.70|CONF_GATE", src)


def test_directional_entry_respects_circuit_breaker(tmp_path):
    from signal_generator import TradeSignal
    eng = _paper_engine(tmp_path)
    eng.rollback_guard.trigger_emergency_stop("parity-test")
    sig = TradeSignal(signal_id="x", timestamp=1, symbol="BTCUSDT", direction="BUY", entry_price=100.0,
                      stop_loss=99.0, take_profit=102.0, atr=1.0, confidence=0.8, consensus_score=0.5,
                      position_size=100.0, risk_dollars=1.0, rationale="parity")
    assert eng.execute_directional_signal(sig, current_price=100.0, timestamp=1_700_000_000_000) is None


def test_backtest_engine_has_no_circuit_breaker_documented():
    """Documenta (verde): BacktestEngine não modela RollbackGuard/circuit breaker (relógio de parede + DB; o DIRECTIONAL do
    piloto também o ignora). Cooldown/cotas existem como opções (cooldown_bars/daily_quota) no modo pilot_parity."""
    from backtesting import BacktestEngine
    params = set(inspect.signature(BacktestEngine.__init__).parameters)
    assert not any(k in params for k in ("rollback_guard", "max_daily_trades", "circuit_breaker"))


def test_backtest_engine_supports_exit_ladder():
    """A ladder é do perfil (STRATEGY_PROFILES, importado) e vale em pilot_parity=True."""
    from backtesting import BacktestEngine
    params = set(inspect.signature(BacktestEngine.__init__).parameters)
    assert {"pilot_parity", "strategy_type"} <= params
    assert "STRATEGY_PROFILES" in _src("backtesting.py")


def test_backtest_engine_has_maker_taker_by_exit_type():
    from backtesting import BacktestEngine
    params = set(inspect.signature(BacktestEngine.__init__).parameters)
    assert "maker_fee_rate" in params or "maker_fee" in params


def test_backtest_engine_models_liquidation():
    assert "LIQUIDATION" in _src("backtesting.py")
    from backtesting import BacktestEngine
    assert "pilot_parity" in set(inspect.signature(BacktestEngine.__init__).parameters)


def test_paper_engine_has_position_timeout():
    s = _src("smart_trade.py")
    assert "HORIZON_TIMEOUT" in s or "max_holding" in s


def test_pilot_swing_uses_closed_candles_only():
    s = _src("scripts/run_paper_simulation.py")
    body = s.split("async def evaluate_swing_opportunities", 1)[1].split("async def run_loop", 1)[0]
    assert "_drop_forming_candle(df, SWING_BAR_MS" in body
    import pandas as pd
    from scripts.run_paper_simulation import PaperTradingDaemon, SWING_BAR_MS
    df = pd.DataFrame({"timestamp": [0, SWING_BAR_MS, 2 * SWING_BAR_MS]})
    assert len(PaperTradingDaemon._drop_forming_candle(df, SWING_BAR_MS, 2 * SWING_BAR_MS + 1)) == 2   # última em formação
    assert len(PaperTradingDaemon._drop_forming_candle(df, SWING_BAR_MS, 3 * SWING_BAR_MS)) == 3        # já fechada


def test_feature_window_effect_400_vs_full_is_small():
    """Efeito da janela de 400 velas (piloto) vs histórico completo (backtest) nas features de decisão."""
    full = pa.load_closed_candles("BTCUSDT", 3000)
    f_full = pa.build_features(full).iloc[-1]
    f_400 = pa.build_features(full.iloc[-400:].reset_index(drop=True)).iloc[-1]
    for k in ("atr_14", "rsi_14", "sma_200", "dist_vwap_pct"):
        a, b = float(f_full[k]), float(f_400[k])
        assert abs(a - b) <= 0.05 * max(abs(a), 1e-9) + 1e-6, f"{k}: full={a} vs 400={b}"


@pytest.mark.xfail(reason="Ladder validada em eval_staged_tp_trailing_swing (TP1 25% @1.5 ATR, TP2 25% @3.0 ATR, trailing 2.5 ATR, "
                          "4h) != ladder do piloto (TP1 40% / TP2 30% / runner 30%; mults 1.5/2.5/4.0 trail 1.2 no DIRECTIONAL).",
                   strict=False)
def test_validated_ladder_parameters_equal_pilot_ladder():
    import smart_trade as st
    src = _src("scripts/eval_staged_tp_trailing_swing.py")
    assert f"fecha {int(st.EXIT_TP1_FRACTION * 100)}%" in src


@pytest.mark.xfail(reason="Gatilho do scalp no piloto (simulation_settings: VWAP>=2.0%, RSI 35/65, volZ>=0.2 + comitê E7 conf>=0.70) "
                          "!= eval_surgical_scalp (volZ>=1.8, VWAP>=1.2%, RSI 32/68, sem comitê). Nem o gatilho nem o comitê-em-15m foram backtestados.",
                   strict=False)
def test_scalp_trigger_equals_validated_scalp_eval():
    ev = _src("scripts/eval_surgical_scalp.py")
    pilot = _src("scripts/run_paper_simulation.py")
    assert "vol_z >= 1.8" not in ev or "vol_z_min" not in pilot
