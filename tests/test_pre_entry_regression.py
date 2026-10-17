"""
tests/test_pre_entry_regression.py - Teste de regressão para proteção contra candles pré-entrada.
Garante que barras históricas com timestamp anterior a entry_timestamp NUNCA acionem
Stop Loss, Take Profit, Trailing Stop ou liquidação em posições abertas por rompimento.
"""

import pytest
import sqlite3
from smart_trade import PaperTradingEngine, STRATEGY_PROFILES
from signal_generator import TradeSignal


def _make_engine(tmp_path):
    db_file = str(tmp_path / "test_pre_entry.db")
    return PaperTradingEngine(initial_capital=10000.0, db_path=db_file, restore=False)


def _sig(symbol="BTWUSDT", direction="BUY", price=2.0, atr=0.02, ts=1792200000000):
    sl = price - 1.5 * atr if direction == "BUY" else price + 1.5 * atr
    tp = price + 2.0 * atr if direction == "BUY" else price - 2.0 * atr
    return TradeSignal(
        signal_id="sig-test",
        timestamp=ts,
        symbol=symbol,
        direction=direction,
        entry_price=price,
        stop_loss=sl,
        take_profit=tp,
        atr=atr,
        confidence=0.88,
        consensus_score=0.35,
        position_size=100.0,
        risk_dollars=4.0,
        rationale="Regressão pré-entrada",
    )


def test_pre_entry_bars_do_not_trigger_sl_on_breakout_long(tmp_path):
    eng = _make_engine(tmp_path)
    now_ts = 1792200000000  # Tempo da decisão
    
    # Abre posição Golden Pot LONG em 2.00 com SL em 1.97
    pos = eng.execute_golden_pot_signal(_sig("BTWUSDT", "BUY", price=2.0, atr=0.02), current_price=2.0, timestamp=now_ts)
    assert pos is not None
    assert pos.status == "OPEN"
    assert pos.stop_loss < 2.0
    sl = pos.stop_loss
    entry_ts = pos.entry_timestamp

    # Simula 5 barras do passado (antes de entry_ts) onde o preço era mais baixo (ex: 1.90, bem abaixo do SL de 1.97)
    for past_ts in range(entry_ts - 5 * 180000, entry_ts, 180000):
        trig = eng.update_market_tick(
            symbol="BTWUSDT",
            high=1.92,
            low=1.90,
            close=1.91,
            timestamp=past_ts,
            open_price=1.915,
        )
        assert trig == [], f"Barra do passado ({past_ts} < {entry_ts}) não pode gerar saída!"
        assert pos.position_id in eng.positions
        assert eng.positions[pos.position_id].status == "OPEN"
        # Garante que extremos da posição não foram corrompidos por preços pré-entrada
        assert eng.positions[pos.position_id].lowest_price >= 1.99

    # Agora simula uma barra válida pós-entrada que atinge o TP1
    trig_tp = eng.update_market_tick(
        symbol="BTWUSDT",
        high=pos.tp1_price + 0.01,
        low=pos.entry_price + 0.01,
        close=pos.tp1_price + 0.005,
        timestamp=entry_ts,
        open_price=pos.entry_price + 0.008,
    )
    assert len(trig_tp) == 1
    assert trig_tp[0].action == "TP1_PARTIAL"
    assert eng.positions[pos.position_id].tp1_hit is True


def test_pre_entry_bars_do_not_trigger_sl_on_breakout_short(tmp_path):
    eng = _make_engine(tmp_path)
    now_ts = 1792200000000

    # Abre posição Scalp SHORT em 100.0 com SL em 101.5
    pos = eng.execute_scalp_signal(_sig("SOLUSDT", "SELL", price=100.0, atr=1.0), current_price=100.0, timestamp=now_ts)
    assert pos is not None
    assert pos.status == "OPEN"
    entry_ts = pos.entry_timestamp

    # Barras pré-entrada tinham preço mais alto (ex: 105.0, bem acima do SL)
    for past_ts in range(entry_ts - 3 * 60000, entry_ts, 60000):
        trig = eng.update_market_tick(
            symbol="SOLUSDT",
            high=106.0,
            low=104.0,
            close=105.0,
            timestamp=past_ts,
            open_price=104.5,
        )
        assert trig == []
        assert eng.positions[pos.position_id].status == "OPEN"
        assert eng.positions[pos.position_id].highest_price <= 101.0


def test_argon_fix1_flow_gate_dedupes_per_bar_ts_and_blocks_stale_entry(tmp_path):
    from scripts.run_paper_simulation import PaperTradingDaemon
    eng = _make_engine(tmp_path)
    calls = []

    class FlippingTape:
        def evaluate(self, symbol, direction, now_ms=None):
            calls.append(now_ms)
            # 1ª chamada no fechamento da vela bloqueia (HOLD); 2ª chamada 9 min depois aprova (BUY)
            vote = "HOLD" if len(calls) == 1 else direction
            return {"vote": vote, "status": "OK", "reason": "test", "features": {}}

    d = PaperTradingDaemon(engine=eng, loader=object(), golden_pot_agent=object(), tape_agent=FlippingTape())
    sig = _sig("SOLUSDT", "BUY", price=100.0, atr=1.0)
    bar_ts = 1792177200000

    assert d._flow_gate("DIRECTIONAL", "SOLUSDT", "BUY", sig, 3.0, 100.0, 1.0, bar_ts=bar_ts) is False
    # Reavaliação da MESMA barra 15s/9min depois deve usar o cache (False) sem chamar o tape nem permitir entrada stale
    assert d._flow_gate("DIRECTIONAL", "SOLUSDT", "BUY", sig, 3.0, 100.0, 1.0, bar_ts=bar_ts) is False
    assert len(calls) == 1
    assert eng.flow_gate_counts()["BLOCKED"] == 1


def test_argon_fix2_trailing_ratchet_waits_for_native_strategy_bar_bucket(tmp_path):
    eng = _make_engine(tmp_path)
    base_ts = 1792159200000  # divisível por 3600_000 (1h)
    pos = eng.execute_directional_signal(_sig("SOLUSDT", "BUY", price=100.0, atr=1.0, ts=base_ts), current_price=100.0, timestamp=base_ts)
    entry_ts = pos.entry_timestamp  # base_ts + 1h

    # 1ª sub-barra de 3m dentro da hora entry_ts atinge TP1 (101.5) e TP2 (102.5), ativando trailing
    eng.update_market_tick("SOLUSDT", high=102.8, low=100.2, close=102.6, timestamp=entry_ts, open_price=100.2)
    assert pos.tp2_hit and pos.trailing_active
    initial_trail = pos.trailing_stop_price

    # 2ª sub-barra de 3m (entry_ts + 3m, AINDA dentro da mesma janela de 1h!) faz máxima em 103.5
    # Não deve elevar o trailing_stop_price ainda (acumula extremo da hora até virar a próxima hora)
    eng.update_market_tick("SOLUSDT", high=103.5, low=102.4, close=103.0, timestamp=entry_ts + 180_000, open_price=102.6)
    assert pos.trailing_stop_price == pytest.approx(initial_trail)

    # Na abertura da PRÓXIMA hora (entry_ts + 1h), o extremo 103.5 da hora fechada ratcheta o trailing!
    eng.update_market_tick("SOLUSDT", high=103.2, low=102.5, close=103.0, timestamp=entry_ts + 3600_000, open_price=103.0)
    assert pos.trailing_stop_price > initial_trail


def test_argon_fix3_golden_pot_agent_prioritizes_swing_breakout_over_scalp_reversion():
    from agents.golden_pot_agent import GoldenPotAgent
    agent = GoldenPotAgent(min_volume_usd=10_000_000.0)
    # Par com forte rompimento (+8% no dia, colado na máxima 96% do range) E esticado +4% acima da VWAP
    opp = agent.evaluate_ticker_opportunity({
        "symbol": "BTWUSDT",
        "last_price": 2.08,
        "vwap_24h": 2.00,
        "high_24h": 2.10,
        "low_24h": 1.60,
        "volume_24h_usd": 50_000_000.0,
        "price_change_pct": 8.0,
    })
    assert opp is not None
    assert opp.setup_type == "SWING_BREAKOUT"
    assert opp.direction == "BUY"


def test_argon_fix4_blocks_same_symbol_collision_across_strategies(tmp_path):
    eng = _make_engine(tmp_path)
    ts = 1792200000000
    p1 = eng.execute_scalp_signal(_sig("SOLUSDT", "BUY", price=100.0, atr=1.0, ts=ts), current_price=100.0, timestamp=ts)
    assert p1 is not None
    # Tentar abrir DIRECTIONAL no mesmo símbolo enquanto SCALP está aberto deve ser rejeitado
    p2 = eng.execute_directional_signal(_sig("SOLUSDT", "BUY", price=100.0, atr=1.0, ts=ts), current_price=100.0, timestamp=ts)
    assert p2 is None
    assert eng.last_open_reject == "margin"


def test_argon_fix5_leverage_policy_scales_atr_threshold_by_timeframe():
    from leverage_policy import decide_leverage
    # ATR% de 1.6% em 5m (>= 1.5% limiar 5m) reduz 4x -> 3x; em 1h (< 3.0% limiar 1h) mantém 4x
    assert decide_leverage(0.85, atr_pct=0.016, regime_alignment=True, timeframe="5m") == 3.0
    assert decide_leverage(0.85, atr_pct=0.016, regime_alignment=True, timeframe="1h") == 4.0
    assert decide_leverage(0.85, atr_pct=0.013, regime_alignment=True, timeframe="3m") == 3.0


def test_argon_fix6_performance_and_learning_consolidate_by_position_id(tmp_path):
    from performance_tracker import PerformanceTracker
    eng = _make_engine(tmp_path)
    recorded_outcomes = []

    class DummyLearner:
        def update_trade_outcome(self, pid, pnl):
            recorded_outcomes.append((pid, pnl))

    eng.learning_tracker = DummyLearner()
    ts = 1792200000000
    pos = eng.execute_directional_signal(_sig("SOLUSDT", "BUY", price=100.0, atr=1.0, ts=ts), current_price=100.0, timestamp=ts, leverage=2.0)
    ets = pos.entry_timestamp

    # Barra 1 pós-entrada: bate TP1 (+40%) e na mesma barra volta ao Breakeven (-$0.05 nos 60% restantes)
    eng.update_market_tick("SOLUSDT", high=pos.tp1_price + 0.2, low=pos.entry_price - 0.1, close=pos.entry_price, timestamp=ets, open_price=pos.entry_price + 0.1)
    assert pos.status == "CLOSED"
    assert pos.realized_pnl > 0.0  # Posição consolidada é VENCEDORA!

    # LearningTracker recebeu o PnL consolidado POSITIVO da posição (e não o -$0.05 do BREAKEVEN_STOP!)
    assert len(recorded_outcomes) == 1
    assert recorded_outcomes[0][0] == pos.position_id
    assert recorded_outcomes[0][1] == pytest.approx(pos.realized_pnl)
    assert recorded_outcomes[0][1] > 0.0

    # PerformanceTracker consolida por position_id: 1 trade vencedor (100% WR, streak WIN=1, zero LOSS)
    pt = PerformanceTracker(db_path=eng.db_path)
    m = pt.compute_metrics_from_db()
    assert m.total_trades == 1
    assert m.winning_trades == 1
    assert m.losing_trades == 0
    assert m.current_streak_type == "WIN"
    assert m.max_consecutive_losses == 0

