"""
Testes de Homologação de Lucratividade - Semana 1:
1. Assimetria Matemática Estrita (R:R >= 2.0 no TP2/TP3 e TP1 com Breakeven)
2. Execução Maker-First em Scalp e Golden Pot (economia de taxas e slippage passivo)
3. Dimensionamento Dinâmico por Volatilidade (1% de Risco por Trade)
"""

import pytest
from smart_trade import (
    PaperTradingEngine,
    STRATEGY_PROFILES,
    TradeSignal,
    MAKER_SLIPPAGE_PCT,
)


def test_strict_mathematical_asymmetry_profiles():
    """Valida assimetria matemática: TP1 < TP2 < TP3 e R:R mínimo por tipo de saída."""
    for name, prof in STRATEGY_PROFILES.items():
        if prof.tp1_fraction == 0.0 and prof.tp2_fraction == 0.0 and prof.tp3_atr_mult == 0.0 and prof.trail_atr_mult > 0.0:
            # Trailing puro (SWING_4H): sem parciais/alvos fixos e trail >= stop loss inicial
            assert prof.tp1_atr_mult == 0.0 and prof.tp2_atr_mult == 0.0
            assert prof.trail_atr_mult >= prof.sl_atr_mult
            continue
        assert prof.tp1_atr_mult < prof.tp2_atr_mult < prof.tp3_atr_mult
        rr_tp2 = prof.tp2_atr_mult / prof.sl_atr_mult
        rr_tp3 = prof.tp3_atr_mult / prof.sl_atr_mult
        if prof.tp1_fraction > 0.0 or prof.tp2_fraction > 0.0:
            # Ladder 3 estágios: TP2 e TP3 devem remunerar favoravelmente o risco
            assert rr_tp2 >= 1.6, f"{name}: TP2 R:R {rr_tp2:.2f} deve ser >= 1.6"
            assert rr_tp3 >= 3.0, f"{name}: TP3 R:R {rr_tp3:.2f} deve ser >= 3.0"
        else:
            # Alvo único limpo (SCALP S4.2, sem BE precoce): TP3 >= 1.5R
            assert rr_tp3 >= 1.5, f"{name}: TP3 R:R {rr_tp3:.2f} deve ser >= 1.5"

    # Golden Pot alinhado para capturar fat-tails com TP3 >= 6.0 ATR
    assert STRATEGY_PROFILES["GOLDEN_POT"].tp1_atr_mult >= 1.5
    assert STRATEGY_PROFILES["GOLDEN_POT"].tp3_atr_mult >= 6.0


def test_maker_first_entry_in_scalp_and_golden_pot():
    """Valida que estratégias intradiárias operam com maker_entry=True para economia de taxas."""
    assert PaperTradingEngine._ENTRY_META["SCALP"]["maker_entry"] is True
    assert PaperTradingEngine._ENTRY_META["GOLDEN_POT"]["maker_entry"] is True


def test_volatility_adjusted_sizing_in_engine(tmp_path):
    """Valida que o dimensionamento varia inversamente com a volatilidade/distância do stop."""
    db_file = str(tmp_path / "test_vol.db")
    engine = PaperTradingEngine(initial_capital=10000.0, db_path=db_file, restore=False)
    engine.risk_per_trade_pct = 0.01  # 1% da banca = $100 em risco
    lev = 2.0
    cap_pct = 0.50

    sig = TradeSignal(
        signal_id="sig-test-vol",
        timestamp=1700000000000,
        symbol="BTCUSDT",
        direction="BUY",
        entry_price=80000.0,
        stop_loss=78000.0,
        take_profit=84000.0,
        atr=1000.0,
        confidence=0.8,
        consensus_score=0.7,
        position_size=5000.0,
        risk_dollars=100.0,
        rationale="teste",
    )

    # 1. Stop curto (distância de 1% do preço): perda de 1% * 2x lev = 2% da margem.
    # Margem ideal = $100 / 0.02 = $5,000.
    margin_short_sl, _ = engine._lot_margin(sig, lev, cap_pct, sl_distance_pct=0.01)
    expected_short = 100.0 / (0.01 * 2.0)
    assert margin_short_sl == pytest.approx(min(expected_short, 10000.0 * cap_pct), rel=1e-2)

    # 2. Stop longo/altcoin volátil (distância de 10% do preço): perda de 10% * 2x lev = 20% da margem.
    # Margem ideal = $100 / 0.20 = $500.
    margin_wide_sl, _ = engine._lot_margin(sig, lev, cap_pct, sl_distance_pct=0.10)
    expected_wide = 100.0 / (0.10 * 2.0)
    assert margin_wide_sl == pytest.approx(expected_wide, rel=1e-2)
    assert margin_wide_sl < margin_short_sl

    # 3. Fallback quando risk_per_trade_pct is None (modo lote fixo clássico)
    engine.risk_per_trade_pct = None
    margin_fixed, _ = engine._lot_margin(sig, lev, cap_pct, sl_distance_pct=0.10)
    assert margin_fixed == engine.paper_lot_usd


def test_open_position_applies_volatility_sizing(tmp_path):
    """Valida que _open_ladder_position aplica dimensionamento dinâmico contido em 1% de risco."""
    db_file = str(tmp_path / "test_open.db")
    engine = PaperTradingEngine(initial_capital=10000.0, db_path=db_file, restore=False)
    engine.risk_per_trade_pct = 0.01

    # Stop com distância de ~2.5% do preço (atr = 2200 para SL de 0.9x ATR = 1980 pts em 80000)
    # Margem ideal = $100 / (0.0247 * 2) = $2,020 (dentro do cap de 25% = $2,500)
    sig = TradeSignal(
        signal_id="sig-test-open",
        timestamp=1700000000000,
        symbol="BTCUSDT",
        direction="BUY",
        entry_price=80000.0,
        stop_loss=78000.0,
        take_profit=84000.0,
        atr=2200.0,
        confidence=0.8,
        consensus_score=0.7,
        position_size=5000.0,
        risk_dollars=100.0,
        rationale="teste",
    )
    pos = engine.execute_scalp_signal(sig, current_price=80000.0, timestamp=1700000000000, leverage=2.0)

    assert pos is not None
    assert pos.entry_price == pytest.approx(80000.0 * (1.0 + MAKER_SLIPPAGE_PCT))
    sl_dist = abs(pos.entry_price - pos.stop_loss) / pos.entry_price
    expected_risk = pos.margin * 2.0 * sl_dist
    assert expected_risk == pytest.approx(50.0, abs=3.0)  # SCALP S4.2: risco 0.50x (0.5% da banca = $50)

    from dataclasses import replace as _dc_replace
    pos_dir = engine.execute_directional_signal(
        _dc_replace(sig, symbol="ETHUSDT", signal_id="sig-test-open-dir"),
        current_price=80000.0, timestamp=1700000000000, leverage=2.0, sl_atr_mult=1.0,
    )
    assert pos_dir is not None
    sl_dist_dir = abs(pos_dir.entry_price - pos_dir.stop_loss) / pos_dir.entry_price
    expected_risk_dir = pos_dir.margin * 2.0 * sl_dist_dir
    assert expected_risk_dir == pytest.approx(100.0, abs=5.0)  # DIRECTIONAL: risco pleno 1.00x ($100)


def test_micro_stop_floor_prevents_notional_explosion(tmp_path):
    """Garante que stops microscópicos (ex: 6 bps no 3m) usem piso de 50 bps no denominador de margem."""
    db_file = str(tmp_path / "test_floor.db")
    engine = PaperTradingEngine(initial_capital=10000.0, db_path=db_file, restore=False)
    engine.risk_per_trade_pct = 0.01

    sig = TradeSignal(
        signal_id="sig-micro", timestamp=1700000000000, symbol="ETHUSDT", direction="SELL",
        entry_price=2400.0, stop_loss=2401.44, take_profit=2390.0, atr=1.6,
        confidence=0.82, consensus_score=-0.6, position_size=1000.0, risk_dollars=100.0, rationale="micro-sl",
    )
    # Sem amortecimento, sl_distance_pct=0.0006 com lev=3x daria ideal_margin = $100 / 0.0018 = $55.555 -> cap 25% ($2.500)
    # Com amortecimento de ruído (< 50 bps), a margem cai drasticamente (<< $1.000) em vez de bater no teto de $2.500
    margin, _ = engine._lot_margin(sig, 3.0, 0.25, sl_distance_pct=0.0006)
    assert margin < 500.0
    assert margin == pytest.approx((100.0 / (0.025 * 3.0)) * 0.20, rel=1e-2)


def test_s1_daemon_scalp_leverage_cap_and_consecutive_stop_cooldown(tmp_path):
    """Valida teto de 3x no Scalp/Swing em modo S1 e cooldown dobrado após 2 STOP_LOSS seguidos."""
    import pandas as pd
    from scripts.run_paper_simulation import (
        PaperTradingDaemon, SCALP_BAR_MS, SCALP_COOLDOWN_BARS, SCALP_STREAK_STOP_COOLDOWN_BARS,
    )
    from smart_trade import PaperTrade

    db_file = str(tmp_path / "test_s1_daemon.db")
    daemon = PaperTradingDaemon(
        engine=PaperTradingEngine(initial_capital=10000.0, db_path=db_file, restore=False),
        risk_per_trade_pct=0.01,
    )
    committee = TradeSignal(
        signal_id="c1", timestamp=1700000000000, symbol="ETHUSDT", direction="SELL",
        entry_price=2400.0, stop_loss=2410.0, take_profit=2370.0, atr=1.5,
        confidence=0.88, consensus_score=-0.7, position_size=100.0, risk_dollars=50.0, rationale="c",
    )
    latest = pd.Series({"close": 2400.0, "sma_200": 2500.0})
    sig, lev = daemon._build_ladder_signal(
        committee, "SCALP", "SELL", 2400.0, atr=1.5, latest=latest, margin_cap=2500.0, rationale="s1"
    )
    # Confiança 0.88 daria 4x, mas modo S1 limita SCALP em 3.0x e aplica piso de ATR (25 bps = 6.0 pts em 2400)
    assert lev <= 3.0
    assert sig.atr >= 2400.0 * 0.0025

    base_bar = 1_000_000
    for idx in (1, 2):
        daemon.engine.trade_history.append(PaperTrade(
            trade_id=f"s{idx}", position_id=f"p{idx}", symbol="ETHUSDT", action="STOP_LOSS",
            price=2405.0, quantity=1.0, fee=1.0, pnl=-10.0,
            timestamp=(base_bar + idx) * SCALP_BAR_MS, rationale="stop",
        ))
    last_stop_bar = base_bar + 2
    # Após 15 barras (> 12 padrão, mas < 24 de streak stop), continua em cooldown!
    assert daemon._scalp_in_cooldown("ETHUSDT", last_stop_bar + SCALP_COOLDOWN_BARS + 3) is True
    assert daemon._scalp_in_cooldown("ETHUSDT", last_stop_bar + SCALP_STREAK_STOP_COOLDOWN_BARS + 1) is False
