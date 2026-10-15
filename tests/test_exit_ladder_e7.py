"""
Regressão do ADDENDUM E7 (2026-10-15): SCALP / GOLDEN_POT / DAY TRADE passam pelo comitê e compartilham a
Smart Trade Exit Ladder (TP1 40% + BE, TP2 30% + trailing, TP3/trailing no runner).
DB temporário + mocks (sem rede, sem tocar no market_cache.db real).
"""
import asyncio
import os
import sqlite3
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import smart_trade as st  # noqa: E402
from smart_trade import PaperTradingEngine, STRATEGY_PROFILES  # noqa: E402
from signal_generator import TradeSignal  # noqa: E402
import run_paper_simulation as rps  # noqa: E402
from run_paper_simulation import PaperTradingDaemon  # noqa: E402

BAR = rps.GP_BAR_MS
TS0 = 1_700_000_000_000


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    for suf in ("", "-wal", "-shm"):
        if os.path.exists(path + suf):
            os.remove(path + suf)


def make_engine(db):
    return PaperTradingEngine(initial_capital=10000.0, db_path=db, max_concurrent_positions=20)


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _sig(symbol="BTCUSDT", direction="BUY", price=100.0, atr=1.0):
    return TradeSignal(
        signal_id="t", timestamp=TS0, symbol=symbol, direction=direction, entry_price=price,
        stop_loss=price - atr if direction == "BUY" else price + atr,
        take_profit=price + 2 * atr if direction == "BUY" else price - 2 * atr, atr=atr,
        confidence=0.9, consensus_score=0.8, position_size=1e9, risk_dollars=1.0, rationale="t",
    )


def mirror(side):
    """Mapeia preços 'LONG' para o lado: SHORT espelha em torno de 100 (entrada 100.01 <-> 99.99)."""
    return (lambda x: x) if side == "BUY" else (lambda x: 200.0 - x)


def tick(e, side, k, hi, lo, close=None, ts_off=0):
    """Barra k (k*15m após TS0) em coordenadas LONG (hi/lo), espelhada para SHORT."""
    m = mirror(side)
    h, l = (hi, lo) if side == "BUY" else (200.0 - lo, 200.0 - hi)
    c = m(close if close is not None else (hi + lo) / 2.0)
    return e.update_market_tick("BTCUSDT", h, l, c, TS0 + k * BAR + ts_off)


def open_scalp(e, side="BUY", **kw):
    sig = _sig(direction=side)
    return e.execute_scalp_signal(sig, 100.0, TS0, **kw)


def non_open(e, pos):
    return [t for t in e.trade_history if t.position_id == pos.position_id and not t.action.startswith("OPEN_")]


# ===========================================================================
# 1. ESCADA DE SAÍDA COMUM
# ===========================================================================
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_ladder_tp1_tp2_tp3_sequence_and_breakeven(temp_db, side):
    e = make_engine(temp_db)
    pos = open_scalp(e, side)
    sgn = 1.0 if side == "BUY" else -1.0
    init = pos.quantity
    assert pos.tp3_price == pytest.approx(pos.entry_price + sgn * 3.0)
    assert (pos.tp1_fraction, pos.tp2_fraction) == (st.EXIT_TP1_FRACTION, st.EXIT_TP2_FRACTION)
    assert pos.runner_fraction == pytest.approx(0.30)

    t = tick(e, side, 1, hi=101.2, lo=100.5)
    assert [x.action for x in t] == ["TP1_PARTIAL"]
    assert pos.tp1_hit and pos.quantity == pytest.approx(init * 0.6)
    be = pos.entry_price * (1.0 + sgn * st.BREAKEVEN_FEE_BUFFER_PCT)
    assert pos.stop_loss == pytest.approx(be)           # breakeven (+fees) após TP1
    assert (pos.stop_loss - pos.entry_price) * sgn > 0

    t = tick(e, side, 2, hi=101.9, lo=101.2)
    assert [x.action for x in t] == ["TP2_PARTIAL"]
    assert pos.tp2_hit and pos.trailing_active and pos.quantity == pytest.approx(init * 0.3)
    assert pos.remaining_fraction == pytest.approx(0.3)

    t = tick(e, side, 3, hi=103.5, lo=102.0)
    assert [x.action for x in t] == ["TAKE_PROFIT_3"]
    assert pos.position_id not in e.positions


def test_ladder_all_three_in_one_strong_bar_long(temp_db):
    e = make_engine(temp_db)
    open_scalp(e, "BUY")
    t = tick(e, "BUY", 1, hi=103.5, lo=100.5)
    assert [x.action for x in t] == ["TP1_PARTIAL", "TP2_PARTIAL", "TAKE_PROFIT_3"]


def test_breakeven_stop_after_tp1(temp_db):
    e = make_engine(temp_db)
    pos = open_scalp(e, "BUY")
    tick(e, "BUY", 1, hi=101.2, lo=100.5)
    t = tick(e, "BUY", 2, hi=100.9, lo=99.9)      # volta e toca o BE
    assert [x.action for x in t] == ["BREAKEVEN_STOP"]
    assert pos.position_id not in e.positions


def test_same_bar_tp1_then_touch_breakeven_is_conservative(temp_db):
    e = make_engine(temp_db)
    open_scalp(e, "BUY")
    t = tick(e, "BUY", 1, hi=101.9, lo=99.95)      # toca TP1/TP2 e o BE na mesma barra
    assert [x.action for x in t] == ["TP1_PARTIAL", "BREAKEVEN_STOP"]


def test_stop_has_priority_over_tp_in_same_bar(temp_db):
    e = make_engine(temp_db)
    open_scalp(e, "BUY")
    t = tick(e, "BUY", 1, hi=101.5, lo=98.5)       # SL (99.11) e TP1 na mesma barra -> SL primeiro
    assert [x.action for x in t] == ["STOP_LOSS"]


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_trailing_rises_never_falls_and_uses_only_previous_bars(temp_db, side):
    e = make_engine(temp_db)
    pos = open_scalp(e, side, tp3_atr_mult=20.0)
    sgn = 1.0 if side == "BUY" else -1.0
    m = mirror(side)
    tick(e, side, 1, hi=101.2, lo=100.5)
    tick(e, side, 2, hi=101.9, lo=101.2)                 # TP2 -> trailing ativo
    assert pos.trailing_active
    trail0 = pos.trailing_stop_price
    assert trail0 == pytest.approx(m(101.01))             # tp2 - 0.8 x ATR
    assert pos.trailing_atr_mult == STRATEGY_PROFILES["SCALP"].trail_atr_mult

    # Barra 3 tem máxima 105 e mínima 103: com stop calculado da PRÓPRIA barra (104.2) sairia; não pode.
    # Ao chegar a barra 3, a barra 2 (fechada, máx 101.9) entra: 101.9 - 0.8 = 101.1; a barra 3 NÃO.
    assert tick(e, side, 3, hi=105.0, lo=103.0) == []
    trail1 = pos.trailing_stop_price
    assert trail1 == pytest.approx(m(101.1))
    assert (trail1 - trail0) * sgn > 0
    assert tick(e, side, 3, hi=104.0, lo=103.5) == []   # tick repetido da MESMA barra (mesmo timestamp de abertura)
    assert pos.trailing_stop_price == pytest.approx(trail1)

    # Barra 4 chega: barra 3 fechou -> trailing sobe para 105 - 0.8 = 104.2
    assert tick(e, side, 4, hi=104.8, lo=104.5) == []
    assert pos.trailing_stop_price == pytest.approx(m(104.2))
    assert (pos.trailing_stop_price - trail1) * sgn > 0
    assert pos.stop_loss == pytest.approx(pos.trailing_stop_price)   # stop efetivo acompanha

    # Barra 5 com máxima menor: trailing NÃO afrouxa; e a mínima 103.9 aciona o stop em 104.2
    t = tick(e, side, 5, hi=104.6, lo=103.9)
    assert [x.action for x in t] == ["TRAILING_STOP"]
    assert t[0].price == pytest.approx(m(104.2), rel=5e-4)   # preenchimento (taker) com slippage perp
    assert pos.position_id not in e.positions


def test_trailing_never_loosens_when_bars_pull_back(temp_db):
    e = make_engine(temp_db)
    pos = open_scalp(e, "BUY", tp3_atr_mult=20.0)
    tick(e, "BUY", 1, hi=101.2, lo=100.5)
    tick(e, "BUY", 2, hi=101.9, lo=101.2)
    levels = []
    for k, (hi, lo) in enumerate([(104.0, 102.5), (103.5, 102.6), (103.0, 102.7), (105.0, 102.8), (103.9, 102.9)], start=3):
        tick(e, "BUY", k, hi, lo)
        levels.append(pos.trailing_stop_price)
    assert all(b >= a for a, b in zip(levels, levels[1:]))


# ===========================================================================
# 2. TAXAS POR TIPO DE SAÍDA, PNL E LEDGER
# ===========================================================================
def _rate(t):
    return t.fee / (t.price * t.quantity)


def test_fees_maker_for_tps_and_taker_for_stops(temp_db):
    e = make_engine(temp_db)
    pos = open_scalp(e, "BUY", tp3_atr_mult=20.0)
    tick(e, "BUY", 1, hi=101.2, lo=100.5)
    tick(e, "BUY", 2, hi=101.9, lo=101.2)
    tick(e, "BUY", 3, hi=105.0, lo=103.0)
    tick(e, "BUY", 4, hi=104.8, lo=104.5)
    tick(e, "BUY", 5, hi=104.6, lo=103.9)
    by = {t.action: t for t in non_open(e, pos)}
    assert _rate(by["TP1_PARTIAL"]) == pytest.approx(e.maker_fee)
    assert _rate(by["TP2_PARTIAL"]) == pytest.approx(e.maker_fee)
    assert _rate(by["TRAILING_STOP"]) == pytest.approx(e.taker_fee)

    e2 = make_engine(tempfile.mktemp(suffix=".db"))
    p2 = open_scalp(e2, "BUY")
    t3 = tick(e2, "BUY", 1, hi=103.5, lo=100.5)[-1]
    assert t3.action == "TAKE_PROFIT_3" and _rate(t3) == pytest.approx(e2.maker_fee)

    e3 = make_engine(tempfile.mktemp(suffix=".db"))
    open_scalp(e3, "BUY")
    assert tick(e3, "BUY", 1, hi=100.5, lo=98.5)[0].action == "STOP_LOSS"
    assert _rate(e3.trade_history[-1]) == pytest.approx(e3.taker_fee)

    e4 = make_engine(tempfile.mktemp(suffix=".db"))
    p4 = open_scalp(e4, "BUY")
    tick(e4, "BUY", 1, hi=101.2, lo=100.5)
    assert tick(e4, "BUY", 2, hi=100.9, lo=99.9)[0].action == "BREAKEVEN_STOP"
    assert _rate(e4.trade_history[-1]) == pytest.approx(e4.taker_fee)

    e5 = make_engine(tempfile.mktemp(suffix=".db"))
    p5 = open_scalp(e5, "BUY")
    t = e5.close_position(p5.position_id, 100.2, TS0 + BAR, reason="MANUAL_CLOSE")
    assert _rate(t) == pytest.approx(e5.taker_fee)


def test_liquidation_is_taker(temp_db):
    e = make_engine(temp_db)
    pos = e.execute_directional_signal(_sig(price=100.0, atr=1.0), 100.0, TS0, leverage=5)
    pos.stop_loss = 0.0
    t = e.update_market_tick("BTCUSDT", 100.0, pos.liquidation_price - 0.01, pos.liquidation_price, TS0 + BAR)
    assert t[0].action == "LIQUIDATION" and _rate(t[0]) == pytest.approx(e.taker_fee)


def test_partials_pnl_sums_to_ledger_and_balance(temp_db):
    e = make_engine(temp_db)
    pos = open_scalp(e, "BUY", tp3_atr_mult=20.0)
    open_fee = pos.fees_paid
    for k, (hi, lo) in enumerate([(101.2, 100.5), (101.9, 101.2), (105.0, 103.0), (104.8, 104.5), (104.6, 103.9)], start=1):
        tick(e, "BUY", k, hi, lo)
    assert pos.position_id not in e.positions
    legs = non_open(e, pos)
    assert [t.action for t in legs] == ["TP1_PARTIAL", "TP2_PARTIAL", "TRAILING_STOP"]
    total = sum(t.pnl for t in legs)
    assert total == pytest.approx(pos.realized_pnl)
    assert total == pytest.approx(e.total_realized_pnl)
    assert e.balance == pytest.approx(10000.0 - open_fee + total)
    assert sum(t.quantity for t in legs) == pytest.approx(pos.initial_quantity)   # frações somam 100%
    e2 = make_engine(temp_db)   # restore -> mesmo saldo (ledger = fonte)
    assert e2.balance == pytest.approx(e.balance)


# ===========================================================================
# 3. PERSISTÊNCIA / MIGRAÇÃO / RESTORE
# ===========================================================================
def test_restore_state_restores_tp3_trailing_and_fractions(temp_db):
    e = make_engine(temp_db)
    pos = open_scalp(e, "BUY", tp3_atr_mult=20.0)
    tick(e, "BUY", 1, hi=101.2, lo=100.5)
    tick(e, "BUY", 2, hi=101.9, lo=101.2)
    tick(e, "BUY", 3, hi=105.0, lo=103.0)
    tick(e, "BUY", 4, hi=104.8, lo=104.5)
    e2 = make_engine(temp_db)
    q = e2.positions[pos.position_id]
    assert q.tp3_price == pytest.approx(pos.tp3_price) and q.tp3_price > 0
    assert q.trailing_active and q.tp1_hit and q.tp2_hit
    assert q.trailing_stop_price == pytest.approx(pos.trailing_stop_price)
    assert q.trailing_atr_mult == pytest.approx(0.8) and q.entry_atr == pytest.approx(1.0)
    assert (q.tp1_fraction, q.tp2_fraction) == (st.EXIT_TP1_FRACTION, st.EXIT_TP2_FRACTION)
    assert q.quantity == pytest.approx(pos.quantity) and q.stop_loss == pytest.approx(pos.stop_loss)
    # continua funcionando após o restart: barra 5 aciona o trailing restaurado
    t = e2.update_market_tick("BTCUSDT", 104.6, 103.9, 104.0, TS0 + 5 * BAR)
    assert [x.action for x in t] == ["TRAILING_STOP"]


def test_to_dict_exposes_ladder_fields(temp_db):
    e = make_engine(temp_db)
    pos = open_scalp(e, "BUY")
    d = pos.to_dict()
    for k in ("tp1", "tp2", "tp3", "tp3_price", "trailing_stop_price", "trailing_active", "trailing_atr_mult",
              "tp1_fraction", "tp2_fraction", "runner_fraction", "remaining_fraction"):
        assert k in d
    assert d["tp1"] == pos.tp1_price and d["tp3"] == pos.tp3_price and d["remaining_fraction"] == pytest.approx(1.0)
    assert "trail_bar_ts" not in d


def test_old_table_migrates_and_legacy_position_keeps_old_tp2_close(temp_db):
    con = sqlite3.connect(temp_db)
    con.execute("""CREATE TABLE paper_positions (position_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, side TEXT NOT NULL,
        entry_price REAL NOT NULL, current_price REAL NOT NULL, quantity REAL NOT NULL, notional REAL NOT NULL,
        stop_loss REAL, take_profit REAL, unrealized_pnl REAL, realized_pnl REAL, funding_collected REAL,
        fees_paid REAL, entry_timestamp INTEGER, strategy_type TEXT, status TEXT NOT NULL,
        tp1_price REAL DEFAULT 0.0, tp1_hit INTEGER DEFAULT 0, tp2_price REAL DEFAULT 0.0,
        initial_quantity REAL DEFAULT 0.0, leverage REAL DEFAULT 1.0)""")
    con.execute(f"INSERT INTO paper_positions VALUES ('old','BTCUSDT','LONG',100,100,0.5,100,100.05,102,0,0,0,0,{TS0},"
                "'SCALP','OPEN',101,1,102,1.0,1.0)")
    con.commit()
    con.close()
    e = make_engine(temp_db)
    p = e.positions["old"]
    assert p.tp3_price == 0.0 and p.tp1_fraction == 0.5 and p.tp2_fraction == 0.0 and not p.trailing_active
    t = e.update_market_tick("BTCUSDT", 102.5, 101.5, 102.4, TS0)
    assert [x.action for x in t] == ["TAKE_PROFIT_2"]   # legado: TP2 encerra o restante


# ===========================================================================
# 4. SL SEMPRE ANTES DA LIQUIDAÇÃO / PERFIS
# ===========================================================================
@pytest.mark.parametrize("kind", ["scalp", "gp", "dir"])
def test_sl_before_liquidation_reduces_leverage_all_strategies(temp_db, kind):
    e = make_engine(temp_db)
    s = _sig(price=100.0, atr=30.0)
    fn = {"scalp": e.execute_scalp_signal, "gp": e.execute_golden_pot_signal, "dir": e.execute_directional_signal}[kind]
    pos = fn(s, 100.0, TS0, leverage=5)
    assert pos is not None and pos.leverage < 5.0
    assert pos.stop_loss > pos.liquidation_price


def test_profiles_single_source_and_ordering():
    assert set(STRATEGY_PROFILES) == {"SCALP", "DIRECTIONAL", "GOLDEN_POT"}
    sc, dt, gp = (STRATEGY_PROFILES[k] for k in ("SCALP", "DIRECTIONAL", "GOLDEN_POT"))
    assert sc.tp3_atr_mult < dt.tp3_atr_mult < gp.tp3_atr_mult           # alvos curtos < médios < maiores
    assert sc.trail_atr_mult < dt.trail_atr_mult < gp.trail_atr_mult
    for p in (sc, dt, gp):
        assert p.tp1_atr_mult < p.tp2_atr_mult < p.tp3_atr_mult and p.min_committee_confidence == 0.70
    assert rps.GP_TP3_ATR_MULT == gp.tp3_atr_mult and rps.SCALP_MIN_COMMITTEE_CONFIDENCE == 0.70


@pytest.mark.parametrize("kind,strategy", [("scalp", "SCALP"), ("gp", "GOLDEN_POT"), ("dir", "DIRECTIONAL")])
def test_every_strategy_uses_its_profile_ladder(temp_db, kind, strategy):
    e = make_engine(temp_db)
    fn = {"scalp": e.execute_scalp_signal, "gp": e.execute_golden_pot_signal, "dir": e.execute_directional_signal}[kind]
    pos = fn(_sig(), 100.0, TS0)
    pr = STRATEGY_PROFILES[strategy]
    assert pos.strategy_type == strategy
    assert pos.tp1_price - pos.entry_price == pytest.approx(pr.tp1_atr_mult)
    assert pos.tp2_price - pos.entry_price == pytest.approx(pr.tp2_atr_mult)
    assert pos.tp3_price - pos.entry_price == pytest.approx(pr.tp3_atr_mult)
    assert pos.trailing_atr_mult == pr.trail_atr_mult and pos.entry_atr == pytest.approx(1.0)


# ===========================================================================
# 5. COMITÊ OBRIGATÓRIO (SCALP, GOLDEN POT, DAY TRADE)
# ===========================================================================
def _committee(direction="BUY", conf=0.82, score=0.47, veto=False):
    async def _f(context, symbol="X", current_price=0.0, atr=0.0, timestamp=0, use_llm=False):
        return TradeSignal(
            signal_id="c", timestamp=timestamp, symbol=symbol, direction="HOLD" if veto else direction,
            entry_price=current_price, stop_loss=current_price - atr, take_profit=current_price + 2 * atr,
            atr=atr, confidence=conf, consensus_score=score, position_size=123456.0, risk_dollars=77.0,
            rationale="comitê-mock", is_vetoed=veto,
        )
    return _f


def _bar_ts(offset=0):
    return (int(time.time() * 1000) // BAR) * BAR - BAR * (1 - offset)


def _sfeat(ts_last, side="BUY", close=100.0, sma=90.0, n=30):
    """Features do setup de scalp (VWAP/RSI/volume) já com sma_200."""
    sgn = 1.0 if side == "BUY" else -1.0
    ts = ts_last - BAR * np.arange(n)[::-1]
    return pd.DataFrame({
        "timestamp": ts, "open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0,
        "sma_200": sma, "rsi_14": 30.0 if side == "BUY" else 70.0, "atr_14": 1.0, "volume_zscore_20": 1.0,
        "dist_vwap_pct": -3.0 * sgn,
    })


def _mfeat(ts_last, side="BUY", close=100.0, sma=90.0, atr=1.0, n=130, vol_z=3.0):
    """Momentum + volatilidade extraordinária na última vela (barras anteriores calmas)."""
    sgn = 1.0 if side == "BUY" else -1.0
    ts = ts_last - BAR * np.arange(n)[::-1]
    base = close - sgn * 3.0
    df = pd.DataFrame({
        "timestamp": ts, "open": base, "high": base + 0.5, "low": base - 0.5, "close": base, "volume": 1.0,
        "sma_200": sma, "rsi_14": 50.0, "atr_14": atr / 3.0, "volume_zscore_20": 0.0, "dist_vwap_pct": 0.0,
        "liquidity_sweep_bull": 0, "liquidity_sweep_bear": 0, "at_bullish_poi": 0, "at_bearish_poi": 0,
    })
    i = df.index[-1]
    df.loc[i, ["open", "close"]] = [close - sgn * 2.4, close]
    df.loc[i, "high"] = close + 0.1 if side == "BUY" else close + 2.5
    df.loc[i, "low"] = close - 2.5 if side == "BUY" else close - 0.1
    df.loc[i, "atr_14"] = atr
    df.loc[i, "volume_zscore_20"] = vol_z
    return df


class FakeScanner:
    def __init__(self, symbols, volume=500e6):
        self.symbols, self.volume = symbols, volume

    async def scan_market(self, force_refresh=False):
        return {"top_scalps": [{"symbol": s, "volume_24h_usd": self.volume} for s in self.symbols]}

    async def close(self):
        pass


class FakeLoader:
    async def get_candles(self, sym, tf, limit=100, exchange=None, **kw):
        n = 300
        ts = _bar_ts() - 3600_000 * np.arange(n)[::-1]
        return pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})

    async def close(self):
        pass


def make_daemon(db, committee, feat, symbols=("AAAUSDT",), volume=500e6):
    eng = make_engine(db)
    d = PaperTradingDaemon(engine=eng, loader=FakeLoader(), golden_pot_agent=FakeScanner(list(symbols), volume))

    async def _fetch(sym, now_ts):
        return pd.DataFrame({"timestamp": [0]})
    d._gp_fetch_closed_candles = _fetch
    d._fetch_closed_candles_15m = _fetch
    d._compute_swing_features = lambda df: feat.copy()
    eng.core.evaluate_market_async = committee
    return d


def opens(engine, prefix):
    return [t for t in engine.trade_history if t.action.startswith(prefix)]


def _swing_daemon(db, direction="BUY", conf=0.8, veto=False):
    d = make_daemon(db, _committee(), _sfeat(_bar_ts()))
    feat = _sfeat(_bar_ts(), close=100.0)
    d._compute_swing_features = lambda df: feat.copy()
    sg = TradeSignal(
        signal_id="s", timestamp=_bar_ts(), symbol="BTCUSDT", direction=direction, entry_price=100.0,
        stop_loss=99.0, take_profit=101.5, atr=1.0, confidence=conf, consensus_score=0.5, position_size=1.0,
        risk_dollars=1.0, rationale="swing-mock", is_vetoed=veto,
    )
    d.engine.core.evaluate_multi_strategy = lambda **kw: {"directional_signal": sg.to_dict()}
    return d


@pytest.mark.parametrize("kind", ["gp", "scalp", "swing"])
@pytest.mark.parametrize("case", ["disagree", "hold", "veto", "lowconf"])
def test_committee_disagrees_or_vetoes_no_strategy_enters(temp_db, kind, case):
    cm = {
        "disagree": _committee("SELL"), "hold": _committee("HOLD"),
        "veto": _committee("BUY", veto=True), "lowconf": _committee("BUY", conf=0.55),
    }[case]
    if kind == "gp":
        d = make_daemon(temp_db, cm, _mfeat(_bar_ts()))
        run(d.evaluate_golden_pot_opportunities())
        assert opens(d.engine, "OPEN_GOLDEN_POT_") == []
    elif kind == "scalp":
        d = make_daemon(temp_db, cm, _sfeat(_bar_ts()))
        run(d.evaluate_scalp_opportunities())
        assert opens(d.engine, "OPEN_SCALP_") == []
    else:
        if case == "disagree":
            pytest.skip("day trade: o sinal do comitê É a direção (não há gatilho independente para divergir)")
        sw = {"disagree": ("SELL", 0.8, False), "hold": ("HOLD", 0.8, False),
              "veto": ("BUY", 0.8, True), "lowconf": ("BUY", 0.55, False)}[case]
        d = _swing_daemon(temp_db, *sw)
        run(d.evaluate_swing_opportunities())
        assert [t for t in d.engine.trade_history if t.action in ("OPEN_LONG", "OPEN_SHORT")] == []


def test_swing_day_trade_enters_with_ladder(temp_db):
    d = _swing_daemon(temp_db, "BUY", 0.8)
    run(d.evaluate_swing_opportunities())
    pos = next(iter(d.engine.positions.values()))
    assert pos.strategy_type == "DIRECTIONAL" and pos.tp3_price > pos.tp2_price > pos.tp1_price > pos.entry_price
    assert pos.stop_loss < pos.entry_price and pos.trailing_atr_mult == STRATEGY_PROFILES["DIRECTIONAL"].trail_atr_mult


def test_scalp_uses_real_committee_values_not_constants(temp_db):
    d = make_daemon(temp_db, _committee("BUY", conf=0.78, score=0.41), _sfeat(_bar_ts()))
    captured = {}
    orig = d.engine.execute_scalp_signal

    def spy(signal, **kw):
        captured["sig"], captured["kw"] = signal, kw
        return orig(signal=signal, **kw)
    d.engine.execute_scalp_signal = spy
    run(d.evaluate_scalp_opportunities())
    assert len(opens(d.engine, "OPEN_SCALP_")) == 1
    sig = captured["sig"]
    assert (sig.confidence, sig.consensus_score) == (0.78, 0.41)
    assert sig.confidence != 0.85 and abs(sig.consensus_score) != 0.80
    assert "comitê-mock" in sig.rationale and sig.risk_dollars != 77.0
    assert sig.risk_dollars == pytest.approx(sig.position_size / sig.entry_price
                                             * STRATEGY_PROFILES["SCALP"].sl_atr_mult * sig.atr)
    pos = next(iter(d.engine.positions.values()))
    assert pos.leverage == 3.0                      # decide_leverage(conf 0.78 -> 3x), a favor da tendência
    assert pos.margin == pytest.approx(100.0)


def test_scalp_no_candidate_no_committee_call(temp_db):
    feat = _sfeat(_bar_ts())
    feat["dist_vwap_pct"] = 0.0                      # sem setup VWAP/RSI
    calls = []

    async def spy(**kw):
        calls.append(kw)
        return await _committee("BUY")(**kw)
    d = make_daemon(temp_db, spy, feat)
    run(d.evaluate_scalp_opportunities())
    assert calls == [] and opens(d.engine, "OPEN_SCALP_") == []


def test_scalp_one_per_bar_and_cooldown(temp_db):
    ts = _bar_ts()
    d = make_daemon(temp_db, _committee("BUY"), _sfeat(ts))
    run(d.evaluate_scalp_opportunities())
    run(d.evaluate_scalp_opportunities())
    assert len(opens(d.engine, "OPEN_SCALP_")) == 1   # 1 entrada por vela (mesmo com 3 símbolos core)
    pos = next(iter(d.engine.positions.values()))
    d.engine.close_position(pos.position_id, pos.entry_price * 0.99, ts + 1000, reason="STOP_LOSS")
    nxt = _sfeat(ts + BAR)
    d._compute_swing_features = lambda df: nxt.copy()
    run(d.evaluate_scalp_opportunities())
    # BTC/SOL (sem fechamento recente) podem entrar; o símbolo fechado fica em cooldown
    syms = [t.symbol for t in opens(d.engine, "OPEN_SCALP_")]
    assert syms.count(pos.symbol) == 1


def test_scalp_requests_400_closed_candles(temp_db):
    calls = []

    class L:
        async def get_candles(self, sym, tf, limit=100, exchange=None, **kw):
            calls.append((tf, limit, exchange))
            n = 300
            cur_open = (int(time.time() * 1000) // BAR) * BAR
            ts = cur_open - BAR * np.arange(n)[::-1]
            return pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})

        async def close(self):
            pass
    d = PaperTradingDaemon(engine=make_engine(temp_db), loader=L(), golden_pot_agent=FakeScanner([]))
    df = run(d._fetch_closed_candles_15m("BTCUSDT", int(time.time() * 1000)))
    assert calls[0] == ("15m", 400, "binance")
    assert int(df["timestamp"].iloc[-1]) + BAR <= int(time.time() * 1000)


# ===========================================================================
# 6. GOLDEN POT: MOMENTUM + VOLATILIDADE EXTRAORDINÁRIA (NÃO MEAN-REVERSION)
# ===========================================================================
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_gp_trigger_fires_on_volatility_expansion_momentum(side):
    r = PaperTradingDaemon.gp_momentum_trigger(_mfeat(_bar_ts(), side=side))
    assert r is not None and r[0] == side and "momentum" in r[1]


@pytest.mark.parametrize("mut,why", [
    ("no_expansion", "ATR sem expansão"),
    ("low_volume", "volume_z baixo"),
    ("no_breakout", "sem rompimento"),
    ("narrow_range", "range < k ATR"),
    ("weak_close", "fecha longe da extrema"),
    ("red_candle", "vela contra"),
])
def test_gp_trigger_rejects_when_any_condition_missing(mut, why):
    f = _mfeat(_bar_ts())
    i = f.index[-1]
    if mut == "no_expansion":
        f["atr_14"] = 1.0
    elif mut == "low_volume":
        f.loc[i, "volume_zscore_20"] = 0.5
    elif mut == "no_breakout":
        f.loc[f.index[-10], "high"] = 120.0              # máxima anterior acima do fechamento
    elif mut == "narrow_range":
        f.loc[i, "low"] = 99.0
        f.loc[i, "atr_14"] = 1.0
        f["atr_14"] = f["atr_14"].where(f.index == i, 0.3)
    elif mut == "weak_close":
        f.loc[i, "high"] = 104.0
    elif mut == "red_candle":
        f.loc[i, "open"] = 101.0
    assert PaperTradingDaemon.gp_momentum_trigger(f) is None, why


def test_gp_trigger_does_not_fire_on_mean_reversion_fade():
    """Setup do GP antigo (VWAP -3% + RSI 25, preço esticado para baixo) SEM expansão/rompimento: não dispara."""
    n = 130
    ts = _bar_ts() - BAR * np.arange(n)[::-1]
    close = np.linspace(110.0, 97.0, n)
    f = pd.DataFrame({
        "timestamp": ts, "open": close + 0.1, "high": close + 0.3, "low": close - 0.3, "close": close,
        "sma_200": 100.0, "rsi_14": 25.0, "atr_14": 0.6, "volume_zscore_20": 2.5, "dist_vwap_pct": -3.5,
        "liquidity_sweep_bull": 1, "liquidity_sweep_bear": 0, "at_bullish_poi": 1, "at_bearish_poi": 0,
    })
    assert PaperTradingDaemon.gp_momentum_trigger(f) is None


def test_gp_needs_committee_in_momentum_direction(temp_db):
    d = make_daemon(temp_db, _committee("SELL"), _mfeat(_bar_ts(), side="BUY"))   # momentum BUY, comitê SELL
    run(d.evaluate_golden_pot_opportunities())
    assert opens(d.engine, "OPEN_GOLDEN_POT_") == []


def test_gp_enters_with_real_values_ladder_and_profile(temp_db):
    d = make_daemon(temp_db, _committee("BUY", conf=0.83, score=0.52), _mfeat(_bar_ts()))
    captured = {}
    orig = d.engine.execute_golden_pot_signal

    def spy(signal, **kw):
        captured["sig"] = signal
        return orig(signal=signal, **kw)
    d.engine.execute_golden_pot_signal = spy
    run(d.evaluate_golden_pot_opportunities())
    assert len(opens(d.engine, "OPEN_GOLDEN_POT_")) == 1
    sig = captured["sig"]
    assert (sig.confidence, sig.consensus_score) == (0.83, 0.52)
    assert "Golden Pot v3" in sig.rationale and "momentum BUY" in sig.rationale
    pos = next(iter(d.engine.positions.values()))
    assert pos.tp3_price - pos.entry_price == pytest.approx(STRATEGY_PROFILES["GOLDEN_POT"].tp3_atr_mult * 1.0)


def test_gp_old_vwap_rsi_fade_no_longer_triggers_daemon(temp_db):
    f = _mfeat(_bar_ts())
    f["atr_14"] = 1.0                                   # sem expansão de volatilidade
    f["dist_vwap_pct"], f["rsi_14"] = -5.0, 15.0        # gatilho do GP v2 (fade) satisfeito
    d = make_daemon(temp_db, _committee("BUY"), f)
    run(d.evaluate_golden_pot_opportunities())
    assert opens(d.engine, "OPEN_GOLDEN_POT_") == []


def test_gp_sweep_poi_alone_is_not_a_trigger(temp_db):
    f = _mfeat(_bar_ts())
    f["atr_14"] = 1.0
    f["liquidity_sweep_bull"], f["at_bullish_poi"] = 1, 1
    d = make_daemon(temp_db, _committee("BUY"), f)
    run(d.evaluate_golden_pot_opportunities())
    assert opens(d.engine, "OPEN_GOLDEN_POT_") == []
