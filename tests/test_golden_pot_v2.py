"""
Regressão do GOLDEN_POT v2 (ADDENDUM E5) + lote=margem/alavancagem do Smart Trader.
Todos os testes usam DB temporário e mocks (sem rede, sem tocar no market_cache.db real).
"""
import asyncio
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from smart_trade import PaperTradingEngine, PaperPosition, MAX_LEVERAGE  # noqa: E402
from signal_generator import TradeSignal  # noqa: E402
import run_paper_simulation as rps  # noqa: E402
from run_paper_simulation import PaperTradingDaemon  # noqa: E402

BAR = rps.GP_BAR_MS


def _bar_ts(offset_bars: int = 0) -> int:
    return (int(time.time() * 1000) // BAR) * BAR - BAR * (1 - offset_bars)  # última vela fechada (+offset)


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


class FakeScanner:
    def __init__(self, symbols, volume=500e6):
        self.symbols, self.volume = symbols, volume

    async def scan_market(self, force_refresh=False):
        return {"top_scalps": [{"symbol": s, "volume_24h_usd": self.volume} for s in self.symbols]}


class FakeLoader:
    async def get_candles(self, *a, **k):
        raise AssertionError("não deve ser chamado (fetch mockado)")

    async def close(self):
        pass


def _feat(ts_last, side="BUY", close=100.0, sma=90.0, rsi=30.0, rsi_prev=29.0, o=None, n=20, atr=1.0, drop=0.0):
    """DataFrame de features sintético já com warmup (últimas n linhas)."""
    ts = ts_last - BAR * np.arange(n)[::-1]
    closes = np.full(n, close, dtype=float)
    closes[-5] = close - drop  # move na última hora = (close - close[-5]) / atr
    df = pd.DataFrame({
        "timestamp": ts, "open": closes - (0.2 if o is None else o), "high": closes + 1, "low": closes - 1,
        "close": closes, "volume": 1.0, "sma_200": sma, "rsi_14": rsi, "atr_14": atr,
        "dist_vwap_pct": -3.0 if side == "BUY" else 3.0, "liquidity_sweep_bull": 0, "liquidity_sweep_bear": 0,
        "at_bullish_poi": 0, "at_bearish_poi": 0,
    })
    df.loc[df.index[-2], "rsi_14"] = rsi_prev
    return df


def _mfeat(ts_last, side="BUY", close=100.0, sma=90.0, atr=1.0, n=130, vol_z=3.0, **_ignored):
    """Features sintéticas COM gatilho de momentum + volatilidade extraordinária (GP v3) na última vela.

    Barras anteriores: calmas (ATR = atr/3, range pequeno, abaixo do rompimento). Última: ATR = atr (expansão 3x),
    range 2.6 ATR, volume_z alto, fecha perto da extrema na direção e rompe a máxima/mínima das 20 anteriores.
    """
    sgn = 1.0 if side == "BUY" else -1.0
    ts = ts_last - BAR * np.arange(n)[::-1]
    base = close - sgn * 3.0
    df = pd.DataFrame({
        "timestamp": ts, "open": base, "high": base + 0.5, "low": base - 0.5, "close": base, "volume": 1.0,
        "sma_200": sma, "rsi_14": 50.0, "atr_14": atr / 3.0, "volume_zscore_20": 0.0,
        "dist_vwap_pct": 0.0, "liquidity_sweep_bull": 0, "liquidity_sweep_bear": 0,
        "at_bullish_poi": 0, "at_bearish_poi": 0,
    })
    i = df.index[-1]
    df.loc[i, ["open", "close"]] = [close - sgn * 2.4, close]
    df.loc[i, "high"] = close + 0.1 if side == "BUY" else close + 2.5
    df.loc[i, "low"] = close - 2.5 if side == "BUY" else close - 0.1
    df.loc[i, "atr_14"] = atr
    df.loc[i, "volume_zscore_20"] = vol_z
    return df


def _committee(direction="BUY", conf=0.82, score=0.47, veto=False, sym="AAAUSDT"):
    def _f(context, symbol=sym, current_price=0.0, atr=0.0, timestamp=0, use_llm=False):
        return TradeSignal(
            signal_id="c", timestamp=timestamp, symbol=symbol, direction="HOLD" if veto else direction,
            entry_price=current_price, stop_loss=current_price - atr, take_profit=current_price + 2 * atr,
            atr=atr, confidence=conf, consensus_score=score, position_size=123456.0, risk_dollars=77.0,
            rationale="comitê-mock", is_vetoed=veto,
        )

    async def _async(**kw):
        return _f(**kw)
    return _async


def make_daemon(db, symbols, feat_df, committee, volume=500e6, engine=None):
    eng = engine or make_engine(db)
    d = PaperTradingDaemon(engine=eng, loader=FakeLoader(), golden_pot_agent=FakeScanner(symbols, volume))

    async def _fetch(sym, now_ts):
        return pd.DataFrame({"timestamp": [0]})
    d._gp_fetch_closed_candles = _fetch
    d._compute_swing_features = lambda df: feat_df.copy()
    eng.core.evaluate_market_async = committee
    return d


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def gp_opens(engine):
    return [t for t in engine.trade_history if t.action.startswith("OPEN_GOLDEN_POT_")]


# ---------------------------------------------------------------------------
def test_committee_disagrees_no_entry(temp_db):
    d = make_daemon(temp_db, ["AAAUSDT"], _mfeat(_bar_ts()), _committee("SELL"))
    run(d.evaluate_golden_pot_opportunities())
    assert gp_opens(d.engine) == []


def test_committee_hold_no_entry(temp_db):
    d = make_daemon(temp_db, ["AAAUSDT"], _mfeat(_bar_ts()), _committee("HOLD"))
    run(d.evaluate_golden_pot_opportunities())
    assert gp_opens(d.engine) == []


def test_risk_sentinel_veto_no_entry(temp_db):
    d = make_daemon(temp_db, ["AAAUSDT"], _mfeat(_bar_ts()), _committee("BUY", veto=True))
    run(d.evaluate_golden_pot_opportunities())
    assert gp_opens(d.engine) == []


def test_low_committee_confidence_no_entry(temp_db):
    d = make_daemon(temp_db, ["AAAUSDT"], _mfeat(_bar_ts()), _committee("BUY", conf=0.55))
    run(d.evaluate_golden_pot_opportunities())
    assert gp_opens(d.engine) == []


def test_agreement_enters_with_real_signal_values_not_constants(temp_db):
    d = make_daemon(temp_db, ["AAAUSDT"], _mfeat(_bar_ts()), _committee("BUY", conf=0.82, score=0.47))
    captured = {}
    orig = d.engine.execute_golden_pot_signal

    def spy(signal, **kw):
        captured["sig"], captured["kw"] = signal, kw
        return orig(signal=signal, **kw)
    d.engine.execute_golden_pot_signal = spy
    run(d.evaluate_golden_pot_opportunities())
    assert len(gp_opens(d.engine)) == 1
    sig = captured["sig"]
    assert sig.confidence == 0.82 and sig.consensus_score == 0.47
    assert (sig.confidence, sig.consensus_score) not in [(0.90, 0.85), (0.90, -0.85)]
    # risk_dollars coerente: (margem*lev / preço) * distância do SL
    notional = sig.position_size
    sl_dist = abs(sig.entry_price - sig.stop_loss)
    assert sig.risk_dollars == pytest.approx(notional / sig.entry_price * sl_dist)
    assert "comitê-mock" in sig.rationale


def test_two_candidates_same_bar_only_one_enters(temp_db):
    d = make_daemon(temp_db, ["AAAUSDT", "BBBUSDT"], _mfeat(_bar_ts()), _committee("BUY"))
    run(d.evaluate_golden_pot_opportunities())
    run(d.evaluate_golden_pot_opportunities())  # novo ciclo na mesma vela
    assert len(gp_opens(d.engine)) == 1


def test_bar_limit_survives_restart(temp_db):
    feat = _mfeat(_bar_ts())
    d = make_daemon(temp_db, ["AAAUSDT"], feat, _committee("BUY"))
    run(d.evaluate_golden_pot_opportunities())
    assert len(gp_opens(d.engine)) == 1
    d2 = make_daemon(temp_db, ["BBBUSDT"], feat, _committee("BUY"))  # engine novo (restore)
    run(d2.evaluate_golden_pot_opportunities())
    assert len(gp_opens(d2.engine)) == 1  # nenhuma segunda entrada na mesma vela


def test_cooldown_respected_after_close_and_restart(temp_db):
    ts = _bar_ts()
    d = make_daemon(temp_db, ["AAAUSDT"], _mfeat(ts), _committee("BUY"))
    run(d.evaluate_golden_pot_opportunities())
    pos = next(iter(d.engine.positions.values()))
    d.engine.close_position(pos.position_id, pos.entry_price * 0.99, ts + 1000, reason="STOP_LOSS")

    # 1 barra depois, mesmo engine: cooldown ativo
    d.engine.daily_golden_pots_count = 0
    feat_next = _mfeat(ts + BAR)
    d._compute_swing_features = lambda df: feat_next.copy()
    run(d.evaluate_golden_pot_opportunities())
    assert len(gp_opens(d.engine)) == 1

    # restart: ledger restaura trade_history -> cooldown continua
    d2 = make_daemon(temp_db, ["AAAUSDT"], _mfeat(ts + 5 * BAR), _committee("BUY"))
    d2.engine.daily_golden_pots_count = 0
    run(d2.evaluate_golden_pot_opportunities())
    assert len(gp_opens(d2.engine)) == 1

    # após N barras: liberado
    d3 = make_daemon(temp_db, ["AAAUSDT"], _mfeat(ts + (rps.GP_COOLDOWN_BARS + 1) * BAR), _committee("BUY"))
    d3.engine.daily_golden_pots_count = 0
    run(d3.evaluate_golden_pot_opportunities())
    assert len(gp_opens(d3.engine)) == 2


def test_tp1_partial_is_not_a_close_for_cooldown(temp_db):
    d = make_daemon(temp_db, ["AAAUSDT"], _mfeat(_bar_ts()), _committee("BUY"))
    assert d._gp_in_cooldown("AAAUSDT", _bar_ts() // BAR) is False


@pytest.mark.parametrize("sym", ["SKHYNIXUSDT", "SAMSUNGUSDT", "KORUUSDT", "SOXLUSDT", "SOXSUSDT",
                                 "TSLAUSDT", "NVDAUSDT", "SKHYUSDT"])
def test_excluded_symbols_ignored(temp_db, sym):
    assert PaperTradingDaemon.is_gp_excluded_symbol(sym)
    d = make_daemon(temp_db, [sym], _mfeat(_bar_ts()), _committee("BUY"))
    run(d.evaluate_golden_pot_opportunities())
    assert gp_opens(d.engine) == []


def test_non_excluded_symbol_not_matched():
    assert not PaperTradingDaemon.is_gp_excluded_symbol("SOLUSDT")
    assert not PaperTradingDaemon.is_gp_excluded_symbol("1000PEPEUSDT")


def test_no_sma200_warmup_skipped(temp_db):
    d = make_daemon(temp_db, ["AAAUSDT"], _mfeat(_bar_ts()), _committee("BUY"))
    d._compute_swing_features = lambda df: pd.DataFrame()  # sem warmup válido
    run(d.evaluate_golden_pot_opportunities())
    assert gp_opens(d.engine) == []


def test_fetch_requests_400_candles_and_drops_open_candle(temp_db):
    calls = []

    class L:
        async def get_candles(self, sym, tf, limit=100, exchange=None, **kw):
            calls.append((tf, limit, exchange))
            n = 300
            now = int(time.time() * 1000)
            cur_open = (now // BAR) * BAR  # vela em formação
            ts = cur_open - BAR * np.arange(n)[::-1] + 0
            return pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})

        async def close(self):
            pass
    d = PaperTradingDaemon(engine=make_engine(temp_db), loader=L(), golden_pot_agent=FakeScanner([]))
    df = run(d._gp_fetch_closed_candles("AAAUSDT", int(time.time() * 1000)))
    assert calls[0] == (rps.GP_TIMEFRAME, rps.GP_CANDLE_LIMIT, "binance") and rps.GP_CANDLE_LIMIT >= 400
    assert int(df["timestamp"].iloc[-1]) + BAR <= int(time.time() * 1000)  # só velas fechadas


def test_short_history_skipped(temp_db):
    class L:
        async def get_candles(self, *a, **k):
            return pd.DataFrame({"timestamp": np.arange(120) * BAR, "close": 1.0})

        async def close(self):
            pass
    d = PaperTradingDaemon(engine=make_engine(temp_db), loader=L(), golden_pot_agent=FakeScanner([]))
    assert run(d._gp_fetch_closed_candles("NEWUSDT", int(time.time() * 1000))) is None


# --- filtro de faca caindo --------------------------------------------------
def test_knife_blocks_long_below_sma200_extreme_rsi_without_reversal():
    f = _feat(_bar_ts(), close=80.0, sma=100.0, rsi=15.0, rsi_prev=16.0, o=-0.2)  # vela vermelha, RSI caindo
    assert PaperTradingDaemon.gp_falling_knife_blocked("BUY", f) is not None


def test_knife_allows_long_with_reversal_confirmation():
    f = _feat(_bar_ts(), close=80.0, sma=100.0, rsi=15.0, rsi_prev=12.0, o=0.2)  # vela verde, RSI virando
    assert PaperTradingDaemon.gp_falling_knife_blocked("BUY", f) is None


def test_knife_blocks_on_atr_drop_last_hour():
    f = _feat(_bar_ts(), close=80.0, sma=100.0, rsi=30.0, rsi_prev=31.0, o=-0.2)
    f.loc[f.index[-5], "close"] = 84.0  # queda de 4 ATR na última hora, vela vermelha
    assert PaperTradingDaemon.gp_falling_knife_blocked("BUY", f) is not None


def test_knife_symmetric_short():
    f = _feat(_bar_ts(), side="SELL", close=120.0, sma=100.0, rsi=85.0, rsi_prev=84.0, o=0.2)  # vela verde, RSI subindo
    assert PaperTradingDaemon.gp_falling_knife_blocked("SELL", f) is not None
    f2 = _feat(_bar_ts(), side="SELL", close=120.0, sma=100.0, rsi=85.0, rsi_prev=88.0, o=-0.2)
    assert PaperTradingDaemon.gp_falling_knife_blocked("SELL", f2) is None


def test_knife_filter_not_applied_to_momentum_trigger_v3(temp_db):
    # E7: o gatilho GP é momentum (não fade); o filtro de faca-caindo (legado do fade) não bloqueia mais a entrada.
    f = _mfeat(_bar_ts(), sma=130.0)  # fecha abaixo da sma_200: regime contra, mas o comitê concorda
    d = make_daemon(temp_db, ["AAAUSDT"], f, _committee("BUY"))
    run(d.evaluate_golden_pot_opportunities())
    assert len(gp_opens(d.engine)) == 1


# --- lote = margem, alavancagem ---------------------------------------------
def _sig(symbol="BTCUSDT", direction="BUY", price=60000.0, atr=600.0, size=1e9):
    return TradeSignal(
        signal_id="t", timestamp=1_700_000_000_000, symbol=symbol, direction=direction, entry_price=price,
        stop_loss=price - atr if direction == "BUY" else price + atr,
        take_profit=price + 2 * atr if direction == "BUY" else price - 2 * atr, atr=atr,
        confidence=0.9, consensus_score=0.8, position_size=size, risk_dollars=1.0, rationale="t",
    )


@pytest.mark.parametrize("kind", ["gp", "scalp", "dir"])
def test_lot_is_100_margin_at_1x(temp_db, kind):
    e = make_engine(temp_db)
    s = _sig("BTCUSDT")
    fn = {"gp": lambda: e.execute_golden_pot_signal(s, 60000.0, 1_700_000_000_000),
          "scalp": lambda: e.execute_scalp_signal(s, 60000.0, 1_700_000_000_000),
          "dir": lambda: e.execute_directional_signal(s, 60000.0, 1_700_000_000_000)}[kind]
    pos = fn()
    assert pos is not None and pos.notional == pytest.approx(100.0) and pos.leverage == 1.0
    assert pos.margin == pytest.approx(100.0) and 0 < pos.quantity < 0.01  # BTC fracionário ok


def test_btc_lot_100_lev5_notional_500(temp_db):
    e = make_engine(temp_db)
    pos = e.execute_directional_signal(_sig(atr=600.0), 60000.0, 1_700_000_000_000, leverage=5)
    assert pos.leverage == 5.0 and pos.notional == pytest.approx(500.0) and pos.margin == pytest.approx(100.0)
    assert e.margin_used == pytest.approx(100.0)
    assert pos.to_dict()["margin"] == pytest.approx(100.0) and pos.to_dict()["leverage"] == 5.0


def test_leverage_hard_cap_and_half_steps(temp_db):
    e = make_engine(temp_db)
    p = e.execute_golden_pot_signal(_sig("ETHUSDT", price=3000.0, atr=30.0), 3000.0, 1_700_000_000_000, leverage=50)
    assert p.leverage <= MAX_LEVERAGE == 5.0
    e2 = make_engine(tempfile.mktemp(suffix=".db"))
    p2 = e2.execute_golden_pot_signal(_sig("ETHUSDT", price=3000.0, atr=30.0), 3000.0, 1_700_000_000_000, leverage=2.7)
    assert p2.leverage == 2.5


def test_leverage_reduced_if_sl_beyond_liquidation(temp_db):
    e = make_engine(temp_db)
    # SL a 15% do preço: com 5x a liquidação (~19.5%)... com 5x liq = -19.5% -> SL (-15%) antes. Usa SL 25%.
    s = _sig(price=100.0, atr=25.0)  # sl = 75
    pos = e.execute_directional_signal(s, 100.0, 1_700_000_000_000, leverage=5)
    assert pos.leverage < 5.0
    assert pos.stop_loss > pos.liquidation_price


def test_liquidation_closes_position(temp_db):
    e = make_engine(temp_db)
    pos = e.execute_directional_signal(_sig(price=100.0, atr=1.0), 100.0, 1_700_000_000_000, leverage=5)
    assert pos.leverage == 5.0
    pos.stop_loss = 0.0  # força SL além da liquidação (defesa em profundidade)
    liq = pos.liquidation_price
    assert liq == pytest.approx(pos.entry_price * (1 - 0.2 + 0.005))
    trig = e.update_market_tick("BTCUSDT", pos.entry_price, liq - 0.01, liq, 1_700_000_000_000 + 1000)
    assert [t.action for t in trig] == ["LIQUIDATION"]
    assert trig[0].pnl < 0 and abs(trig[0].pnl) <= 100.0 + 1.0  # perda ~ margem


def test_restore_keeps_leverage_and_margin(temp_db):
    e = make_engine(temp_db)
    pos = e.execute_directional_signal(_sig(atr=600.0), 60000.0, 1_700_000_000_000, leverage=4)
    e2 = make_engine(temp_db)
    q = e2.positions[pos.position_id]
    assert q.leverage == 4.0 and q.margin == pytest.approx(100.0)
    assert e2.margin_used == pytest.approx(e.margin_used)


def test_old_table_migrates_with_default_leverage(temp_db):
    import sqlite3
    con = sqlite3.connect(temp_db)
    con.execute("""CREATE TABLE paper_positions (position_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, side TEXT NOT NULL,
        entry_price REAL NOT NULL, current_price REAL NOT NULL, quantity REAL NOT NULL, notional REAL NOT NULL,
        stop_loss REAL, take_profit REAL, unrealized_pnl REAL, realized_pnl REAL, funding_collected REAL,
        fees_paid REAL, entry_timestamp INTEGER, strategy_type TEXT, status TEXT NOT NULL)""")
    con.execute("INSERT INTO paper_positions VALUES ('old','BTCUSDT','LONG',100,100,1,100,90,120,0,0,0,0,1,'DIRECTIONAL','OPEN')")
    con.commit()
    con.close()
    e = make_engine(temp_db)
    assert e.positions["old"].leverage == 1.0 and e.positions["old"].margin == 100.0


def test_paper_lot_runtime_change_and_validation(temp_db):
    e = make_engine(temp_db)
    assert e.get_simulation_settings()["paper_lot_usd"] == 100.0
    e.update_simulation_settings({"paper_lot_usd": 250})
    assert e.paper_lot_usd == 250.0
    p = e.execute_directional_signal(_sig(), 60000.0, 1_700_000_000_000)
    assert p.notional == pytest.approx(250.0)
    # alteração externa (UI -> DB) é lida por sync_settings_from_db sem reiniciar
    import sqlite3
    with sqlite3.connect(temp_db) as con:
        con.execute("UPDATE simulation_settings SET value='40' WHERE key='paper_lot_usd'")
    e.sync_settings_from_db()
    assert e.paper_lot_usd == 40.0
    with sqlite3.connect(temp_db) as con:
        con.execute("UPDATE simulation_settings SET value='5' WHERE key='paper_lot_usd'")
    e.sync_settings_from_db()
    assert e.paper_lot_usd == 40.0  # inválido ignorado
    for bad in (9.99, 1000.01, "abc", float("nan")):
        with pytest.raises(ValueError):
            e.update_simulation_settings({"paper_lot_usd": bad})


def test_gp_daemon_uses_policy_leverage_from_real_confidence(temp_db):
    d = make_daemon(temp_db, ["AAAUSDT"], _mfeat(_bar_ts(), close=100.0, atr=1.0), _committee("BUY", conf=0.85))
    run(d.evaluate_golden_pot_opportunities())
    pos = next(iter(d.engine.positions.values()))
    assert pos.leverage == 2.0  # GP v4: teto de alavancagem 2x para altcoins voláteis 5m
    assert pos.notional == pytest.approx(200.0) and pos.margin == pytest.approx(100.0)


def test_gp_low_liquidity_forces_1x(temp_db):
    d = make_daemon(temp_db, ["AAAUSDT"], _mfeat(_bar_ts()), _committee("BUY", conf=0.95), volume=30e6)
    run(d.evaluate_golden_pot_opportunities())
    assert next(iter(d.engine.positions.values())).leverage == 1.0
