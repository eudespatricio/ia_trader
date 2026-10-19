"""
test_golden_pot_forensic_protections.py - Testes de Regressão das 4 Travas Forenses do Golden Pot (ADDENDUM E9).

Cobre:
1. Trava de Latência de Barra (Stale Execution Guard: descarta se vela fechou há > 45s).
2. Filtro de Exaustão de Clímax Vendedor (bloqueia venda no fundo com RSI < 15 ou queda acumulada > 2.5 ATR).
3. Filtro de Exaustão de Clímax Comprador (bloqueia compra no topo com RSI > 85 ou alta acumulada > 2.5 ATR).
4. Teto de Alavancagem Moderado (máx 2x em altcoins 5m, independente de alta confiança).
5. Stop Loss Dinâmico Adaptativo (acomoda a amplitude máxima recente para absorver pavios).
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

from smart_trade import PaperTradingEngine, STRATEGY_PROFILES
from signal_generator import TradeSignal
import run_paper_simulation as rps
from run_paper_simulation import PaperTradingDaemon

BAR = rps.GP_BAR_MS


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    for suf in ("", "-wal", "-shm"):
        if os.path.exists(path + suf):
            os.remove(path + suf)


class FakeScanner:
    def __init__(self, symbols, volume=500e6):
        self.symbols, self.volume = symbols, volume

    async def scan_market(self, force_refresh=False):
        return {"top_scalps": [{"symbol": s, "volume_24h_usd": self.volume} for s in self.symbols]}

    async def close(self):
        pass


class FakeLoader:
    async def get_candles(self, *a, **k):
        return pd.DataFrame({"timestamp": [0]})

    async def close(self):
        pass


def make_daemon(db, symbols, feat_df, committee, volume=500e6):
    eng = PaperTradingEngine(initial_capital=10000.0, db_path=db, max_concurrent_positions=20)
    d = PaperTradingDaemon(engine=eng, loader=FakeLoader(), golden_pot_agent=FakeScanner(symbols, volume))
    d._gp_fetch_closed_candles = lambda sym, now_ts: asyncio.sleep(0, result=pd.DataFrame({"timestamp": [0]}))
    d._compute_swing_features = lambda df: feat_df.copy()
    eng.core.evaluate_market_async = committee
    return d


def _committee(direction="BUY", conf=0.85, score=0.50):
    async def _f(context, symbol="AAAUSDT", current_price=0.0, atr=0.0, timestamp=0, use_llm=False):
        return TradeSignal(
            signal_id="c", timestamp=timestamp, symbol=symbol, direction=direction,
            entry_price=current_price, stop_loss=current_price - atr, take_profit=current_price + 2 * atr,
            atr=atr, confidence=conf, consensus_score=score, position_size=100.0, risk_dollars=10.0,
            rationale="comitê-mock", is_vetoed=False,
        )
    return _f


def _mfeat(ts_last, side="BUY", close=100.0, atr=1.0, rsi=50.0, n=130):
    sgn = 1.0 if side == "BUY" else -1.0
    ts = ts_last - BAR * np.arange(n)[::-1]
    base = close - sgn * 2.0
    df = pd.DataFrame({
        "timestamp": ts, "open": base, "high": base + 0.3, "low": base - 0.3, "close": base, "volume": 1.0,
        "sma_200": 90.0, "rsi_14": rsi, "atr_14": atr / 3.0, "volume_zscore_20": 0.0,
        "dist_vwap_pct": 0.0, "liquidity_sweep_bull": 0, "liquidity_sweep_bear": 0,
        "at_bullish_poi": 0, "at_bearish_poi": 0,
    })
    i = df.index[-1]
    df.loc[i, ["open", "close"]] = [close - sgn * 1.5, close]
    df.loc[i, "high"] = close + 0.1 if side == "BUY" else close + 1.6
    df.loc[i, "low"] = close - 1.6 if side == "BUY" else close - 0.1
    df.loc[i, "atr_14"] = atr
    df.loc[i, "volume_zscore_20"] = 2.5
    return df


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# --- TESTES DE REGRESSÃO FORENSE ---

def test_gp_stale_candle_rejection(temp_db):
    """Vela fechada há mais de 45 segundos deve ser descartada por obsolescência."""
    now_ts = int(time.time() * 1000)
    # Candle fechado há 90 segundos
    stale_ts = now_ts - BAR - 90_000
    feat = _mfeat(stale_ts, side="BUY")
    d = make_daemon(temp_db, ["AAAUSDT"], feat, _committee("BUY"))
    d.stale_bar_max_age_ms = 45_000  # ativa verificação estrita
    run(d.evaluate_golden_pot_opportunities())
    assert len(d.engine.positions) == 0


def test_gp_fresh_candle_passes_latency_guard(temp_db):
    """Vela fechada há menos de 45 segundos deve passar pela trava de latência."""
    now_ts = int(time.time() * 1000)
    # Candle fechado há 15 segundos
    fresh_ts = now_ts - BAR - 15_000
    feat = _mfeat(fresh_ts, side="BUY")
    d = make_daemon(temp_db, ["AAAUSDT"], feat, _committee("BUY"))
    d.stale_bar_max_age_ms = 45_000
    run(d.evaluate_golden_pot_opportunities())
    assert len(d.engine.positions) == 1


def test_gp_climax_exhaustion_dump_blocks_sell(temp_db):
    """Queda violenta em 3 barras com RSI sobrevendido (< 15) bloqueia venda no fundo (AGTUSDT)."""
    now_ts = int(time.time() * 1000)
    fresh_ts = now_ts - BAR - 10_000
    feat = _mfeat(fresh_ts, side="SELL", rsi=12.0)
    # Simula queda forte de 4 ATR nas últimas 3 barras
    feat.loc[feat.index[-4], "close"] = 104.0
    feat.loc[feat.index[-1], "close"] = 100.0  # queda de 4 ATR
    d = make_daemon(temp_db, ["AAAUSDT"], feat, _committee("SELL"))
    run(d.evaluate_golden_pot_opportunities())
    assert len(d.engine.positions) == 0


def test_gp_climax_exhaustion_pump_blocks_buy(temp_db):
    """Alta explosiva em 3 barras com RSI sobrecomprado (> 85) bloqueia compra no topo (BTWUSDT)."""
    now_ts = int(time.time() * 1000)
    fresh_ts = now_ts - BAR - 10_000
    feat = _mfeat(fresh_ts, side="BUY", rsi=88.0)
    feat.loc[feat.index[-4], "close"] = 96.0
    feat.loc[feat.index[-1], "close"] = 100.0  # alta de 4 ATR
    d = make_daemon(temp_db, ["AAAUSDT"], feat, _committee("BUY"))
    run(d.evaluate_golden_pot_opportunities())
    assert len(d.engine.positions) == 0


def test_gp_leverage_is_capped_at_2x(temp_db):
    """Altcoins no Golden Pot não devem exceder 2x de alavancagem mesmo com confiança máxima (0.99)."""
    now_ts = int(time.time() * 1000)
    fresh_ts = now_ts - BAR - 10_000
    feat = _mfeat(fresh_ts, side="BUY", rsi=55.0)
    d = make_daemon(temp_db, ["AAAUSDT"], feat, _committee("BUY", conf=0.99))
    run(d.evaluate_golden_pot_opportunities())
    pos = next(iter(d.engine.positions.values()))
    assert pos.leverage == 2.0


def test_gp_adaptive_sl_absorbs_wide_wicks(temp_db):
    """Stop loss adaptativo deve se expandir quando pavios recentes forem maiores que 1.5 ATR."""
    now_ts = int(time.time() * 1000)
    fresh_ts = now_ts - BAR - 10_000
    atr = 1.0
    feat = _mfeat(fresh_ts, side="BUY", close=100.0, atr=atr, rsi=55.0)
    # Cria uma vela recente com amplitude de 2.2 ATR no lado inferior (pavios longos típicos de altcoin)
    i_prev = feat.index[-2]
    feat.loc[i_prev, "high"] = 98.5
    feat.loc[i_prev, "low"] = 96.3  # amplitude 2.2 ATR (98.5 - 96.3 = 2.2 ATR)
    d = make_daemon(temp_db, ["AAAUSDT"], feat, _committee("BUY"))
    run(d.evaluate_golden_pot_opportunities())
    pos = next(iter(d.engine.positions.values()))
    # SL deve ser > 1.5 ATR (perfil básico) para cobrir o pavio de 2.2 ATR
    sl_dist = abs(pos.entry_price - pos.stop_loss)
    assert sl_dist > 1.5 * atr
    assert sl_dist >= 2.2 * rps.GP_ADAPTIVE_SL_MARGIN * atr * 0.99
