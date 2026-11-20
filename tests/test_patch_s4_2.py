"""
Suíte unitária dedicada ao Patch S4.2 (v1.0.7):
  1. Paridade same_bar_safe (closed_safe) no BacktestEngine.
  2. Saída limpa alvo único do SCALP (tp1/tp2_fraction=0.0, sem armadilha BE) + risco 0.50x.
  3. Trava de correlação MAX_SAME_DIR_MAJORS=2 no DIRECTIONAL.
  4. Gate de fluxo fail-closed nos majors vs política híbrida M4 (0.50x) no GOLDEN_POT.
  5. Gate Macro BTC ±1.5% SMA200 no GOLDEN_POT.
  6. Gatilho Squeeze BB/Keltner + ML 15m (>= 0.45) universal para BTC, ETH e SOL.

Usa apenas DBs temporários (tmp_path) e mocks determinísticos; nunca toca datasets/market_cache.db.
"""

import asyncio
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "scripts"))

from backtesting import BacktestEngine  # noqa: E402
import run_paper_simulation as rps  # noqa: E402
from run_paper_simulation import PaperTradingDaemon, SCALP_BAR_MS, SWING_BAR_MS, GP_BAR_MS  # noqa: E402
from signal_generator import TradeSignal  # noqa: E402
from smart_trade import PaperTradingEngine, STRATEGY_PROFILES  # noqa: E402


def _make_engine(db_path: str, risk_pct: float = 0.01) -> PaperTradingEngine:
    eng = PaperTradingEngine(initial_capital=10_000.0, db_path=db_path, restore=False)
    eng.risk_per_trade_pct = risk_pct
    return eng


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _sig(symbol: str = "BTCUSDT", direction: str = "BUY", price: float = 100.0, atr: float = 1.0,
         sl_mult: float = 1.4, tp_mult: float = 2.2) -> TradeSignal:
    sgn = 1.0 if direction == "BUY" else -1.0
    return TradeSignal(
        signal_id=f"sig-{symbol}-{direction}",
        timestamp=1_700_000_000_000,
        symbol=symbol,
        direction=direction,
        entry_price=price,
        stop_loss=price - sgn * sl_mult * atr,
        take_profit=price + sgn * tp_mult * atr,
        atr=atr,
        confidence=0.82,
        consensus_score=0.50 * sgn,
        position_size=100.0,
        risk_dollars=1.0,
        rationale="test-s42",
    )


def _recent_bar_ts(bar_ms: int) -> int:
    now_ms = int(time.time() * 1000)
    return ((now_ms // bar_ms) - 1) * bar_ms


# ---------------------------------------------------------------------------
# 1. Paridade BacktestEngine: closed_safe na mesma barra do TP1
# ---------------------------------------------------------------------------
def test_backtest_same_bar_be_closed_safe_parity():
    base_ts = 1_700_000_000_000
    rows = [
        # Barra 0: sinal BUY em close=100.0 -> entra em barra 1 (open=100.0)
        {"timestamp": base_ts + 0 * SWING_BAR_MS, "open": 99.8, "high": 100.2, "low": 99.7, "close": 100.0, "atr_14": 1.0},
        # Barra 1: atinge TP1 (+1.5 ATR ~101.52), pavio inicial toca 99.8 (< BE ~100.06, > SL ~99.02) e fecha salvo a 100.60
        {"timestamp": base_ts + 1 * SWING_BAR_MS, "open": 100.0, "high": 101.8, "low": 99.8, "close": 100.6, "atr_14": 1.0},
        # Barras 2..4: expansão até TP3 (+4.0 ATR ~104.02)
        {"timestamp": base_ts + 2 * SWING_BAR_MS, "open": 100.6, "high": 102.8, "low": 100.5, "close": 102.6, "atr_14": 1.0},
        {"timestamp": base_ts + 3 * SWING_BAR_MS, "open": 102.6, "high": 104.5, "low": 102.4, "close": 104.2, "atr_14": 1.0},
        {"timestamp": base_ts + 4 * SWING_BAR_MS, "open": 104.2, "high": 104.6, "low": 103.8, "close": 104.3, "atr_14": 1.0},
    ]
    df = pd.DataFrame(rows)
    sigs = np.array([1, 0, 0, 0, 0], dtype=np.int32)
    eng = BacktestEngine(initial_capital=10_000.0, timeframe="1h", pilot_parity=True, risk_per_trade=None)
    res = eng.run(df, sigs, atr_col="atr_14")
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.exit_reason != "BREAKEVEN_STOP"
    assert tr.exit_reason == "TAKE_PROFIT_3"
    assert tr.exit_timestamp > int(df["timestamp"].iloc[1])


# ---------------------------------------------------------------------------
# 2. SCALP: saída limpa alvo único (sem TP1 BE) + risco 0.50x
# ---------------------------------------------------------------------------
def test_scalp_single_target_clean_exit_no_tp1_be(tmp_path):
    ts0 = 1_700_000_000_000
    eng_s = _make_engine(str(tmp_path / "scalp.db"), 0.01)
    pos = eng_s.execute_scalp_signal(_sig("ETHUSDT", "BUY", 100.0, 1.0), 100.0, ts0, leverage=2.0)
    assert pos is not None
    assert pos.tp1_fraction == pytest.approx(0.0)
    assert pos.tp2_fraction == pytest.approx(0.0)
    assert pos.tp3_price == pytest.approx(pos.entry_price + STRATEGY_PROFILES["SCALP"].tp3_atr_mult * 1.0)

    # Barra 1: passa por +1.6/+2.0 ATR e recua até +0.01 ATR (antigo BE +4bps estoparia); deve permanecer aberta
    t1 = eng_s.update_market_tick("ETHUSDT", high=101.7, low=100.01, close=100.5, timestamp=ts0 + SCALP_BAR_MS, open_price=100.2)
    assert t1 == []
    assert pos.position_id in eng_s.positions
    assert pos.tp1_hit is False

    # Barra 2: atinge +2.2 ATR -> fecha 100% como TAKE_PROFIT_3 (maker)
    t2 = eng_s.update_market_tick("ETHUSDT", high=102.3, low=100.5, close=102.2, timestamp=ts0 + 2 * SCALP_BAR_MS, open_price=100.5)
    assert len(t2) == 1
    assert t2[0].action == "TAKE_PROFIT_3"
    assert t2[0].quantity == pytest.approx(pos.initial_quantity)
    assert pos.position_id not in eng_s.positions

    # Compara risco 0.50x do SCALP vs 1.00x do DIRECTIONAL sob mesmo SL (1.4 ATR), alavancagem (2.0x) e capital ($10k)
    eng_s2 = _make_engine(str(tmp_path / "scalp2.db"), 0.01)
    eng_d = _make_engine(str(tmp_path / "dir.db"), 0.01)
    pos_s2 = eng_s2.execute_scalp_signal(_sig("SOLUSDT", "BUY", 100.0, 1.0), 100.0, ts0 + 5 * SCALP_BAR_MS, leverage=2.0)
    pos_d = eng_d.execute_directional_signal(_sig("SOLUSDT", "BUY", 100.0, 1.0, sl_mult=1.4), 100.0, ts0 + 5 * SCALP_BAR_MS, leverage=2.0)
    assert pos_s2 is not None and pos_d is not None
    assert pos_s2.margin == pytest.approx(0.50 * pos_d.margin, rel=0.02)


# ---------------------------------------------------------------------------
# 3. DIRECTIONAL: trava MAX_SAME_DIR_MAJORS=2
# ---------------------------------------------------------------------------
class _SwingLoader:
    def __init__(self, last_ts: int):
        self.last_ts = last_ts

    async def get_candles(self, sym, tf, limit=400, exchange=None, **kw):
        n = 260
        ts = self.last_ts - SWING_BAR_MS * np.arange(n)[::-1]
        close = np.linspace(105.0, 100.0, n)
        return pd.DataFrame({
            "timestamp": ts, "open": close + 0.2, "high": close + 0.5,
            "low": close - 0.5, "close": close, "volume": np.full(n, 100.0),
        })

    async def close(self):
        pass


class _MatchTape:
    def evaluate(self, symbol, direction, now_ms=None, allow_live_fallback=True):
        return {
            "vote": direction, "status": "OK", "reason": "ok",
            "features": {"taker_imb_5m": 0.3, "book_imb_10": 0.2, "microprice_dev": 0.5, "spread_bps": 1.0},
        }


def test_directional_corr_cap_blocks_third_same_dir_major(tmp_path):
    ts_bar = _recent_bar_ts(SWING_BAR_MS)

    def _make_swing_daemon(db_name: str, out_dir: str = "SELL"):
        eng = PaperTradingEngine(initial_capital=10_000.0, db_path=str(tmp_path / db_name), restore=False)
        d = PaperTradingDaemon(
            engine=eng,
            loader=_SwingLoader(ts_bar),
            golden_pot_agent=object(),
            tape_agent=_MatchTape(),
            require_strict_tape_majors=True,
        )

        def _eval_ms(**kw):
            sym = kw["symbol"]
            px = float(kw["current_price"])
            sgn = 1.0 if out_dir == "BUY" else -1.0
            sg = TradeSignal(
                signal_id=f"sw-{sym}-{out_dir}", timestamp=kw["timestamp"], symbol=sym, direction=out_dir,
                entry_price=px, stop_loss=px - sgn * 1.0, take_profit=px + sgn * 4.0, atr=1.0,
                confidence=0.85, consensus_score=0.55 * sgn, position_size=1.0, risk_dollars=1.0, rationale="mock",
            )
            return {"directional_signal": sg.to_dict()}

        d.engine.core.evaluate_multi_strategy = _eval_ms
        return d

    # Caso A: 2 SHORT abertos (BTCUSDT, ETHUSDT) -> bloqueia 3o SHORT (SOLUSDT)
    d1 = _make_swing_daemon("cap2_a.db", "SELL")
    d1.engine.execute_directional_signal(_sig("BTCUSDT", "SELL", 100.0, 1.0, sl_mult=1.0), 100.0, ts_bar - 2 * SWING_BAR_MS)
    d1.engine.execute_directional_signal(_sig("ETHUSDT", "SELL", 100.0, 1.0, sl_mult=1.0), 100.0, ts_bar - 2 * SWING_BAR_MS)
    _run(d1.evaluate_swing_opportunities())
    open_syms_a = {p.symbol for p in d1.engine.positions.values() if p.status == "OPEN"}
    assert open_syms_a == {"BTCUSDT", "ETHUSDT"}

    # Caso B: apenas 1 SHORT aberto (BTCUSDT) -> 2o SHORT (ETHUSDT) entra, 3o (SOLUSDT) bloqueia
    d2 = _make_swing_daemon("cap2_b.db", "SELL")
    d2.engine.execute_directional_signal(_sig("BTCUSDT", "SELL", 100.0, 1.0, sl_mult=1.0), 100.0, ts_bar - 2 * SWING_BAR_MS)
    _run(d2.evaluate_swing_opportunities())
    open_syms_b = {p.symbol for p in d2.engine.positions.values() if p.status == "OPEN"}
    assert open_syms_b == {"BTCUSDT", "ETHUSDT"}

    # Caso C: 2 SHORT abertos (BTC, ETH) mas sinal BUY em SOL -> direção oposta entra
    d3 = _make_swing_daemon("cap2_c.db", "BUY")
    d3.engine.execute_directional_signal(_sig("BTCUSDT", "SELL", 100.0, 1.0, sl_mult=1.0), 100.0, ts_bar - 2 * SWING_BAR_MS)
    d3.engine.execute_directional_signal(_sig("ETHUSDT", "SELL", 100.0, 1.0, sl_mult=1.0), 100.0, ts_bar - 2 * SWING_BAR_MS)
    _run(d3.evaluate_swing_opportunities())
    open_syms_c = {p.symbol for p in d3.engine.positions.values() if p.status == "OPEN"}
    assert open_syms_c == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


# ---------------------------------------------------------------------------
# 4. Gate estrito em majors vs híbrido M4 (0.50x) no GOLDEN_POT
# ---------------------------------------------------------------------------
class _UnavailTape:
    def evaluate(self, symbol, direction, now_ms=None, allow_live_fallback=True):
        return {"vote": "HOLD", "status": "UNAVAILABLE", "reason": "symbol_not_covered", "features": {}}


def test_flow_gate_strict_majors_vs_gp_hybrid_half_size(tmp_path):
    eng = PaperTradingEngine(initial_capital=10_000.0, db_path=str(tmp_path / "fg.db"), restore=False)
    d = PaperTradingDaemon(
        engine=eng, loader=object(), golden_pot_agent=object(), tape_agent=_UnavailTape(),
        require_strict_tape_majors=True,
    )
    sig = _sig("BTCUSDT", "BUY", 100.0, 1.0)
    assert d._flow_gate("DIRECTIONAL", "BTCUSDT", "BUY", sig, 2.0, 100.0, 1.0) is False
    assert d._flow_gate("SCALP", "ETHUSDT", "BUY", sig, 2.0, 100.0, 1.0) is False
    assert d._flow_gate("GOLDEN_POT", "PEPEUSDT", "BUY", sig, 2.0, 100.0, 1.0) is True

    # Verifica redução de 0.50x na margem do GOLDEN_POT com risk_per_trade_pct=0.01
    eng_full = _make_engine(str(tmp_path / "gp_full.db"), 0.01)
    eng_half = _make_engine(str(tmp_path / "gp_half.db"), 0.01)
    p_full = eng_full.execute_golden_pot_signal(_sig("PEPEUSDT", "SELL", 100.0, 1.0, sl_mult=1.5), 100.0, 1_700_000_000_000, leverage=2.0, risk_scale_mult=1.0)
    p_half = eng_half.execute_golden_pot_signal(_sig("PEPEUSDT", "SELL", 100.0, 1.0, sl_mult=1.5), 100.0, 1_700_000_000_000, leverage=2.0, risk_scale_mult=0.50)
    assert p_full is not None and p_half is not None
    assert p_half.margin == pytest.approx(0.50 * p_full.margin, rel=1e-4)


# ---------------------------------------------------------------------------
# 5. GOLDEN_POT: alinhamento Macro BTC ±1.5% SMA200
# ---------------------------------------------------------------------------
def _gp_momentum_df(ts_last: int, side: str = "BUY", close: float = 100.0, sma: float = 95.0, atr: float = 1.0) -> pd.DataFrame:
    n = 130
    sgn = 1.0 if side == "BUY" else -1.0
    ts = ts_last - GP_BAR_MS * np.arange(n)[::-1]
    base = close - sgn * 3.0
    df = pd.DataFrame({
        "timestamp": ts, "open": base, "high": base + 0.5, "low": base - 0.5, "close": base, "volume": 1.0,
        "sma_200": sma, "rsi_14": 55.0 if side == "BUY" else 45.0, "atr_14": atr / 3.0,
        "volume_zscore_20": 0.0, "dist_vwap_pct": 0.0,
        "liquidity_sweep_bull": 0, "liquidity_sweep_bear": 0, "at_bullish_poi": 0, "at_bearish_poi": 0,
    })
    i = df.index[-1]
    df.loc[i, ["open", "close"]] = [close - sgn * 2.4, close]
    df.loc[i, "high"] = close + 0.1 if side == "BUY" else close + 2.5
    df.loc[i, "low"] = close - 2.5 if side == "BUY" else close - 0.1
    df.loc[i, "atr_14"] = atr
    df.loc[i, "volume_zscore_20"] = 3.0
    return df


class _OneScanner:
    async def scan_market(self, force_refresh=False):
        return {"top_swings": [{"symbol": "PEPEUSDT", "volume_24h_usd": 500e6}], "top_scalps": []}

    async def close(self):
        pass


def test_gp_btc_macro_gate_blocks_counter_trend(tmp_path):
    ts_gp = _recent_bar_ts(GP_BAR_MS)

    def _run_gp(side: str, btc_dist: float, db_name: str):
        eng = _make_engine(str(tmp_path / db_name), 0.01)
        d = PaperTradingDaemon(
            engine=eng, loader=object(), golden_pot_agent=_OneScanner(), tape_agent=_MatchTape(),
            enforce_gp_btc_macro=True,
        )
        d.stale_bar_max_age_ms = 0
        gp_df = _gp_momentum_df(ts_gp, side=side)

        async def _fetch_gp(sym, now_ts):
            return gp_df

        async def _btc_dist(sym, now_ts):
            return btc_dist

        async def _comm(context, symbol="X", current_price=0.0, atr=0.0, timestamp=0, use_llm=False):
            return _sig(symbol, side, current_price, atr)

        d._gp_fetch_closed_candles = _fetch_gp
        d._compute_swing_features = lambda df: df.copy()
        d._get_symbol_1h_dist_sma200 = _btc_dist
        d.engine.core.evaluate_market_async = _comm
        _run(d.evaluate_golden_pot_opportunities())
        return [t for t in d.engine.trade_history if t.action.startswith("OPEN_GOLDEN_POT_")]

    assert _run_gp("BUY", -0.025, "gp_buy_blk.db") == []     # BUY bloqueado com BTC -2.5%
    assert _run_gp("SELL", +0.025, "gp_sell_blk.db") == []   # SELL bloqueado com BTC +2.5%
    assert len(_run_gp("SELL", -0.025, "gp_sell_ok.db")) == 1  # SELL alinhado permitido


# ---------------------------------------------------------------------------
# 6. SCALP Squeeze BB/Keltner + ML 15m (>= 0.45) universal em BTC/ETH/SOL
# ---------------------------------------------------------------------------
def _squeeze_df(ts_last: int, vol_z: float = 1.5, atr_val: float = 0.45) -> pd.DataFrame:
    n = 35
    ts = ts_last - SCALP_BAR_MS * np.arange(n)[::-1]
    close = np.full(n, 100.0, dtype=float)
    open_ = np.full(n, 100.0, dtype=float)
    high = np.full(n, 100.05, dtype=float)
    low = np.full(n, 99.95, dtype=float)
    # Barra final rompe Keltner superior (kc_hi ~= 100 + 1.5*0.45 = 100.675)
    open_[-1] = 100.1
    close[-1] = 101.2
    high[-1] = 101.3
    low[-1] = 100.0
    vz = np.full(n, 0.1, dtype=float)
    vz[-1] = vol_z
    return pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high, "low": low, "close": close,
        "volume": np.full(n, 100.0), "atr_14": np.full(n, atr_val), "sma_200": np.full(n, 99.0),
        "volume_zscore_20": vz,
    })


def test_scalp_squeeze_keltner_and_ml15_all_majors_eligible(tmp_path):
    # Barra 15m fechada há 10s (< SCALP_STALE_BAR_MAX_AGE_MS = 180s)
    ts_15 = int(time.time() * 1000) - SCALP_BAR_MS - 10_000
    sq_df = _squeeze_df(ts_15)
    trig = PaperTradingDaemon.scalp_squeeze_keltner_trigger(sq_df, h1_dist_sma200=0.01)
    assert trig is not None and trig[0] == "BUY"

    # Garante que se os modelos 15m existem no repo, são carregados
    eng0 = PaperTradingEngine(initial_capital=10_000.0, db_path=str(tmp_path / "sc0.db"), restore=False)
    d0 = PaperTradingDaemon(engine=eng0, loader=object(), golden_pot_agent=object(), tape_agent=_MatchTape())
    if os.path.exists(os.path.join(ROOT, "models", "lgbm_15m_v1.joblib")):
        assert len(d0._scalp_ml15_models) >= 1
        probs = d0._predict_scalp_ml15_probs(sq_df.iloc[-1])
        assert probs is not None and len(probs) == 2
        assert all(0.0 <= p <= 1.0 and np.isfinite(p) for p in probs)

    def _run_scalp(prob_pair, only_btc: bool, db_name: str):
        eng = _make_engine(str(tmp_path / db_name), 0.01)
        d = PaperTradingDaemon(
            engine=eng, loader=object(), golden_pot_agent=object(), tape_agent=_MatchTape(),
            require_strict_tape_majors=True, use_scalp_squeeze_ml15=True,
        )
        evaluated = []
        ml_calls = []

        async def _fetch_15(sym, now_ts):
            evaluated.append(sym)
            if only_btc and sym != "BTCUSDT":
                return None
            return sq_df

        async def _h1(sym, now_ts):
            return 0.01

        async def _comm(context, symbol="X", current_price=0.0, atr=0.0, timestamp=0, use_llm=False):
            return _sig(symbol, "BUY", current_price, atr)

        def _pred(latest):
            ml_calls.append(prob_pair)
            return prob_pair

        d._scalp_fetch_closed_candles = _fetch_15
        d._compute_swing_features = lambda df: df.copy()
        d._get_symbol_1h_dist_sma200 = _h1
        d._predict_scalp_ml15_probs = _pred
        d.engine.core.evaluate_market_async = _comm
        _run(d.evaluate_scalp_opportunities())
        return evaluated, ml_calls, [t for t in d.engine.trade_history if t.action.startswith("OPEN_SCALP_")]

    ev_blk, ml_blk, tr_blk = _run_scalp((0.40, 0.20), only_btc=False, db_name="sc_blk.db")
    assert set(ev_blk) == {"ETHUSDT", "SOLUSDT", "BTCUSDT"}
    assert len(ml_blk) == 3   # todas as 3 moedas chegaram no filtro ML 15m e foram barradas (< 0.45)
    assert tr_blk == []

    # Com prob >= 0.45 e apenas BTCUSDT rompendo, BTCUSDT entra normalmente (sem isolamento)
    ev_btc, ml_btc, tr_btc = _run_scalp((0.52, 0.20), only_btc=True, db_name="sc_btc.db")
    assert set(ev_btc) == {"ETHUSDT", "SOLUSDT", "BTCUSDT"}
    assert len(ml_btc) == 1
    assert len(tr_btc) == 1 and tr_btc[0].symbol == "BTCUSDT"


def test_sqlite_fd_no_leak_trackers(tmp_path):
    """Garante que conexões SQLite via db_conn fecham os descritores de arquivo após o bloco with."""
    import gc
    from auth_module import AuthManager
    from learning_tracker import LearningTracker
    from performance_tracker import PerformanceTracker

    db = tmp_path / "leak_check.db"
    lt = LearningTracker(db_path=str(db))
    pt = PerformanceTracker(db_path=str(db))
    pt_ro = PerformanceTracker(db_path=str(db), read_only=True)
    auth = AuthManager(db_path=str(db))

    gc.collect()
    fd_before = len(os.listdir("/proc/self/fd"))

    for _ in range(40):
        with lt._get_connection():
            pass
        lt.get_calibrated_weights_for_regime("BEAR_TREND")
        lt.get_agent_calibration_for_regime("BEAR_TREND")
        with pt._get_connection():
            pass
        with pt_ro._get_connection():
            pass
        with auth._get_connection():
            pass

    gc.collect()
    fd_after = len(os.listdir("/proc/self/fd"))
    assert fd_after - fd_before <= 2

