"""Funil de sinais (instrumentação): DB temporário + mocks; nenhuma mudança de comportamento de trading."""
import asyncio
import os
import sqlite3
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import smart_trade as st  # noqa: E402
import test_golden_pot_v2 as g  # noqa: E402  (helpers: make_daemon, _mfeat, _committee, _bar_ts, run, temp_db)
from signal_generator import TradeSignal  # noqa: E402

temp_db = g.temp_db  # fixture reutilizada


def table(db):
    with sqlite3.connect(db) as c:
        return {(s, st_): n for _, s, st_, n in c.execute("SELECT hour_ms,strategy,stage,count FROM signal_funnel")}


def gp(db, feat=None, committee=None, symbols=("AAAUSDT",)):
    d = g.make_daemon(db, list(symbols), feat if feat is not None else g._mfeat(g._bar_ts()),
                      committee or g._committee("BUY"))
    return d


def flush_table(d):
    d.engine.funnel_flush()
    return table(d.engine.db_path)


def test_constants_contract():
    assert st.FUNNEL_SUCCESS_STAGES == ["evaluated", "candidate", "committee_agree", "confidence_ok", "risk_ok",
                                        "gate_passed", "executed"]
    assert st.FUNNEL_REJECT_REASONS == ["no_data", "no_trigger", "committee_disagree", "low_confidence", "veto",
                                        "cooldown", "bar_limit", "quota", "margin", "breaker", "gate_blocked",
                                        "excluded_symbol", "other"]


def test_table_schema(temp_db):
    e = g.make_engine(temp_db)
    with sqlite3.connect(temp_db) as c:
        cols = [(r[1], r[2], r[5]) for r in c.execute("PRAGMA table_info(signal_funnel)")]
    assert cols == [("hour_ms", "INTEGER", 1), ("strategy", "TEXT", 2), ("stage", "TEXT", 3), ("count", "INTEGER", 0)] \
        or {c[0] for c in cols} == {"hour_ms", "strategy", "stage", "count"}
    del e


def test_dedupe_per_bar(temp_db):
    e = g.make_engine(temp_db)
    assert e.funnel_inc("SCALP", "BTCUSDT", 1000, "evaluated") is True
    assert e.funnel_inc("SCALP", "BTCUSDT", 1000, "evaluated") is False
    assert e.funnel_inc("SCALP", "BTCUSDT", 2000, "evaluated") is True      # outra barra
    assert e.funnel_inc("SCALP", "ETHUSDT", 1000, "evaluated") is True      # outro símbolo
    assert e.funnel_inc("SCALP", "BTCUSDT", 1000, "candidate") is True      # outro estágio
    e.funnel_flush()
    assert table(temp_db)[("SCALP", "evaluated")] == 3


def test_upsert_sums_per_hour_and_new_hour_new_row(temp_db):
    e = g.make_engine(temp_db)
    t = [3600.0 * 100 + 5]
    e._funnel_clock = lambda: t[0]
    e.funnel_inc("SCALP", "A", 1, "evaluated")
    e.funnel_flush()
    e.funnel_inc("SCALP", "A", 2, "evaluated")
    e.funnel_flush()
    t[0] += 3600
    e.funnel_inc("SCALP", "A", 3, "evaluated")
    e.funnel_flush()
    with sqlite3.connect(temp_db) as c:
        rows = c.execute("SELECT hour_ms,count FROM signal_funnel ORDER BY hour_ms").fetchall()
    assert rows == [(3600 * 100 * 1000, 2), (3600 * 101 * 1000, 1)]


def test_dedupe_pruned_by_age(temp_db):
    e = g.make_engine(temp_db)
    t = [1_000_000.0]
    e._funnel_clock = lambda: t[0]
    e.funnel_inc("SCALP", "A", 1, "evaluated")
    t[0] += st.FUNNEL_DEDUPE_MAX_AGE_S + 10
    e.funnel_flush()
    assert e._funnel_seen == {}


def test_auto_flush_every_30s(temp_db):
    e = g.make_engine(temp_db)
    t = [5_000_000.0]
    e._funnel_clock = lambda: t[0]
    e._funnel_last_flush = t[0]
    e.funnel_inc("SCALP", "A", 1, "evaluated")
    assert table(temp_db) == {}
    t[0] += 31
    e.funnel_inc("SCALP", "A", 2, "evaluated")
    assert table(temp_db)[("SCALP", "evaluated")] == 2


def test_thread_safety(temp_db):
    e = g.make_engine(temp_db)

    def work(k):
        for i in range(200):
            e.funnel_inc("SCALP", f"S{k}", i, "evaluated")
    ts = [threading.Thread(target=work, args=(k,)) for k in range(5)]
    [x.start() for x in ts]
    [x.join() for x in ts]
    e.funnel_flush()
    assert table(temp_db)[("SCALP", "evaluated")] == 1000


def test_db_failure_never_breaks_and_retries(temp_db, monkeypatch):
    e = g.make_engine(temp_db)
    e.funnel_inc("SCALP", "A", 1, "evaluated")
    real = e._get_connection

    def boom():
        raise sqlite3.OperationalError("disk I/O")
    monkeypatch.setattr(e, "_get_connection", boom)
    e.funnel_flush()  # não levanta
    monkeypatch.setattr(e, "_get_connection", real)
    e.funnel_flush()
    assert table(temp_db)[("SCALP", "evaluated")] == 1


def test_trading_unaffected_when_funnel_raises(temp_db):
    d = gp(temp_db)

    def boom(*a, **k):
        raise RuntimeError("funil quebrou")
    d.engine.funnel_inc = boom
    d.engine.funnel_reject = boom
    g.run(d.evaluate_golden_pot_opportunities())
    assert len(g.gp_opens(d.engine)) == 1


def test_gp_full_path_executed_once_and_deduped(temp_db):
    d = gp(temp_db)
    g.run(d.evaluate_golden_pot_opportunities())
    g.run(d.evaluate_golden_pot_opportunities())  # mesma barra reavaliada
    t = flush_table(d)
    for stage in st.FUNNEL_SUCCESS_STAGES:
        assert t[("GOLDEN_POT", stage)] == 1, stage
    assert t[("GOLDEN_POT", "reject:margin")] == 1  # 2º ciclo: par já tem posição aberta
    assert len(g.gp_opens(d.engine)) == 1


@pytest.mark.parametrize("committee,reason", [
    (g._committee("BUY", veto=True), "veto"),
    (g._committee("SELL"), "committee_disagree"),
    (g._committee("BUY", conf=0.55), "low_confidence"),
])
def test_gp_committee_rejects(temp_db, committee, reason):
    d = gp(temp_db, committee=committee)
    g.run(d.evaluate_golden_pot_opportunities())
    t = flush_table(d)
    assert t[("GOLDEN_POT", f"reject:{reason}")] == 1
    assert ("GOLDEN_POT", "executed") not in t and ("GOLDEN_POT", "risk_ok") not in t
    assert t[("GOLDEN_POT", "candidate")] == 1


def test_gp_no_trigger(temp_db):
    d = gp(temp_db, feat=g._mfeat(g._bar_ts(), vol_z=0.0))
    g.run(d.evaluate_golden_pot_opportunities())
    t = flush_table(d)
    assert t[("GOLDEN_POT", "reject:no_trigger")] == 1 and ("GOLDEN_POT", "candidate") not in t


def test_gp_no_data_excluded_cooldown_other(temp_db):
    d = gp(temp_db)

    async def none(sym, now):
        return None
    d._gp_fetch_closed_candles = none
    g.run(d.evaluate_golden_pot_opportunities())
    assert flush_table(d)[("GOLDEN_POT", "reject:no_data")] == 1

    d = g.make_daemon(temp_db + "x", ["TSLAUSDT", "AAAUSDT"], g._mfeat(g._bar_ts()), g._committee("BUY"))
    try:
        d._gp_in_cooldown = lambda s, b: True
        g.run(d.evaluate_golden_pot_opportunities())
        t = flush_table(d)
        assert t[("GOLDEN_POT", "reject:excluded_symbol")] == 1
        assert t[("GOLDEN_POT", "reject:cooldown")] == 1

        async def bad(**kw):
            raise RuntimeError("x")
        d.engine.core.evaluate_market_async = bad
        d._gp_in_cooldown = lambda s, b: False
        g.run(d.evaluate_golden_pot_opportunities())
        assert flush_table(d)[("GOLDEN_POT", "reject:other")] == 1
    finally:
        for suf in ("", "-wal", "-shm"):
            if os.path.exists(temp_db + "x" + suf):
                os.remove(temp_db + "x" + suf)


def test_gp_gate_blocked_and_post_gate_margin_not_executed(temp_db):
    d = gp(temp_db)
    d._flow_gate = lambda *a, **k: False
    g.run(d.evaluate_golden_pot_opportunities())
    t = flush_table(d)
    assert t[("GOLDEN_POT", "reject:gate_blocked")] == 1 and ("GOLDEN_POT", "gate_passed") not in t

    d = gp(temp_db)
    d._flow_gate = lambda *a, **k: True

    def refuse(**kw):
        d.engine.last_open_reject = "margin"
        return None
    d.engine.execute_golden_pot_signal = refuse
    g.run(d.evaluate_golden_pot_opportunities())
    t = flush_table(d)
    assert t[("GOLDEN_POT", "gate_passed")] == 1
    assert ("GOLDEN_POT", "executed") not in t
    assert t[("GOLDEN_POT", "reject:margin")] == 1


def test_gp_quota_breaker_margin_precheck(temp_db):
    d = gp(temp_db)
    d.engine.can_execute_golden_pot = lambda ts, notional=0.0: (False, "Limite diário independente")
    g.run(d.evaluate_golden_pot_opportunities())
    d.engine.can_execute_golden_pot = lambda ts, notional=0.0: (False, "Bloqueado pelo Circuit Breaker (Fase 7): x")
    g.run(d.evaluate_golden_pot_opportunities())
    d.engine.can_execute_golden_pot = lambda ts, notional=0.0: (True, "OK")
    d.engine.max_concurrent_positions = 0
    g.run(d.evaluate_golden_pot_opportunities())
    t = flush_table(d)
    assert t[("GOLDEN_POT", "reject:quota")] == 1
    assert t[("GOLDEN_POT", "reject:breaker")] == 1
    assert t[("GOLDEN_POT", "reject:margin")] == 1


def test_engine_open_reject_reasons(temp_db):
    d = gp(temp_db)
    e = d.engine
    sig = TradeSignal(signal_id="c", timestamp=1, symbol="AAAUSDT", direction="BUY", entry_price=100.0,
                      stop_loss=99.0, take_profit=102.0, atr=1.0, confidence=0.8, consensus_score=0.5,
                      position_size=100.0, risk_dollars=1.0, rationale="x", is_vetoed=False)
    ts = g._bar_ts()
    e.rollback_guard = type("G", (), {"can_open_new_position": lambda self, *a, **k: (False, "cb")})()
    assert e.execute_golden_pot_signal(sig, 100.0, ts, leverage=1.0) is None and e.last_open_reject == "breaker"
    e.rollback_guard = None
    e.daily_golden_pots_count = 0
    e._check_or_reset_daily_quota(ts)
    e.daily_golden_pots_count = e.max_daily_golden_pots
    assert e.execute_golden_pot_signal(sig, 100.0, ts, leverage=1.0) is None and e.last_open_reject == "quota"
    e.daily_golden_pots_count = 0
    assert e.execute_golden_pot_signal(sig, 100.0, ts, leverage=1.0) is not None


def test_scalp_funnel_no_trigger_and_executed(temp_db):
    d = gp(temp_db)
    d.engine.max_concurrent_positions = 20

    async def fetch(sym, now):
        import pandas as pd
        return pd.DataFrame({"timestamp": [0]})
    d._fetch_closed_candles_15m = fetch
    d._scalp_fetch_closed_candles = fetch
    feat = g._feat(g._bar_ts(), side="BUY", close=100.0, rsi=20.0)
    feat["volume_zscore_20"] = 1.0
    d._compute_swing_features = lambda df: feat.copy()
    g.run(d.evaluate_scalp_opportunities())
    t = flush_table(d)
    assert t[("SCALP", "executed")] == 1
    assert t[("SCALP", "evaluated")] >= 1

    # sem gatilho (volume baixo) em barra nova => reject:no_trigger
    feat2 = g._feat(g._bar_ts() + g.BAR, side="BUY", close=100.0, rsi=50.0)
    feat2["volume_zscore_20"] = -5.0
    d._compute_swing_features = lambda df: feat2.copy()
    d.engine.positions.clear()
    g.run(d.evaluate_scalp_opportunities())
    assert flush_table(d)[("SCALP", "reject:no_trigger")] >= 1


def test_daytrade_rejects_and_executed(temp_db):
    d = gp(temp_db)
    d.core_symbols = ["BTCUSDT"]
    feat = g._feat(g._bar_ts(), n=210)

    class L:
        async def get_candles(self, *a, **k):
            import pandas as pd
            return pd.DataFrame({"timestamp": range(250)})
        async def close(self):
            pass
    d.loader = L()
    d._drop_forming_candle = staticmethod(lambda df, b, n: df)
    d._compute_swing_features = lambda df: feat.copy()
    base = dict(symbol="BTCUSDT", direction="HOLD", confidence=0.9, is_vetoed=False, entry_price=100.0,
                stop_loss=99.0, take_profit=102.0, atr=1.0, consensus_score=0.5, signal_id="s", timestamp=1,
                position_size=1.0, risk_dollars=1.0, rationale="r")
    res = {"directional_signal": base}
    d.engine.core.evaluate_multi_strategy = lambda **k: res
    g.run(d.evaluate_swing_opportunities())
    assert flush_table(d)[("DAY_TRADE", "reject:committee_disagree")] == 1
    res["directional_signal"] = {**base, "direction": "BUY", "is_vetoed": True}
    g.run(d.evaluate_swing_opportunities())
    res["directional_signal"] = {**base, "direction": "BUY", "confidence": 0.01}
    g.run(d.evaluate_swing_opportunities())
    t = flush_table(d)
    assert t[("DAY_TRADE", "evaluated")] == 1 and t[("DAY_TRADE", "candidate")] == 1
    assert t[("DAY_TRADE", "reject:veto")] == 1 and t[("DAY_TRADE", "reject:low_confidence")] == 1
    assert ("DAY_TRADE", "executed") not in t


def test_shutdown_flushes_buffer(temp_db):
    d = gp(temp_db)

    async def noop(*a, **k):
        return None
    d.update_active_positions = noop
    d.evaluate_golden_pot_opportunities = noop
    d.evaluate_scalp_opportunities = noop
    d.evaluate_swing_opportunities = noop
    d.scanner.close = noop
    d.engine.funnel_inc("SCALP", "A", 1, "evaluated")
    assert table(temp_db) == {}
    g.run(d.run_loop(max_cycles=1))
    assert table(temp_db)[("SCALP", "evaluated")] == 1
