import asyncio
import sqlite3
import time
import warnings
from types import SimpleNamespace

import pytest

from agents.technical_agent import TechnicalAgent, _disable_estimator_parallelism
from smart_trade import PaperTradingEngine, FUNNEL_REJECT_REASONS
from rollback_guard import RollbackGuard, DefensiveRiskPolicy, RiskPolicy
from scripts.run_paper_simulation import PaperTradingDaemon
from performance_tracker import PerformanceTracker
import flask_interface


_MISSING = object()

_BASE_FEATURES = {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "timestamp": 1000,
    "open": 104.0,
    "high": 106.0,
    "low": 103.0,
    "close": 105.0,
    "volume": 1000.0,
    "sma_200": 100.0,
    "dist_sma_200": 0.05,
    "rsi_14": 45.0,
    "macd": 0.0,
    "macd_signal": 0.0,
    "macd_hist": 0.0,
    "atr_14": 1.0,
    "atr": 1.0,
    "volume_zscore_20": 1.0,
    "dist_support_pct": 1.0,
    "dist_resistance_pct": 1.0,
    "liquidity_sweep_bull": 0,
    "liquidity_sweep_bear": 0,
    "macro_bull": True,
    "ema_50": 104.0,
    "ema_20": 104.0,
    "adx_14": 20.0,
}


def _value(result, key):
    if isinstance(result, dict):
        return result[key]
    return getattr(result, key)


def _call_evaluate(agent, features):
    try:
        return agent.evaluate(**features)
    except TypeError as exc:
        msg = str(exc)
        if "market_data" not in msg and "positional" not in msg and "unexpected keyword" not in msg:
            raise
        try:
            return agent.evaluate(features)
        except TypeError:
            return agent.evaluate(market_data=features)


def _evaluate(agent, **overrides):
    features = dict(_BASE_FEATURES)
    for key, value in overrides.items():
        if value is _MISSING:
            features.pop(key, None)
        else:
            features[key] = value
    return _call_evaluate(agent, features)


def _insert_flexible(conn, table, values):
    info = list(conn.execute(f"PRAGMA table_info({table})"))
    names = {row[1] for row in info}
    row = {k: v for k, v in values.items() if k in names}
    for cid, name, typ, notnull, default, pk in info:
        if name in row or pk or default is not None or not notnull:
            continue
        upper = (typ or "").upper()
        if "INT" in upper:
            row[name] = 0
        elif "TEXT" in upper:
            row[name] = ""
        else:
            row[name] = 0.0
    if row:
        sql = f"INSERT INTO {table} ({','.join(row)}) VALUES ({','.join('?' for _ in row)})"
        conn.execute(sql, list(row.values()))


def test_counter_trend_sweeps_are_blocked():
    agent = TechnicalAgent(confidence_threshold=0.55, enable_sweep_trigger=True)

    r = _evaluate(
        agent,
        macro_bull=False,
        close=95.0,
        sma_200=100.0,
        dist_sma_200=-0.05,
        rsi_14=55.0,
        liquidity_sweep_bull=1,
        dist_support_pct=0.2,
        volume_zscore_20=2.0,
    )
    assert _value(r, "direction") == "HOLD"

    r = _evaluate(
        agent,
        macro_bull=True,
        close=105.0,
        sma_200=100.0,
        dist_sma_200=0.05,
        rsi_14=45.0,
        liquidity_sweep_bear=1,
        dist_resistance_pct=0.2,
        volume_zscore_20=2.0,
    )
    assert _value(r, "direction") == "HOLD"


@pytest.mark.parametrize(
    "volume_z, expected_direction, expected_conf",
    [
        (0.4, "HOLD", None),
        (1.1, "BUY", 0.68),
        (1.6, "BUY", 0.72),
        (_MISSING, "BUY", 0.68),
    ],
)
def test_aligned_bull_sweep_gating(volume_z, expected_direction, expected_conf):
    agent = TechnicalAgent(confidence_threshold=0.55, enable_sweep_trigger=True)
    r = _evaluate(
        agent,
        macro_bull=True,
        close=105.0,
        sma_200=100.0,
        dist_sma_200=0.05,
        rsi_14=45.0,
        liquidity_sweep_bull=1,
        dist_support_pct=0.3,
        volume_zscore_20=volume_z,
    )
    assert _value(r, "direction") == expected_direction
    if expected_conf is not None:
        assert _value(r, "confidence") == pytest.approx(expected_conf)


def test_aligned_bear_sweep_gating():
    agent = TechnicalAgent(confidence_threshold=0.55, enable_sweep_trigger=True)
    r = _evaluate(
        agent,
        macro_bull=False,
        close=95.0,
        sma_200=100.0,
        dist_sma_200=-0.05,
        rsi_14=55.0,
        liquidity_sweep_bear=1,
        dist_resistance_pct=0.2,
        volume_zscore_20=1.2,
    )
    assert _value(r, "direction") == "SELL"
    assert _value(r, "confidence") == pytest.approx(0.68)


def test_disable_estimator_parallelism_handles_cycles():
    est = SimpleNamespace(n_jobs=12)
    inner = SimpleNamespace(n_jobs=12)
    wrapper = SimpleNamespace(n_jobs=12, estimator=inner)
    child_a = SimpleNamespace(n_jobs=12)
    child_b = SimpleNamespace(n_jobs=12)
    holder = SimpleNamespace(n_jobs=12, estimators_=[child_a, child_b])
    pipe = SimpleNamespace(n_jobs=12)
    pipe.steps = [("self", pipe)]

    for obj in (est, wrapper, holder, pipe):
        _disable_estimator_parallelism(obj)

    assert est.n_jobs == 1
    assert wrapper.n_jobs == 1 and inner.n_jobs == 1
    assert holder.n_jobs == 1 and child_a.n_jobs == 1 and child_b.n_jobs == 1
    assert pipe.n_jobs == 1


def test_evaluate_emits_no_sklearn_parallel_warnings():
    agent = TechnicalAgent(confidence_threshold=0.55)
    features = dict(_BASE_FEATURES)
    model = getattr(agent, "model", None)
    names = getattr(model, "feature_names", None)
    if names is None:
        names = getattr(model, "feature_names_in_", None)
    if names is not None:
        for name in names:
            features.setdefault(str(name), 0.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _call_evaluate(agent, features)
    assert not [w for w in caught if "sklearn.utils.parallel" in str(w.message)]


class _LoaderSpy:
    def __init__(self):
        self.get_candles_calls = 0
        self.close_calls = 0

    async def get_candles(self, *args, **kwargs):
        self.get_candles_calls += 1
        return []

    async def close(self):
        self.close_calls += 1


class _ScannerSpy:
    def __init__(self):
        self.scan_calls = 0

    async def scan_market(self, *args, **kwargs):
        self.scan_calls += 1
        return []

    async def scan_golden_pot(self, *args, **kwargs):
        self.scan_calls += 1
        return []


class _TapeSpy:
    def __init__(self):
        self.calls = 0

    async def scan_market(self, *args, **kwargs):
        self.calls += 1
        return []

    async def scan_tape(self, *args, **kwargs):
        self.calls += 1
        return []

    async def get_tape(self, *args, **kwargs):
        self.calls += 1
        return []

    async def evaluate(self, *args, **kwargs):
        self.calls += 1
        return None


def _wrap_core_calls(daemon):
    core = getattr(getattr(daemon, "engine", None), "core", None)
    if core is None:
        return
    calls = []
    for name in dir(core):
        if not name.startswith(("evaluate_", "process_", "execute_")):
            continue
        original = getattr(core, name)
        if not callable(original):
            continue
        if asyncio.iscoroutinefunction(original):
            async def wrapper(*args, _original=original, **kwargs):
                calls.append(name)
                return await _original(*args, **kwargs)
        else:
            def wrapper(*args, _original=original, **kwargs):
                calls.append(name)
                return _original(*args, **kwargs)
        setattr(core, name, wrapper)
    core._p0_calls = calls


def test_breaker_short_circuits_daemon_and_is_flushed(tmp_path):
    db = str(tmp_path / "cb.db")
    eng = PaperTradingEngine(initial_capital=10000.0, db_path=db, restore=False)
    guard = RollbackGuard(db_path=db)
    eng.rollback_guard = guard

    guard.evaluate_telemetry({"current_drawdown_pct": 12.0, "total_trades": 5})
    assert guard.status.state == "TRIPPED"
    assert guard.status.is_trading_allowed is False

    loader = _LoaderSpy()
    scanner = _ScannerSpy()
    tape = _TapeSpy()
    daemon = PaperTradingDaemon(
        engine=eng,
        loader=loader,
        golden_pot_agent=scanner,
        tape_agent=tape,
    )
    _wrap_core_calls(daemon)

    asyncio.run(daemon.evaluate_swing_opportunities())
    asyncio.run(daemon.evaluate_scalp_opportunities())
    asyncio.run(daemon.evaluate_golden_pot_opportunities())
    assert loader.get_candles_calls == 0
    assert scanner.scan_calls == 0

    eng.funnel_flush()
    with sqlite3.connect(db) as conn:
        stages = [row[0] for row in conn.execute("SELECT stage FROM signal_funnel")]
    assert any("reject:breaker" in str(stage) for stage in stages)

    guard.status.state = "CLEAR"
    guard.status.is_trading_allowed = True
    asyncio.run(daemon.evaluate_swing_opportunities())
    assert loader.get_candles_calls > 0


def _seed_performance_db(db):
    conn = sqlite3.connect(db)
    try:
        for i in range(1, 33):
            pnl = 50.0 if i <= 20 else -40.0
            ts = 1000.0 + i
            _insert_flexible(
                conn,
                "paper_positions",
                {
                    "id": i,
                    "position_id": i,
                    "trade_id": i,
                    "symbol": "BTCUSDT",
                    "side": "LONG",
                    "direction": "LONG",
                    "status": "CLOSED",
                    "state": "CLOSED",
                    "entry_price": 100.0,
                    "exit_price": 100.0,
                    "opened_price": 100.0,
                    "closed_price": 100.0,
                    "quantity": 1.0,
                    "pnl": pnl,
                    "realized_pnl": pnl,
                    "opened_at": ts,
                    "closed_at": ts,
                    "timestamp": ts,
                },
            )
            _insert_flexible(
                conn,
                "paper_trades",
                {
                    "trade_id": 1000 + i,
                    "position_id": i,
                    "symbol": "BTCUSDT",
                    "action": "CLOSE_LONG",
                    "price": 100.0,
                    "quantity": 1.0,
                    "fee": 0.0,
                    "pnl": pnl,
                    "timestamp": ts,
                    "rationale": "test",
                },
            )
        conn.commit()
    finally:
        conn.close()


def _read_events(db, guard):
    events = getattr(guard, "circuit_breaker_events", None)
    if events is not None:
        return [str(getattr(event, "event", event)) for event in events]
    with sqlite3.connect(db) as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(circuit_breaker_events)")]
        if not cols:
            return []
        col = "event" if "event" in cols else "event_type" if "event_type" in cols else cols[0]
        return [str(row[0]) for row in conn.execute(f"SELECT {col} FROM circuit_breaker_events")]


def test_performance_degradation_and_recovery_round_trip(tmp_path):
    db = str(tmp_path / "perf.db")
    PaperTradingEngine(initial_capital=10000.0, db_path=db, restore=False)
    _seed_performance_db(db)

    m = PerformanceTracker(db_path=db).compute_metrics_from_db()
    assert m.win_rate > 50.0
    assert m.profit_factor > 1.5
    assert m.rolling_trades_count == 15
    assert m.rolling_win_rate == pytest.approx(20.0)
    assert m.rolling_profit_factor < 0.60

    rb_db = str(tmp_path / "rb.db")
    guard = RollbackGuard(
        db_path=rb_db,
        max_total_drawdown_pct=50.0,
        max_consecutive_losses=50,
    )
    telemetry = {
        "total_trades": getattr(m, "total_trades", 0),
        "win_rate": getattr(m, "win_rate", 0.0),
        "profit_factor": getattr(m, "profit_factor", 0.0),
        "rolling_trades_count": getattr(m, "rolling_trades_count", 0),
        "rolling_win_rate": getattr(m, "rolling_win_rate", 0.0),
        "rolling_profit_factor": getattr(m, "rolling_profit_factor", 0.0),
        "current_drawdown_pct": 0.0,
    }
    guard.evaluate_telemetry(telemetry)
    assert guard.status.is_rollback_active is True
    assert isinstance(guard.get_risk_policy(), DefensiveRiskPolicy)

    guard.evaluate_telemetry(
        {
            "total_trades": 32,
            "win_rate": 62.5,
            "profit_factor": 1.6,
            "rolling_trades_count": 15,
            "rolling_win_rate": 60.0,
            "rolling_profit_factor": 1.5,
            "current_drawdown_pct": 1.0,
        }
    )
    assert guard.status.is_rollback_active is False
    assert isinstance(guard.get_risk_policy(), RiskPolicy)
    assert "DEGRADATION_ROLLBACK_RECOVERED" in _read_events(rb_db, guard)


def test_funnel_reject_reasons_are_exact(tmp_path):
    db = str(tmp_path / "funnel.db")
    eng = PaperTradingEngine(initial_capital=10000.0, db_path=db, restore=False)

    for reason in (
        "stale_bar",
        "stale_candle",
        "climax_exhaustion",
        "confidence",
        "stale_data",
        "tape_veto",
        "kill_switch",
    ):
        assert reason in FUNNEL_REJECT_REASONS

    eng.funnel_reject("SCALP", "BTCUSDT", 1000, "stale_bar")
    eng.funnel_flush()
    with sqlite3.connect(db) as conn:
        stages = [row[0] for row in conn.execute("SELECT stage FROM signal_funnel")]
    assert any(str(stage) == "reject:stale_bar" for stage in stages)
    assert not any(str(stage) == "reject:other" for stage in stages)


def test_flask_refresh_and_chart_always_close_loader(tmp_path, monkeypatch):
    import data.data_loader as data_loader_mod

    db = str(tmp_path / "flask.db")
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE candles ("
            "exchange TEXT, symbol TEXT, timeframe TEXT, timestamp INTEGER, "
            "open REAL, high REAL, low REAL, close REAL, volume REAL, quote_volume REAL)"
        )
    monkeypatch.setattr(flask_interface, "DB_PATH", db)
    monkeypatch.setattr(flask_interface, "fetch_binance_open_candle", lambda *a, **k: None)
    monkeypatch.setattr(flask_interface, "get_realtime_ticker", lambda *a, **k: {})

    class LoaderSpy:
        fail = False
        close_calls = 0
        instances = []

        def __init__(self, *args, **kwargs):
            self.__class__.instances.append(self)

        async def get_candles(self, *args, **kwargs):
            if self.__class__.fail:
                raise RuntimeError("boom")
            return [
                {
                    "timestamp": 1000 + i,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 10.0,
                    "quote_volume": 1000.0,
                }
                for i in range(3)
            ]

        async def close(self):
            self.__class__.close_calls += 1

    monkeypatch.setattr(data_loader_mod, "DataLoader", LoaderSpy)
    if hasattr(flask_interface, "DataLoader"):
        monkeypatch.setattr(flask_interface, "DataLoader", LoaderSpy)

    client = flask_interface.app.test_client()
    endpoints = (
        "/api/refresh-symbol/BTCUSDT/1h",
        "/api/chart/BTCUSDT/1h?refresh=1",
    )
    for fail in (False, True):
        for endpoint in endpoints:
            LoaderSpy.fail = fail
            LoaderSpy.close_calls = 0
            response = client.get(endpoint)
            assert response.status_code != 404
            assert LoaderSpy.close_calls == 1
