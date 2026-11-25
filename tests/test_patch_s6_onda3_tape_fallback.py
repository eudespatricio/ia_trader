"""S6 Onda 3 / Frente B — TapeReading live REST fallback (cache, triggers, guards) + collector GP coverage.

Contrato congelado; DBs só em tmp_path; HTTP sempre interceptado; FDs via /proc/self/fd."""
import json
import os
import sqlite3
import sys
import urllib.error

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import tape_reading_agent as tra  # noqa: E402
from agents.tape_reading_agent import TapeReadingAgent  # noqa: E402

NOW = 1_800_000_000_000
SCHEMA = """
CREATE TABLE book_features (ts_ms INTEGER, symbol TEXT, mid REAL, spread_bps REAL, microprice_dev_bps REAL,
  bid_sz1 REAL, ask_sz1 REAL, bid_n2 REAL, ask_n2 REAL, bid_n5 REAL, ask_n5 REAL, bid_n10 REAL, ask_n10 REAL,
  bid_n25 REAL, ask_n25 REAL, bid_n50 REAL, ask_n50 REAL, bid_n100 REAL, ask_n100 REAL, PRIMARY KEY (ts_ms, symbol));
CREATE TABLE tick_agg (minute_ms INTEGER, symbol TEXT, buy_qty REAL, sell_qty REAL, buy_notional REAL,
  sell_notional REAL, n_trades INTEGER, vwap REAL, saturated INTEGER, PRIMARY KEY (minute_ms, symbol));
CREATE TABLE binance_series (ts_ms INTEGER, symbol TEXT, kind TEXT, v1 REAL, v2 REAL, v3 REAL,
  PRIMARY KEY (ts_ms, symbol, kind));
"""
# (age_s do minuto, buy_notional, sell_notional, n_trades, saturated)
GOOD_TICKS = [(10, 600.0, 400.0, 20, 0), (70, 600.0, 400.0, 20, 0), (130, 600.0, 400.0, 20, 0)]


def build_db(path, now=NOW, sym="BTCUSDT", book_age=10, ticks=GOOD_TICKS, series=None):
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA)
    if book_age is not None:
        for k in range(4):
            con.execute("INSERT INTO book_features (ts_ms,symbol,mid,spread_bps,microprice_dev_bps,bid_n10,ask_n10) "
                        "VALUES (?,?,?,?,?,?,?)", (now - int(book_age * 1000) - k * 5000, sym, 100.0, 1.0, 1.0, 600.0, 400.0))
    for age, buy, sell, n, sat in ticks:
        con.execute("INSERT INTO tick_agg VALUES (?,?,1,1,?,?,?,100,?)", (now - int(age * 1000), sym, buy, sell, n, sat))
    for age, v2, v3 in (series or []):
        con.execute("INSERT INTO binance_series VALUES (?,?,'taker',1,?,?)", (now - int(age * 1000), sym, v2, v3))
    con.commit()
    con.close()
    return str(path)


def open_fds(path):
    """FDs deste processo apontando para o DB (ou -wal/-shm)."""
    out = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            tgt = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if tgt.startswith(str(path)):
            out.append(tgt)
    return out


class FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Net:
    """Substitui urllib.request.urlopen: registra (url, timeout, fds do DB abertos no momento da chamada)."""

    def __init__(self, spread_bps=1.0, trade_age_ms=1000, n_trades=40, exc=None, watch_db=None):
        self.spread_bps, self.trade_age_ms, self.n_trades = spread_bps, trade_age_ms, n_trades
        self.exc, self.watch_db, self.calls = exc, watch_db, []

    def __call__(self, req, timeout=None, *a, **kw):
        url = getattr(req, "full_url", req)
        self.calls.append((url, timeout, open_fds(self.watch_db) if self.watch_db else []))
        if self.exc is not None:
            raise self.exc
        if "depth" in url:
            h = 0.005 * self.spread_bps
            return FakeResp({"bids": [[str(100 - h), "10"], [str(99.5), "5"]], "asks": [[str(100 + h), "10"], [str(100.5), "5"]]})
        trades = [{"T": NOW - self.trade_age_ms - i * 500, "p": "100", "q": "1", "m": bool(i % 3 == 0)}
                  for i in range(self.n_trades)][::-1]
        return FakeResp(trades)


@pytest.fixture
def net(monkeypatch):
    n = Net()
    monkeypatch.setattr(tra.urllib.request, "urlopen", n)
    return n


def live_agent(tmp_path, monkeypatch, **db_kw):
    p = build_db(tmp_path / "micro.db", **db_kw)
    monkeypatch.setattr(tra, "MICROSTRUCTURE_DB", p)
    return TapeReadingAgent(db_path=p)


def spy_fallback(agent, monkeypatch, watch_db=None):
    calls = []

    def fake(sym, ts, reason):
        calls.append((sym, ts, reason, open_fds(watch_db) if watch_db else []))
        return {"agent_name": "x", "symbol": sym, "vote": "HOLD", "status": "UNAVAILABLE", "confidence": 0.0,
                "features": {}, "reason": "SENTINEL:" + reason, "timestamp": ts}

    monkeypatch.setattr(agent, "_evaluate_live_binance_fallback", fake)
    return calls


def test_module_constants_and_ctor_defaults():
    assert tra.LIVE_FALLBACK_TIMEOUT_S == 2.5
    assert tra.LIVE_FALLBACK_CACHE_TTL_MS == 10_000
    assert tra.LIVE_FALLBACK_ERR_COOLDOWN_MS == 15_000
    a = TapeReadingAgent(db_path="/nonexistent/x.db")
    assert a._fallback_cache_ttl_ms == 10_000 and a._fallback_err_cooldown_ms == 15_000
    b = TapeReadingAgent(db_path="/nonexistent/x.db", fallback_cache_ttl_ms=7, fallback_err_cooldown_ms=9)
    assert (b._fallback_cache_ttl_ms, b._fallback_err_cooldown_ms) == (7, 9)


def test_fallback_uses_timeout_on_both_http_calls(net):
    r = TapeReadingAgent(db_path="/nonexistent/x.db")._evaluate_live_binance_fallback("DOGEUSDT", NOW, "symbol_not_covered")
    assert r["status"] == "OK" and r["features"]["source"] == "binance_live_rest"
    assert len(net.calls) == 2
    assert {c[1] for c in net.calls} == {tra.LIVE_FALLBACK_TIMEOUT_S}


def test_ok_cached_within_ttl_and_refetched_at_boundary(net):
    a = TapeReadingAgent(db_path="/nonexistent/x.db")
    first = a._evaluate_live_binance_fallback("DOGEUSDT", NOW, "r")
    assert len(net.calls) == 2
    second = a._evaluate_live_binance_fallback("DOGEUSDT", NOW + 9_999, "r")
    assert len(net.calls) == 2 and second["vote"] == first["vote"] and second["status"] == "OK"
    a._evaluate_live_binance_fallback("DOGEUSDT", NOW + 10_000, "r")  # age == ttl -> expirado
    assert len(net.calls) == 4


def test_cache_is_per_symbol(net):
    a = TapeReadingAgent(db_path="/nonexistent/x.db")
    a._evaluate_live_binance_fallback("DOGEUSDT", NOW, "r")
    a._evaluate_live_binance_fallback("LINKUSDT", NOW + 1, "r")
    assert len(net.calls) == 4
    assert {"DOGEUSDT" in c[0] for c in net.calls} == {True, False}


def test_negative_age_not_served_from_cache(net):
    a = TapeReadingAgent(db_path="/nonexistent/x.db")
    a._evaluate_live_binance_fallback("DOGEUSDT", NOW, "r")
    a._evaluate_live_binance_fallback("DOGEUSDT", NOW - 1, "r")  # relógio voltou: não usar cache
    assert len(net.calls) == 4


def test_ttl_zero_disables_ok_cache(net):
    a = TapeReadingAgent(db_path="/nonexistent/x.db", fallback_cache_ttl_ms=0)
    a._evaluate_live_binance_fallback("DOGEUSDT", NOW, "r")
    a._evaluate_live_binance_fallback("DOGEUSDT", NOW, "r")
    assert len(net.calls) == 4


def test_cached_result_isolated_from_caller_mutation(net):
    a = TapeReadingAgent(db_path="/nonexistent/x.db")
    r1 = a._evaluate_live_binance_fallback("DOGEUSDT", NOW, "r")
    r1["features"]["poison"] = 1
    r2 = a._evaluate_live_binance_fallback("DOGEUSDT", NOW + 1, "r")
    assert "poison" not in r2["features"]
    assert a.last_result is r2 or a.last_result == r2


def test_exception_cached_with_error_cooldown_and_uses_reason(net, monkeypatch, caplog):
    net.exc = urllib.error.URLError("boom")
    a = TapeReadingAgent(db_path="/nonexistent/x.db")
    r = a._evaluate_live_binance_fallback("DOGEUSDT", NOW, "book_stale")
    assert (r["status"], r["vote"], r["reason"]) == ("UNAVAILABLE", "HOLD", "book_stale")
    n = len(net.calls)
    assert n >= 1
    r2 = a._evaluate_live_binance_fallback("DOGEUSDT", NOW + 14_999, "book_stale")
    assert len(net.calls) == n and r2["status"] == "UNAVAILABLE"
    a._evaluate_live_binance_fallback("DOGEUSDT", NOW + 15_000, "book_stale")
    assert len(net.calls) == 2 * n


def test_error_cooldown_independent_from_ok_ttl(net):
    net.exc = OSError("down")
    a = TapeReadingAgent(db_path="/nonexistent/x.db", fallback_cache_ttl_ms=1_000, fallback_err_cooldown_ms=60_000)
    a._evaluate_live_binance_fallback("DOGEUSDT", NOW, "r")
    n = len(net.calls)
    a._evaluate_live_binance_fallback("DOGEUSDT", NOW + 30_000, "r")  # > ttl OK, < cooldown
    assert len(net.calls) == n


def test_failure_then_recovery_after_cooldown(net):
    a = TapeReadingAgent(db_path="/nonexistent/x.db")
    net.exc = OSError("down")
    assert a._evaluate_live_binance_fallback("DOGEUSDT", NOW, "r")["status"] == "UNAVAILABLE"
    net.exc = None
    ok = a._evaluate_live_binance_fallback("DOGEUSDT", NOW + 20_000, "r")
    assert ok["status"] == "OK"


def test_empty_depth_is_unavailable(net, monkeypatch):
    monkeypatch.setattr(tra.urllib.request, "urlopen", lambda req, timeout=None: FakeResp({"bids": [], "asks": []}))
    r = TapeReadingAgent(db_path="/nonexistent/x.db")._evaluate_live_binance_fallback("DOGEUSDT", NOW, "no_book_data")
    assert (r["status"], r["reason"]) == ("UNAVAILABLE", "no_book_data")


def test_stale_tick_fallback_guard_preserved(net):
    net.trade_age_ms = 10 * 60_000  # todos os trades fora da janela de 5m
    r = TapeReadingAgent(db_path="/nonexistent/x.db")._evaluate_live_binance_fallback("DOGEUSDT", NOW, "r")
    assert (r["status"], r["reason"]) == ("UNAVAILABLE", "stale_tick_fallback")


@pytest.mark.parametrize("sym,spread,wide", [
    ("DOGEUSDT", 10.0, False), ("DOGEUSDT", 14.0, False), ("DOGEUSDT", 20.0, True),
    ("BTCUSDT", 4.0, False), ("BTCUSDT", 8.0, True), ("SOLUSDT", 10.0, True),
])
def test_spread_cap_majors_5bps_alts_15bps(net, sym, spread, wide):
    net.spread_bps = spread
    r = TapeReadingAgent(db_path="/nonexistent/x.db")._evaluate_live_binance_fallback(sym, NOW, "r")
    assert r["status"] == "OK"
    assert (r["reason"] == "spread_wide") is wide
    if wide:
        assert r["vote"] == "HOLD"


BAD = {
    "no_book_data": (dict(book_age=None), "no_book_data", None),
    "book_stale": (dict(book_age=200), "book_stale", "book_age_s"),
    "no_tick_data": (dict(ticks=[]), "no_tick_data", "book_age_s"),
    "tick_stale": (dict(ticks=[(200, 600.0, 400.0, 20, 0), (250, 600.0, 400.0, 20, 0), (290, 600.0, 400.0, 20, 0)]),
                   "tick_stale", "tick_age_s"),
    "n_min<3": (dict(ticks=GOOD_TICKS[:2]), "tick_insufficient", "tick_minutes"),
    "n_trades<30": (dict(ticks=[(10, 600.0, 400.0, 5, 0), (70, 600.0, 400.0, 5, 0), (130, 600.0, 400.0, 5, 0)]),
                    "tick_insufficient", "tick_minutes"),
    "sat>0.5": (dict(ticks=[(10 + 50 * i, 600.0, 400.0, 20, 1 if i < 3 else 0) for i in range(4)]),
                "tick_insufficient", "tick_minutes"),
    "zero_volume": (dict(ticks=[(10, 0.0, 0.0, 20, 0), (70, 0.0, 0.0, 20, 0), (130, 0.0, 0.0, 20, 0)]),
                    "tick_zero_volume", "tick_minutes"),
}


@pytest.mark.parametrize("case", list(BAD))
def test_live_call_triggers_fallback_with_exact_reason(tmp_path, monkeypatch, case):
    kw, reason, _ = BAD[case]
    a = live_agent(tmp_path, monkeypatch, **kw)
    calls = spy_fallback(a, monkeypatch, watch_db=a.db_path)
    r = a.evaluate("BTCUSDT", "BUY", now_ms=NOW, allow_live_fallback=True)
    assert len(calls) == 1
    assert calls[0][:3] == ("BTCUSDT", NOW, reason)
    assert calls[0][3] == [], "conexão RO do SQLite deve estar fechada antes do REST"
    assert r["reason"] == "SENTINEL:" + reason
    assert open_fds(a.db_path) == []


@pytest.mark.parametrize("case", list(BAD))
def test_now_ms_none_is_live_and_triggers(tmp_path, monkeypatch, case):
    import time as _t
    kw, reason, _ = BAD[case]
    a = live_agent(tmp_path, monkeypatch, now=int(_t.time() * 1000), **kw)
    calls = spy_fallback(a, monkeypatch)
    a.evaluate("BTCUSDT", "BUY")
    assert [c[2] for c in calls] == [reason]


@pytest.mark.parametrize("case", list(BAD))
def test_backtest_never_touches_network_and_preserves_reason(tmp_path, monkeypatch, net, case):
    kw, reason, feat_key = BAD[case]
    a = live_agent(tmp_path, monkeypatch, **kw)
    calls = spy_fallback(a, monkeypatch)
    r = a.evaluate("BTCUSDT", "BUY", now_ms=NOW)  # allow_live_fallback=False
    assert calls == [] and net.calls == []
    assert (r["status"], r["vote"], r["reason"]) == ("UNAVAILABLE", "HOLD", reason)
    if feat_key:
        assert feat_key in r["features"]
    if case == "book_stale":
        assert r["features"]["book_age_s"] == pytest.approx(200.0, abs=0.1)
    if case == "n_min<3":
        assert r["features"]["tick_minutes"] == 2


@pytest.mark.parametrize("case", list(BAD))
def test_foreign_db_path_never_goes_live_even_if_allowed(tmp_path, monkeypatch, net, case):
    kw, reason, _ = BAD[case]
    p = build_db(tmp_path / "other.db", **kw)  # MICROSTRUCTURE_DB NÃO patched -> db_path != MICRO
    a = TapeReadingAgent(db_path=p)
    calls = spy_fallback(a, monkeypatch)
    r = a.evaluate("BTCUSDT", "BUY", now_ms=NOW, allow_live_fallback=True)
    assert calls == [] and net.calls == [] and r["reason"] == reason


def test_symbol_not_covered_live_vs_backtest(tmp_path, monkeypatch, net):
    a = live_agent(tmp_path, monkeypatch)
    calls = spy_fallback(a, monkeypatch)
    a.evaluate("DOGEUSDT", "BUY", now_ms=NOW, allow_live_fallback=True)
    assert [c[:3] for c in calls] == [("DOGEUSDT", NOW, "symbol_not_covered")]
    calls.clear()
    r = a.evaluate("LINKUSDT", "BUY", now_ms=NOW)
    assert calls == [] and net.calls == []
    assert (r["status"], r["reason"]) == ("UNAVAILABLE", "symbol_not_covered")


def test_healthy_local_data_ok_without_fallback(tmp_path, monkeypatch, net):
    a = live_agent(tmp_path, monkeypatch)
    calls = spy_fallback(a, monkeypatch)
    r = a.evaluate("BTCUSDT", "BUY", now_ms=NOW, allow_live_fallback=True)
    assert r["status"] == "OK" and r["features"]["taker_source"] == "tick_agg"
    assert calls == [] and net.calls == []


@pytest.mark.parametrize("kw", [
    dict(book_age=120),                                                        # == BOOK_MAX_AGE_S: ainda fresco
    dict(ticks=[(10, 600.0, 400.0, 10, 0), (70, 600.0, 400.0, 10, 0), (130, 600.0, 400.0, 10, 0)]),  # 3 min / 30 trades
    dict(ticks=[(10 + 50 * i, 600.0, 400.0, 20, 1 if i < 2 else 0) for i in range(4)]),              # sat == 0.50
])
def test_boundaries_stay_local(tmp_path, monkeypatch, net, kw):
    a = live_agent(tmp_path, monkeypatch, **kw)
    calls = spy_fallback(a, monkeypatch)
    r = a.evaluate("BTCUSDT", "BUY", now_ms=NOW, allow_live_fallback=True)
    assert r["status"] == "OK" and calls == [] and net.calls == []


def test_priority_local_binance_series_before_rest(tmp_path, monkeypatch, net):
    a = live_agent(tmp_path, monkeypatch, ticks=[], series=[(60, 700.0, 300.0)])
    calls = spy_fallback(a, monkeypatch)
    r = a.evaluate("BTCUSDT", "BUY", now_ms=NOW, allow_live_fallback=True)
    assert r["status"] == "OK"
    assert r["features"]["taker_source"] == "binance_series_5m"
    assert calls == [] and net.calls == []


def test_priority_series_rescues_bad_ticks_too(tmp_path, monkeypatch, net):
    a = live_agent(tmp_path, monkeypatch, ticks=GOOD_TICKS[:2], series=[(60, 700.0, 300.0)])
    calls = spy_fallback(a, monkeypatch)
    r = a.evaluate("BTCUSDT", "BUY", now_ms=NOW, allow_live_fallback=True)
    assert r["status"] == "OK" and r["features"]["taker_source"] == "binance_series_5m" and calls == []


def test_series_with_fresh_taker_but_stale_book_still_falls_back(tmp_path, monkeypatch):
    a = live_agent(tmp_path, monkeypatch, book_age=300, ticks=[], series=[(60, 700.0, 300.0)])
    calls = spy_fallback(a, monkeypatch)
    a.evaluate("BTCUSDT", "BUY", now_ms=NOW, allow_live_fallback=True)
    assert [c[2] for c in calls] == ["book_stale"]


def test_stale_series_does_not_block_fallback(tmp_path, monkeypatch):
    a = live_agent(tmp_path, monkeypatch, ticks=[], series=[(1_000, 700.0, 300.0)])  # 1000 s > 720 s
    calls = spy_fallback(a, monkeypatch)
    a.evaluate("BTCUSDT", "BUY", now_ms=NOW, allow_live_fallback=True)
    assert [c[2] for c in calls] == ["no_tick_data"]


def test_real_fallback_runs_after_db_closed_and_no_fd_leak(tmp_path, monkeypatch):
    a = live_agent(tmp_path, monkeypatch, book_age=None)
    n = Net(watch_db=a.db_path)
    monkeypatch.setattr(tra.urllib.request, "urlopen", n)
    r = a.evaluate("BTCUSDT", "BUY", now_ms=NOW, allow_live_fallback=True)
    assert r["status"] == "OK" and r["features"]["source"] == "binance_live_rest"
    assert n.calls and all(c[2] == [] for c in n.calls)
    assert all(c[1] == tra.LIVE_FALLBACK_TIMEOUT_S for c in n.calls)
    for i in range(1, 30):
        a.evaluate("BTCUSDT", "BUY", now_ms=NOW + i * 11_000, allow_live_fallback=True)
    assert open_fds(a.db_path) == []


def test_evaluate_repeated_calls_hit_cache_not_http(tmp_path, monkeypatch, net):
    a = live_agent(tmp_path, monkeypatch, book_age=None)
    for i in range(5):
        a.evaluate("BTCUSDT", "BUY", now_ms=NOW + i * 1_000, allow_live_fallback=True)
    assert len(net.calls) == 2  # depth + aggTrades, uma vez


def test_evaluate_http_failure_returns_original_reason_and_closes_db(tmp_path, monkeypatch):
    a = live_agent(tmp_path, monkeypatch, book_age=200)
    n = Net(exc=OSError("down"), watch_db=a.db_path)
    monkeypatch.setattr(tra.urllib.request, "urlopen", n)
    r = a.evaluate("BTCUSDT", "BUY", now_ms=NOW, allow_live_fallback=True)
    assert (r["status"], r["reason"]) == ("UNAVAILABLE", "book_stale")
    assert all(c[2] == [] for c in n.calls) and open_fds(a.db_path) == []


mc = pytest.importorskip("scripts.microstructure_collector")


def make_cache(path, rows):
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE golden_pots (symbol TEXT, timestamp INTEGER)")
    con.executemany("INSERT INTO golden_pots VALUES (?,?)", rows)
    con.commit()
    con.close()
    return str(path)


def test_collector_constants():
    assert mc.DEFAULT_GP_TOP_N == 6 and mc.MAX_ACTIVE_SYMBOLS == 16
    assert len(mc.BASE_SYMBOLS) + mc.DEFAULT_GP_TOP_N <= mc.MAX_ACTIVE_SYMBOLS


def test_gp_default_top6_ordering_and_limit(tmp_path, monkeypatch):
    alts = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT", "FFFUSDT", "GGGUSDT", "HHHUSDT"]
    p = make_cache(tmp_path / "c.db", [(s, 1000 + i) for i, s in enumerate(alts)])
    monkeypatch.setattr(mc, "CACHE_DB", p)
    out = mc.get_active_golden_pot_symbols()
    assert out == list(reversed(alts))[:6]
    assert mc.get_active_golden_pot_symbols(top_n=2) == ["HHHUSDT", "GGGUSDT"]
    assert mc.get_active_golden_pot_symbols(top_n=0) == []
    assert open_fds(p) == []


def test_gp_tie_on_timestamp_broken_by_count(tmp_path, monkeypatch):
    rows = [("LOWUSDT", 500), ("HIGHUSDT", 500), ("HIGHUSDT", 400), ("HIGHUSDT", 300)]
    monkeypatch.setattr(mc, "CACHE_DB", make_cache(tmp_path / "c.db", rows))
    assert mc.get_active_golden_pot_symbols(top_n=2) == ["HIGHUSDT", "LOWUSDT"]


def test_gp_excludes_base_and_tradfi_without_consuming_slots(tmp_path, monkeypatch):
    excluded = ["BTCUSDT", "ethusdt", "SOLUSDT", "MUUSDT", "NBISUSDT", "TSLAUSDT"]
    good = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT", "FFFUSDT"]
    rows = [(s, 9000 - i) for i, s in enumerate(excluded)] + [(s, 100 - i) for i, s in enumerate(good)]
    monkeypatch.setattr(mc, "CACHE_DB", make_cache(tmp_path / "c.db", rows))
    out = mc.get_active_golden_pot_symbols()
    assert out == good
    assert not {s.upper() for s in out} & {s.upper() for s in excluded}


def test_gp_dedups_mixed_case(tmp_path, monkeypatch):
    rows = [("dogeusdt", 900), ("DOGEUSDT", 800), ("DogeUSDT", 700), ("LINKUSDT", 600), ("linkusdt", 500)]
    monkeypatch.setattr(mc, "CACHE_DB", make_cache(tmp_path / "c.db", rows))
    out = mc.get_active_golden_pot_symbols(top_n=6)
    assert [s.upper() for s in out] == ["DOGEUSDT", "LINKUSDT"]
    assert len(set(out)) == 2


def test_gp_missing_empty_or_corrupt_db_returns_empty_without_fd_leak(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "CACHE_DB", str(tmp_path / "absent.db"))
    assert mc.get_active_golden_pot_symbols() == []
    empty = tmp_path / "empty.db"
    sqlite3.connect(str(empty)).close()  # sem tabela golden_pots
    monkeypatch.setattr(mc, "CACHE_DB", str(empty))
    assert mc.get_active_golden_pot_symbols() == []
    assert open_fds(str(empty)) == []
    zero = tmp_path / "zero.db"
    p = make_cache(zero, [])
    monkeypatch.setattr(mc, "CACHE_DB", p)
    assert mc.get_active_golden_pot_symbols() == [] and open_fds(p) == []
