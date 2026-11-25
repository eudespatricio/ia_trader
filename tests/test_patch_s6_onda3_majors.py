"""S6.3 Onda 3 majors — independent tests written from the frozen contracts (tmp_path DBs only)."""
import math
import os
import sqlite3
import time
import urllib.error
from datetime import datetime, timezone
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import pytest

import scripts.backfill_candidate_majors as bf
import scripts.eval_majors_expansion_s6_oos as ev
import scripts.majors_s6_sim as ms

H, M15, DAY = 3_600_000, 900_000, 86_400_000
T0 = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
def _kline_fetcher(tf_ms, first, n, calls=None, dup=False):
    """Serves Binance-like klines [first, first+n*tf) honoring startTime/endTime/limit."""
    def fetch(url):
        q = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        if calls is not None:
            calls.append(q)
        s, e, lim = int(q["startTime"]), int(q["endTime"]), int(q["limit"])
        assert lim <= 1500
        ts = [first + i * tf_ms for i in range(n) if s <= first + i * tf_ms <= e][:lim]
        rows = [[t, "1.0", "2.0", "0.5", "1.5", "10.0", t + tf_ms - 1, "15.0", 1, "0", "0", "0"] for t in ts]
        return (rows[::-1] + rows[:1]) if dup else rows
    return fetch

def _fund_fetcher(first, n, step=8 * H, calls=None, mark=""):
    def fetch(url):
        q = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        if calls is not None:
            calls.append(q)
        s, e, lim = int(q["startTime"]), int(q["endTime"]), int(q["limit"])
        assert lim <= 1000
        ts = [first + i * step for i in range(n) if s <= first + i * step <= e][:lim]
        return [{"symbol": "X", "fundingTime": t, "fundingRate": "0.0001", "markPrice": mark} for t in ts]
    return fetch

def _mk_db(path, wal=False):
    con = sqlite3.connect(path)
    if wal:
        con.execute("PRAGMA journal_mode=WAL")
    con.executescript(
        "CREATE TABLE candles(exchange TEXT, symbol TEXT, timeframe TEXT, timestamp INTEGER, open REAL, high REAL,"
        " low REAL, close REAL, volume REAL, quote_volume REAL, PRIMARY KEY(exchange,symbol,timeframe,timestamp));"
        "CREATE TABLE funding_rates(exchange TEXT, symbol TEXT, timestamp INTEGER, funding_rate REAL, mark_price REAL,"
        " PRIMARY KEY(exchange,symbol,timestamp));"
        "CREATE TABLE paper_positions(id INTEGER PRIMARY KEY, symbol TEXT, status TEXT);"
        "CREATE TABLE simulation_settings(key TEXT PRIMARY KEY, value TEXT);"
        "INSERT INTO paper_positions(symbol,status) VALUES('BTCUSDT','OPEN'),('ETHUSDT','OPEN'),('SOLUSDT','CLOSED');"
        "INSERT INTO simulation_settings VALUES('pilot_start_utc','2026-01-02T03:04:05Z');")
    con.commit()
    return con

def _count(path, table):
    c = sqlite3.connect(path)
    try:
        return c.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    finally:
        c.close()

def _candle_rows(sym, tf, tf_ms, n, first=T0, close=1.5):
    return [("binance", sym, tf, first + i * tf_ms, 1.0, 2.0, 0.5, close, 10.0, 15.0) for i in range(n)]
def test_constants_contract():
    assert bf.CANDIDATE_MAJORS == ("XRPUSDT", "NEARUSDT", "DOGEUSDT", "LINKUSDT", "AAVEUSDT")
    s1, s15, sf = (bf.parse_utc_ms(x) for x in (bf.DEFAULT_1H_START_UTC, bf.DEFAULT_15M_START_UTC, bf.DEFAULT_FUNDING_START_UTC))
    assert s1 < s15 and sf <= s15 and s1 % H == 0 and s15 % M15 == 0

def test_klines_paginate_over_1500_sorted_deduped_schema():
    calls = []
    rows = bf.fetch_klines_paginated("xrpusdt", "1h", T0, T0 + 3200 * H, fetcher=_kline_fetcher(H, T0, 3200, calls), sleep_s=0)
    assert len(calls) >= 3 and len(rows) == 3200
    assert [r[3] for r in rows] == [T0 + i * H for i in range(3200)]
    assert rows[0] == ("binance", "XRPUSDT", "1h", T0, 1.0, 2.0, 0.5, 1.5, 10.0, 15.0)
    assert all(isinstance(v, float) for v in rows[0][4:])

def test_klines_dedupe_unsorted_page():
    rows = bf.fetch_klines_paginated("LINKUSDT", "1h", T0, T0 + 10 * H, fetcher=_kline_fetcher(H, T0, 10, dup=True), sleep_s=0)
    assert [r[3] for r in rows] == [T0 + i * H for i in range(10)]

def test_klines_exclude_bars_closing_after_end_ms():
    # Server returns bar opening at T0+3h; it closes at T0+4h > end_ms=T0+3.5h → must be dropped.
    rows = bf.fetch_klines_paginated("XRPUSDT", "1h", T0, T0 + 3 * H + H // 2, fetcher=_kline_fetcher(H, T0, 10), sleep_s=0)
    assert [r[3] for r in rows] == [T0, T0 + H, T0 + 2 * H]

def test_klines_exclude_currently_forming_candle():
    now = int(time.time() * 1000)
    cur_open = now // H * H
    first = cur_open - 4 * H
    rows = bf.fetch_klines_paginated("XRPUSDT", "1h", first, now + 6 * H, fetcher=_kline_fetcher(H, first, 7), sleep_s=0)
    assert rows and max(r[3] for r in rows) == cur_open - H
    assert all(r[3] + H - 1 < now for r in rows)

def test_klines_retry_transient_then_ok_and_persistent_failure(monkeypatch):
    monkeypatch.setattr(bf.time, "sleep", lambda s: None)
    inner, state = _kline_fetcher(H, T0, 5), {"n": 0}

    def flaky(url):
        state["n"] += 1
        if state["n"] <= 2:
            raise urllib.error.HTTPError(url, 429 if state["n"] == 1 else 503, "x", None, None)
        return inner(url)

    assert len(bf.fetch_klines_paginated("XRPUSDT", "1h", T0, T0 + 5 * H, fetcher=flaky, sleep_s=0)) == 5 and state["n"] == 3

    def dead(url):
        raise urllib.error.HTTPError(url, 503, "down", None, None)

    with pytest.raises(RuntimeError):
        bf.fetch_klines_paginated("XRPUSDT", "1h", T0, T0 + 5 * H, fetcher=dead, sleep_s=0)

def test_funding_paginate_1000_sorted_deduped_schema(monkeypatch):
    calls = []
    f = _fund_fetcher(T0, 2500, calls=calls, mark="123.5")
    rows = bf.fetch_funding_paginated("nearusdt", T0, T0 + 2500 * 8 * H, fetcher=f, sleep_s=0)
    assert len(calls) >= 3 and len(rows) == 2500
    assert [r[2] for r in rows] == [T0 + i * 8 * H for i in range(2500)]
    assert rows[0] == ("binance", "NEARUSDT", T0, 0.0001, 123.5)
    # empty markPrice → stored as None (never fabricated)
    assert bf.fetch_funding_paginated("NEARUSDT", T0, T0 + 8 * H, fetcher=_fund_fetcher(T0, 2), sleep_s=0)[0][4] is None

def test_create_online_backup_includes_wal_content(tmp_path):
    db = str(tmp_path / "live.db")
    con = _mk_db(db, wal=True)  # connection stays open: data may only live in the WAL
    bf.upsert_candles_batched(db, _candle_rows("XRPUSDT", "1h", H, 50))
    path = bf.create_online_backup(db, str(tmp_path / "bk"))
    assert os.path.dirname(path) == str(tmp_path / "bk") and os.path.exists(path)
    bc = sqlite3.connect(path)
    assert bc.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert bc.execute("SELECT count(*) FROM candles").fetchone()[0] == 50
    assert bc.execute("SELECT count(*) FROM paper_positions").fetchone()[0] == 3
    bc.close(); con.close()
    with pytest.raises(FileNotFoundError):
        bf.create_online_backup(str(tmp_path / "missing.db"), str(tmp_path / "bk2"))

def test_upsert_idempotent_multibatch_and_preserves_state(tmp_path, monkeypatch):
    db = str(tmp_path / "m.db")
    _mk_db(db).close()
    commits, real = [], sqlite3.connect

    def spy(*a, **k):
        c = real(*a, **k)
        c.set_trace_callback(lambda s: commits.append(s) if s.strip().upper().startswith("COMMIT") else None)
        return c

    monkeypatch.setattr(bf.sqlite3, "connect", spy)
    rows = _candle_rows("AAVEUSDT", "15m", M15, 2500)
    assert bf.upsert_candles_batched(db, rows, batch_size=1000) == 2500
    assert len(commits) >= 3  # 3 short transactions, not one giant write
    monkeypatch.setattr(bf.sqlite3, "connect", real)
    assert bf.upsert_candles_batched(db, rows, batch_size=1000) == 2500
    assert _count(db, "candles") == 2500  # idempotent
    bf.upsert_candles_batched(db, _candle_rows("AAVEUSDT", "15m", M15, 5, close=9.0))
    c = sqlite3.connect(db)
    assert c.execute("SELECT count(*), min(close) FROM candles WHERE timestamp<?", (T0 + 5 * M15,)).fetchone() == (5, 9.0)
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    c.close()
    fr = [("binance", "AAVEUSDT", T0 + i * 8 * H, 0.0001, None) for i in range(1200)]
    assert bf.upsert_funding_batched(db, fr, batch_size=500) == 1200
    assert bf.upsert_funding_batched(db, fr, batch_size=500) == 1200 and _count(db, "funding_rates") == 1200
    assert _count(db, "paper_positions") == 3 and _count(db, "simulation_settings") == 1
    assert bf.upsert_candles_batched(db, []) == 0 and bf.upsert_funding_batched(db, []) == 0

def test_upsert_failed_batch_rolls_back_only_that_batch(tmp_path):
    db = str(tmp_path / "r.db")
    _mk_db(db).close()
    rows = _candle_rows("XRPUSDT", "1h", H, 1500)
    rows[1200] = rows[1200][:-1]  # wrong arity inside 2nd batch
    with pytest.raises(sqlite3.Error):
        bf.upsert_candles_batched(db, rows, batch_size=1000)
    assert _count(db, "candles") == 1000

def test_verify_integrity_counts_gaps_dups_and_read_only(tmp_path):
    db = str(tmp_path / "v.db")
    con = _mk_db(db)
    ts = [T0 + i * H for i in range(10) if i not in (4, 5, 6)] + [T0 + 20 * H]  # gaps: 3 bars, then 10 bars
    con.executemany(bf.KLINE_SQL, [("binance", "XRPUSDT", "1h", t, 1, 2, .5, 1, 1, 1) for t in ts])
    con.execute(bf.KLINE_SQL, ("bybit", "XRPUSDT", "1h", T0, 1, 2, .5, 1, 1, 1))  # cross-exchange duplicate
    con.executemany(bf.KLINE_SQL, [("binance", "XRPUSDT", "15m", T0 + i * M15, 1, 2, .5, 1, 1, 1) for i in range(8)])
    con.executemany(bf.FUNDING_SQL, [("binance", "XRPUSDT", T0 + i * 8 * H, 1e-4, None) for i in range(3)])
    con.commit(); con.close()
    before = open(db, "rb").read()
    res = bf.verify_backfill_integrity(db, ["xrpusdt", "NEARUSDT"], T0, T0)
    assert open(db, "rb").read() == before  # strictly read-only
    x = res["symbols"]["XRPUSDT"]
    assert (x["1h"]["count"], x["1h"]["gap_events"], x["1h"]["max_gap_bars"]) == (8, 2, 10)
    assert (x["15m"]["count"], x["15m"]["gap_events"], x["15m"]["max_gap_bars"]) == (8, 0, 0)
    assert x["funding_rates"]["count"] == 3
    assert res["symbols"]["NEARUSDT"]["1h"]["count"] == 0
    assert res["cross_exchange_duplicates"] == 1 and res["integrity_check"] == "ok"
    assert res["pilot_start_utc"] == "2026-01-02T03:04:05Z" and res["open_positions_count"] == 2

def test_verify_integrity_without_pilot_tables(tmp_path):
    db = str(tmp_path / "bare.db")
    con = _mk_db(db)
    con.executescript("DROP TABLE paper_positions; DROP TABLE simulation_settings;")
    con.commit(); con.close()
    res = bf.verify_backfill_integrity(db, ["XRPUSDT"], T0, T0)
    assert res["open_positions_count"] is None and res["pilot_start_utc"] is None and res["integrity_check"] == "ok"
# ───────────────────────── ro_connect ─────────────────────────
def test_ro_connect_blocks_writes(tmp_path):
    db = str(tmp_path / "ro.db")
    _mk_db(db).close()
    con = ms.ro_connect(db)
    assert con.execute("SELECT count(*) FROM paper_positions").fetchone()[0] == 3
    assert con.execute("PRAGMA query_only").fetchone()[0] == 1
    for sql in ("DELETE FROM paper_positions", "INSERT INTO simulation_settings VALUES('k','v')", "DROP TABLE candles"):
        with pytest.raises(sqlite3.OperationalError):
            con.execute(sql)
    con.close()
    assert _count(db, "paper_positions") == 3
    with pytest.raises(sqlite3.OperationalError):
        ms.ro_connect(str(tmp_path / "absent.db"))  # ro mode never creates a DB
# ───────────────────────── fit_purged_ensemble ─────────────────────────
COLS, NB = ["f1", "f2"], 700

def _sd(seed, n=NB, shift=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"timestamp": T0 + (np.arange(n) + shift) * H, "open": close, "high": close * 1.004,
                       "low": close * 0.996, "close": close, "atr_14": close * 0.01,
                       "f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    return SimpleNamespace(df=df)

def _fit(sds, cutoff, **kw):
    return ev.fit_purged_ensemble(sds, list(sds), COLS, cutoff, 1.5, 1.0, False, **kw)

def test_fit_zero_leakage_and_no_joblib_and_per_symbol_labels(monkeypatch):
    import joblib

    def boom(*a, **k):
        raise AssertionError("joblib.load must never be called")

    monkeypatch.setattr(joblib, "load", boom)
    sds = {"AAA": _sd(1), "BBB": _sd(2, shift=3)}
    cutoff = int(sds["AAA"].df["timestamp"].iloc[500])
    models, n = _fit(sds, cutoff)
    assert len(models) == 2
    # horizon=5 labels dropped PER SYMBOL: (500-5) rows each; a cross-symbol concat would give a larger count.
    n_exp = (500 - 5) + (int((sds["BBB"].df["timestamp"] < cutoff).sum()) - 5)
    assert n == n_exp
    # poison everything at/after the cutoff (incl. the boundary row) and append later rows
    pois = {}
    for s, sd in sds.items():
        d = sd.df.copy()
        m = d["timestamp"] >= cutoff
        d.loc[m, ["close", "open", "high", "low"]] *= 37.0
        d.loc[m, ["f1", "f2"]] = 1e6
        extra = d.tail(50).copy(); extra["timestamp"] += 1000 * H; extra[["f1", "f2"]] = -1e6
        pois[s] = SimpleNamespace(df=pd.concat([d, extra], ignore_index=True))
    models_p, n_p = _fit(pois, cutoff)
    assert n_p == n
    probe = pd.DataFrame(np.random.default_rng(9).normal(size=(64, 2)), columns=COLS)
    for a, b in zip(models, models_p):
        assert np.allclose(a.predict_proba(probe), b.predict_proba(probe), rtol=0, atol=1e-12)

def test_fit_start_ms_window_and_empty_training_raises():
    sds = {"AAA": _sd(3), "BBB": _sd(4)}
    ts = sds["AAA"].df["timestamp"]
    _, n = _fit(sds, int(ts.iloc[500]), start_ms=int(ts.iloc[100]))
    assert n == 2 * (400 - 5)
    with pytest.raises(RuntimeError):
        _fit(sds, int(ts.iloc[0]))
# ───────────────────────── FastCachedEnricher parity / lookahead ─────────────────────────
D0 = int(datetime(2025, 6, 10, tzinfo=timezone.utc).timestamp() * 1000)
NOW = D0 + 13 * H + 35 * 60_000

def _build_dbs(tmp_path, tag, future=False):
    cache, micro = str(tmp_path / f"cache_{tag}.db"), str(tmp_path / f"micro_{tag}.db")
    c = sqlite3.connect(cache)
    c.executescript(
        "CREATE TABLE macro_daily_history(series_name TEXT, date_str TEXT, timestamp_ms INTEGER, value REAL);"
        "CREATE TABLE macro_events_calendar(event_name TEXT, timestamp_ms INTEGER);"
        "CREATE TABLE funding_rates(exchange TEXT, symbol TEXT, timestamp INTEGER, funding_rate REAL, mark_price REAL);")
    mac = []
    for i in range(1, 31):
        t = D0 - i * DAY
        for name, base in (("DXY", 100.0), ("VIX", 18.0), ("SP500", 5000.0), ("FEAR_AND_GREED", 50.0)):
            mac.append((name, datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%d"), t, base + math.sin(i) * base * 0.02))
    c.executemany("INSERT INTO macro_daily_history VALUES(?,?,?,?)", mac)
    c.executemany("INSERT INTO macro_events_calendar VALUES(?,?)", [("US CPI", D0), ("FOMC", D0 + 9 * DAY)])
    c.executemany("INSERT INTO funding_rates VALUES('binance','BTCUSDT',?,?,NULL)",
                  [(D0 + 8 * H - i * 8 * H, 1e-4 * math.cos(i)) for i in range(100)])
    c.executemany("INSERT INTO funding_rates VALUES('binance','ETHUSDT',?,?,NULL)",
                  [(D0 - 2 * DAY - i * 8 * H, 2e-4 * math.sin(i)) for i in range(40)])  # stale → micro derivs path
    mc = sqlite3.connect(micro)
    mc.executescript(
        "CREATE TABLE book_features(symbol TEXT, ts_ms INTEGER, spread_bps REAL, bid_n10 REAL, ask_n10 REAL);"
        "CREATE TABLE derivs(symbol TEXT, ts_ms INTEGER, funding_rate REAL);"
        "CREATE TABLE tick_agg(symbol TEXT, minute_ms INTEGER, buy_notional REAL, sell_notional REAL, saturated INTEGER);"
        "CREATE TABLE binance_series(symbol TEXT, kind TEXT, ts_ms INTEGER, v2 REAL, v3 REAL);")
    mc.execute("INSERT INTO book_features VALUES('BTCUSDT',?,1.5,120.0,100.0)", (NOW - 30_000,))
    mc.executemany("INSERT INTO tick_agg VALUES('BTCUSDT',?,?,?,0)",
                   [(NOW - k * 60_000, 100.0 + k, 80.0 + 2 * k) for k in range(1, 7)])
    mc.executemany("INSERT INTO derivs VALUES('ETHUSDT',?,?)", [(NOW - 3_600_000 * k, 1e-4 * (k % 4)) for k in range(1, 40)])
    mc.executemany("INSERT INTO binance_series VALUES('ETHUSDT','taker',?,?,?)",
                   [(NOW - 600_000 - k * 300_000, 50.0 + k, 40.0) for k in range(5)])
    mc.executemany("INSERT INTO binance_series VALUES('XRPUSDT','taker',?,?,40.0)", [(NOW - 300_000, 90.0), (NOW - 600_000, 70.0), (NOW - 3 * H, 60.0)])
    if future:  # everything strictly after NOW (or beyond the calendar window) is mutated/added
        c.executemany("INSERT INTO macro_daily_history VALUES(?,?,?,?)", [("VIX", "f", D0, 99.0), ("DXY", "f2", D0 + DAY, 1.0)])
        c.execute("UPDATE macro_events_calendar SET timestamp_ms=timestamp_ms+86400000 WHERE event_name='FOMC'")
        c.executemany("INSERT INTO funding_rates VALUES('binance',?,?,?,NULL)",
                      [(s, NOW + k * 8 * H, 0.05) for s in ("BTCUSDT", "ETHUSDT") for k in range(1, 4)])
        mc.execute("INSERT INTO book_features VALUES('BTCUSDT',?,500.0,1.0,900.0)", (NOW + 10_000,))
        mc.executemany("INSERT INTO tick_agg VALUES('BTCUSDT',?,9e9,1.0,0)", [(NOW + k * 60_000,) for k in range(0, 4)])
        mc.executemany("INSERT INTO derivs VALUES('ETHUSDT',?,0.09)", [(NOW + k * H,) for k in range(1, 4)])
        mc.executemany("INSERT INTO binance_series VALUES('ETHUSDT','taker',?,9e9,1.0)", [(NOW + k * 300_000,) for k in range(1, 4)])
    c.commit(); mc.commit(); c.close(); mc.close()
    return cache, micro

def _assert_ctx_equal(a, b, tol=1e-12):
    assert set(a) == set(b)
    for k in a:
        if isinstance(a[k], (int, float, np.number)) and not isinstance(a[k], bool):
            if isinstance(a[k], float) and math.isnan(a[k]):
                assert math.isnan(b[k]), k
            else:
                assert abs(float(a[k]) - float(b[k])) <= tol, k
        else:
            assert a[k] == b[k], k

@pytest.mark.parametrize("sym", ["BTCUSDT", "ETHUSDT", "NEARUSDT", "XRPUSDT"])
def test_fast_enricher_exact_parity_with_live(tmp_path, sym):
    cache, micro = _build_dbs(tmp_path, "p")
    fast, live = ms.FastCachedEnricher(cache, micro), ms.LiveContextEnricher(cache, micro)
    try:
        latest = {"close": 100.0, "timestamp": NOW - H}
        cf, cl = fast.enrich(sym, latest, None, NOW), live.enrich(sym, latest, None, NOW)
        _assert_ctx_equal(cf, cl)
        if sym == "BTCUSDT":  # non-vacuous: every causal source contributed
            for k in ("dxy_ret_5d", "vix_value", "fear_and_greed", "funding_zscore", "micro_book_imb_10", "micro_taker_imb_5m"):
                assert k in cf, k
            assert cf["is_cpi_fomc_window"] is True
        if sym == "ETHUSDT":
            assert cf["funding_source"] == "micro_derivs" and "micro_taker_imb_5m" in cf
        if sym in ("NEARUSDT", "XRPUSDT"):  # XRP: LIMIT 3 window reaches -3h row → live yields nothing; fast must match
            assert "funding_rate" not in cf and "micro_taker_imb_5m" not in cf
    finally:
        fast.close()

@pytest.mark.parametrize("sym", ["BTCUSDT", "ETHUSDT"])
def test_fast_enricher_zero_lookahead(tmp_path, sym):
    (base_c, base_m), (fut_c, fut_m) = _build_dbs(tmp_path, "a"), _build_dbs(tmp_path, "b", future=True)
    e1, e2, e3 = ms.FastCachedEnricher(base_c, base_m), ms.FastCachedEnricher(fut_c, fut_m), ms.LiveContextEnricher(fut_c, fut_m)
    try:
        latest = {"close": 100.0}
        ref = e1.enrich(sym, latest, None, NOW)
        _assert_ctx_equal(ref, e2.enrich(sym, latest, None, NOW))
        _assert_ctx_equal(ref, e3.enrich(sym, latest, None, NOW))
    finally:
        e1.close(); e2.close()
def _cand(sym, d, ts=T0, **kw):
    return {"sym": sym, "ts": ts, "d": d, **kw}

def _out(c, hold=10 * H, r=0.5, mf=0.01, reason="TP"):
    return {"exit_ts": c["ts"] + hold, "close_t": c["ts"] + hold, "r": r, "sl_pct": 0.01, "margin_frac": mf, "reason": reason}

def _sim(cands, outcome=_out, **kw):
    kw.setdefault("bootstrap_iters", 0)
    return ms.simulate_portfolio_with_caps(cands, outcome, **kw)

@pytest.mark.parametrize("cap", [2, 3, 4])
def test_corr_cap_per_direction_not_blocking_opposite(cap):
    cands = [_cand(f"L{i}", 1) for i in range(5)] + [_cand(f"S{i}", -1) for i in range(2)]
    res = _sim(cands, corr_cap=cap, max_net_risk_r=None)
    longs = [t for t in res["trades"] if t["d"] == 1]
    assert len(longs) == cap and len([t for t in res["trades"] if t["d"] == -1]) == 2
    assert res["blocked_counts"]["corr_cap"] == 5 - cap and res["blocked_counts"]["net_risk_r"] == 0

def test_corr_cap_releases_after_positions_close():
    cands = [_cand("A", 1), _cand("B", 1), _cand("C", 1, ts=T0 + 11 * H)]
    res = _sim(cands, corr_cap=2)
    assert len(res["trades"]) == 3 and res["blocked_counts"]["corr_cap"] == 0

def test_net_risk_unit_weights_allows_hedges():
    cands = [_cand(s, d, ts=T0 + i * 60_000) for i, (s, d) in enumerate(
        [("L1", 1), ("L2", 1), ("S1", -1), ("S2", -1), ("S3", -1)])]
    res = _sim(cands, corr_cap=10, max_net_risk_r=1.0)
    assert [t["sym"] for t in res["trades"]] == ["L1", "S1", "S2"]  # net: 1, 0, -1 (|net|<=1); L2 & S3 breach
    assert res["blocked_counts"]["net_risk_r"] == 2 and res["blocked_counts"]["corr_cap"] == 0

def test_net_risk_with_beta_map_differs_from_unit_weights():
    cands = [_cand(s, 1, ts=T0 + i * 60_000) for i, s in enumerate("ABCD")] + [_cand("H", -1, ts=T0 + 10 * 60_000)]
    beta = {"A": 1.0, "B": 0.4, "C": 0.4, "D": 0.4, "H": 2.0}
    res_b = _sim(cands, corr_cap=10, max_net_risk_r=1.5, beta_map=beta)
    assert [t["sym"] for t in res_b["trades"]] == ["A", "B", "H"]  # 1.0, 1.4, (C,D → 1.8 blocked), hedge → |1.4-2.0|=0.6
    assert res_b["blocked_counts"]["net_risk_r"] == 2
    res_u = _sim(cands, corr_cap=10, max_net_risk_r=1.5)
    assert [t["sym"] for t in res_u["trades"]] == ["A", "H"] and res_u["blocked_counts"]["net_risk_r"] == 3

def test_margin_util_cap_and_release():
    cands = [_cand("A", 1), _cand("B", 1), _cand("C", 1, ts=T0 + 11 * H)]
    res = _sim(cands, corr_cap=10, max_margin_util=0.5, outcome=lambda c: _out(c, mf=0.3))
    assert [t["sym"] for t in res["trades"]] == ["A", "C"] and res["blocked_counts"]["margin"] == 1

def test_cooldown_fn_blocks_same_symbol_and_receives_sl_streak():
    cands = [_cand("A", 1, ts=T0), _cand("A", 1, ts=T0 + 2 * H)]
    short = lambda c: _out(c, hold=H // 2)  # noqa: E731
    assert len(_sim(cands, outcome=short)["trades"]) == 2
    res = _sim(cands, outcome=short, cooldown_fn=lambda c, o, streak: 4 * H)
    assert len(res["trades"]) == 1 and res["blocked_counts"]["cooldown"] == 1
    seen = []
    seq = [_cand("A", 1, ts=T0 + i * 10 * H) for i in range(4)]
    reasons = iter(["SL", "SL", "TP", "SL"])
    _sim(seq, outcome=lambda c: _out(c, hold=H, reason=next(reasons)), cooldown_fn=lambda c, o, s: seen.append(s) or 0)
    assert seen == [1, 2, 0, 1]

def test_outcome_none_skips_trade_and_empty_input():
    assert _sim([_cand("A", 1)], outcome=lambda c: None)["trades"] == []
    assert _sim([])["trades_count"] == 0
# ───────────────────────── evaluate_go_gate ─────────────────────────
BASE = {"sum_r": 10.0, "trades_per_day": 1.0, "pf": 1.30, "max_dd_pct": 8.0, "avg_r": 0.1, "pf_ex_top1": 1.2}
GOOD = {"sum_r": 15.0, "trades_per_day": 1.5, "pf": 1.5, "pf_ex_top1": 1.3, "avg_r": 0.2, "max_dd_pct": 9.0, "n": 20, "dsr_passed_95pct": True}
MARG = {"XRPUSDT": {"avg_r": 0.1, "delta_sum_r": 1.0, "n": 20},
        "NEARUSDT": {"avg_r": -0.1, "delta_sum_r": -2.0, "delta_sharpe": -0.1, "delta_calmar": -0.1, "n": 10}}

def test_go_gate_accepts_and_partitions_symbols():
    g = ms.evaluate_go_gate(GOOD, BASE, MARG)
    assert g["verdict"] == "GO" and g["passed"] is True
    assert g["promoted_symbols"] == ["XRPUSDT"] and g["rejected_symbols"] == ["NEARUSDT"]

def test_go_gate_boundaries_pass_exactly_at_thresholds():
    edge = {**GOOD, "pf": 1.25, "pf_ex_top1": 1.10, "max_dd_pct": 14.99, "n": 15}
    assert ms.evaluate_go_gate(edge, BASE, MARG)["verdict"] == "GO"

@pytest.mark.parametrize("field,val", [
    ("pf", 1.2499), ("pf_ex_top1", 1.0999), ("max_dd_pct", 15.0), ("max_dd_pct", 40.0), ("sum_r", 10.0), ("sum_r", 9.0),
    ("trades_per_day", 1.0), ("trades_per_day", 0.5), ("avg_r", 0.0), ("avg_r", -0.05),
    ("n", 14), ("dsr_passed_95pct", False)])
def test_go_gate_rejects_each_threshold(field, val):
    g = ms.evaluate_go_gate({**GOOD, field: val}, BASE, MARG)
    assert g["verdict"] == "NO-GO" and g["passed"] is False and g["reasons"]

def test_go_gate_rejects_when_no_symbol_passes_marginal_filter():
    bad = {"NEARUSDT": MARG["NEARUSDT"], "DOGEUSDT": {"avg_r": -0.3, "delta_sum_r": -1.0, "n": 5}}
    g = ms.evaluate_go_gate(GOOD, BASE, bad)
    assert g["verdict"] == "NO-GO" and g["promoted_symbols"] == [] and sorted(g["rejected_symbols"]) == ["DOGEUSDT", "NEARUSDT"]

def test_go_gate_ml8_degradation():
    ok = {"sum_r": 9.8, "pf": 1.28, "max_dd_pct": 8.5}
    assert ms.evaluate_go_gate(GOOD, BASE, MARG, baseline_ml8_oos=ok)["verdict"] == "GO"
    for bad in ({**ok, "sum_r": 5.0}, {**ok, "pf": 1.0}, {**ok, "max_dd_pct": 16.0}):
        assert ms.evaluate_go_gate(GOOD, BASE, MARG, baseline_ml8_oos=bad)["verdict"] == "NO-GO"
