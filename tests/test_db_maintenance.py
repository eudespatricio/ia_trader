import os
import sqlite3

from data.db_maintenance import find_sparse_symbols, purge_sparse_symbols, wal_checkpoint


def _make_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA wal_autocheckpoint=0;")
    conn.execute(
        "CREATE TABLE candles (exchange TEXT, symbol TEXT, timeframe TEXT, timestamp INTEGER, close REAL, "
        "PRIMARY KEY (exchange, symbol, timeframe, timestamp))"
    )
    rows = [("binance", "BTCUSDT", "15m", i, 1.0) for i in range(3)]  # protegido, esparso
    rows += [("binance", "QUSDT", "15m", i, 1.0) for i in range(5)]
    rows += [("binance", "ZZZUSDT", "15m", i, 1.0) for i in range(600)]
    conn.executemany("INSERT INTO candles VALUES (?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def test_wal_checkpoint_truncates(tmp_path):
    db = str(tmp_path / "t.db")
    keep = sqlite3.connect(db)
    keep.execute("PRAGMA journal_mode=WAL;")
    keep.execute("PRAGMA wal_autocheckpoint=0;")
    keep.execute("CREATE TABLE x (a)")
    keep.executemany("INSERT INTO x VALUES (?)", [("y" * 100,)] * 2000)
    keep.commit()
    assert os.path.getsize(db + "-wal") > 0
    res = wal_checkpoint(db)
    keep.close()
    assert res["ok"] and res["mode"] == "TRUNCATE" and res["busy"] == 0
    assert res["wal_after"] == 0


def test_wal_checkpoint_busy_does_not_raise(tmp_path):
    db = str(tmp_path / "t.db")
    w = sqlite3.connect(db)
    w.execute("PRAGMA journal_mode=WAL;")
    w.execute("PRAGMA wal_autocheckpoint=0;")
    w.execute("CREATE TABLE x (a)")
    w.commit()
    w.execute("INSERT INTO x VALUES (1)")
    w.commit()
    r = sqlite3.connect(db)
    r.execute("BEGIN")
    r.execute("SELECT * FROM x").fetchall()  # leitor ativo
    w.execute("INSERT INTO x VALUES (2)")
    w.commit()
    res = wal_checkpoint(db, busy_timeout_ms=100)
    r.close()
    w.close()
    assert res["ok"] is True  # nunca levanta; PASSIVE como fallback


def test_wal_checkpoint_missing_db(tmp_path):
    res = wal_checkpoint(str(tmp_path / "nope.db"))
    assert res["ok"] is False


def test_find_and_purge_protects_core(tmp_path):
    db = str(tmp_path / "c.db")
    _make_db(db)
    found = find_sparse_symbols(db, "15m", 500)
    assert [f["symbol"] for f in found] == ["QUSDT"]
    dry = purge_sparse_symbols(db, "15m", 500, dry_run=True)
    assert dry["deleted_rows"] == 0
    real = purge_sparse_symbols(db, "15m", 500, dry_run=False)
    assert real["deleted_rows"] == 5
    conn = sqlite3.connect(db)
    left = dict(conn.execute("SELECT symbol, COUNT(*) FROM candles GROUP BY symbol").fetchall())
    conn.close()
    assert left == {"BTCUSDT": 3, "ZZZUSDT": 600}
