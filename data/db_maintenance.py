"""Manutenção segura do SQLite de cache (WAL checkpoint e purge de símbolos esparsos).

Nunca remove dados dos símbolos protegidos (BTC/ETH/SOL). O purge é dry-run por padrão.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

PROTECTED_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT"})


def wal_checkpoint(db_path: str, busy_timeout_ms: int = 500, truncate_threshold_bytes: int = 0) -> Dict[str, Any]:
    """Executa checkpoint do WAL tolerante a busy.

    Tenta TRUNCATE (com busy_timeout curto para não bloquear escritores/leitores);
    se ocupado, faz PASSIVE (nunca bloqueia). Nunca levanta exceção.
    """
    res: Dict[str, Any] = {"ok": False, "mode": None, "busy": None, "wal_before": 0, "wal_after": 0, "error": None}
    wal_path = db_path + "-wal"
    try:
        if not os.path.exists(db_path):
            res["error"] = "db_missing"
            return res
        res["wal_before"] = os.path.getsize(wal_path) if os.path.exists(wal_path) else 0
        if res["wal_before"] <= truncate_threshold_bytes:
            res.update(ok=True, mode="SKIPPED")
            return res
        conn = sqlite3.connect(db_path, timeout=busy_timeout_ms / 1000.0)
        try:
            conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)};")
            for mode in ("TRUNCATE", "PASSIVE"):
                try:
                    busy, _log, _ckpt = conn.execute(f"PRAGMA wal_checkpoint({mode});").fetchone()
                except sqlite3.OperationalError as e:  # database is locked
                    res["error"] = str(e)
                    continue
                res.update(mode=mode, busy=int(busy), ok=True, error=None)
                if busy == 0:
                    break
        finally:
            conn.close()
        res["wal_after"] = os.path.getsize(wal_path) if os.path.exists(wal_path) else 0
    except Exception as e:  # noqa: BLE001 - manutenção jamais pode derrubar o supervisor
        res["error"] = str(e)
    return res


def find_sparse_symbols(
    db_path: str,
    timeframe: str = "15m",
    min_samples: int = 500,
    protected: Iterable[str] = PROTECTED_SYMBOLS,
) -> List[Dict[str, Any]]:
    """Lista (exchange, symbol) com menos de `min_samples` candles no timeframe. Somente leitura."""
    prot = {p.upper() for p in protected}
    conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True, timeout=5.0)
    try:
        rows = conn.execute(
            "SELECT exchange, symbol, COUNT(*), MIN(timestamp), MAX(timestamp) FROM candles "
            "WHERE timeframe = ? GROUP BY exchange, symbol HAVING COUNT(*) < ? ORDER BY COUNT(*)",
            (timeframe, min_samples),
        ).fetchall()
    finally:
        conn.close()
    return [
        {"exchange": r[0], "symbol": r[1], "count": r[2], "min_ts": r[3], "max_ts": r[4]}
        for r in rows
        if r[1].upper() not in prot
    ]


def purge_sparse_symbols(
    db_path: str,
    timeframe: str = "15m",
    min_samples: int = 500,
    dry_run: bool = True,
    protected: Iterable[str] = PROTECTED_SYMBOLS,
) -> Dict[str, Any]:
    """Remove candles de símbolos esparsos (ex.: originados do GoldenPot scanner). Dry-run por padrão.

    Símbolos protegidos (BTC/ETH/SOL) jamais são removidos, independentemente do número de amostras.
    """
    targets = find_sparse_symbols(db_path, timeframe, min_samples, protected)
    deleted = 0
    if not dry_run and targets:
        prot = {p.upper() for p in protected}
        conn = sqlite3.connect(db_path, timeout=15.0)
        try:
            with conn:
                for t in targets:
                    if t["symbol"].upper() in prot:
                        continue
                    cur = conn.execute(
                        "DELETE FROM candles WHERE exchange = ? AND symbol = ? AND timeframe = ?",
                        (t["exchange"], t["symbol"], timeframe),
                    )
                    deleted += cur.rowcount
        finally:
            conn.close()
    return {"dry_run": dry_run, "targets": targets, "deleted_rows": deleted}
