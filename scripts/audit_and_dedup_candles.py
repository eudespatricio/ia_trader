"""scripts/audit_and_dedup_candles.py
Audit & deduplicate market_cache candles preserving Binance primary history and seams.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Set


def _ro_uri(path: str) -> str:
    return f"file:{os.path.abspath(path)}?mode=ro"


def audit_databases(cache_db: str, micro_db: str = "") -> Dict[str, Any]:
    """Audits market_cache.db and microstructure.db using read-only connections."""
    conn = sqlite3.connect(_ro_uri(cache_db), uri=True)
    try:
        cur = conn.cursor()

        # 1. Exact cross-exchange duplicates (COUNT > 1 per symbol, timeframe, timestamp)
        cur.execute("""
            SELECT symbol, timeframe, count(*)
            FROM (
                SELECT symbol, timeframe, timestamp FROM candles
                GROUP BY symbol, timeframe, timestamp HAVING count(*) > 1
            ) GROUP BY symbol, timeframe
        """)
        exact_groups = {f"{r[0]}:{r[1]}": r[2] for r in cur.fetchall()}

        cur.execute("""
            SELECT c1.rowid FROM candles c1
            WHERE c1.exchange != 'binance'
              AND EXISTS (
                  SELECT 1 FROM candles c2
                  WHERE c2.exchange = 'binance' AND c2.symbol = c1.symbol
                    AND c2.timeframe = c1.timeframe AND c2.timestamp = c1.timestamp
              )
        """)
        exact_rowids = [r[0] for r in cur.fetchall()]

        # 2. Redundant 1d non-binance UTC+8 offset rows within binance 1d date range
        cur.execute("""
            SELECT c1.rowid, c1.symbol FROM candles c1
            JOIN (
                SELECT symbol,
                       min(date(timestamp/1000, 'unixepoch')) AS min_d,
                       max(date(timestamp/1000, 'unixepoch')) AS max_d
                FROM candles WHERE exchange = 'binance' AND timeframe = '1d'
                GROUP BY symbol
            ) b ON c1.symbol = b.symbol
            WHERE c1.timeframe = '1d' AND c1.exchange != 'binance'
              AND c1.timestamp % 86400000 != 0
              AND date(c1.timestamp/1000, 'unixepoch') BETWEEN b.min_d AND b.max_d
        """)
        offset_rows = cur.fetchall()
        offset_rowids = [r[0] for r in offset_rows]
        offset_counts = dict(Counter(r[1] for r in offset_rows))

        removable_set: Set[int] = set(exact_rowids) | set(offset_rowids)

        # 3. Non-overlapping multi-exchange seams (excluding removable rows)
        cur.execute("CREATE TEMP TABLE temp_removable(rowid INTEGER PRIMARY KEY)")
        if removable_set:
            cur.executemany("INSERT INTO temp_removable VALUES (?)", [(r,) for r in removable_set])

        cur.execute("""
            SELECT symbol, timeframe, exchange,
                   min(timestamp), max(timestamp),
                   date(min(timestamp)/1000, 'unixepoch'),
                   date(max(timestamp)/1000, 'unixepoch'),
                   count(*), avg(volume)
            FROM candles
            WHERE rowid NOT IN (SELECT rowid FROM temp_removable)
            GROUP BY symbol, timeframe, exchange
            ORDER BY symbol, timeframe, min(timestamp)
        """)
        grouped_ex = defaultdict(list)
        for r in cur.fetchall():
            grouped_ex[(r[0], r[1])].append(r)

        seams: List[Dict[str, Any]] = []
        for (sym, tf), ex_list in grouped_ex.items():
            if len(ex_list) > 1:
                non_overlap = all(ex_list[i][4] < ex_list[i + 1][3] for i in range(len(ex_list) - 1))
                if non_overlap:
                    v1, v2 = ex_list[0][8] or 1.0, ex_list[1][8] or 1.0
                    ratio = round(v2 / v1 if v1 > 0 else 0.0, 2)
                    seams.append({
                        "symbol": sym,
                        "timeframe": tf,
                        "exchanges": [e[2] for e in ex_list],
                        "date_ranges": {e[2]: f"{e[5]}..{e[6]}" for e in ex_list},
                        "avg_volumes": {e[2]: round(e[8] or 0.0, 2) for e in ex_list},
                        "vol_ratio": ratio,
                    })

        # 4. funding_rates and macro_daily_history checks
        def _chk_dup(query: str) -> int:
            try:
                cur.execute(query)
                return cur.fetchone()[0]
            except sqlite3.OperationalError:
                return 0

        funding_dups = _chk_dup("SELECT count(*) FROM (SELECT symbol, timestamp FROM funding_rates GROUP BY symbol, timestamp HAVING count(*) > 1)")
        macro_dups = _chk_dup("SELECT count(*) FROM (SELECT series_name, date_str FROM macro_daily_history GROUP BY series_name, date_str HAVING count(*) > 1)")
    finally:
        conn.close()

    # 5. microstructure.db integrity
    micro_report: Dict[str, Any] = {}
    if micro_db and os.path.exists(micro_db):
        m_conn = sqlite3.connect(_ro_uri(micro_db), uri=True)
        try:
            m_cur = m_conn.cursor()
            for tbl in ["binance_series", "book_features", "derivs", "tick_agg", "liquidations"]:
                try:
                    m_cur.execute(f"SELECT count(*) FROM {tbl}")
                    cnt = m_cur.fetchone()[0]
                    m_cur.execute(f"PRAGMA table_info({tbl})")
                    pks = [c[1] for c in m_cur.fetchall() if c[5] > 0]
                    pk_dups = 0
                    if pks:
                        pk_c = ", ".join(pks)
                        m_cur.execute(f"SELECT count(*) FROM (SELECT {pk_c} FROM {tbl} GROUP BY {pk_c} HAVING count(*) > 1)")
                        pk_dups = m_cur.fetchone()[0]
                    micro_report[tbl] = {"row_count": cnt, "pk_duplicates": pk_dups}
                except sqlite3.OperationalError as e:
                    micro_report[tbl] = {"row_count": 0, "pk_duplicates": 0, "error": str(e)}
        finally:
            m_conn.close()

    return {
        "exact_duplicates": {
            "grouped_counts": exact_groups,
            "total_rowids": len(exact_rowids),
            "rowids": exact_rowids,
        },
        "offset_1d_duplicates": {
            "counts_by_symbol": offset_counts,
            "total_rowids": len(offset_rowids),
            "rowids": offset_rowids,
        },
        "removable_rowids": sorted(removable_set),
        "total_removable": len(removable_set),
        "non_overlapping_seams": seams,
        "funding_rates_duplicates": funding_dups,
        "macro_daily_duplicates": macro_dups,
        "microstructure": micro_report,
    }


def deduplicate_candles(cache_db: str, backup_dir: str) -> Dict[str, Any]:
    """Creates an online backup and deletes duplicate non-binance candle rows in a WAL transaction."""
    audit = audit_databases(cache_db, "")
    removable = audit.get("removable_rowids", [])
    if not removable:
        return {
            "status": "CLEAN",
            "deleted_exact": 0,
            "deleted_1d_offset": 0,
            "total_deleted": 0,
            "integrity_check": "ok",
        }

    os.makedirs(backup_dir, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_file = os.path.join(backup_dir, f"pre_dedup_{ts_str}.db")

    src = sqlite3.connect(cache_db, timeout=30.0)
    dst = None
    try:
        dst = sqlite3.connect(backup_file)
        with dst:
            src.backup(dst)
    finally:
        if dst is not None:
            dst.close()
        src.close()

    conn = sqlite3.connect(cache_db, timeout=30.0, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.executemany("DELETE FROM candles WHERE rowid = ?", [(r,) for r in removable])
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        cur = conn.cursor()
        cur.execute("PRAGMA integrity_check")
        res = cur.fetchone()
    finally:
        conn.close()

    if not res or str(res[0]).lower() != "ok":
        raise RuntimeError(f"Integrity check failed: {res}")

    return {
        "status": "DEDUPLICATED",
        "backup_file": backup_file,
        "deleted_exact": audit["exact_duplicates"]["total_rowids"],
        "deleted_1d_offset": audit["offset_1d_duplicates"]["total_rowids"],
        "total_deleted": len(removable),
        "integrity_check": res[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and deduplicate market_cache candles")
    parser.add_argument("--check", action="store_true", default=False, help="Run audit check (default)")
    parser.add_argument("--apply", action="store_true", default=False, help="Perform deduplication")
    parser.add_argument("--cache-db", default="/home/eudespatricio/ia_trader/datasets/market_cache.db")
    parser.add_argument("--micro-db", default="/home/eudespatricio/ia_trader/datasets/microstructure.db")
    parser.add_argument("--backup-dir", default="/home/eudespatricio/ia_trader/datasets/backups")
    args = parser.parse_args()

    if args.apply:
        res = deduplicate_candles(args.cache_db, args.backup_dir)
        print(json.dumps(res, indent=2))
    else:
        audit = audit_databases(args.cache_db, args.micro_db)
        summary = {
            "exact_duplicates": audit["exact_duplicates"]["grouped_counts"],
            "exact_duplicates_total": audit["exact_duplicates"]["total_rowids"],
            "offset_1d_duplicates": audit["offset_1d_duplicates"]["counts_by_symbol"],
            "offset_1d_duplicates_total": audit["offset_1d_duplicates"]["total_rowids"],
            "total_removable_candles": audit["total_removable"],
            "funding_rates_duplicates": audit["funding_rates_duplicates"],
            "macro_daily_duplicates": audit["macro_daily_duplicates"],
            "seams_detected": len(audit["non_overlapping_seams"]),
            "non_overlapping_seams": audit["non_overlapping_seams"],
            "microstructure": audit["microstructure"],
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
