"""scripts/backfill_candidate_majors.py
Idempotent Binance Futures historical backfill (1h, 15m, funding_rates) for candidate majors.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

CANDIDATE_MAJORS: Tuple[str, ...] = ("XRPUSDT", "NEARUSDT", "DOGEUSDT", "LINKUSDT", "AAVEUSDT")
DEFAULT_1H_START_UTC = "2020-11-15T00:00:00Z"
DEFAULT_15M_START_UTC = "2025-10-01T00:00:00Z"
DEFAULT_FUNDING_START_UTC = "2020-11-15T00:00:00Z"
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "datasets", "market_cache.db")
DEFAULT_BACKUP_DIR = os.path.join(BASE_DIR, "datasets", "backups")

FAPI_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
FAPI_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
TF_MS: Dict[str, int] = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}

KLINE_SQL = (
    "INSERT OR REPLACE INTO candles "
    "(exchange, symbol, timeframe, timestamp, open, high, low, close, volume, quote_volume) "
    "VALUES (?,?,?,?,?,?,?,?,?,?)"
)
FUNDING_SQL = (
    "INSERT OR REPLACE INTO funding_rates "
    "(exchange, symbol, timestamp, funding_rate, mark_price) VALUES (?,?,?,?,?)"
)


def parse_utc_ms(utc_str: str) -> int:
    """Parse an ISO-8601 UTC string into epoch milliseconds."""
    dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ro_uri(path: str) -> str:
    return f"file:{os.path.abspath(path)}?mode=ro"


def _default_fetcher(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "ia-trader-backfill/1.0"})
    with urllib.request.urlopen(req, timeout=20.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_json_with_retry(
    url: str, fetcher: Optional[Callable[[str], Any]] = None,
    max_retries: int = 5, base_backoff_s: float = 0.6,
) -> Any:
    """Fetch JSON with exponential backoff on HTTP 429/5xx and transient errors."""
    fn = fetcher if fetcher is not None else _default_fetcher
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            raw = fn(url)
            if isinstance(raw, (bytes, bytearray)):
                return json.loads(raw.decode("utf-8"))
            return json.loads(raw) if isinstance(raw, str) else raw
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise
            ra = exc.headers.get("Retry-After") if exc.headers else None
            delay = float(ra) if ra and ra.isdigit() else base_backoff_s * (2 ** attempt)
            if exc.code == 429:
                delay = max(delay, 2.0)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_exc = exc
            delay = base_backoff_s * (2 ** attempt)
        if attempt + 1 < max_retries:
            time.sleep(delay)
    raise RuntimeError(f"Request failed after {max_retries} attempts for {url}: {last_exc}") from last_exc


def create_online_backup(db_path: str, backup_dir: str) -> str:
    """Create a safe online SQLite backup using Connection.backup()."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found: {db_path}")
    os.makedirs(backup_dir, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = os.path.join(backup_dir, f"pre_s6_3_backfill_{ts_str}.db")
    src = sqlite3.connect(db_path, timeout=30.0)
    dst: Optional[sqlite3.Connection] = None
    try:
        src.execute("PRAGMA busy_timeout=30000")
        dst = sqlite3.connect(backup_path, timeout=30.0)
        with dst:
            src.backup(dst, pages=2000, sleep=0.02)
    finally:
        if dst is not None:
            dst.close()
        src.close()
    return backup_path


def fetch_klines_paginated(
    symbol: str, interval: str, start_ms: int, end_ms: int,
    fetcher: Optional[Callable[[str], Any]] = None, sleep_s: float = 0.12,
) -> List[Tuple[str, str, str, int, float, float, float, float, float, float]]:
    """Paginate Binance Futures klines excluding forming candles, sorted/deduped by open_ts."""
    if interval not in TF_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    tf_ms, sym = TF_MS[interval], symbol.upper()
    cur_start, end_ms = int(start_ms), int(end_ms)
    by_ts: Dict[int, Tuple[str, str, str, int, float, float, float, float, float, float]] = {}

    while cur_start + tf_ms <= end_ms:
        url = (
            f"{FAPI_KLINES_URL}?symbol={sym}&interval={interval}"
            f"&startTime={cur_start}&endTime={end_ms}&limit=1500"
        )
        data = _fetch_json_with_retry(url, fetcher=fetcher)
        if not data:
            break
        now_ms = int(time.time() * 1000)
        for k in data:
            open_ts, close_ts = int(k[0]), int(k[6])
            if close_ts < now_ms and open_ts >= start_ms and open_ts + tf_ms <= end_ms:
                by_ts[open_ts] = (
                    "binance", sym, interval, open_ts,
                    float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), float(k[7]),
                )
        last_open = int(data[-1][0])
        if len(data) < 1500 or last_open + tf_ms <= cur_start:
            break
        cur_start = last_open + tf_ms
        if sleep_s > 0:
            time.sleep(sleep_s)
    return [by_ts[ts] for ts in sorted(by_ts)]


def fetch_funding_paginated(
    symbol: str, start_ms: int, end_ms: int,
    fetcher: Optional[Callable[[str], Any]] = None, sleep_s: float = 0.12,
) -> List[Tuple[str, str, int, float, Optional[float]]]:
    """Paginate Binance Futures funding rates sorted/deduped by fundingTime."""
    sym = symbol.upper()
    cur_start, end_ms = int(start_ms), int(end_ms)
    by_ts: Dict[int, Tuple[str, str, int, float, Optional[float]]] = {}

    while cur_start <= end_ms:
        url = f"{FAPI_FUNDING_URL}?symbol={sym}&startTime={cur_start}&endTime={end_ms}&limit=1000"
        data = _fetch_json_with_retry(url, fetcher=fetcher)
        if not data:
            break
        for item in data:
            ft = int(item["fundingTime"])
            if start_ms <= ft <= end_ms:
                mp_raw = item.get("markPrice")
                by_ts[ft] = ("binance", sym, ft, float(item["fundingRate"]), float(mp_raw) if mp_raw else None)
        last_ft = int(data[-1]["fundingTime"])
        if len(data) < 1000 or last_ft + 1 <= cur_start:
            break
        cur_start = last_ft + 1
        if sleep_s > 0:
            time.sleep(sleep_s)
    return [by_ts[ts] for ts in sorted(by_ts)]


def _upsert_batched(db_path: str, sql: str, rows: Sequence[Tuple], batch_size: int) -> int:
    if not rows:
        return 0
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
    total = 0
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.executemany(sql, batch)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            total += len(batch)
            time.sleep(0.02)
    finally:
        conn.close()
    return total


def upsert_candles_batched(db_path: str, rows: Sequence[Tuple], batch_size: int = 1000) -> int:
    """Upsert candle rows in short WAL transactions to avoid starving live writers."""
    return _upsert_batched(db_path, KLINE_SQL, rows, batch_size)


def upsert_funding_batched(db_path: str, rows: Sequence[Tuple], batch_size: int = 1000) -> int:
    """Upsert funding rate rows in short WAL transactions."""
    return _upsert_batched(db_path, FUNDING_SQL, rows, batch_size)


def _safe_query_one(cur: sqlite3.Cursor, query: str) -> Any:
    try:
        cur.execute(query)
        row = cur.fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        return None


def verify_backfill_integrity(
    db_path: str, symbols: Sequence[str], start_1h_ms: int, start_15m_ms: int,
) -> Dict[str, Any]:
    """Run read-only verification of counts, gaps, duplicates, integrity, and pilot state."""
    conn = sqlite3.connect(_ro_uri(db_path), uri=True, timeout=30.0)
    try:
        cur = conn.cursor()
        per_symbol: Dict[str, Any] = {}
        for raw_sym in symbols:
            sym = raw_sym.upper()
            sym_info: Dict[str, Any] = {}
            for tf, start_req in (("1h", start_1h_ms), ("15m", start_15m_ms)):
                tf_ms = TF_MS[tf]
                cur.execute(
                    "SELECT timestamp FROM candles WHERE exchange='binance' AND symbol=? AND timeframe=? ORDER BY timestamp ASC",
                    (sym, tf),
                )
                ts_list = [r[0] for r in cur.fetchall()]
                gaps, max_gap_bars = 0, 0
                for idx in range(1, len(ts_list)):
                    diff = ts_list[idx] - ts_list[idx - 1]
                    if diff > tf_ms:
                        gaps += 1
                        max_gap_bars = max(max_gap_bars, int(diff // tf_ms) - 1)
                sym_info[tf] = {
                    "count": len(ts_list), "min_ts": ts_list[0] if ts_list else None,
                    "max_ts": ts_list[-1] if ts_list else None, "requested_start_ms": start_req,
                    "gap_events": gaps, "max_gap_bars": max_gap_bars,
                    "max_gap_hours": round((max_gap_bars * tf_ms) / 3_600_000.0, 2),
                }
            cur.execute(
                "SELECT count(*), min(timestamp), max(timestamp) FROM funding_rates WHERE exchange='binance' AND symbol=?",
                (sym,),
            )
            f_cnt, f_min, f_max = cur.fetchone()
            sym_info["funding_rates"] = {"count": f_cnt, "min_ts": f_min, "max_ts": f_max}
            per_symbol[sym] = sym_info

        cur.execute(
            "SELECT count(*) FROM (SELECT symbol, timeframe, timestamp FROM candles "
            "GROUP BY symbol, timeframe, timestamp HAVING count(*) > 1)"
        )
        cross_exchange_dups = cur.fetchone()[0]
        cur.execute("PRAGMA integrity_check")
        integrity_res = cur.fetchone()[0]
        pilot_start_utc = _safe_query_one(cur, "SELECT value FROM simulation_settings WHERE key='pilot_start_utc'")
        open_positions_count = _safe_query_one(cur, "SELECT count(*) FROM paper_positions WHERE status='OPEN'")
        open_symbols: List[str] = []
        try:
            cur.execute("SELECT symbol FROM paper_positions WHERE status='OPEN' ORDER BY symbol")
            open_symbols = [r[0] for r in cur.fetchall()]
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()

    return {
        "db_path": db_path, "integrity_check": integrity_res,
        "cross_exchange_duplicates": cross_exchange_dups, "pilot_start_utc": pilot_start_utc,
        "open_positions_count": open_positions_count, "open_position_symbols": open_symbols,
        "symbols": per_symbol,
    }


def _parse_symbols(raw: Optional[Sequence[str]]) -> List[str]:
    if not raw:
        return list(CANDIDATE_MAJORS)
    out: List[str] = []
    for item in raw:
        for part in item.split(","):
            s = part.strip().upper()
            if s and s not in out:
                out.append(s)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Idempotent Binance Futures backfill for candidate majors")
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--backup-dir", default=DEFAULT_BACKUP_DIR)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--skip-backup", action="store_true", default=False)
    parser.add_argument("--dedup", action="store_true", default=False)
    parser.add_argument("--verify-only", action="store_true", default=False)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--start-1h-utc", default=DEFAULT_1H_START_UTC)
    parser.add_argument("--start-15m-utc", default=DEFAULT_15M_START_UTC)
    parser.add_argument("--start-funding-utc", default=DEFAULT_FUNDING_START_UTC)
    args = parser.parse_args()

    symbols = _parse_symbols(args.symbols)
    start_1h_ms, start_15m_ms = parse_utc_ms(args.start_1h_utc), parse_utc_ms(args.start_15m_utc)
    start_fund_ms, now_ms = parse_utc_ms(args.start_funding_utc), int(time.time() * 1000)

    report: Dict[str, Any] = {"symbols": symbols, "backup_path": None, "ingested": {}, "dedup": None, "verification": None}
    if not args.verify_only:
        if not args.skip_backup:
            report["backup_path"] = create_online_backup(args.db_path, args.backup_dir)
        for sym in symbols:
            rows_1h = fetch_klines_paginated(sym, "1h", start_1h_ms, now_ms)
            up_1h = upsert_candles_batched(args.db_path, rows_1h)
            rows_15m = fetch_klines_paginated(sym, "15m", start_15m_ms, now_ms)
            up_15m = upsert_candles_batched(args.db_path, rows_15m)
            rows_fund = fetch_funding_paginated(sym, start_fund_ms, now_ms)
            up_fund = upsert_funding_batched(args.db_path, rows_fund)
            report["ingested"][sym] = {
                "1h_fetched": len(rows_1h), "1h_upserted": up_1h,
                "15m_fetched": len(rows_15m), "15m_upserted": up_15m,
                "funding_fetched": len(rows_fund), "funding_upserted": up_fund,
            }
        if args.dedup:
            from scripts.audit_and_dedup_candles import deduplicate_candles
            report["dedup"] = deduplicate_candles(args.db_path, args.backup_dir)

    report["verification"] = verify_backfill_integrity(args.db_path, symbols, start_1h_ms, start_15m_ms)
    payload = json.dumps(report, indent=2)
    if args.json_out:
        out_dir = os.path.dirname(os.path.abspath(args.json_out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(payload)
    print(payload)


if __name__ == "__main__":
    main()
