"""
scripts/microstructure_collector.py — Coletor contínuo de microestrutura (dados públicos, SEM chave de API).

Constrói o histórico próprio que as APIs gratuitas não fornecem (livro de ofertas, fluxo de agressão, derivativos):
  * book_features  — a cada BOOK_EVERY s: spread, microprice, imbalance e profundidade (notional) por janela em bps (Bitget).
  * tick_agg       — agregação por minuto das negociações (compra/venda agressora, nº de trades, VWAP, flag de saturação).
  * derivs         — a cada DERIV_EVERY s: funding, open interest, mark/index e volume 24h (Bitget ticker).
  * binance_series — razão long/short (global e top traders), taker buy/sell e open interest (Binance, 5m).
                     Backfill inicial de ~30 dias (limite da Binance) + atualização incremental.
Banco separado (datasets/microstructure.db, WAL) para não competir com o cache do piloto.
Limitação conhecida: liquidações não têm endpoint REST histórico; ficam fora (exigiria WebSocket).

Uso:
  .venv/bin/python scripts/microstructure_collector.py           # roda continuamente
  .venv/bin/python scripts/microstructure_collector.py --status  # contagens e frescor
"""
import json
import logging
import os
import signal
import sqlite3
import sys
import threading
import time
import urllib.parse
import urllib.request
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from db_conn import connect as _db_connect
from scripts.binance_ws_collector import BinanceFuturesWSCollector

log = logging.getLogger("collector")
try:
    from agents.golden_pot_agent import NON_CRYPTO_BASE_EXCLUSIONS
except Exception as exc:  # noqa: BLE001
    log.error("Failed to import NON_CRYPTO_BASE_EXCLUSIONS: %s", exc)
    NON_CRYPTO_BASE_EXCLUSIONS = frozenset()

DB_PATH = os.path.join(ROOT, "datasets", "microstructure.db")
CACHE_DB = os.path.join(ROOT, "datasets", "market_cache.db")
BASE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SYMBOLS = list(BASE_SYMBOLS)
DEFAULT_GP_TOP_N = 6
MAX_ACTIVE_SYMBOLS = 16
BITGET = "https://api.bitget.com/api/v2/mix/market"
BINANCE = "https://fapi.binance.com/futures/data"
BINANCE_FAPI = "https://fapi.binance.com/fapi/v1"
PT = "USDT-FUTURES"
BINANCE_ONLY_SYMBOLS = set()
_BITGET_BACKOFF_UNTIL: Dict[str, float] = {}


def _bitget_in_backoff(sym: str) -> bool:
    return time.time() < _BITGET_BACKOFF_UNTIL.get(sym, 0.0)


def _set_bitget_backoff(sym: str, ttl_s: float = 300.0) -> None:
    _BITGET_BACKOFF_UNTIL[sym] = time.time() + ttl_s
EXCLUDED_GP_BASES = frozenset({
    "SKHYNIX", "SKHY", "SAMSUNG", "KORU", "SOXL", "SOXS", "TQQQ", "SQQQ", "SPXL", "SPXS", "UVXY",
    "TSLA", "NVDA", "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NFLX", "AMD", "INTC",
    "COIN", "MSTR", "HOOD", "PLTR", "BABA", "TSM", "SPY", "QQQ", "IWM", "GLD", "SLV", "USO",
    "SNXX", "SNDK", "UNITREE", "OPENAI", "SPACEX", "ANTHROPIC", "STRIPE", "FIGMA",
})
_ALL_EXCLUDED_GP_BASES = EXCLUDED_GP_BASES | NON_CRYPTO_BASE_EXCLUSIONS
BOOK_EVERY = 15.0
FILLS_EVERY = 6.0
DERIV_EVERY = 60.0
BINANCE_EVERY = 600.0
SYMBOLS_REFRESH_EVERY = 120.0
BOOK_WINDOWS_FINE = (2, 5, 10)        # bps (precision scale1, ~±13 bps no BTC)
BOOK_WINDOWS_COARSE = (25, 50, 100)   # bps (precision scale2, ~±120 bps no BTC)
BINANCE_KINDS = {
    "globalLongShortAccountRatio": "ls_global",
    "topLongShortPositionRatio": "ls_top_pos",
    "topLongShortAccountRatio": "ls_top_acc",
    "takerlongshortRatio": "taker",
    "openInterestHist": "oi",
}

STOP = threading.Event()

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS book_features (
  ts_ms INTEGER, symbol TEXT, mid REAL, spread_bps REAL, microprice_dev_bps REAL,
  bid_sz1 REAL, ask_sz1 REAL,
  bid_n2 REAL, ask_n2 REAL, bid_n5 REAL, ask_n5 REAL, bid_n10 REAL, ask_n10 REAL,
  bid_n25 REAL, ask_n25 REAL, bid_n50 REAL, ask_n50 REAL, bid_n100 REAL, ask_n100 REAL,
  PRIMARY KEY (ts_ms, symbol));
CREATE TABLE IF NOT EXISTS tick_agg (
  minute_ms INTEGER, symbol TEXT, buy_qty REAL, sell_qty REAL, buy_notional REAL, sell_notional REAL,
  n_trades INTEGER, vwap REAL, saturated INTEGER, PRIMARY KEY (minute_ms, symbol));
CREATE TABLE IF NOT EXISTS derivs (
  ts_ms INTEGER, symbol TEXT, funding_rate REAL, open_interest REAL, mark_price REAL, index_price REAL,
  last_price REAL, quote_volume_24h REAL, PRIMARY KEY (ts_ms, symbol));
CREATE TABLE IF NOT EXISTS liquidations (
  ts_ms INTEGER, symbol TEXT, side TEXT, price REAL, qty REAL, notional_usd REAL,
  PRIMARY KEY (ts_ms, symbol, side, price)
);
CREATE TABLE IF NOT EXISTS binance_series (
  ts_ms INTEGER, symbol TEXT, kind TEXT, v1 REAL, v2 REAL, v3 REAL, PRIMARY KEY (ts_ms, symbol, kind));
CREATE TABLE IF NOT EXISTS active_symbols (
  symbol TEXT PRIMARY KEY, updated_at INTEGER);
"""


def _is_excluded_gp(sym: str) -> bool:
    s = str(sym or "").upper().strip()
    for suf in ("USDT", "USDC", "USD", "PERP"):
        if s.endswith(suf) and len(s) > len(suf):
            s = s[:-len(suf)]
            break
    return s in _ALL_EXCLUDED_GP_BASES


def get_active_golden_pot_symbols(top_n: int = DEFAULT_GP_TOP_N) -> list:
    """Descobre até top_n altcoins mais frequentes/recentes na tabela golden_pots de market_cache.db."""
    if top_n <= 0 or not os.path.exists(CACHE_DB):
        return []
    conn = None
    try:
        conn = _db_connect(f"file:{CACHE_DB}?mode=ro", uri=True, timeout=5.0)
        rows = conn.execute(
            "SELECT symbol, count(*) as cnt "
            "FROM golden_pots "
            "GROUP BY symbol "
            "ORDER BY max(timestamp) DESC, cnt DESC "
            "LIMIT 60"
        ).fetchall()
        base_set = {s.upper() for s in BASE_SYMBOLS}
        seen = set()
        out = []
        limit_n = int(top_n)
        for r in rows:
            if not r or not r[0]:
                continue
            sym = str(r[0]).upper().strip()
            if not sym or sym in base_set or sym in seen:
                continue
            if _is_excluded_gp(sym):
                continue
            seen.add(sym)
            out.append(sym)
            if len(out) >= limit_n:
                break
        return out
    except Exception as e:
        log.warning("Falha ao consultar golden_pots no market_cache.db: %s", e)
        return []
    finally:
        if conn is not None:
            conn.close()


def connect():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("PRAGMA busy_timeout=30000")
    c.executescript(SCHEMA)
    return c


def get_json(url, params, timeout=8, retries=2):
    q = url + "?" + urllib.parse.urlencode(params)
    last = None
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(q, headers={"User-Agent": "ia-trader-collector/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception as e:   # noqa: BLE001
            last = e
            time.sleep(0.5 * (i + 1))
    raise RuntimeError(f"{url} falhou: {last}")


def depth_notional(levels, mid, windows, side):
    """Soma de notional (preço*qty) dentro de cada janela em bps a partir do mid. NaN se o livro não cobre a janela."""
    out = {}
    if not levels:
        return {w: None for w in windows}
    far = abs(float(levels[-1][0]) - mid) / mid * 1e4
    for w in windows:
        if far < w:
            out[w] = None
            continue
        tot = 0.0
        for p, q in levels:
            p, q = float(p), float(q)
            if abs(p - mid) / mid * 1e4 <= w:
                tot += p * q
        out[w] = tot
    return out


def book_snapshot_binance(sym):
    d = get_json(BINANCE_FAPI + "/depth", {"symbol": sym, "limit": 500}, retries=1)
    asks, bids = d.get("asks") or [], d.get("bids") or []
    if not asks or not bids:
        raise RuntimeError(f"binance depth vazio para {sym}")
    ba, bb = float(asks[0][0]), float(bids[0][0])
    sa, sb = float(asks[0][1]), float(bids[0][1])
    mid = (ba + bb) / 2
    micro = (ba * sb + bb * sa) / (sa + sb) if (sa + sb) > 0 else mid
    f_a = depth_notional(asks, mid, BOOK_WINDOWS_FINE, "ask")
    f_b = depth_notional(bids, mid, BOOK_WINDOWS_FINE, "bid")
    c_a = depth_notional(asks, mid, BOOK_WINDOWS_COARSE, "ask")
    c_b = depth_notional(bids, mid, BOOK_WINDOWS_COARSE, "bid")
    ts = int(d.get("T") or d.get("E") or time.time() * 1000)
    return (ts, sym, mid, (ba - bb) / mid * 1e4, (micro - mid) / mid * 1e4, sb, sa,
            f_b[2], f_a[2], f_b[5], f_a[5], f_b[10], f_a[10],
            c_b[25], c_a[25], c_b[50], c_a[50], c_b[100], c_a[100])


def book_snapshot(sym):
    if sym in BINANCE_ONLY_SYMBOLS or _bitget_in_backoff(sym):
        return book_snapshot_binance(sym)
    try:
        fine = get_json(BITGET + "/merge-depth", {"symbol": sym, "productType": PT, "precision": "scale1", "limit": "max"}, retries=1)["data"]
        coarse = get_json(BITGET + "/merge-depth", {"symbol": sym, "productType": PT, "precision": "scale2", "limit": "max"}, retries=1)["data"]
        # topo do livro preciso: usa a melhor precisão disponível
        top = get_json(BITGET + "/merge-depth", {"symbol": sym, "productType": PT, "precision": "scale0", "limit": "5"}, retries=1)["data"]
        ba, bb = float(top["asks"][0][0]), float(top["bids"][0][0])
        sa, sb = float(top["asks"][0][1]), float(top["bids"][0][1])
        mid = (ba + bb) / 2
        micro = (ba * sb + bb * sa) / (sa + sb) if (sa + sb) > 0 else mid
        f_a = depth_notional(fine["asks"], mid, BOOK_WINDOWS_FINE, "ask")
        f_b = depth_notional(fine["bids"], mid, BOOK_WINDOWS_FINE, "bid")
        c_a = depth_notional(coarse["asks"], mid, BOOK_WINDOWS_COARSE, "ask")
        c_b = depth_notional(coarse["bids"], mid, BOOK_WINDOWS_COARSE, "bid")
        ts = int(top.get("ts") or time.time() * 1000)
        return (ts, sym, mid, (ba - bb) / mid * 1e4, (micro - mid) / mid * 1e4, sb, sa,
                f_b[2], f_a[2], f_b[5], f_a[5], f_b[10], f_a[10],
                c_b[25], c_a[25], c_b[50], c_a[50], c_b[100], c_a[100])
    except Exception:
        _set_bitget_backoff(sym)
        return book_snapshot_binance(sym)


class TickAggregator:
    def __init__(self):
        self._lock = threading.Lock()
        self.last_id = {}
        self.buckets = {}   # (sym, minute_ms) -> dict

    def ingest_ws_trade(self, sym: str, ts_ms: int, price: float, qty: float, is_buyer_maker: bool) -> None:
        m = ts_ms - ts_ms % 60000
        with self._lock:
            b = self.buckets.setdefault((sym, m), dict(bq=0.0, sq=0.0, bn=0.0, sn=0.0, n=0, sat=0))
            if is_buyer_maker:
                b["sq"] += qty
                b["sn"] += price * qty
            else:
                b["bq"] += qty
                b["bn"] += price * qty
            b["n"] += 1

    def _poll_binance(self, sym):
        rows = get_json(BINANCE_FAPI + "/aggTrades", {"symbol": sym, "limit": 1000}, retries=1)
        if not rows:
            return 0
        with self._lock:
            last_seen = self.last_id.get(("binance", sym), 0)
        new = [r for r in rows if int(r["a"]) > last_seen]
        if not new:
            return 0
        all_new = new
        max_id = max(int(r["a"]) for r in new)
        saturated = (len(new) == len(rows) and last_seen != 0)
        pages = 1
        while saturated and pages < 3:
            next_rows = get_json(
                BINANCE_FAPI + "/aggTrades",
                {"symbol": sym, "limit": 1000, "fromId": max_id + 1},
                retries=1,
            )
            if not next_rows:
                break
            next_new = [r for r in next_rows if int(r["a"]) > max_id]
            if not next_new:
                break
            all_new.extend(next_new)
            max_id = max(int(r["a"]) for r in next_new)
            saturated = len(next_new) == len(next_rows)
            pages += 1
        first_new_id = min(int(r["a"]) for r in all_new)
        gap_suspected = (last_seen != 0 and first_new_id > last_seen + 1)
        still_saturated = saturated or gap_suspected
        with self._lock:
            for r in sorted(all_new, key=lambda r: int(r["a"])):
                ts = int(r["T"])
                m = ts - ts % 60000
                b = self.buckets.setdefault((sym, m), dict(bq=0.0, sq=0.0, bn=0.0, sn=0.0, n=0, sat=0))
                p, q = float(r["p"]), float(r["q"])
                # m=True no aggTrades da Binance significa que o comprador é o maker -> agressão de VENDA (sell)
                if not r.get("m", False):
                    b["bq"] += q; b["bn"] += p * q
                else:
                    b["sq"] += q; b["sn"] += p * q
                b["n"] += 1
            if still_saturated:
                last_ts = int(max(all_new, key=lambda r: int(r["a"]))["T"])
                self.buckets[(sym, last_ts - last_ts % 60000)]["sat"] = 1
            self.last_id[("binance", sym)] = max(int(r["a"]) for r in all_new)
        return len(all_new)

    def poll(self, sym):
        if sym in BINANCE_ONLY_SYMBOLS or _bitget_in_backoff(sym):
            return self._poll_binance(sym)
        try:
            rows = get_json(BITGET + "/fills", {"symbol": sym, "productType": PT, "limit": "100"}, retries=1)["data"]
        except Exception:
            _set_bitget_backoff(sym)
            return self._poll_binance(sym)
        with self._lock:
            last_seen = self.last_id.setdefault(("bitget", sym), 0)
        new = [r for r in rows if int(r["tradeId"]) > last_seen]
        if not new:
            return 0
        saturated = (len(new) == len(rows) and last_seen != 0)
        with self._lock:
            for r in sorted(new, key=lambda r: int(r["tradeId"])):
                ts = int(r["ts"])
                m = ts - ts % 60000
                b = self.buckets.setdefault((sym, m), dict(bq=0.0, sq=0.0, bn=0.0, sn=0.0, n=0, sat=0))
                p, q = float(r["price"]), float(r["size"])
                if r["side"] == "buy":
                    b["bq"] += q; b["bn"] += p * q
                else:
                    b["sq"] += q; b["sn"] += p * q
                b["n"] += 1
            if saturated:
                last_ts = int(max(new, key=lambda r: int(r["tradeId"]))["ts"])
                self.buckets[(sym, last_ts - last_ts % 60000)]["sat"] = 1
            self.last_id[("bitget", sym)] = max(int(r["tradeId"]) for r in new)
        return len(new)

    def flush(self, conn, force=False):
        now_min = int(time.time() * 1000) // 60000 * 60000
        with self._lock:
            out = []
            for (sym, m), b in list(self.buckets.items()):
                if force or m < now_min:
                    tot_q = b["bq"] + b["sq"]
                    vwap = (b["bn"] + b["sn"]) / tot_q if tot_q > 0 else None
                    out.append((m, sym, b["bq"], b["sq"], b["bn"], b["sn"], b["n"], vwap, b["sat"]))
                    del self.buckets[(sym, m)]
        if out:
            conn.executemany("INSERT OR REPLACE INTO tick_agg VALUES (?,?,?,?,?,?,?,?,?)", out)
            conn.commit()
        return len(out)


def deriv_snapshot_binance(sym):
    prem = get_json(BINANCE_FAPI + "/premiumIndex", {"symbol": sym}, retries=1)
    oi = get_json(BINANCE_FAPI + "/openInterest", {"symbol": sym}, retries=1)
    t24 = get_json(BINANCE_FAPI + "/ticker/24hr", {"symbol": sym}, retries=1)
    ts = int(prem.get("time") or time.time() * 1000)
    return (
        ts, sym,
        float(prem.get("lastFundingRate") or 0.0),
        float(oi.get("openInterest") or 0.0),
        float(prem.get("markPrice") or 0.0),
        float(prem.get("indexPrice") or 0.0),
        float(t24.get("lastPrice") or prem.get("markPrice") or 0.0),
        float(t24.get("quoteVolume") or 0.0),
    )


def deriv_snapshot(sym):
    if sym in BINANCE_ONLY_SYMBOLS or _bitget_in_backoff(sym):
        return deriv_snapshot_binance(sym)
    try:
        d = get_json(BITGET + "/ticker", {"symbol": sym, "productType": PT}, retries=1)["data"][0]
        oi = get_json(BITGET + "/open-interest", {"symbol": sym, "productType": PT}, retries=1)["data"]["openInterestList"][0]["size"]
        return (int(d["ts"]), sym, float(d["fundingRate"]), float(oi), float(d["markPrice"]), float(d["indexPrice"]),
                float(d["lastPr"]), float(d["quoteVolume"]))
    except Exception:
        _set_bitget_backoff(sym)
        return deriv_snapshot_binance(sym)


def binance_fetch(conn, sym, endpoint, kind, start_ms=None, end_ms=None, limit=500):
    p = {"symbol": sym, "period": "5m", "limit": limit}
    if start_ms:
        p["startTime"] = start_ms
    if end_ms:
        p["endTime"] = end_ms
    rows = get_json(f"{BINANCE}/{endpoint}", p)
    out = []
    for r in rows:
        ts = int(r["timestamp"])
        if kind == "oi":
            out.append((ts, sym, kind, float(r["sumOpenInterest"]), float(r["sumOpenInterestValue"]), None))
        elif kind == "taker":
            out.append((ts, sym, kind, float(r["buySellRatio"]), float(r["buyVol"]), float(r["sellVol"])))
        else:
            out.append((ts, sym, kind, float(r["longShortRatio"]), float(r["longAccount"]), float(r["shortAccount"])))
    if out:
        conn.executemany("INSERT OR IGNORE INTO binance_series VALUES (?,?,?,?,?,?)", out)
        conn.commit()
    return len(out), (min(r[0] for r in out) if out else None)


def binance_backfill_and_update():
    """Thread separada: backfill de ~30 dias (1ª vez) e atualização incremental a cada BINANCE_EVERY."""
    conn = connect()
    horizon_ms = 30 * 24 * 3600 * 1000
    first = True
    while not STOP.is_set():
        now = int(time.time() * 1000)
        for sym in SYMBOLS:
            for endpoint, kind in BINANCE_KINDS.items():
                if STOP.is_set():
                    return
                try:
                    last = conn.execute("SELECT MAX(ts_ms), MIN(ts_ms) FROM binance_series WHERE symbol=? AND kind=?", (sym, kind)).fetchone()
                    mx, mn = last
                    if mx is None:
                        # backfill pra trás, em páginas de 500 pontos de 5m
                        end = now
                        total = 0
                        while end > now - horizon_ms and not STOP.is_set():
                            n, oldest = binance_fetch(conn, sym, endpoint, kind, end_ms=end)
                            total += n
                            if not n or oldest is None:
                                break
                            end = oldest - 1
                            time.sleep(0.25)
                        log.info("backfill %s %s: %d pontos", sym, kind, total)
                    else:
                        n, _ = binance_fetch(conn, sym, endpoint, kind, start_ms=mx + 1, limit=500)
                        if n:
                            log.debug("binance %s %s +%d", sym, kind, n)
                        time.sleep(0.2)
                except Exception as e:   # noqa: BLE001
                    log.warning("binance %s %s: %s", sym, kind, e)
        first = False
        STOP.wait(BINANCE_EVERY)


def status():
    if not os.path.exists(DB_PATH):
        print("microstructure.db não encontrado:", DB_PATH)
        return
    conn = _db_connect(f"file:{os.path.abspath(DB_PATH)}?mode=ro", uri=True, timeout=5.0)
    try:
        now = time.time() * 1000
        print("microstructure.db:", DB_PATH, f"({os.path.getsize(DB_PATH) / 1e6:.1f} MB)")
        tables_cols = (
            ("book_features", "ts_ms"),
            ("tick_agg", "minute_ms"),
            ("derivs", "ts_ms"),
            ("binance_series", "ts_ms"),
            ("liquidations", "ts_ms"),
        )
        for t, col in tables_cols:
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone()
            if not exists:
                print(f"  {t:15s} linhas=        0  cobertura=          -  último=tabela ausente")
                continue
            n, mn, mx = conn.execute(f"SELECT COUNT(*), MIN({col}), MAX({col}) FROM {t}").fetchone()
            age = f"{(now - mx) / 1000:.0f}s atrás" if mx else "-"
            span = f"{(mx - mn) / 86400000:.2f} dias" if mn else "-"
            print(f"  {t:15s} linhas={n:9d}  cobertura={span:>11s}  último={age}")
        for r in conn.execute("SELECT kind, symbol, COUNT(*) FROM binance_series GROUP BY kind, symbol ORDER BY kind, symbol"):
            print("   binance", r)
        print("Saturação tick_agg 24h (saturados / minutos):")
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            total, sat = conn.execute(
                "SELECT COUNT(*), SUM(CASE WHEN saturated=1 THEN 1 ELSE 0 END) FROM tick_agg "
                "WHERE symbol=? AND minute_ms >= ?",
                (sym, now - 24 * 3600 * 1000),
            ).fetchone()
            total = total or 0
            sat = sat or 0
            pct = f"{100.0 * sat / total:.1f}%" if total else "-"
            print(f"  {sym:10s} minutos={total:5d}  saturados={sat:5d}  ({pct})")
    finally:
        conn.close()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [collector] %(message)s")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    conn = connect()
    agg = TickAggregator()
    threading.Thread(target=binance_backfill_and_update, daemon=True, name="binance").start()
    active_symbols = list(BASE_SYMBOLS)
    ws_collector = BinanceFuturesWSCollector(symbols=active_symbols, on_trade=agg.ingest_ws_trade, stop_event=STOP)
    ws_collector.start()
    log.info("Coletor iniciado: %s | book=%ss fills=%ss deriv=%ss", active_symbols, BOOK_EVERY, FILLS_EVERY, DERIV_EVERY)
    due = {"book": 0.0, "fills": 0.0, "deriv": 0.0, "report": time.time() + 60, "symbols": 0.0}
    counts = dict(book=0, fills=0, deriv=0, errors=0)
    while not STOP.is_set():
        now = time.time()
        if now >= due["symbols"]:
            due["symbols"] = now + SYMBOLS_REFRESH_EVERY
            gp_syms = get_active_golden_pot_symbols(top_n=DEFAULT_GP_TOP_N)
            current_syms = list(dict.fromkeys(BASE_SYMBOLS + gp_syms))[:MAX_ACTIVE_SYMBOLS]
            if set(current_syms) != set(active_symbols):
                log.info("Expansão dinâmica de cobertura: %s -> %s", active_symbols, current_syms)
                active_symbols = current_syms
                ws_collector.update_symbols(active_symbols)
            try:
                conn.executemany(
                    "INSERT OR REPLACE INTO active_symbols VALUES (?,?)",
                    [(s, int(now * 1000)) for s in active_symbols]
                )
                conn.commit()
            except Exception as e:
                log.warning("Erro ao atualizar active_symbols: %s", e)

        if now >= due["book"]:
            due["book"] = now + BOOK_EVERY
            rows = []
            for s in list(active_symbols):
                try:
                    rows.append(book_snapshot(s))
                except Exception as e:   # noqa: BLE001
                    counts["errors"] += 1; log.warning("book %s: %s", s, e)
            if rows:
                conn.executemany("INSERT OR REPLACE INTO book_features VALUES (" + ",".join("?" * 19) + ")", rows)
                conn.commit(); counts["book"] += len(rows)
        if now >= due["fills"]:
            due["fills"] = now + FILLS_EVERY
            for s in list(active_symbols):
                try:
                    if ws_collector.is_symbol_live(s, max_age_s=20.0):
                        # WebSocket está alimentando tick_agg em tempo real (sat=0, sem REST).
                        pass
                    else:
                        counts["fills"] += agg.poll(s)
                except Exception as e:   # noqa: BLE001
                    counts["errors"] += 1; log.warning("fills %s: %s", s, e)
            agg.flush(conn)
            liq_rows = ws_collector.drain_liquidations()
            if liq_rows:
                conn.executemany("INSERT OR IGNORE INTO liquidations VALUES (?,?,?,?,?,?)", liq_rows)
                conn.commit()
        if now >= due["deriv"]:
            due["deriv"] = now + DERIV_EVERY
            rows = []
            for s in list(active_symbols):
                try:
                    rows.append(deriv_snapshot(s))
                except Exception as e:   # noqa: BLE001
                    counts["errors"] += 1; log.warning("deriv %s: %s", s, e)
            if rows:
                conn.executemany("INSERT OR REPLACE INTO derivs VALUES (?,?,?,?,?,?,?,?)", rows)
                conn.commit(); counts["deriv"] += len(rows)
        if now >= due["report"]:
            due["report"] = now + 300
            log.info("heartbeat: %s (símbolos ativos=%s)", counts, active_symbols)
        STOP.wait(1.0)
    ws_collector.stop()
    agg.flush(conn, force=True)
    conn.close()
    log.info("Coletor encerrado.")


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    else:
        main()
