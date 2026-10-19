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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "datasets", "microstructure.db")
CACHE_DB = os.path.join(ROOT, "datasets", "market_cache.db")
BASE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
SYMBOLS = list(BASE_SYMBOLS)
BITGET = "https://api.bitget.com/api/v2/mix/market"
BINANCE = "https://fapi.binance.com/futures/data"
PT = "USDT-FUTURES"
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

log = logging.getLogger("collector")
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
CREATE TABLE IF NOT EXISTS binance_series (
  ts_ms INTEGER, symbol TEXT, kind TEXT, v1 REAL, v2 REAL, v3 REAL, PRIMARY KEY (ts_ms, symbol, kind));
CREATE TABLE IF NOT EXISTS active_symbols (
  symbol TEXT PRIMARY KEY, updated_at INTEGER);
"""


def get_active_golden_pot_symbols(top_n: int = 3) -> list:
    """Descobre até top_n altcoins mais frequentes/recentes na tabela golden_pots de market_cache.db."""
    if not os.path.exists(CACHE_DB):
        return []
    try:
        with sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True, timeout=5.0) as conn:
            rows = conn.execute("""
                SELECT symbol, count(*) as cnt 
                FROM golden_pots 
                WHERE symbol NOT IN ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')
                GROUP BY symbol 
                ORDER BY max(timestamp) DESC, cnt DESC 
                LIMIT ?
            """, (top_n,)).fetchall()
            return [str(r[0]).upper() for r in rows if r[0]]
    except Exception as e:
        log.warning("Falha ao consultar golden_pots no market_cache.db: %s", e)
        return []


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


def book_snapshot(sym):
    fine = get_json(BITGET + "/merge-depth", {"symbol": sym, "productType": PT, "precision": "scale1", "limit": "max"})["data"]
    coarse = get_json(BITGET + "/merge-depth", {"symbol": sym, "productType": PT, "precision": "scale2", "limit": "max"})["data"]
    # topo do livro preciso: usa a melhor precisão disponível
    top = get_json(BITGET + "/merge-depth", {"symbol": sym, "productType": PT, "precision": "scale0", "limit": "5"})["data"]
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


class TickAggregator:
    def __init__(self):
        self.last_id = {s: 0 for s in BASE_SYMBOLS}
        self.buckets = {}   # (sym, minute_ms) -> dict

    def poll(self, sym):
        rows = get_json(BITGET + "/fills", {"symbol": sym, "productType": PT, "limit": "100"})["data"]
        last_seen = self.last_id.setdefault(sym, 0)
        new = [r for r in rows if int(r["tradeId"]) > last_seen]
        if not new:
            return 0
        saturated = (len(new) == len(rows) and last_seen != 0)
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
        self.last_id[sym] = max(int(r["tradeId"]) for r in new)
        return len(new)

    def flush(self, conn, force=False):
        now_min = int(time.time() * 1000) // 60000 * 60000
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


def deriv_snapshot(sym):
    d = get_json(BITGET + "/ticker", {"symbol": sym, "productType": PT})["data"][0]
    oi = get_json(BITGET + "/open-interest", {"symbol": sym, "productType": PT})["data"]["openInterestList"][0]["size"]
    return (int(d["ts"]), sym, float(d["fundingRate"]), float(oi), float(d["markPrice"]), float(d["indexPrice"]),
            float(d["lastPr"]), float(d["quoteVolume"]))


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
    conn = connect()
    now = time.time() * 1000
    print("microstructure.db:", DB_PATH, f"({os.path.getsize(DB_PATH) / 1e6:.1f} MB)")
    for t, col in (("book_features", "ts_ms"), ("tick_agg", "minute_ms"), ("derivs", "ts_ms"), ("binance_series", "ts_ms")):
        n, mn, mx = conn.execute(f"SELECT COUNT(*), MIN({col}), MAX({col}) FROM {t}").fetchone()
        age = f"{(now - mx) / 1000:.0f}s atrás" if mx else "-"
        span = f"{(mx - mn) / 86400000:.2f} dias" if mn else "-"
        print(f"  {t:15s} linhas={n:9d}  cobertura={span:>11s}  último={age}")
    for r in conn.execute("SELECT kind, symbol, COUNT(*) FROM binance_series GROUP BY kind, symbol ORDER BY kind, symbol"):
        print("   binance", r)


def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [collector] %(message)s")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    conn = connect()
    agg = TickAggregator()
    threading.Thread(target=binance_backfill_and_update, daemon=True, name="binance").start()
    active_symbols = list(BASE_SYMBOLS)
    log.info("Coletor iniciado: %s | book=%ss fills=%ss deriv=%ss", active_symbols, BOOK_EVERY, FILLS_EVERY, DERIV_EVERY)
    due = {"book": 0.0, "fills": 0.0, "deriv": 0.0, "report": time.time() + 60, "symbols": 0.0}
    counts = dict(book=0, fills=0, deriv=0, errors=0)
    while not STOP.is_set():
        now = time.time()
        if now >= due["symbols"]:
            due["symbols"] = now + SYMBOLS_REFRESH_EVERY
            gp_syms = get_active_golden_pot_symbols(top_n=3)
            current_syms = list(dict.fromkeys(BASE_SYMBOLS + gp_syms))
            if set(current_syms) != set(active_symbols):
                log.info("Expansão dinâmica de cobertura: %s -> %s", active_symbols, current_syms)
                active_symbols = current_syms
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
                    counts["fills"] += agg.poll(s)
                except Exception as e:   # noqa: BLE001
                    counts["errors"] += 1; log.warning("fills %s: %s", s, e)
            agg.flush(conn)
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
    agg.flush(conn, force=True)
    conn.close()
    log.info("Coletor encerrado.")


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    else:
        main()
