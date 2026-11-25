"""Hub de Preços em Tempo Real (Binance Futures WS + Batch REST Fallback).
Conexão WS única em thread daemon asyncio com TTL de interesse (120s) e debounce de 0.5s.
NUNCA inicia thread/WS ao importar o módulo.
"""
from __future__ import annotations

import asyncio, json, logging, math, threading, time, urllib.request
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("ia_trader.live_price_hub")
WS_BASE_URL = "wss://fstream.binance.com/market/stream"
REST_BATCH_PRICE_URL = "https://fapi.binance.com/fapi/v1/ticker/price"


class LivePriceHub:
    """Cache RAM thread-safe de tickers 24h e velas abertas em tempo real."""

    def __init__(self, ws_base_url: str = WS_BASE_URL, interest_ttl_s: float = 120.0,
                 reconnect_debounce_s: float = 0.5, rest_batch_ttl_s: float = 1.5) -> None:
        self.ws_base_url, self.interest_ttl_s = ws_base_url, float(interest_ttl_s)
        self.reconnect_debounce_s, self.rest_batch_ttl_s = float(reconnect_debounce_s), float(rest_batch_ttl_s)
        self._lock, self._rest_lock = threading.Lock(), threading.Lock()
        self._stop_event, self._thread, self._running = threading.Event(), None, False
        self._symbol_interest: Dict[str, float] = {}
        self._kline_interest: Dict[Tuple[str, str], float] = {}
        self._tickers: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._klines: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}
        self._rest_batch_ts, self._rest_batch_prices = 0.0, {}

    @property
    def is_running(self) -> bool:
        with self._lock:
            return bool(self._running and self._thread is not None and self._thread.is_alive())

    def start(self) -> None:
        with self._lock:
            if self._running and self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._running = True
            self._thread = threading.Thread(target=self._run_thread, name="LivePriceHubWS", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        with self._lock:
            self._running = False
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            if thread is not None and self._thread is thread and not thread.is_alive():
                self._thread = None

    def clear(self) -> None:
        with self._lock:
            self._tickers.clear(); self._klines.clear()
            self._symbol_interest.clear(); self._kline_interest.clear()
        with self._rest_lock:
            self._rest_batch_ts = 0.0; self._rest_batch_prices.clear()

    def watch_symbols(self, symbols: Iterable[str]) -> None:
        expiry = time.monotonic() + self.interest_ttl_s
        with self._lock:
            for raw in symbols:
                sym = str(raw or "").strip().upper()
                if sym:
                    self._symbol_interest[sym] = expiry

    def watch_kline(self, symbol: str, timeframe: str) -> None:
        sym, tf = str(symbol or "").strip().upper(), str(timeframe or "").strip().lower()
        if not sym:
            return
        expiry = time.monotonic() + self.interest_ttl_s
        with self._lock:
            self._symbol_interest[sym] = expiry
            if tf:
                self._kline_interest[(sym, tf)] = expiry

    def inject_price(
        self, symbol: str, price: float, open_24h: Optional[float] = None,
        high_24h: Optional[float] = None, low_24h: Optional[float] = None,
        volume_24h: float = 0.0, change_pct_24h: Optional[float] = None, ts_ms: Optional[int] = None,
    ) -> None:
        sym, px = str(symbol or "").strip().upper(), float(price)
        if not sym or not math.isfinite(px) or px <= 0.0:
            return
        o24 = float(open_24h) if (open_24h is not None and float(open_24h) > 0) else px
        h24 = float(high_24h) if (high_24h is not None and float(high_24h) > 0) else px
        l24 = float(low_24h) if (low_24h is not None and float(low_24h) > 0) else px
        chg = round(float(change_pct_24h), 2) if change_pct_24h is not None else (
            round(((px - o24) / o24) * 100.0, 2) if o24 > 0 else 0.0
        )
        payload = {
            "symbol": sym, "last_price": px, "open_24h": o24, "high_24h": h24, "low_24h": l24,
            "volume_24h": float(volume_24h or 0.0), "change_pct_24h": chg,
            "ts_ms": int(ts_ms) if ts_ms is not None else int(time.time() * 1000),
        }
        with self._lock:
            self._tickers[sym] = (time.monotonic(), payload)

    def inject_kline(self, symbol: str, timeframe: str, candle: Dict[str, Any]) -> None:
        sym, tf = str(symbol or "").strip().upper(), str(timeframe or "").strip().lower()
        if not sym or not tf or not isinstance(candle, dict):
            return
        ts = int(candle.get("timestamp") or 0) or (int(candle["time"]) * 1000 if candle.get("time") else 0)
        norm = {
            "symbol": sym, "timeframe": tf, "timestamp": ts,
            "open": float(candle.get("open", 0.0)), "high": float(candle.get("high", 0.0)),
            "low": float(candle.get("low", 0.0)), "close": float(candle.get("close", 0.0)),
            "volume": float(candle.get("volume", 0.0)), "quote_volume": float(candle.get("quote_volume", 0.0)),
            "is_closed": bool(candle.get("is_closed", False)),
        }
        with self._lock:
            self._klines[(sym, tf)] = (time.monotonic(), norm)

    def has_fresh_price(self, symbol: str, max_age_s: float = 10.0) -> bool:
        sym = str(symbol or "").strip().upper()
        if not sym:
            return False
        now = time.monotonic()
        with self._lock:
            entry = self._tickers.get(sym)
            return bool(entry and (now - entry[0]) <= max_age_s and float(entry[1].get("last_price", 0.0)) > 0.0)

    def get_ticker(self, symbol: str, max_age_s: float = 10.0) -> Optional[Dict[str, Any]]:
        sym = str(symbol or "").strip().upper()
        if not sym:
            return None
        now = time.monotonic()
        with self._lock:
            if self._running:
                self._symbol_interest[sym] = now + self.interest_ttl_s
            entry = self._tickers.get(sym)
            if entry and (now - entry[0]) <= max_age_s and float(entry[1].get("last_price", 0.0)) > 0.0:
                return dict(entry[1])
            return None

    def get_open_candle(self, symbol: str, timeframe: str, max_age_s: float = 10.0) -> Optional[Dict[str, Any]]:
        sym, tf = str(symbol or "").strip().upper(), str(timeframe or "").strip().lower()
        if not sym or not tf:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._klines.get((sym, tf))
            if entry and (now - entry[0]) <= max_age_s and float(entry[1].get("close", 0.0)) > 0.0:
                return dict(entry[1])
            return None

    def _fetch_rest_batch_prices(self) -> Dict[str, float]:
        now = time.monotonic()
        with self._rest_lock:
            if self._rest_batch_prices and (now - self._rest_batch_ts) <= self.rest_batch_ttl_s:
                return dict(self._rest_batch_prices)
            try:
                req = urllib.request.Request(REST_BATCH_PRICE_URL, headers={"User-Agent": "ia_trader/1.0"})
                with urllib.request.urlopen(req, timeout=2.0) as resp:
                    raw = json.loads(resp.read().decode())
                parsed = {str(it["symbol"]).upper(): float(it["price"]) for it in (raw if isinstance(raw, list) else [])
                          if it.get("symbol") and float(it.get("price") or 0.0) > 0.0}
                if parsed:
                    self._rest_batch_prices, self._rest_batch_ts = parsed, time.monotonic()
                    return dict(parsed)
            except Exception as exc:
                logger.debug("LivePriceHub batch REST fallback falhou: %s", exc)
            return dict(self._rest_batch_prices)

    def get_prices(self, symbols: Iterable[str], max_age_s: float = 10.0, allow_rest_fallback: bool = True) -> Dict[str, float]:
        wanted = list(dict.fromkeys(str(s or "").strip().upper() for s in symbols if str(s or "").strip()))
        if not wanted:
            return {}
        now, result, missing = time.monotonic(), {}, []
        with self._lock:
            for sym in wanted:
                entry = self._tickers.get(sym)
                if entry and (now - entry[0]) <= max_age_s and float(entry[1].get("last_price", 0.0)) > 0.0:
                    result[sym] = float(entry[1]["last_price"])
                else:
                    missing.append(sym)
        if missing and allow_rest_fallback:
            batch = self._fetch_rest_batch_prices()
            for sym in missing:
                if batch.get(sym, 0.0) > 0.0:
                    result[sym] = batch[sym]
        return result

    def _desired_streams(self) -> Tuple[str, ...]:
        now = time.monotonic()
        with self._lock:
            self._symbol_interest = {s: exp for s, exp in self._symbol_interest.items() if exp > now}
            self._kline_interest = {k: exp for k, exp in self._kline_interest.items() if exp > now}
            streams = {f"{s.lower()}@miniTicker" for s in self._symbol_interest}
            streams.update(f"{s.lower()}@kline_{tf}" for s, tf in self._kline_interest)
        return tuple(sorted(streams))

    def _handle_ws_message(self, raw_msg: str) -> None:
        try: msg = json.loads(raw_msg)
        except Exception: return
        if not isinstance(msg, dict): return
        p = msg.get("data") if isinstance(msg.get("data"), dict) else msg
        ev = str(p.get("e") or "")
        try:
            if ev == "24hrMiniTicker" and p.get("s"):
                self.inject_price(
                    str(p["s"]), float(p.get("c") or 0.0), open_24h=float(p.get("o") or 0.0),
                    high_24h=float(p.get("h") or 0.0), low_24h=float(p.get("l") or 0.0),
                    volume_24h=float(p.get("v") or 0.0), ts_ms=int(p.get("E") or int(time.time() * 1000)),
                )
            elif ev == "kline" and isinstance(p.get("k"), dict):
                k = p["k"]
                sym, tf = str(k.get("s") or p.get("s") or "").upper(), str(k.get("i") or "").lower()
                if sym and tf:
                    self.inject_kline(sym, tf, {
                        "timestamp": int(k.get("t") or 0), "open": float(k.get("o") or 0.0),
                        "high": float(k.get("h") or 0.0), "low": float(k.get("l") or 0.0),
                        "close": float(k.get("c") or 0.0), "volume": float(k.get("v") or 0.0),
                        "quote_volume": float(k.get("q") or 0.0), "is_closed": bool(k.get("x", False)),
                    })
        except (TypeError, ValueError):
            return

    def _run_thread(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_loop())
        except Exception as exc:
            logger.debug("LivePriceHub loop encerrado: %s", exc)
        finally:
            try: loop.close()
            except Exception: pass
            with self._lock:
                self._running = False

    async def _ws_loop(self) -> None:
        try:
            import websockets
        except Exception as exc:
            logger.warning("LivePriceHub: websockets indisponível (%s)", exc)
            return
        backoff_s = 1.0
        while not self._stop_event.is_set():
            streams = self._desired_streams()
            if not streams:
                await asyncio.sleep(0.25)
                continue
            url = f"{self.ws_base_url}?streams={'/'.join(streams)}"
            reconnecting_for_streams, pending_since = False, None
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=2) as ws:
                    backoff_s = 1.0
                    while not self._stop_event.is_set():
                        try:
                            self._handle_ws_message(await asyncio.wait_for(ws.recv(), timeout=0.25))
                        except asyncio.TimeoutError:
                            pass
                        if self._desired_streams() != streams:
                            pending_since = pending_since or time.monotonic()
                            if (time.monotonic() - pending_since) >= self.reconnect_debounce_s:
                                reconnecting_for_streams = True
                                break
                        else:
                            pending_since = None
            except Exception as exc:
                if not self._stop_event.is_set():
                    logger.debug("LivePriceHub WS desconectado (%s), reconectando em %.1fs", exc, backoff_s)
            if not reconnecting_for_streams and not self._stop_event.is_set():
                end_t = time.monotonic() + backoff_s
                while not self._stop_event.is_set() and time.monotonic() < end_t:
                    await asyncio.sleep(min(0.25, max(0.0, end_t - time.monotonic())))
                backoff_s = min(backoff_s * 2.0, 15.0)


LIVE_PRICE_HUB = LivePriceHub()


def start_live_price_hub() -> LivePriceHub:
    LIVE_PRICE_HUB.start(); return LIVE_PRICE_HUB
