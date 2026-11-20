import asyncio
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("collector.ws")


class BinanceFuturesWSCollector:
    def __init__(
        self,
        symbols: Iterable[str],
        on_trade: Callable[[str, int, float, float, bool], None],
        stop_event: Optional[threading.Event] = None,
        ws_base_url: str = "wss://fstream.binance.com/market/stream",
    ):
        self._symbols_lock = threading.Lock()
        self._symbols: set = {s.upper() for s in symbols if s}
        self.on_trade = on_trade
        self.stop_event = stop_event or threading.Event()
        self.ws_base_url = ws_base_url.rstrip("/")

        self._last_msg_lock = threading.Lock()
        self.last_msg_ts: Dict[str, float] = {}
        self._last_any_msg: float = 0.0

        self._liq_lock = threading.Lock()
        self._liq_buffer: List[Tuple[int, str, str, float, float, float]] = []

        self._dropped_trades: int = 0
        self._last_drop_log_ts: float = 0.0
        self._reconnect_needed = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def parse_stream_payload(raw: Any) -> Optional[Tuple[str, Any]]:
        if isinstance(raw, str) or isinstance(raw, bytes):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return None
        if not isinstance(raw, dict):
            return None

        # Combined stream envelope {"stream": "...", "data": {...}}
        if "stream" in raw and "data" in raw:
            d = raw["data"]
        else:
            d = raw

        if not isinstance(d, dict):
            return None

        event_type = d.get("e")
        if event_type == "aggTrade":
            try:
                sym = str(d.get("s", "")).upper()
                ts_ms = int(d.get("T", 0))
                price = float(d.get("p", 0.0))
                qty = float(d.get("q", 0.0))
                is_buyer_maker = bool(d.get("m", False))
            except (TypeError, ValueError):
                return None
            if not sym or ts_ms <= 0 or price <= 0.0 or qty <= 0.0:
                return None
            return (
                "aggTrade",
                (sym, ts_ms, price, qty, is_buyer_maker),
            )
        elif event_type == "forceOrder":
            o = d.get("o") or {}
            if not isinstance(o, dict):
                return None

            try:
                sym = str(o.get("s", "")).upper()
                side = str(o.get("S", "")).upper()
                price = float(o.get("ap") or o.get("p") or 0.0)
                qty = float(o.get("z") or o.get("l") or o.get("q") or 0.0)
                ts_ms = int(o.get("T") or d.get("E") or 0)
                notional = price * qty
            except (TypeError, ValueError):
                return None

            if (
                sym
                and side in ("BUY", "SELL")
                and price > 0
                and qty > 0
                and ts_ms > 0
            ):
                return (
                    "forceOrder",
                    (ts_ms, sym, side, price, qty, notional),
                )
        return None

    def handle_message(
        self,
        raw: Any,
        trade_queue: Optional[asyncio.Queue] = None,
    ) -> Optional[str]:
        parsed = self.parse_stream_payload(raw)
        if parsed is None:
            return None

        event_type, data = parsed
        now = time.monotonic()

        if event_type == "aggTrade":
            sym, ts_ms, price, qty, is_buyer_maker = data
            with self._last_msg_lock:
                self.last_msg_ts[sym] = now
                self._last_any_msg = now
            if trade_queue is not None:
                if trade_queue.full():
                    try:
                        trade_queue.get_nowait()
                        trade_queue.task_done()
                    except asyncio.QueueEmpty:
                        pass
                    self._dropped_trades += 1
                    if now - self._last_drop_log_ts >= 30.0:
                        self._last_drop_log_ts = now
                        logger.warning(
                            f"WS trade dispatch queue full; dropped {self._dropped_trades} oldest trades"
                        )
                try:
                    trade_queue.put_nowait(data)
                except asyncio.QueueFull:
                    pass
            else:
                try:
                    self.on_trade(sym, ts_ms, price, qty, is_buyer_maker)
                except Exception as e:
                    logger.warning(f"on_trade callback failed for {sym}: {e}")
            return "aggTrade"
        elif event_type == "forceOrder":
            ts_ms, sym, side, price, qty, notional = data
            with self._symbols_lock:
                symbols = set(self._symbols)
            if sym not in symbols:
                return None

            with self._liq_lock:
                self._liq_buffer.append((ts_ms, sym, side, price, qty, notional))
                if len(self._liq_buffer) > 5000:
                    self._liq_buffer = self._liq_buffer[-5000:]
            with self._last_msg_lock:
                self._last_any_msg = now
            return "forceOrder"
        return None

    def drain_liquidations(self) -> List[Tuple[int, str, str, float, float, float]]:
        with self._liq_lock:
            result = self._liq_buffer[:]
            self._liq_buffer.clear()
        return result

    def update_symbols(self, symbols: Iterable[str]) -> bool:
        new_symbols = {s.upper() for s in symbols if s}
        with self._symbols_lock:
            current = set(self._symbols)
            if new_symbols == current:
                return False
            self._symbols = new_symbols
        self._reconnect_needed.set()
        return True

    def is_symbol_live(self, symbol: str, max_age_s: float = 20.0) -> bool:
        sym = symbol.upper()
        with self._symbols_lock:
            if sym not in self._symbols:
                return False
        if self.stop_event.is_set():
            return False
        with self._last_msg_lock:
            ts = self.last_msg_ts.get(sym)
        if ts is None:
            return False
        return (time.monotonic() - ts) <= max_age_s

    def build_stream_url(self) -> str:
        with self._symbols_lock:
            syms = sorted(self._symbols)
        streams = [f"{s.lower()}@aggTrade" for s in syms] + ["!forceOrder@arr"]
        return f"{self.ws_base_url}?streams=" + "/".join(streams)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self._run_loop()),
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)

    async def _dispatch_worker(self, trade_queue: asyncio.Queue) -> None:
        while not self.stop_event.is_set() or not trade_queue.empty():
            try:
                item = await asyncio.wait_for(trade_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            sym, ts_ms, price, qty, is_buyer_maker = item
            try:
                self.on_trade(sym, ts_ms, price, qty, is_buyer_maker)
            except Exception as e:
                logger.warning(f"on_trade callback failed for {sym}: {e}")
            finally:
                trade_queue.task_done()

    async def _run_loop(self) -> None:
        import websockets

        backoff = 1.0
        connected_at = 0.0
        trade_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        worker_task = asyncio.create_task(self._dispatch_worker(trade_queue))

        try:
            while not self.stop_event.is_set():
                if self._reconnect_needed.is_set():
                    self._reconnect_needed.clear()

                url = self.build_stream_url()
                try:
                    async with websockets.connect(
                        url,
                        ping_interval=20,
                        ping_timeout=20,
                        open_timeout=10,
                        max_size=2**22,
                    ) as ws:
                        connected_at = time.monotonic()

                        while not self.stop_event.is_set():
                            if self._reconnect_needed.is_set():
                                break
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
                            except asyncio.TimeoutError:
                                if time.monotonic() - connected_at > 60.0:
                                    backoff = 1.0
                                continue
                            except Exception as recv_err:
                                logger.warning(f"WebSocket recv error: {recv_err}")
                                break

                            if raw is not None:
                                try:
                                    self.handle_message(raw, trade_queue=trade_queue)
                                    if time.monotonic() - connected_at > 60.0:
                                        backoff = 1.0
                                except Exception as e:
                                    logger.warning(f"handle_message error: {e}")

                        if self.stop_event.is_set():
                            connected_at = 0.0
                            break

                        if self._reconnect_needed.is_set():
                            self._reconnect_needed.clear()
                            connected_at = 0.0
                            continue

                        if connected_at > 0.0 and (time.monotonic() - connected_at) > 60.0:
                            backoff = 1.0
                        else:
                            backoff = min(backoff * 2, 30.0)
                        connected_at = 0.0

                        for _ in range(int(backoff * 10)):
                            if self.stop_event.is_set():
                                break
                            await asyncio.sleep(0.1)

                except Exception as e:
                    logger.warning(f"WebSocket connection error: {e}")
                    if self.stop_event.is_set():
                        connected_at = 0.0
                        break
                    if self._reconnect_needed.is_set():
                        self._reconnect_needed.clear()
                        connected_at = 0.0
                        continue
                    if connected_at > 0.0 and (time.monotonic() - connected_at) > 60.0:
                        backoff = 1.0
                    else:
                        backoff = min(backoff * 2, 30.0)
                    connected_at = 0.0
                    for _ in range(int(backoff * 10)):
                        if self.stop_event.is_set():
                            break
                        await asyncio.sleep(0.1)
        finally:
            worker_task.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)
            while not trade_queue.empty():
                try:
                    sym, ts_ms, price, qty, is_buyer_maker = trade_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    self.on_trade(sym, ts_ms, price, qty, is_buyer_maker)
                except Exception as e:
                    logger.warning(f"on_trade callback failed for {sym}: {e}")
                finally:
                    trade_queue.task_done()
