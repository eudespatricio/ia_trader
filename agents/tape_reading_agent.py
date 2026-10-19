"""
agents/tape_reading_agent.py - Confirmação de fluxo (Tape Reading) para o gate de entrada (ADDENDUM E8).

Lê SOMENTE LEITURA (sqlite `mode=ro`) o banco datasets/microstructure.db alimentado por
scripts/microstructure_collector.py (tabelas book_features e tick_agg). Regras PRÉ-REGISTRADAS e fixas
(constantes abaixo; não otimizar). Sem lookahead: só linhas com ts <= now_ms.

Voto:
  BUY  : taker_imb_5m >= +TAKER_IMB_MIN  E book_imb_10 >= +BOOK_IMB_MIN  E microprice_dev >= 0
  SELL : espelhado (<= -TAKER_IMB_MIN, <= -BOOK_IMB_MIN, microprice_dev <= 0)
  HOLD : caso contrário, ou spread_bps > SPREAD_MAX_BPS (motivo 'spread_wide').
Status 'UNAVAILABLE' (fail-open no gate): símbolo fora da cobertura do coletor, DB ausente/ilegível,
dados ausentes, insuficientes ou velhos (book > BOOK_MAX_AGE_S, tick > TICK_MAX_AGE_S).
"""
from __future__ import annotations

import logging
import os
import sqlite3
from db_conn import connect as _db_connect
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ia_trader.tape_reading")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MICROSTRUCTURE_DB = os.path.join(BASE, "datasets", "microstructure.db")

COVERED_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT"})   # = BASE SYMBOLS do coletor
TAKER_WINDOW_MIN = 5            # soma dos últimos 5 minutos de tick_agg
TAKER_MIN_MINUTES = 3           # mínimo de minutos distintos com dados na janela
TAKER_MIN_TRADES = 30           # mínimo de trades somados na janela
BOOK_SNAPSHOTS = 4              # média dos últimos 4 snapshots
TAKER_IMB_MIN = 0.15
BOOK_IMB_MIN = 0.10
MICROPRICE_DEV_MIN_BPS = 0.0
SPREAD_MAX_BPS = 5.0
BOOK_MAX_AGE_S = 120
TICK_MAX_AGE_S = 180

# Item 2 & 3: Score contínuo e absorção institucional
FLOW_SCORE_MIN = 0.12
ABSORPTION_TAKER_THRESHOLD = 0.30
ABSORPTION_BOOK_THRESHOLD = 0.15


class TapeReadingAgent:
    name = "TapeReadingAgent"

    def __init__(
        self,
        db_path: Optional[str] = None,
        use_continuous_flow: bool = True,
        covered_symbols: Optional[set] = None,
    ):
        self.db_path = db_path or MICROSTRUCTURE_DB
        self.use_continuous_flow = use_continuous_flow
        self.covered_symbols = set(covered_symbols) if covered_symbols else set(COVERED_SYMBOLS)
        self._last_refresh_ms: int = 0
        self.last_result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    def _unavailable(self, symbol: str, reason: str, now_ms: int, features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._finish({
            "agent_name": self.name, "symbol": symbol, "vote": "HOLD", "status": "UNAVAILABLE",
            "confidence": 0.0, "features": features or {}, "reason": reason, "timestamp": now_ms,
        })

    def _finish(self, res: Dict[str, Any]) -> Dict[str, Any]:
        self.last_result = res
        return res

    def _connect_ro(self) -> sqlite3.Connection:
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(self.db_path)
        con = _db_connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0)
        con.execute("PRAGMA busy_timeout=5000")
        return con

    def add_covered_symbol(self, sym: str) -> None:
        self.covered_symbols.add(str(sym).upper())

    def set_covered_symbols(self, symbols: Any) -> None:
        self.covered_symbols = {str(s).upper() for s in symbols}

    def refresh_covered_symbols_from_db(self, now_ms: Optional[int] = None) -> None:
        """Sincroniza símbolos monitorados a partir da tabela active_symbols gerenciada pelo coletor."""
        now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        self._last_refresh_ms = now_ms
        try:
            con = self._connect_ro()
            rows = con.execute(
                "SELECT symbol FROM active_symbols WHERE updated_at >= ?",
                (now_ms - 3600_000,)
            ).fetchall()
            con.close()
            for r in rows:
                if r[0]:
                    self.covered_symbols.add(str(r[0]).upper())
        except Exception:
            pass

    def evaluate(self, symbol: str, direction: str = "", now_ms: Optional[int] = None) -> Dict[str, Any]:
        """Retorna dict AgentSignal-like. `direction` é só informativo (o voto não depende dele)."""
        now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        sym = str(symbol).upper()
        if sym not in self.covered_symbols:
            if (now_ms - self._last_refresh_ms) > 30_000:
                self.refresh_covered_symbols_from_db(now_ms)
            if sym not in self.covered_symbols:
                return self._unavailable(sym, "symbol_not_covered", now_ms)
        try:
            con = self._connect_ro()
        except Exception as e:  # noqa: BLE001
            return self._unavailable(sym, f"db_unavailable:{type(e).__name__}", now_ms)
        try:
            books = con.execute(
                "SELECT ts_ms, spread_bps, microprice_dev_bps, bid_n10, ask_n10 FROM book_features "
                "WHERE symbol=? AND ts_ms<=? ORDER BY ts_ms DESC LIMIT ?", (sym, now_ms, BOOK_SNAPSHOTS)).fetchall()
            win_start = now_ms - TAKER_WINDOW_MIN * 60_000
            ticks = con.execute(
                "SELECT minute_ms, buy_notional, sell_notional, n_trades FROM tick_agg "
                "WHERE symbol=? AND minute_ms<=? AND minute_ms>? ORDER BY minute_ms DESC",
                (sym, now_ms, win_start)).fetchall()
        except Exception as e:  # noqa: BLE001
            return self._unavailable(sym, f"db_error:{type(e).__name__}", now_ms)
        finally:
            con.close()

        if not books:
            return self._unavailable(sym, "no_book_data", now_ms)
        if not ticks:
            return self._unavailable(sym, "no_tick_data", now_ms)
        book_age = (now_ms - int(books[0][0])) / 1000.0
        tick_age = (now_ms - int(ticks[0][0])) / 1000.0
        feats: Dict[str, Any] = {"book_age_s": round(book_age, 1), "tick_age_s": round(tick_age, 1)}
        if book_age > BOOK_MAX_AGE_S:
            return self._unavailable(sym, "book_stale", now_ms, feats)
        if tick_age > TICK_MAX_AGE_S:
            return self._unavailable(sym, "tick_stale", now_ms, feats)

        n_min = len(ticks)
        n_trades = sum(int(r[3] or 0) for r in ticks)
        feats.update({"tick_minutes": n_min, "n_trades_5m": n_trades})
        if n_min < TAKER_MIN_MINUTES or n_trades < TAKER_MIN_TRADES:
            return self._unavailable(sym, "tick_insufficient", now_ms, feats)
        buy = sum(float(r[1] or 0.0) for r in ticks)
        sell = sum(float(r[2] or 0.0) for r in ticks)
        if buy + sell <= 0.0:
            return self._unavailable(sym, "tick_zero_volume", now_ms, feats)
        taker = (buy - sell) / (buy + sell)

        imbs: List[float] = []
        for r in books:
            b, a = r[3], r[4]
            if b is None or a is None or (float(b) + float(a)) <= 0.0:
                continue
            imbs.append((float(b) - float(a)) / (float(b) + float(a)))
        devs = [float(r[2]) for r in books if r[2] is not None]
        spread = books[0][1]
        if not imbs or not devs or spread is None:
            return self._unavailable(sym, "book_incomplete", now_ms, feats)
        book_imb = sum(imbs) / len(imbs)
        mdev = sum(devs) / len(devs)
        spread = float(spread)

        mdev_norm = max(-1.0, min(1.0, mdev))
        flow_score = 0.60 * taker + 0.30 * book_imb + 0.10 * mdev_norm
        feats.update({
            "taker_imb_5m": taker,
            "book_imb_10": book_imb,
            "microprice_dev": mdev,
            "spread_bps": spread,
            "flow_score": round(flow_score, 4),
        })

        is_bullish_absorption = (taker <= -ABSORPTION_TAKER_THRESHOLD and book_imb >= ABSORPTION_BOOK_THRESHOLD)
        is_bearish_absorption = (taker >= ABSORPTION_TAKER_THRESHOLD and book_imb <= -ABSORPTION_BOOK_THRESHOLD)

        if spread > SPREAD_MAX_BPS:
            vote, reason = "HOLD", "spread_wide"
        elif is_bullish_absorption:
            vote, reason = "BUY", "absorption_buy_confirmed"
        elif is_bearish_absorption:
            vote, reason = "SELL", "absorption_sell_confirmed"
        elif self.use_continuous_flow:
            if flow_score >= FLOW_SCORE_MIN:
                vote, reason = "BUY", "flow_buy_confirmed"
            elif flow_score <= -FLOW_SCORE_MIN:
                vote, reason = "SELL", "flow_sell_confirmed"
            else:
                vote, reason = "HOLD", "no_flow_confluence"
        else:
            if taker >= TAKER_IMB_MIN and book_imb >= BOOK_IMB_MIN and mdev >= MICROPRICE_DEV_MIN_BPS:
                vote, reason = "BUY", "flow_buy_confirmed"
            elif taker <= -TAKER_IMB_MIN and book_imb <= -BOOK_IMB_MIN and mdev <= -MICROPRICE_DEV_MIN_BPS:
                vote, reason = "SELL", "flow_sell_confirmed"
            else:
                vote, reason = "HOLD", "no_flow_confluence"

        conf = 0.0
        if vote != "HOLD":
            if "absorption" in reason:
                conf = float(min(1.0, 0.5 * (abs(taker) / 0.50) + 0.5 * (abs(book_imb) / 0.30)))
            elif self.use_continuous_flow:
                conf = float(min(1.0, abs(flow_score) / (2.0 * FLOW_SCORE_MIN)))
            else:
                conf = float(min(1.0, 0.5 * min(1.0, abs(taker) / (2 * TAKER_IMB_MIN))
                                 + 0.5 * min(1.0, abs(book_imb) / (2 * BOOK_IMB_MIN))))
        return self._finish({
            "agent_name": self.name, "symbol": sym, "vote": vote, "status": "OK", "confidence": conf,
            "features": feats, "reason": reason, "timestamp": now_ms,
        })
