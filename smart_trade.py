"""
smart_trade.py - Motor de Execução e Simulação em Tempo Real (Paper Trading Engine).
Fase 6 do Projeto IA Trader.

Princípios de Arquitetura e Governança:
1. Modo Estrito SOMENTE SIMULAÇÃO (Strict Paper Trading):
   - Proibição absoluta de ordens reais ou chamadas de trading autenticadas com chaves ativas.
   - Qualquer tentativa de desativar a simulação sem auditoria prévia lança PermissionError.
2. Microestrutura Realista com Custos Institucionais:
   - Taxas taker reais (0.06%) e slippage modelado (0.02% spot, 0.03% perp).
   - Acúmulo de funding rate real (crédito/débito a cada 8h de mercado).
   - Colisão conservadora intra-barra (Stop Loss assumido primeiro em caso de overlap).
3. Persistência SQLite Multi-Thread com WAL Mode:
   - Grava posições, histórico de trades e curva de equity em tempo real em `market_cache.db`.
   - Permite que o Dashboard Flask (Fase 8) renderize telemetria ao vivo sem locks.
4. Suporte Multiestratégia:
   - Execução de Carry Delta-Neutro (Spot + Perp) e Swings Direcionais com ATR dinâmico.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
import time
import uuid
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from core import TradingCore
from signal_generator import TradeSignal

logger = logging.getLogger("ia_trader.smart_trade")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [smart_trade] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ============================================================================
# ESTRUTURAS DE DADOS: POSIÇÕES E TRADES EM PAPER TRADING
# ============================================================================

@dataclass
class PaperPosition:
    position_id: str
    symbol: str
    side: str                          # "LONG", "SHORT", "DELTA_NEUTRAL_CARRY"
    entry_price: float
    current_price: float
    quantity: float
    notional: float
    stop_loss: float
    take_profit: float
    highest_price: float
    lowest_price: float
    trailing_stop_pct: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    funding_collected: float = 0.0
    fees_paid: float = 0.0
    entry_timestamp: int = 0
    strategy_type: str = "DIRECTIONAL" # "DIRECTIONAL", "CARRY", "SCALP"
    status: str = "OPEN"               # "OPEN", "CLOSED"
    tp1_price: float = 0.0
    tp1_hit: bool = False
    tp2_price: float = 0.0
    initial_quantity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



@dataclass
class PaperTrade:
    trade_id: str
    position_id: str
    symbol: str
    action: str                        # "OPEN_LONG", "OPEN_SHORT", "CLOSE_LONG", "CLOSE_SHORT", "OPEN_CARRY", "CLOSE_CARRY", "STOP_LOSS", "TAKE_PROFIT", "FUNDING"
    price: float
    quantity: float
    fee: float
    pnl: float
    timestamp: int
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# MOTOR DE PAPER TRADING: PaperTradingEngine
# ============================================================================

class PaperTradingEngine:
    """
    Motor institucional de simulação e roteamento de ordens em Paper Trading.
    Opera 100% isolado de chaves privadas e fundos reais.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        db_path: Optional[str] = None,
        taker_fee: float = 0.0006,       # 0.06% taker fee
        maker_fee: float = 0.0002,       # 0.02% maker fee
        slippage_spot: float = 0.0002,   # 0.02% slippage spot
        slippage_perp: float = 0.0003,   # 0.03% slippage perp
        simulation_mode: bool = True,
        core: Optional[TradingCore] = None,
        max_concurrent_positions: Optional[int] = None,
        max_margin_utilization_pct: Optional[float] = None,
        restore: bool = True,
    ):
        # TRAVA DE SEGURANÇA DA REGRA DE OURO
        if not simulation_mode:
            raise PermissionError("O modo live trading com capital real está bloqueado pelo protocolo de governança da Fase 6.")

        self.simulation_mode = True
        self.initial_capital = float(initial_capital)
        self.balance = float(initial_capital)
        self.taker_fee = taker_fee
        self.maker_fee = maker_fee
        self.slippage_spot = slippage_spot
        self.slippage_perp = slippage_perp

        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "datasets", "market_cache.db"
        )
        self.core = core or TradingCore(capital=self.initial_capital, simulation_mode=True)

        self.positions: Dict[str, PaperPosition] = {}
        self.closed_positions: List[PaperPosition] = []
        self.trade_history: List[PaperTrade] = []
        self.total_fees_paid: float = 0.0
        self.total_realized_pnl: float = 0.0

        # Controle de Cota Diária de Scalper Intradiário (Fase 6 & Governança)
        self.min_daily_scalps: int = 2
        self.max_daily_scalps: int = 5
        self.daily_scalps_count: int = 0
        self.current_day_utc: str = ""

        # Controle de Cota Diária Independente de Golden Pots (Varredura de Alta Convicção)
        # Golden Pots NÃO impactam e NÃO consomem a cota de scalps diários padrões!
        self.max_daily_golden_pots: int = 3
        self.daily_golden_pots_count: int = 0

        # Controle Estrito de Exposição e Margem (Preservação de Capital sem Alavancagem Excessiva)
        self.max_concurrent_positions: Optional[int] = max_concurrent_positions
        self.max_margin_utilization_pct: float = 0.85 if max_margin_utilization_pct is None else float(max_margin_utilization_pct)

        self._lock = threading.RLock()

        # Otimização: Compressão inteligente de snapshots de portfólio no SQLite
        self._last_snapshot_time: float = 0.0
        self._last_snapshot_equity: float = -1.0
        self._last_snapshot_open_count: int = -1

        # Telemetria & Circuit Breakers (Fase 7) & Aprendizado Adaptativo
        try:
            from performance_tracker import PerformanceTracker
            from rollback_guard import RollbackGuard
            from learning_tracker import LearningTracker
            self.performance_tracker = PerformanceTracker(db_path=self.db_path, initial_capital=self.initial_capital)
            self.rollback_guard = RollbackGuard(db_path=self.db_path)
            self.learning_tracker = LearningTracker(db_path=self.db_path)
        except Exception as e:
            logger.warning(f"Aviso ao inicializar módulos da Fase 7: {e}")
            self.performance_tracker = None
            self.rollback_guard = None
            self.learning_tracker = None


        self._init_database_tables()

        # A5: reinício do daemon não pode zerar saldo/posições
        if restore:
            try:
                self.restore_state()
            except Exception as e:
                logger.error(f"Falha ao restaurar estado do SQLite (iniciando limpo): {e}")

        logger.info(f"PaperTradingEngine inicializado com capital de ${self.initial_capital:,.2f} USD em modo SIMULAÇÃO.")


    def _get_connection(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        con = sqlite3.connect(self.db_path, timeout=15.0)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_database_tables(self) -> None:
        """Cria as tabelas de telemetria e estado de paper trading no SQLite."""
        with self._get_connection() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS paper_positions (
                    position_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    notional REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    funding_collected REAL,
                    fees_paid REAL,
                    entry_timestamp INTEGER,
                    strategy_type TEXT,
                    status TEXT NOT NULL,
                    tp1_price REAL DEFAULT 0.0,
                    tp1_hit INTEGER DEFAULT 0,
                    tp2_price REAL DEFAULT 0.0,
                    initial_quantity REAL DEFAULT 0.0
                )
            """)
            # Migrações seguras de colunas caso a tabela já exista de sessões anteriores
            for col, col_type in [
                ("tp1_price", "REAL DEFAULT 0.0"),
                ("tp1_hit", "INTEGER DEFAULT 0"),
                ("tp2_price", "REAL DEFAULT 0.0"),
                ("initial_quantity", "REAL DEFAULT 0.0"),
            ]:
                try:
                    con.execute(f"ALTER TABLE paper_positions ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass

            con.execute("""
                CREATE TABLE IF NOT EXISTS paper_trades (
                    trade_id TEXT PRIMARY KEY,
                    position_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    fee REAL NOT NULL,
                    pnl REAL NOT NULL,
                    timestamp INTEGER NOT NULL,
                    rationale TEXT
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS paper_portfolio (
                    timestamp INTEGER PRIMARY KEY,
                    equity REAL NOT NULL,
                    balance REAL NOT NULL,
                    margin_used REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl_cum REAL NOT NULL,
                    total_fees_paid REAL NOT NULL,
                    open_positions_count INTEGER NOT NULL
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS simulation_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur = con.cursor()
            cur.execute("SELECT count(*) FROM simulation_settings")
            if cur.fetchone()[0] == 0:
                defaults = [
                    ("operation_mode", "INSTITUTIONAL"),
                    ("max_daily_golden_pots", "3"),
                    ("max_daily_scalps", "5"),
                    ("max_concurrent_positions", "5"),
                    ("max_margin_utilization_pct", "0.85"),
                    ("scalp_vwap_dev_pct", "2.0"),
                    ("scalp_rsi_oversold", "35.0"),
                    ("scalp_rsi_overbought", "65.0"),
                    ("scalp_volume_z", "0.2"),
                    ("reset_daily_quotas", "0"),
                ]
                cur.executemany("INSERT OR IGNORE INTO simulation_settings (key, value) VALUES (?, ?)", defaults)
            con.commit()

    # =========================================================================
    # RESTAURAÇÃO DE ESTADO (A5) E RESET MANUAL
    # =========================================================================

    @staticmethod
    def _ts_ms(ts: Union[int, float]) -> float:
        """Normaliza timestamp (s ou ms) para ms."""
        ts = float(ts or 0)
        return ts if ts > 1e11 else ts * 1000.0

    @staticmethod
    def _set_setting(cur: sqlite3.Cursor, key: str, value: str) -> None:
        cur.execute("""
            INSERT INTO simulation_settings (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
        """, (key, value))

    @staticmethod
    def _max_rowid(cur: sqlite3.Cursor, table: str) -> int:
        cur.execute(f"SELECT COALESCE(MAX(rowid), 0) FROM {table}")
        return int(cur.fetchone()[0])

    def manual_reset(self) -> None:
        """
        Reset manual INTENCIONAL: volta ao capital inicial, sem posições/histórico em memória,
        e grava um marco (rowid) no SQLite para que restart posterior NÃO ressuscite o passado.
        Linhas antigas do DB são preservadas para auditoria.
        """
        with self._lock:
            with self._get_connection() as con:
                cur = con.cursor()
                self._set_setting(cur, "ledger_epoch_trade_rowid", str(self._max_rowid(cur, "paper_trades")))
                self._set_setting(cur, "ledger_epoch_pos_rowid", str(self._max_rowid(cur, "paper_positions")))
                self._set_setting(cur, "quota_epoch_trade_rowid", str(self._max_rowid(cur, "paper_trades")))
                self._set_setting(cur, "MANUAL_RESET", "0")
                con.commit()
            self.balance = self.initial_capital
            self.positions = {}
            self.closed_positions = []
            self.trade_history = []
            self.total_fees_paid = 0.0
            self.total_realized_pnl = 0.0
            self.daily_scalps_count = 0
            self.daily_golden_pots_count = 0
            self.current_day_utc = ""
            logger.warning("[MANUAL_RESET] Estado de paper trading reiniciado intencionalmente.")

    def restore_state(self) -> Dict[str, Any]:
        """
        Recarrega do SQLite: saldo (ledger paper_trades), posições OPEN, histórico, fees/PnL
        acumulados e contadores de cota diária. Respeita:
          - flag simulation_settings.MANUAL_RESET == '1' (reset intencional -> começa limpo);
          - marco ledger_epoch_* (ignora tudo anterior a um reset manual prévio);
          - marco quota_epoch_trade_rowid (reset manual de cotas no mesmo dia);
          - virada de dia UTC (cotas só contam trades do último dia registrado).
        Saldo = capital inicial - taxas de abertura (OPEN_*) + PnL líquido de saídas (inclui TP1_PARTIAL).
        """
        with self._lock:
            with self._get_connection() as con:
                cur = con.cursor()
                cur.execute("SELECT key, value FROM simulation_settings")
                st = dict(cur.fetchall())
                flag = str(st.get("MANUAL_RESET", "0")).strip().lower()
                if flag in ("1", "true", "yes"):
                    con.commit()
                    self.manual_reset()
                    self.sync_settings_from_db()
                    return {"restored": False, "reason": "MANUAL_RESET"}

                t_epoch = int(float(st.get("ledger_epoch_trade_rowid", 0) or 0))
                p_epoch = int(float(st.get("ledger_epoch_pos_rowid", 0) or 0))
                q_epoch = int(float(st.get("quota_epoch_trade_rowid", 0) or 0))

                cur.execute(
                    "SELECT rowid, trade_id, position_id, symbol, action, price, quantity, fee, pnl, timestamp, rationale "
                    "FROM paper_trades WHERE rowid > ? ORDER BY rowid", (t_epoch,))
                trade_rows = cur.fetchall()

                cur.execute(
                    "SELECT position_id, symbol, side, entry_price, current_price, quantity, notional, stop_loss, "
                    "take_profit, unrealized_pnl, realized_pnl, funding_collected, fees_paid, entry_timestamp, "
                    "strategy_type, status, tp1_price, tp1_hit, tp2_price, initial_quantity "
                    "FROM paper_positions WHERE rowid > ? ORDER BY rowid", (p_epoch,))
                pos_rows = cur.fetchall()

            if not trade_rows and not pos_rows:
                return {"restored": False, "reason": "EMPTY_DB"}

            history: List[PaperTrade] = []
            balance = self.initial_capital
            fees = 0.0
            realized = 0.0
            for r in trade_rows:
                tr = PaperTrade(
                    trade_id=r[1], position_id=r[2], symbol=r[3], action=r[4], price=float(r[5]),
                    quantity=float(r[6]), fee=float(r[7]), pnl=float(r[8]), timestamp=int(r[9]),
                    rationale=r[10] or "",
                )
                history.append(tr)
                fees += tr.fee
                if tr.action.startswith("OPEN_"):
                    balance -= tr.fee
                else:
                    balance += tr.pnl
                    realized += tr.pnl

            open_pos: Dict[str, PaperPosition] = {}
            closed_pos: List[PaperPosition] = []
            for r in pos_rows:
                f = lambda v: float(v) if v is not None else 0.0
                entry, cur_px = f(r[3]), f(r[4])
                pos = PaperPosition(
                    position_id=r[0], symbol=r[1], side=r[2], entry_price=entry, current_price=cur_px,
                    quantity=f(r[5]), notional=f(r[6]), stop_loss=f(r[7]), take_profit=f(r[8]),
                    highest_price=max(entry, cur_px), lowest_price=min(entry, cur_px) if cur_px > 0 else entry,
                    unrealized_pnl=f(r[9]), realized_pnl=f(r[10]), funding_collected=f(r[11]),
                    fees_paid=f(r[12]), entry_timestamp=int(r[13] or 0), strategy_type=r[14] or "DIRECTIONAL",
                    status=r[15], tp1_price=f(r[16]), tp1_hit=bool(r[17]), tp2_price=f(r[18]),
                    initial_quantity=f(r[19]),
                )
                if pos.status == "OPEN":
                    open_pos[pos.position_id] = pos
                else:
                    closed_pos.append(pos)

            self.balance = balance
            self.total_fees_paid = fees
            self.total_realized_pnl = realized
            self.trade_history = history
            self.positions = open_pos
            self.closed_positions = closed_pos

            # Cotas diárias: conta aberturas do último dia UTC registrado, após o marco de reset de cotas
            self.daily_scalps_count = 0
            self.daily_golden_pots_count = 0
            self.current_day_utc = ""
            if history:
                last_day = self._format_day_utc(max(self._ts_ms(t.timestamp) for t in history) / 1000.0)
                self.current_day_utc = last_day
                with self._get_connection() as con:
                    cur = con.cursor()
                    cur.execute(
                        "SELECT action, timestamp FROM paper_trades WHERE rowid > ? AND rowid > ? "
                        "AND (action LIKE 'OPEN_SCALP_%' OR action LIKE 'OPEN_GOLDEN_POT_%')",
                        (t_epoch, q_epoch))
                    for action, ts in cur.fetchall():
                        if self._format_day_utc(self._ts_ms(ts) / 1000.0) != last_day:
                            continue
                        if action.startswith("OPEN_SCALP_"):
                            self.daily_scalps_count += 1
                        else:
                            self.daily_golden_pots_count += 1

            self.sync_settings_from_db()  # limites atuais + eventual reset de cotas pendente
            logger.info(
                f"[RESTORE] Saldo ${self.balance:,.2f} | {len(open_pos)} posições abertas | "
                f"{len(history)} trades | cotas {self.daily_scalps_count}/{self.daily_golden_pots_count} ({self.current_day_utc})"
            )
            return {"restored": True, "balance": self.balance, "open_positions": len(open_pos),
                    "trades": len(history)}

    def get_simulation_settings(self) -> Dict[str, Any]:
        """Recupera configurações e limites de simulação ativas do SQLite."""
        defaults = {
            "operation_mode": "INSTITUTIONAL",
            "max_daily_golden_pots": 3,
            "max_daily_scalps": 5,
            "max_concurrent_positions": self.max_concurrent_positions,
            "max_margin_utilization_pct": self.max_margin_utilization_pct,
            "scalp_vwap_dev_pct": 2.0,
            "scalp_rsi_oversold": 35.0,
            "scalp_rsi_overbought": 65.0,
            "scalp_volume_z": 0.2,
        }
        try:
            with self._get_connection() as con:
                cur = con.cursor()
                cur.execute("SELECT key, value FROM simulation_settings")
                for k, v in cur.fetchall():
                    if k in ("max_daily_golden_pots", "max_daily_scalps", "max_concurrent_positions"):
                        defaults[k] = int(v)
                    elif k in ("scalp_vwap_dev_pct", "scalp_rsi_oversold", "scalp_rsi_overbought", "scalp_volume_z", "max_margin_utilization_pct"):
                        defaults[k] = float(v)
                    else:
                        defaults[k] = v
        except Exception as e:
            logger.debug(f"Aviso ao ler simulation_settings: {e}")

        defaults["daily_golden_pots_count"] = self.daily_golden_pots_count
        defaults["daily_scalps_count"] = self.daily_scalps_count
        defaults["margin_used"] = round(self.margin_used, 2)
        defaults["available_margin"] = round(self.available_margin, 2)
        defaults["open_positions_count"] = len([p for p in self.positions.values() if p.status == "OPEN"])
        return defaults

    def update_simulation_settings(self, settings: Dict[str, Any], reset_quotas: bool = False) -> Dict[str, Any]:
        """Atualiza configurações de simulação no SQLite e propaga aos parâmetros internos."""
        with self._lock:
            with self._get_connection() as con:
                cur = con.cursor()
                for k, v in settings.items():
                    cur.execute("""
                        INSERT INTO simulation_settings (key, value, updated_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
                    """, (str(k), str(v)))
                if reset_quotas:
                    cur.execute("""
                        INSERT INTO simulation_settings (key, value, updated_at)
                        VALUES ('reset_daily_quotas', '1', CURRENT_TIMESTAMP)
                        ON CONFLICT(key) DO UPDATE SET value = '1', updated_at = CURRENT_TIMESTAMP
                    """)
                    self.daily_scalps_count = 0
                    self.daily_golden_pots_count = 0
                con.commit()
            self.sync_settings_from_db()
        return self.get_simulation_settings()

    def sync_settings_from_db(self) -> Dict[str, Any]:
        """Sincroniza parâmetros de limites e modo do SQLite com o motor em execução."""
        with self._lock:
            try:
                with self._get_connection() as con:
                    cur = con.cursor()
                    cur.execute("SELECT key, value FROM simulation_settings")
                    settings_dict = dict(cur.fetchall())

                    if settings_dict.get("reset_daily_quotas") == "1":
                        self.daily_scalps_count = 0
                        self.daily_golden_pots_count = 0
                        cur.execute("UPDATE simulation_settings SET value = '0' WHERE key = 'reset_daily_quotas'")
                        # Persiste o marco do reset para que restore_state não recontabilize as cotas
                        self._set_setting(cur, "quota_epoch_trade_rowid", str(self._max_rowid(cur, "paper_trades")))
                        con.commit()
                        logger.info("🔄 [RESET DE COTAS] Contadores diários zerados sob demanda do usuário.")

                    if "max_daily_golden_pots" in settings_dict:
                        self.max_daily_golden_pots = int(settings_dict["max_daily_golden_pots"])
                    if "max_daily_scalps" in settings_dict:
                        self.max_daily_scalps = int(settings_dict["max_daily_scalps"])
                    if "max_concurrent_positions" in settings_dict:
                        self.max_concurrent_positions = int(settings_dict["max_concurrent_positions"])
                    if "max_margin_utilization_pct" in settings_dict:
                        self.max_margin_utilization_pct = float(settings_dict["max_margin_utilization_pct"])
                    if "operation_mode" in settings_dict:
                        self.operation_mode = settings_dict["operation_mode"]

                    return settings_dict
            except Exception as e:
                logger.debug(f"Aviso ao sincronizar simulation_settings: {e}")
                return {}

    @property
    def equity(self) -> float:
        """Patrimônio líquido total (saldo disponível + PnL não-realizado de posições abertas)."""
        with self._lock:
            unrealized = sum(p.unrealized_pnl for p in self.positions.values() if p.status == "OPEN")
            return self.balance + unrealized

    @property
    def margin_used(self) -> float:
        """Margem comprometida em posições abertas."""
        with self._lock:
            return sum(p.notional for p in self.positions.values() if p.status == "OPEN")

    @property
    def available_margin(self) -> float:
        """Margem livre disponível para novas alocações sem alavancagem excessiva."""
        max_allowed = self.balance * self.max_margin_utilization_pct
        return max(0.0, max_allowed - self.margin_used)

    def _format_day_utc(self, timestamp: int) -> str:
        """Converte timestamp (ms ou s) para data UTC formatada YYYY-MM-DD."""
        if timestamp <= 0:
            return datetime.now(timezone.utc).strftime("%Y-%m-%d")
        ts_sec = timestamp / 1000.0 if timestamp > 1e11 else float(timestamp)
        return datetime.fromtimestamp(ts_sec, tz=timezone.utc).strftime("%Y-%m-%d")

    def _check_or_reset_daily_quota(self, timestamp: int) -> str:
        """Verifica a virada do dia UTC e reseta a contagem de scalps diários e golden pots se necessário."""
        day_str = self._format_day_utc(timestamp)
        # Avança apenas para frente: timestamps de velas antigas (ex.: 23:45 do dia anterior)
        # não podem reabrir o dia anterior e zerar a cota duas vezes na virada UTC.
        if day_str > self.current_day_utc:
            self.current_day_utc = day_str
            self.daily_scalps_count = 0
            self.daily_golden_pots_count = 0
            logger.info(f"Virada de dia UTC detectada ({day_str}): Contadores de scalps diários e golden pots resetados para 0.")
        return self.current_day_utc

    def can_execute_scalp(self, timestamp: int, notional: float = 0.0) -> Tuple[bool, str]:
        """
        Trava de segurança: Garante no máximo max_daily_scalps operações por dia,
        respeita o teto de posições simultâneas (max_concurrent_positions),
        respeita o teto de margem da carteira (max_margin_utilization_pct)
        e checa status do Circuit Breaker / RollbackGuard (Fase 7).
        Retorna (True, 'OK') se autorizado, ou (False, motivo) se bloqueado pelo teto.
        Thread-safe com proteção de lock.
        """
        with self._lock:
            # Trava 1: Cota diária estrita
            self._check_or_reset_daily_quota(timestamp)
            if self.daily_scalps_count >= self.max_daily_scalps:
                return False, f"Limite diário estrito de {self.max_daily_scalps} scalps atingido para {self.current_day_utc}."

            # Trava 2: Posições simultâneas abertas (se configurado)
            if self.max_concurrent_positions is not None and self.max_concurrent_positions > 0:
                open_count = len([p for p in self.positions.values() if p.status == "OPEN"])
                if open_count >= self.max_concurrent_positions:
                    return False, f"Teto de posições simultâneas atingido ({open_count}/{self.max_concurrent_positions}). Aguardando encerramento."

            # Trava 3: Margem livre disponível
            max_margin = self.balance * self.max_margin_utilization_pct
            if notional > 0.0 and (self.margin_used + notional) > max_margin:
                return False, f"Margem livre insuficiente: Alocado ${self.margin_used:,.2f} + Ordem ${notional:,.2f} > Teto ${max_margin:,.2f} (Disponível: ${self.available_margin:,.2f})."

            # Trava 4: Circuit Breakers e Cooldown (Fase 7)
            if getattr(self, "rollback_guard", None):
                allowed, reason = self.rollback_guard.can_open_new_position("SCALP", strategy="SCALP")
                if not allowed:
                    return False, f"Bloqueado pelo Circuit Breaker (Fase 7): {reason}"

            return True, "OK"

    def can_execute_golden_pot(self, timestamp: int, notional: float = 0.0) -> Tuple[bool, str]:
        """
        Trava de segurança para Golden Pots: Cota separada e independente dos scalps normais,
        respeita o teto de posições simultâneas (max_concurrent_positions),
        respeita o teto de margem da carteira (max_margin_utilization_pct)
        e checa status de Circuit Breaker / RollbackGuard (Fase 7).
        Thread-safe com proteção de lock.
        """
        with self._lock:
            # Trava 1: Cota diária independente
            self._check_or_reset_daily_quota(timestamp)
            if self.daily_golden_pots_count >= self.max_daily_golden_pots:
                return False, f"Limite diário independente de {self.max_daily_golden_pots} Golden Pots atingido para {self.current_day_utc}."

            # Trava 2: Posições simultâneas abertas (se configurado)
            if self.max_concurrent_positions is not None and self.max_concurrent_positions > 0:
                open_count = len([p for p in self.positions.values() if p.status == "OPEN"])
                if open_count >= self.max_concurrent_positions:
                    return False, f"Teto de posições simultâneas atingido ({open_count}/{self.max_concurrent_positions}). Aguardando encerramento."

            # Trava 3: Margem livre disponível
            max_margin = self.balance * self.max_margin_utilization_pct
            if notional > 0.0 and (self.margin_used + notional) > max_margin:
                return False, f"Margem livre insuficiente: Alocado ${self.margin_used:,.2f} + Ordem ${notional:,.2f} > Teto ${max_margin:,.2f} (Disponível: ${self.available_margin:,.2f})."

            # Trava 4: Circuit Breakers e Cooldown (Fase 7)
            if getattr(self, "rollback_guard", None):
                allowed, reason = self.rollback_guard.can_open_new_position("GOLDEN_POT", strategy="GOLDEN_POT")
                if not allowed:
                    return False, f"Bloqueado pelo Circuit Breaker (Fase 7): {reason}"

            return True, "OK"


    def get_daily_scalp_status(self) -> Dict[str, Any]:
        """Retorna telemetria da cota diária de scalps (mínimo 2, máximo 5)."""
        with self._lock:
            return {
                "current_day_utc": self.current_day_utc,
                "daily_scalps_count": self.daily_scalps_count,
                "min_daily_scalps": self.min_daily_scalps,
                "max_daily_scalps": self.max_daily_scalps,
                "can_trade_more": self.daily_scalps_count < self.max_daily_scalps,
                "target_reached": self.daily_scalps_count >= self.min_daily_scalps,
                "quota_status": (
                    "LIMITE_ATINGIDO" if self.daily_scalps_count >= self.max_daily_scalps
                    else ("NA_FAIXA_IDEAL" if self.daily_scalps_count >= self.min_daily_scalps
                          else "ABAIXO_DO_MINIMO")
                ),
            }

    def get_daily_golden_pot_status(self) -> Dict[str, Any]:
        """Retorna telemetria da cota diária exclusiva e independente de Golden Pots."""
        with self._lock:
            return {
                "current_day_utc": self.current_day_utc,
                "daily_golden_pots_count": self.daily_golden_pots_count,
                "max_daily_golden_pots": self.max_daily_golden_pots,
                "can_trade_more": self.daily_golden_pots_count < self.max_daily_golden_pots,
                "quota_status": (
                    "LIMITE_ATINGIDO" if self.daily_golden_pots_count >= self.max_daily_golden_pots
                    else "DISPONIVEL"
                ),
            }

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Calcula métricas agregadas de desempenho da carteira simulada."""
        with self._lock:
            eq = self.equity
            ret_pct = ((eq - self.initial_capital) / self.initial_capital) * 100.0
            
            # Win rate por position_id: TP1_PARTIAL + fechamento final = 1 único trade.
            # Posições ainda abertas (só com parcial) não contam como trade finalizado.
            pnl_by_pos: Dict[str, float] = {}
            for t in self.trade_history:
                if t.action.startswith("OPEN_") or t.position_id in self.positions:
                    continue
                pnl_by_pos[t.position_id] = pnl_by_pos.get(t.position_id, 0.0) + t.pnl
            closed_trades = [v for v in pnl_by_pos.values() if v != 0.0]
            winning = [v for v in closed_trades if v > 0.0]
            win_rate = (len(winning) / len(closed_trades) * 100.0) if closed_trades else 0.0

            return {
                "initial_capital": self.initial_capital,
                "current_balance": round(self.balance, 2),
                "current_equity": round(eq, 2),
                "net_profit": round(eq - self.initial_capital, 2),
                "return_pct": round(ret_pct, 2),
                "realized_pnl": round(self.total_realized_pnl, 2),
                "unrealized_pnl": round(sum(p.unrealized_pnl for p in self.positions.values()), 2),
                "total_fees_paid": round(self.total_fees_paid, 2),
                "open_positions": len(self.positions),
                "margin_used": round(self.margin_used, 2),
                "available_margin": round(self.available_margin, 2),
                "max_concurrent_positions": self.max_concurrent_positions,
                "max_margin_utilization_pct": self.max_margin_utilization_pct,
                "margin_utilization_pct": round((self.margin_used / self.balance * 100.0) if self.balance > 0 else 0.0, 1),
                "total_trades_count": len(self.trade_history),
                "win_rate_pct": round(win_rate, 1),
                "simulation_mode": True,
                "min_daily_scalps": self.min_daily_scalps,
                "max_daily_scalps": self.max_daily_scalps,
                "daily_scalps_count": self.daily_scalps_count,
                "current_day_utc": self.current_day_utc,
                "daily_scalps_target_reached": self.daily_scalps_count >= self.min_daily_scalps,
                "daily_scalps_limit_reached": self.daily_scalps_count >= self.max_daily_scalps,
                "max_daily_golden_pots": self.max_daily_golden_pots,
                "daily_golden_pots_count": self.daily_golden_pots_count,
                "daily_golden_pots_limit_reached": self.daily_golden_pots_count >= self.max_daily_golden_pots,
                "circuit_breaker": self.rollback_guard.status.to_dict() if getattr(self, "rollback_guard", None) else {},
            }



    # =========================================================================
    # ROTEAMENTO E EXECUÇÃO DE ORDENS SIMULADAS
    # =========================================================================

    def execute_directional_signal(
        self,
        signal: TradeSignal,
        current_price: float,
        timestamp: int,
    ) -> Optional[PaperPosition]:
        """Abre ou gerencia uma posição direcional a partir de um sinal do comitê."""
        with self._lock:
            if signal.direction not in ["BUY", "SELL"] or signal.is_vetoed:
                return None

            symbol = signal.symbol
            # Verifica se já há posição aberta no par
            existing = [p for p in self.positions.values() if p.symbol == symbol and p.strategy_type == "DIRECTIONAL" and p.status == "OPEN"]
            if existing:
                return None

            if self.max_concurrent_positions is not None and self.max_concurrent_positions > 0:
                open_count = len([p for p in self.positions.values() if p.status == "OPEN"])
                if open_count >= self.max_concurrent_positions:
                    logger.info(f"[{symbol}] Ordem Direcional Rejeitada: Teto de posições simultâneas atingido ({open_count}/{self.max_concurrent_positions}).")
                    return None

            # Dimensionamento de lote com limite estrito de margem livre
            max_margin = self.balance * self.max_margin_utilization_pct
            free_margin = max(0.0, max_margin - self.margin_used)
            notional = min(signal.position_size, free_margin, self.balance * 0.5)
            if notional < 10.0:  # Mínimo de $10 USD
                logger.info(f"[{symbol}] Ordem Direcional Rejeitada: Margem livre insuficiente (${free_margin:,.2f}).")
                return None

            side = "LONG" if signal.direction == "BUY" else "SHORT"
            slip = self.slippage_perp if side == "SHORT" else self.slippage_spot
            exec_price = current_price * (1.0 + slip) if side == "LONG" else current_price * (1.0 - slip)
            qty = notional / exec_price
            fee = notional * self.taker_fee

            if (self.margin_used + notional + fee) > max_margin:
                logger.info(f"[{symbol}] Ordem Direcional Rejeitada: Excede margem máxima permitida.")
                return None

            pos_id = str(uuid.uuid4())[:8]
            pos = PaperPosition(
                position_id=pos_id,
                symbol=symbol,
                side=side,
                entry_price=exec_price,
                current_price=exec_price,
                quantity=qty,
                notional=notional,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                highest_price=exec_price,
                lowest_price=exec_price,
                fees_paid=fee,
                entry_timestamp=timestamp,
                strategy_type="DIRECTIONAL",
                status="OPEN",
            )

            trade = PaperTrade(
                trade_id=str(uuid.uuid4())[:8],
                position_id=pos_id,
                symbol=symbol,
                action=f"OPEN_{side}",
                price=exec_price,
                quantity=qty,
                fee=fee,
                pnl=0.0,
                timestamp=timestamp,
                rationale=signal.rationale,
            )

            try:
                self._save_position(pos)
                self._save_trade(trade)
            except Exception as exc:
                logger.error(f"Falha ao persistir ordem direcional no SQLite: {exc}. Rollback executado.")
                return None

            self.balance -= fee
            self.total_fees_paid += fee
            self.positions[pos_id] = pos
            self.trade_history.append(trade)

            logger.info(f"[{symbol}] Ordem Simulada {side} ABERTA @ ${exec_price:,.2f} | Qtd: {qty:.4f} | Risco: ${signal.risk_dollars:.2f}")
            return pos

    def execute_scalp_signal(
        self,
        signal: TradeSignal,
        current_price: float,
        timestamp: int,
        tp1_atr_mult: float = 1.0,
        tp2_atr_mult: float = 1.8,
        sl_atr_mult: float = 0.9,
    ) -> Optional[PaperPosition]:
        """
        Executa um scalp intradiário cirúrgico (15m) com cota estrita de 2 a 5 trades por dia.
        Usa taxas de ordem Maker (0.02%) e saída em múltiplos estágios (TP1 50% + SL Breakeven).
        Thread-safe com proteção de lock e garantia de integridade transacional.
        """
        with self._lock:
            if self.max_concurrent_positions is not None and self.max_concurrent_positions > 0:
                open_count = len([p for p in self.positions.values() if p.status == "OPEN"])
                if open_count >= self.max_concurrent_positions:
                    logger.info(f"[{signal.symbol}] Scalp Rejeitado: Teto de posições simultâneas atingido ({open_count}/{self.max_concurrent_positions}).")
                    return None

            # Dimensionamento de lote para scalp (máx 25% do saldo e respeitando margem livre)
            max_margin = self.balance * self.max_margin_utilization_pct
            free_margin = max(0.0, max_margin - self.margin_used)
            notional = min(signal.position_size, free_margin, self.balance * 0.25)
            if notional < 10.0:
                logger.info(f"[{signal.symbol}] Scalp Rejeitado: Margem livre insuficiente (${free_margin:,.2f}).")
                return None

            can_exec, reason = self.can_execute_scalp(timestamp, notional=notional)
            if not can_exec:
                logger.info(f"[{signal.symbol}] Scalp Rejeitado: {reason}")
                return None

            if signal.direction not in ["BUY", "SELL"] or signal.is_vetoed:
                return None

            symbol = signal.symbol
            # Verifica se já há scalp aberto no par
            existing = [p for p in self.positions.values() if p.symbol == symbol and p.strategy_type == "SCALP" and p.status == "OPEN"]
            if existing:
                return None

            side = "LONG" if signal.direction == "BUY" else "SHORT"
            
            # Execução Maker para Day Trade (slippage mínimo 0.01%)
            maker_slip = 0.0001
            exec_price = current_price * (1.0 + maker_slip) if side == "LONG" else current_price * (1.0 - maker_slip)
            qty = notional / exec_price
            fee = notional * self.maker_fee  # Taxa Maker de 0.02%

            if (self.margin_used + notional + fee) > max_margin:
                logger.info(f"[{symbol}] Scalp Rejeitado: Excede margem máxima permitida.")
                return None

            atr = signal.atr if signal.atr > 0 else (current_price * 0.005)
            if side == "LONG":
                sl_price = exec_price - (sl_atr_mult * atr)
                tp1_p = exec_price + (tp1_atr_mult * atr)
                tp2_p = exec_price + (tp2_atr_mult * atr)
            else:
                sl_price = exec_price + (sl_atr_mult * atr)
                tp1_p = exec_price - (tp1_atr_mult * atr)
                tp2_p = exec_price - (tp2_atr_mult * atr)

            pos_id = str(uuid.uuid4())[:8]
            pos = PaperPosition(
                position_id=pos_id,
                symbol=symbol,
                side=side,
                entry_price=exec_price,
                current_price=exec_price,
                quantity=qty,
                notional=notional,
                stop_loss=sl_price,
                take_profit=tp2_p,
                highest_price=exec_price,
                lowest_price=exec_price,
                fees_paid=fee,
                entry_timestamp=timestamp,
                strategy_type="SCALP",
                status="OPEN",
                tp1_price=tp1_p,
                tp1_hit=False,
                tp2_price=tp2_p,
                initial_quantity=qty,
            )

            next_quota_num = self.daily_scalps_count + 1
            trade = PaperTrade(
                trade_id=str(uuid.uuid4())[:8],
                position_id=pos_id,
                symbol=symbol,
                action=f"OPEN_SCALP_{side}",
                price=exec_price,
                quantity=qty,
                fee=fee,
                pnl=0.0,
                timestamp=timestamp,
                rationale=f"Scalp Diário ({next_quota_num}/{self.max_daily_scalps}) | {signal.rationale}",
            )

            # Persistência atômica no SQLite com integridade e rollback garantidos
            try:
                self._save_position(pos)
                self._save_trade(trade)
            except Exception as exc:
                logger.error(f"Falha ao persistir scalp no SQLite: {exc}. Rollback do estado em memória.")
                return None

            self.balance -= fee
            self.total_fees_paid += fee
            self.daily_scalps_count += 1
            self.positions[pos_id] = pos
            self.trade_history.append(trade)

            logger.info(f"[{symbol}] Scalp {side} #{self.daily_scalps_count} ABERTO @ ${exec_price:,.2f} | TP1: ${tp1_p:,.2f} | TP2: ${tp2_p:,.2f} | SL: ${sl_price:,.2f}")
            return pos

    def execute_golden_pot_signal(
        self,
        signal: TradeSignal,
        current_price: float,
        timestamp: int,
        tp1_atr_mult: float = 1.2,
        tp2_atr_mult: float = 2.5,
        sl_atr_mult: float = 1.0,
    ) -> Optional[PaperPosition]:
        """
        Executa operação originada de detecção do GoldenPotAgent (Golden Pot de Alta Convicção).
        Possui cota independente (máx 3/dia) que NÃO consome a cota de scalps diários padrões!
        Utiliza microestrutura Maker (0.02%) e saídas parciais (TP1 50% + Breakeven stop + TP2).
        Thread-safe com proteção de lock.
        """
        with self._lock:
            if self.max_concurrent_positions is not None and self.max_concurrent_positions > 0:
                open_count = len([p for p in self.positions.values() if p.status == "OPEN"])
                if open_count >= self.max_concurrent_positions:
                    logger.info(f"[{signal.symbol}] Golden Pot Rejeitado: Teto de posições simultâneas atingido ({open_count}/{self.max_concurrent_positions}).")
                    return None

            # Dimensionamento de lote para Golden Pot (máx 20% do saldo e respeitando margem livre)
            max_margin = self.balance * self.max_margin_utilization_pct
            free_margin = max(0.0, max_margin - self.margin_used)
            notional = min(signal.position_size, free_margin, self.balance * 0.20)
            if notional < 10.0:
                logger.info(f"[{signal.symbol}] Golden Pot Rejeitado: Margem livre insuficiente (${free_margin:,.2f}).")
                return None

            can_exec, reason = self.can_execute_golden_pot(timestamp, notional=notional)
            if not can_exec:
                logger.info(f"[{signal.symbol}] Golden Pot Rejeitado: {reason}")
                return None

            if signal.direction not in ["BUY", "SELL"] or signal.is_vetoed:
                return None

            symbol = signal.symbol
            # Verifica se já há golden pot aberto no par
            existing = [p for p in self.positions.values() if p.symbol == symbol and p.strategy_type == "GOLDEN_POT" and p.status == "OPEN"]
            if existing:
                return None

            side = "LONG" if signal.direction == "BUY" else "SHORT"
            
            # Execução Maker para Golden Pot (slippage mínimo 0.01%)
            maker_slip = 0.0001
            exec_price = current_price * (1.0 + maker_slip) if side == "LONG" else current_price * (1.0 - maker_slip)
            qty = notional / exec_price
            fee = notional * self.maker_fee  # Taxa Maker 0.02%

            if (self.margin_used + notional + fee) > max_margin:
                logger.info(f"[{symbol}] Golden Pot Rejeitado: Excede margem máxima permitida.")
                return None

            atr = signal.atr if signal.atr > 0 else (current_price * 0.008)
            if side == "LONG":
                sl_price = exec_price - (sl_atr_mult * atr)
                tp1_p = exec_price + (tp1_atr_mult * atr)
                tp2_p = exec_price + (tp2_atr_mult * atr)
            else:
                sl_price = exec_price + (sl_atr_mult * atr)
                tp1_p = exec_price - (tp1_atr_mult * atr)
                tp2_p = exec_price - (tp2_atr_mult * atr)

            pos_id = str(uuid.uuid4())[:8]
            pos = PaperPosition(
                position_id=pos_id,
                symbol=symbol,
                side=side,
                entry_price=exec_price,
                current_price=exec_price,
                quantity=qty,
                notional=notional,
                stop_loss=sl_price,
                take_profit=tp2_p,
                highest_price=exec_price,
                lowest_price=exec_price,
                fees_paid=fee,
                entry_timestamp=timestamp,
                strategy_type="GOLDEN_POT",
                status="OPEN",
                tp1_price=tp1_p,
                tp1_hit=False,
                tp2_price=tp2_p,
                initial_quantity=qty,
            )

            next_quota_num = self.daily_golden_pots_count + 1
            trade = PaperTrade(
                trade_id=str(uuid.uuid4())[:8],
                position_id=pos_id,
                symbol=symbol,
                action=f"OPEN_GOLDEN_POT_{side}",
                price=exec_price,
                quantity=qty,
                fee=fee,
                pnl=0.0,
                timestamp=timestamp,
                rationale=f"Golden Pot ({next_quota_num}/{self.max_daily_golden_pots}) | {signal.rationale}",
            )

            # Persistência atômica no SQLite com integridade e rollback garantidos
            try:
                self._save_position(pos)
                self._save_trade(trade)
            except Exception as exc:
                logger.error(f"Falha ao persistir Golden Pot no SQLite: {exc}. Rollback do estado em memória.")
                return None

            self.balance -= fee
            self.total_fees_paid += fee
            self.daily_golden_pots_count += 1
            self.positions[pos_id] = pos
            self.trade_history.append(trade)

            logger.info(f"[{symbol}] Golden Pot {side} #{self.daily_golden_pots_count} ABERTO @ ${exec_price:,.2f} | TP1: ${tp1_p:,.2f} | TP2: ${tp2_p:,.2f} | SL: ${sl_price:,.2f}")
            return pos

    def execute_carry_allocation(
        self,
        symbol: str,
        current_price: float,
        timestamp: int,
        allocation_usd: float = 2000.0,
        rationale: str = "Carry Delta-Neutro Spot+Perp Ativado",
    ) -> Optional[PaperPosition]:
        """Abre estrutura delta-neutra spot long + perp short."""
        with self._lock:
            # Verifica se já há carry ativo no par
            existing = [p for p in self.positions.values() if p.symbol == symbol and p.strategy_type == "CARRY" and p.status == "OPEN"]
            if existing:
                return None

            # Respeita o teto global de margem (evita somar exposição acima do capital)
            free_margin = max(0.0, self.balance * self.max_margin_utilization_pct - self.margin_used)
            usable_alloc = min(allocation_usd, self.balance * 0.8, free_margin)
            if usable_alloc < 50.0:
                return None

            # 4 pernas: 50% Spot Long, 50% Perp Short
            leg_notional = usable_alloc / 2.0
            spot_price = current_price * (1.0 + self.slippage_spot)
            perp_price = current_price * (1.0 - self.slippage_perp)

            spot_qty = leg_notional / spot_price
            perp_qty = leg_notional / perp_price

            spot_fee = leg_notional * self.taker_fee
            perp_fee = leg_notional * self.taker_fee
            total_open_fees = spot_fee + perp_fee

            if self.balance < usable_alloc + total_open_fees:
                return None

            pos_id = str(uuid.uuid4())[:8]
            pos = PaperPosition(
                position_id=pos_id,
                symbol=symbol,
                side="DELTA_NEUTRAL_CARRY",
                entry_price=current_price,
                current_price=current_price,
                quantity=spot_qty,
                notional=usable_alloc,
                stop_loss=0.0,
                take_profit=0.0,
                highest_price=current_price,
                lowest_price=current_price,
                fees_paid=total_open_fees,
                entry_timestamp=timestamp,
                strategy_type="CARRY",
                status="OPEN",
            )

            trade = PaperTrade(
                trade_id=str(uuid.uuid4())[:8],
                position_id=pos_id,
                symbol=symbol,
                action="OPEN_CARRY",
                price=current_price,
                quantity=spot_qty,
                fee=total_open_fees,
                pnl=0.0,
                timestamp=timestamp,
                rationale=f"{rationale} (Spot Long: ${leg_notional:.2f}, Perp Short: ${leg_notional:.2f})",
            )

            try:
                self._save_position(pos)
                self._save_trade(trade)
            except Exception as exc:
                logger.error(f"Falha ao persistir carry no SQLite: {exc}. Rollback executado.")
                return None

            self.balance -= total_open_fees
            self.total_fees_paid += total_open_fees
            self.positions[pos_id] = pos
            self.trade_history.append(trade)

            logger.info(f"[{symbol}] Estrutura Delta-Neutra CARRY Aberta | Notional: ${usable_alloc:,.2f} USD")
            return pos

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        timestamp: int,
        reason: str = "MANUAL_CLOSE",
    ) -> Optional[PaperTrade]:
        """Encerra uma posição simulada, deduz taxas de saída e liquida PnL."""
        with self._lock:
            if position_id not in self.positions:
                return None

            pos = self.positions[position_id]
            if pos.status != "OPEN":
                return None

            # A1: apenas TP2 (limit order) de SCALP/GOLDEN_POT é maker. SL, BREAKEVEN_STOP e
            # MANUAL são ordens a mercado: taker + slippage perp.
            is_scalp_like = pos.strategy_type in ["SCALP", "GOLDEN_POT"]
            is_maker_exit = is_scalp_like and reason == "TAKE_PROFIT_2"
            fee_rate = self.maker_fee if is_maker_exit else self.taker_fee
            if is_maker_exit:
                slip = 0.0001
            elif is_scalp_like:
                slip = self.slippage_perp
            else:
                slip = self.slippage_perp if pos.side == "SHORT" else self.slippage_spot

            if pos.side == "LONG":
                exec_price = exit_price * (1.0 - slip)
                exit_fee = (pos.quantity * exec_price) * fee_rate
                gross_pnl = (exec_price - pos.entry_price) * pos.quantity
            elif pos.side == "SHORT":
                exec_price = exit_price * (1.0 + slip)
                exit_fee = (pos.quantity * exec_price) * fee_rate
                gross_pnl = (pos.entry_price - exec_price) * pos.quantity
            else:  # DELTA_NEUTRAL_CARRY
                exec_price = exit_price
                exit_fee = pos.notional * self.taker_fee
                gross_pnl = pos.funding_collected  # PnL no carry advém estritamente do funding acumulado

            net_pnl = gross_pnl - exit_fee
            prev_status = pos.status
            prev_realized = pos.realized_pnl
            prev_fees = pos.fees_paid
            prev_curr_price = pos.current_price

            pos.realized_pnl += net_pnl
            pos.fees_paid += exit_fee
            pos.status = "CLOSED"
            pos.current_price = exec_price

            trade = PaperTrade(
                trade_id=str(uuid.uuid4())[:8],
                position_id=pos.position_id,
                symbol=pos.symbol,
                action=reason,
                price=exec_price,
                quantity=pos.quantity,
                fee=exit_fee,
                pnl=net_pnl,
                timestamp=timestamp,
                rationale=f"Posição {pos.side} ({pos.strategy_type}) encerrada por {reason} | PnL Líquido: ${net_pnl:+.2f}",
            )

            try:
                self._save_position(pos)
                self._save_trade(trade)
            except Exception as exc:
                logger.error(f"Falha ao persistir encerramento de posição no SQLite: {exc}. Rollback executado.")
                pos.status = prev_status
                pos.realized_pnl = prev_realized
                pos.fees_paid = prev_fees
                pos.current_price = prev_curr_price
                return None

            # net_pnl já está líquido da taxa de saída: não subtrair exit_fee novamente
            self.balance += net_pnl
            self.total_fees_paid += exit_fee
            self.total_realized_pnl += net_pnl

            self.trade_history.append(trade)
            self.closed_positions.append(pos)
            del self.positions[position_id]

            logger.info(f"[{pos.symbol}] Posição {pos.side} FECHADA ({reason}) @ ${exec_price:,.2f} | PnL: ${net_pnl:+.2f}")

            # Telemetria & Circuit Breakers (Fase 7)
            if getattr(self, "performance_tracker", None):
                try:
                    metrics = self.performance_tracker.compute_metrics_from_db()
                    if getattr(self, "rollback_guard", None):
                        self.rollback_guard.evaluate_telemetry(metrics)
                except Exception as _e:
                    logger.debug(f"Aviso telemetria pós-trade: {_e}")

            # Aprendizado Adaptativo Contínuo (LearningTracker)
            if getattr(self, "learning_tracker", None):
                try:
                    self.learning_tracker.update_trade_outcome(pos.position_id, net_pnl)
                except Exception as _e:
                    logger.debug(f"Aviso learning pós-trade: {_e}")

            return trade



    # =========================================================================
    # ATUALIZAÇÃO INTRA-BARRA E ROLAGEM DE MERCADO
    # =========================================================================

    def update_market_tick(
        self,
        symbol: str,
        high: float,
        low: float,
        close: float,
        timestamp: int,
        funding_rate_period: float = 0.0,
    ) -> List[PaperTrade]:
        # Todo o ciclo de leitura/mutação roda sob RLock (reentrante com close_position),
        # evitando "dict changed size during iteration" e leituras inconsistentes de
        # equity/margin em threads concorrentes (Flask, supervisor).
        with self._lock:
            return self._update_market_tick_locked(symbol, high, low, close, timestamp, funding_rate_period)

    def _update_market_tick_locked(
        self,
        symbol: str,
        bar_high: float,
        bar_low: float,
        close: float,
        timestamp: int,
        funding_rate_period: float = 0.0,
    ) -> List[PaperTrade]:
        """
        Atualiza posições com a vela de mercado e verifica SL/TP intra-barra.
        Aplica o funding rate se for momento de liquidação (a cada 8h).
        Suporta saídas parciais (TP1 50% + SL Breakeven) para Scalps Intradiários.
        Thread-safe com proteção de lock.
        """
        with self._lock:
            triggered_trades: List[PaperTrade] = []
            open_pos_ids = list(self.positions.keys())

        for pos_id in open_pos_ids:
            if pos_id not in self.positions:
                continue
            pos = self.positions[pos_id]
            if pos.symbol != symbol or pos.status != "OPEN":
                continue

            # A3: vela anterior/simultânea à entrada não pode disparar SL/TP com extremos
            # que ocorreram antes da posição existir -> usa apenas o close.
            if self._ts_ms(timestamp) <= self._ts_ms(pos.entry_timestamp):
                high = low = close
            else:
                high, low = bar_high, bar_low

            pos.current_price = close
            pos.highest_price = max(pos.highest_price, high)
            pos.lowest_price = min(pos.lowest_price, low)

            # 1. Estratégia SCALP ou GOLDEN_POT (Saídas Parciais TP1 50% + Breakeven Protegido + TP2)
            if pos.strategy_type in ["SCALP", "GOLDEN_POT"]:
                strat_name = "Golden Pot" if pos.strategy_type == "GOLDEN_POT" else "Scalp"
                if pos.side == "LONG":
                    pos.unrealized_pnl = (close - pos.entry_price) * pos.quantity
                    # Gatilho TP1 Parcial (50%)
                    if not pos.tp1_hit and high >= pos.tp1_price:
                        pos.tp1_hit = True
                        partial_qty = pos.initial_quantity * 0.5
                        pnl_partial = (pos.tp1_price - pos.entry_price) * partial_qty
                        fee_partial = (pos.tp1_price * partial_qty) * self.maker_fee
                        net_partial = pnl_partial - fee_partial

                        pos.quantity -= partial_qty
                        pos.realized_pnl += net_partial
                        pos.fees_paid += fee_partial
                        self.balance += net_partial  # net_partial já deduz fee_partial
                        self.total_fees_paid += fee_partial
                        self.total_realized_pnl += net_partial

                        # Move SL para Breakeven protegido (+0.04% para cobrir taxas)
                        pos.stop_loss = pos.entry_price * 1.0004

                        t_tp1 = PaperTrade(
                            trade_id=str(uuid.uuid4())[:8],
                            position_id=pos.position_id,
                            symbol=pos.symbol,
                            action="TP1_PARTIAL",
                            price=pos.tp1_price,
                            quantity=partial_qty,
                            fee=fee_partial,
                            pnl=net_partial,
                            timestamp=timestamp,
                            rationale=f"TP1 {strat_name} Parcial (50%) atingido @ ${pos.tp1_price:,.2f} | SL elevado para Breakeven",
                        )
                        self.trade_history.append(t_tp1)
                        self._save_trade(t_tp1)
                        triggered_trades.append(t_tp1)
                        logger.info(f"[{pos.symbol}] TP1 {strat_name} (50%) EXECUTADO @ ${pos.tp1_price:,.2f} | PnL: ${net_partial:+.2f} | SL -> Breakeven")
                        pos.unrealized_pnl = (close - pos.entry_price) * pos.quantity  # recalcula com qtd remanescente

                    # Gatilho TP2 (encerra os 50% restantes)
                    if pos.tp1_hit and high >= pos.tp2_price:
                        t = self.close_position(pos_id, pos.tp2_price, timestamp, reason="TAKE_PROFIT_2")
                        if t: triggered_trades.append(t)
                        continue

                    # Gatilho Stop Loss (ou Breakeven se TP1 já bateu)
                    if low <= pos.stop_loss:
                        reason = "BREAKEVEN_STOP" if pos.tp1_hit else "STOP_LOSS"
                        t = self.close_position(pos_id, pos.stop_loss, timestamp, reason=reason)
                        if t: triggered_trades.append(t)
                        continue

                elif pos.side == "SHORT":
                    pos.unrealized_pnl = (pos.entry_price - close) * pos.quantity
                    # Gatilho TP1 Parcial (50%)
                    if not pos.tp1_hit and low <= pos.tp1_price:
                        pos.tp1_hit = True
                        partial_qty = pos.initial_quantity * 0.5
                        pnl_partial = (pos.entry_price - pos.tp1_price) * partial_qty
                        fee_partial = (pos.tp1_price * partial_qty) * self.maker_fee
                        net_partial = pnl_partial - fee_partial

                        pos.quantity -= partial_qty
                        pos.realized_pnl += net_partial
                        pos.fees_paid += fee_partial
                        self.balance += net_partial  # net_partial já deduz fee_partial
                        self.total_fees_paid += fee_partial
                        self.total_realized_pnl += net_partial

                        # Move SL para Breakeven protegido (-0.04% para cobrir taxas)
                        pos.stop_loss = pos.entry_price * 0.9996

                        t_tp1 = PaperTrade(
                            trade_id=str(uuid.uuid4())[:8],
                            position_id=pos.position_id,
                            symbol=pos.symbol,
                            action="TP1_PARTIAL",
                            price=pos.tp1_price,
                            quantity=partial_qty,
                            fee=fee_partial,
                            pnl=net_partial,
                            timestamp=timestamp,
                            rationale=f"TP1 {strat_name} Parcial (50%) atingido @ ${pos.tp1_price:,.2f} | SL reduzido para Breakeven",
                        )
                        self.trade_history.append(t_tp1)
                        self._save_trade(t_tp1)
                        triggered_trades.append(t_tp1)
                        logger.info(f"[{pos.symbol}] TP1 {strat_name} (50%) EXECUTADO @ ${pos.tp1_price:,.2f} | PnL: ${net_partial:+.2f} | SL -> Breakeven")
                        pos.unrealized_pnl = (pos.entry_price - close) * pos.quantity  # recalcula com qtd remanescente

                    # Gatilho TP2 (encerra os 50% restantes)
                    if pos.tp1_hit and low <= pos.tp2_price:
                        t = self.close_position(pos_id, pos.tp2_price, timestamp, reason="TAKE_PROFIT_2")
                        if t: triggered_trades.append(t)
                        continue

                    # Gatilho Stop Loss (ou Breakeven se TP1 já bateu)
                    if high >= pos.stop_loss:
                        reason = "BREAKEVEN_STOP" if pos.tp1_hit else "STOP_LOSS"
                        t = self.close_position(pos_id, pos.stop_loss, timestamp, reason=reason)
                        if t: triggered_trades.append(t)
                        continue

                self._save_position(pos)
                continue

            # 2. Estratégia DIRECTIONAL Padrão
            elif pos.side == "LONG":
                pos.unrealized_pnl = (close - pos.entry_price) * pos.quantity
                # Checagem de Stop Loss intra-barra
                if low <= pos.stop_loss:
                    t = self.close_position(pos_id, pos.stop_loss, timestamp, reason="STOP_LOSS")
                    if t: triggered_trades.append(t)
                    continue
                # Checagem de Take Profit intra-barra
                elif high >= pos.take_profit:
                    t = self.close_position(pos_id, pos.take_profit, timestamp, reason="TAKE_PROFIT")
                    if t: triggered_trades.append(t)
                    continue

            elif pos.side == "SHORT":
                pos.unrealized_pnl = (pos.entry_price - close) * pos.quantity
                # Checagem de Stop Loss intra-barra
                if high >= pos.stop_loss:
                    t = self.close_position(pos_id, pos.stop_loss, timestamp, reason="STOP_LOSS")
                    if t: triggered_trades.append(t)
                    continue
                # Checagem de Take Profit intra-barra
                elif low <= pos.take_profit:
                    t = self.close_position(pos_id, pos.take_profit, timestamp, reason="TAKE_PROFIT")
                    if t: triggered_trades.append(t)
                    continue

            elif pos.side == "DELTA_NEUTRAL_CARRY":
                # No carry delta-neutro, a oscilação de preço tem delta zero
                if funding_rate_period != 0.0:
                    # Notional perp short recebe o funding
                    perp_notional = pos.notional / 2.0
                    earned = perp_notional * funding_rate_period
                    pos.funding_collected += earned
                    pos.unrealized_pnl = pos.funding_collected
                    logger.debug(f"[{symbol}] Funding Carry Acumulado: +${earned:.4f} (Taxa: {funding_rate_period:+.4%})")

            self._save_position(pos)

        # Registra snapshot de carteira no SQLite
        self._record_portfolio_snapshot(timestamp)
        return triggered_trades

    # =========================================================================
    # PERSISTÊNCIA EM BANCO DE DADOS SQLITE (WAL MODE)
    # =========================================================================

    def _save_position(self, pos: PaperPosition) -> None:
        with self._get_connection() as con:
            con.execute("""
                INSERT OR REPLACE INTO paper_positions (
                    position_id, symbol, side, entry_price, current_price, quantity,
                    notional, stop_loss, take_profit, unrealized_pnl, realized_pnl,
                    funding_collected, fees_paid, entry_timestamp, strategy_type, status,
                    tp1_price, tp1_hit, tp2_price, initial_quantity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pos.position_id, pos.symbol, pos.side, pos.entry_price, pos.current_price,
                pos.quantity, pos.notional, pos.stop_loss, pos.take_profit,
                pos.unrealized_pnl, pos.realized_pnl, pos.funding_collected,
                pos.fees_paid, pos.entry_timestamp, pos.strategy_type, pos.status,
                pos.tp1_price, 1 if pos.tp1_hit else 0, pos.tp2_price, pos.initial_quantity
            ))
            con.commit()


    def _save_trade(self, trade: PaperTrade) -> None:
        with self._get_connection() as con:
            con.execute("""
                INSERT OR REPLACE INTO paper_trades (
                    trade_id, position_id, symbol, action, price, quantity, fee, pnl, timestamp, rationale
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.trade_id, trade.position_id, trade.symbol, trade.action,
                trade.price, trade.quantity, trade.fee, trade.pnl, trade.timestamp, trade.rationale
            ))
            con.commit()

    def _record_portfolio_snapshot(self, timestamp: int, force: bool = False) -> None:
        """
        Registra snapshot do portfólio no SQLite.
        Aplica compressão inteligente: grava apenas se houver variação no equity >= $0.01,
        mudança no número de posições abertas, transcurso de >= 300 segundos, ou se force=True.
        Reduz I/O em disco em > 95%.
        """
        curr_equity = self.equity
        open_count = len(self.positions)
        time_diff = abs(timestamp - self._last_snapshot_time) if self._last_snapshot_time > 0 else 999999.0
        is_ms = (timestamp > 1e11)
        time_diff_sec = (time_diff / 1000.0) if is_ms else float(time_diff)

        equity_changed = abs(curr_equity - self._last_snapshot_equity) >= 0.01
        count_changed = (open_count != self._last_snapshot_open_count)
        time_elapsed = time_diff_sec >= 300.0

        if not force and self._last_snapshot_time > 0:
            if not (equity_changed or count_changed or time_elapsed):
                return

        self._last_snapshot_time = float(timestamp)
        self._last_snapshot_equity = curr_equity
        self._last_snapshot_open_count = open_count

        with self._get_connection() as con:
            con.execute("""
                INSERT OR REPLACE INTO paper_portfolio (
                    timestamp, equity, balance, margin_used, unrealized_pnl,
                    realized_pnl_cum, total_fees_paid, open_positions_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, curr_equity, self.balance, self.margin_used,
                sum(p.unrealized_pnl for p in self.positions.values()),
                self.total_realized_pnl, self.total_fees_paid, open_count
            ))
            con.commit()
