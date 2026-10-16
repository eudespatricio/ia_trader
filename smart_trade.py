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
from db_conn import connect as _db_connect
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
from leverage_policy import MAX_LEVERAGE, clamp_leverage

# Lote padrão = MARGEM (capital alocado, USD) por trade em TODA ordem do paper trading
# (GP, Scalp, Directional); notional = lote x alavancagem (decidida pelo Smart Trader, <= MAX_LEVERAGE).
# Configurável em runtime via simulation_settings["paper_lot_usd"] (validado em [MIN, MAX]).
# Taxas/slippage são proporcionais ao notional -> expectância em % independe do lote e da alavancagem.
PAPER_LOT_USD = 100.0
PAPER_LOT_MIN_USD = 10.0
PAPER_LOT_MAX_USD = 1000.0
MAINTENANCE_MARGIN_PCT = 0.005   # 0.5% de margem de manutenção (preço de liquidação aproximado)

# ----------------------------------------------------------------------------
# SMART TRADE EXIT LADDER (ADDENDUM E7, 2026-10-15) - COMUM a SCALP, GOLDEN_POT e DIRECTIONAL.
#   TP1 (EXIT_TP1_FRACTION)  -> parcial maker; SL vai a breakeven (+BREAKEVEN_FEE_BUFFER_PCT).
#   TP2 (EXIT_TP2_FRACTION)  -> parcial maker; ATIVA o trailing stop no runner restante.
#   Runner (1 - TP1 - TP2)   -> sai em TP3 (maker) OU no trailing stop (taker), o que vier antes.
# Trailing: distância = trailing_atr_mult x ATR_de_entrada (congelado); o stop só é atualizado a partir de
# barras JÁ FECHADAS (extremos da barra corrente só entram quando chega a barra seguinte) e nunca afrouxa.
# As estratégias diferem SOMENTE pelos multiplicadores abaixo (e pelo gatilho de oportunidade).
# ----------------------------------------------------------------------------
EXIT_TP1_FRACTION = 0.40
EXIT_TP2_FRACTION = 0.30
EXIT_RUNNER_FRACTION = round(1.0 - EXIT_TP1_FRACTION - EXIT_TP2_FRACTION, 10)   # 0.30
BREAKEVEN_FEE_BUFFER_PCT = 0.0004   # SL pós-TP1 = entrada +/- 0.04% (cobre taxas)
PARTIAL_TRADE_ACTIONS = frozenset({"TP1_PARTIAL", "TP2_PARTIAL"})   # saídas parciais (não fecham a posição)
# Saídas por ordem limite (maker). SL/BREAKEVEN_STOP/TRAILING_STOP/MANUAL_CLOSE/LIQUIDATION = taker.
MAKER_EXIT_REASONS = frozenset({"TAKE_PROFIT_1", "TAKE_PROFIT_2", "TAKE_PROFIT_3"})


# ----------------------------------------------------------------------------
# PARIDADE BACKTEST x PILOTO (docs/PARIDADE_BACKTEST_PILOTO.md) - convenções únicas
# ----------------------------------------------------------------------------
# P2 ENTRADA: o chamador informa o timestamp (abertura) da vela de DECISÃO t, já FECHADA. Em cripto 24/7,
#   close(t) == open(t+1); o piloto entra a mercado nesse instante ao preço close(t) (+ slippage) e registra
#   entry_timestamp = ts(t) + bar_ms == ts(t+1), exatamente a convenção do BacktestEngine (entrada no open t+1).
#   A vela t+1 é, portanto, a primeira vela pós-entrada e é processada com high/low completos.
# P4 TIMEOUT: posição aberta há >= max_holding_bars barras do timeframe de decisão é fechada a mercado
#   (taker + slippage, motivo HORIZON_TIMEOUT) no open da barra seguinte ao prazo. Mesmo valor do A/B/backtest
#   (48 barras) no DIRECTIONAL 1h (=48h). SCALP/GP decidem em 15m: mantém-se o MESMO nº de barras (48 x 15m = 12h),
#   i.e. o horizonte é definido em barras do timeframe de decisão, como no backtest (max_holding_bars).
DIRECTIONAL_BAR_MS = 60 * 60 * 1000
INTRADAY_BAR_MS = 15 * 60 * 1000
DIRECTIONAL_MAX_HOLDING_BARS = 48
SCALP_MAX_HOLDING_BARS = 48
GOLDEN_POT_MAX_HOLDING_BARS = 48
# SLIPPAGE de ordens a MERCADO (entrada taker, SL/BE/TRAILING/TIMEOUT/MANUAL): único valor, simétrico long/short,
# igual ao BacktestEngine (0.02% por fill). Ordens limite (maker) mantêm 0.01%. (slippage_spot/perp ficam só no carry.)
MARKET_SLIPPAGE_PCT = 0.0002
MAKER_SLIPPAGE_PCT = 0.0001
# P5 FUNDING: liquidação a cada 8h DECORRIDAS desde a entrada (entry + k*8h) - mesma regra do BacktestEngine
# (bars_held % bars_per_funding == 0, i.e. por tempo e independente do timeframe). funding>0: long paga / short
# recebe; <0 inverso. Taxa REAL = último registro da tabela `funding_rates` do cache SQLite (sem rede) vigente
# naquele instante (<= 8h de idade); ausente -> FUNDING_FALLBACK_RATE_8H (mesmo 0.01%/8h do BacktestEngine).
# (A bolsa liquida em 00/08/16h UTC; o nº esperado de liquidações por posição é o mesmo - convenção de paridade.)
FUNDING_INTERVAL_MS = 8 * 60 * 60 * 1000
FUNDING_FALLBACK_RATE_8H = 0.0001
FUNDING_RATE_MATCH_TOLERANCE_MS = 10 * 60 * 1000   # tolerância de jitter do timestamp da taxa
FUNDING_MAX_SETTLEMENTS_PER_TICK = 12
NON_CLOSING_TRADE_ACTIONS = PARTIAL_TRADE_ACTIONS | frozenset({"FUNDING"})   # linhas de ledger que NÃO fecham a posição


# --- Funil de sinais (instrumentação pura; NÃO altera decisões de trading) ---
FUNNEL_SUCCESS_STAGES: List[str] = [
    "evaluated", "candidate", "committee_agree", "confidence_ok", "risk_ok", "gate_passed", "executed",
]
FUNNEL_REJECT_REASONS: List[str] = [
    "no_data", "no_trigger", "committee_disagree", "low_confidence", "veto", "cooldown", "bar_limit",
    "quota", "margin", "breaker", "gate_blocked", "excluded_symbol", "other",
]
FUNNEL_FLUSH_INTERVAL_S: float = 30.0
FUNNEL_DEDUPE_MAX_AGE_S: float = 6 * 3600.0


@dataclass(frozen=True)
class StrategyProfile:
    """Multiplicadores de ATR (sobre o preço de entrada) de uma estratégia. Mesma mecânica para todas."""
    sl_atr_mult: float
    tp1_atr_mult: float
    tp2_atr_mult: float
    tp3_atr_mult: float
    trail_atr_mult: float
    min_committee_confidence: float = 0.70
    bar_ms: int = DIRECTIONAL_BAR_MS                       # duração da barra de decisão (P2/P4)
    max_holding_bars: int = DIRECTIONAL_MAX_HOLDING_BARS   # timeout por nº de barras (P4)


STRATEGY_PROFILES: Dict[str, StrategyProfile] = {
    # alvos curtos
    "SCALP": StrategyProfile(sl_atr_mult=0.9, tp1_atr_mult=1.0, tp2_atr_mult=1.8, tp3_atr_mult=3.0, trail_atr_mult=0.8,
                             bar_ms=INTRADAY_BAR_MS, max_holding_bars=SCALP_MAX_HOLDING_BARS),
    # day trade 1h (SL efetivo = o do comitê/SignalGenerator; sl_atr_mult é o fallback)
    "DIRECTIONAL": StrategyProfile(sl_atr_mult=1.0, tp1_atr_mult=1.5, tp2_atr_mult=2.5, tp3_atr_mult=4.0, trail_atr_mult=1.2,
                                   bar_ms=DIRECTIONAL_BAR_MS, max_holding_bars=DIRECTIONAL_MAX_HOLDING_BARS),
    # maior potencial: TP3 bem mais longo e trailing mais largo
    "GOLDEN_POT": StrategyProfile(sl_atr_mult=1.0, tp1_atr_mult=1.2, tp2_atr_mult=2.5, tp3_atr_mult=6.0, trail_atr_mult=1.5,
                                  bar_ms=INTRADAY_BAR_MS, max_holding_bars=GOLDEN_POT_MAX_HOLDING_BARS),
}

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
    leverage: float = 1.0
    # --- Smart Trade Exit Ladder (E7) ---
    tp3_price: float = 0.0
    tp2_hit: bool = False
    tp1_fraction: float = EXIT_TP1_FRACTION     # fração da quantidade inicial realizada no TP1
    tp2_fraction: float = EXIT_TP2_FRACTION     # fração da quantidade inicial realizada no TP2
    trailing_active: bool = False
    trailing_stop_price: float = 0.0            # 0.0 = ainda sem nível calculado
    trailing_atr_mult: float = 0.0
    entry_atr: float = 0.0                      # ATR congelado na entrada (base do trailing)
    # estado em memória do trailing (barras fechadas); reconstruído em restore_state a partir do stop
    trail_ref_extreme: float = 0.0
    trail_bar_ts: float = 0.0
    trail_bar_high: float = 0.0
    trail_bar_low: float = 0.0
    # estado em memória de funding/timeout (P4/P5); reconstruído em restore_state
    funding_cursor_ms: float = 0.0    # última liquidação de funding já aplicada (0 = usa entry_timestamp)
    last_bar_close: float = 0.0       # close da última barra FECHADA processada (== open da seguinte em 24/7)

    @property
    def runner_fraction(self) -> float:
        return max(0.0, 1.0 - self.tp1_fraction - self.tp2_fraction)

    @property
    def remaining_fraction(self) -> float:
        base = self.initial_quantity if self.initial_quantity > 0 else self.quantity
        return (self.quantity / base) if base > 0 else 0.0

    @property
    def margin(self) -> float:
        """Margem (capital alocado) = notional / alavancagem."""
        lev = self.leverage if self.leverage and self.leverage >= 1.0 else 1.0
        return self.notional / lev

    @property
    def liquidation_price(self) -> float:
        """Preço de liquidação aproximado (0.0 se sem alavancagem ou não-direcional)."""
        if self.leverage is None or self.leverage <= 1.0 or self.side not in ("LONG", "SHORT"):
            return 0.0
        return PaperPosition.calc_liquidation_price(self.side, self.entry_price, self.leverage)

    @staticmethod
    def calc_liquidation_price(side: str, entry: float, leverage: float) -> float:
        if side == "LONG":
            return entry * (1.0 - 1.0 / leverage + MAINTENANCE_MARGIN_PCT)
        return entry * (1.0 + 1.0 / leverage - MAINTENANCE_MARGIN_PCT)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("trail_ref_extreme", "trail_bar_ts", "trail_bar_high", "trail_bar_low",
                  "funding_cursor_ms", "last_bar_close"):
            d.pop(k, None)   # estado interno de barras, não é contrato público
        d["margin"] = self.margin
        d["liquidation_price"] = self.liquidation_price
        d["tp1"], d["tp2"], d["tp3"] = self.tp1_price, self.tp2_price, self.tp3_price
        d["runner_fraction"] = self.runner_fraction
        d["remaining_fraction"] = self.remaining_fraction
        return d



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
        slippage_perp: float = 0.0003,   # 0.03% slippage perp (somente pernas do carry)
        simulation_mode: bool = True,
        core: Optional[TradingCore] = None,
        max_concurrent_positions: Optional[int] = None,
        max_margin_utilization_pct: Optional[float] = None,
        restore: bool = True,
        slippage_market: Optional[float] = None,
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
        # Slippage único (long == short) das ordens a mercado direcionais/scalp/GP = BacktestEngine (P-slip).
        self.slippage_market = MARKET_SLIPPAGE_PCT if slippage_market is None else float(slippage_market)
        self.default_funding_rate_8h = FUNDING_FALLBACK_RATE_8H

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

        self.paper_lot_usd: float = PAPER_LOT_USD

        self._lock = threading.RLock()

        # Funil de sinais: buffer em memória + dedupe por (strategy, symbol, bar_ts, stage)
        self._funnel_lock = threading.Lock()
        self._funnel_buffer: Dict[Tuple[int, str, str], int] = {}
        self._funnel_seen: Dict[Tuple[str, str, int, str], float] = {}
        self._funnel_last_flush: float = time.time()
        self._funnel_clock = time.time
        self.last_open_reject: str = "other"

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
        con = _db_connect(self.db_path, timeout=15.0)
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
                ("leverage", "REAL DEFAULT 1.0"),
                # Smart Trade Exit Ladder (E7). Defaults = comportamento legado (TP1 50% + TP2 fecha o restante).
                ("tp3_price", "REAL DEFAULT 0.0"),
                ("tp2_hit", "INTEGER DEFAULT 0"),
                ("tp1_fraction", "REAL DEFAULT 0.5"),
                ("tp2_fraction", "REAL DEFAULT 0.0"),
                ("trailing_active", "INTEGER DEFAULT 0"),
                ("trailing_stop_price", "REAL DEFAULT 0.0"),
                ("trailing_atr_mult", "REAL DEFAULT 0.0"),
                ("entry_atr", "REAL DEFAULT 0.0"),
            ]:
                try:
                    con.execute(f"ALTER TABLE paper_positions ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass

            con.execute("""
                CREATE TABLE IF NOT EXISTS flow_gate_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms INTEGER, symbol TEXT, strategy_type TEXT, direction TEXT,
                    decision TEXT, tape_vote TEXT, tape_status TEXT, tape_reason TEXT,
                    taker_imb_5m REAL, book_imb_10 REAL, microprice_dev REAL, spread_bps REAL,
                    entry_price REAL, stop_loss REAL, tp1 REAL, tp2 REAL, tp3 REAL, atr REAL,
                    confidence REAL, consensus_score REAL, leverage REAL
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS signal_funnel (
                    hour_ms INTEGER NOT NULL,
                    strategy TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY (hour_ms, strategy, stage)
                )
            """)
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
            cur.execute("INSERT OR IGNORE INTO simulation_settings (key, value) VALUES ('paper_lot_usd', ?)",
                        (str(PAPER_LOT_USD),))
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
                    "strategy_type, status, tp1_price, tp1_hit, tp2_price, initial_quantity, COALESCE(leverage, 1.0), "
                    "COALESCE(tp3_price, 0.0), COALESCE(tp2_hit, 0), COALESCE(tp1_fraction, 0.5), "
                    "COALESCE(tp2_fraction, 0.0), COALESCE(trailing_active, 0), COALESCE(trailing_stop_price, 0.0), "
                    "COALESCE(trailing_atr_mult, 0.0), COALESCE(entry_atr, 0.0) "
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
                    initial_quantity=f(r[19]), leverage=max(1.0, f(r[20])),
                    tp3_price=f(r[21]), tp2_hit=bool(r[22]), tp1_fraction=f(r[23]), tp2_fraction=f(r[24]),
                    trailing_active=bool(r[25]), trailing_stop_price=f(r[26]), trailing_atr_mult=f(r[27]),
                    entry_atr=f(r[28]),
                )
                if pos.trailing_active and pos.trailing_stop_price > 0 and pos.trailing_atr_mult > 0:
                    # referência = extremo das barras fechadas que gerou o stop (stop +/- distância).
                    # A barra em formação no momento do restart é descartada (só afrouxa informação, nunca vaza).
                    d_trail = pos.trailing_atr_mult * pos.entry_atr
                    pos.trail_ref_extreme = (pos.trailing_stop_price + d_trail if pos.side == "LONG"
                                             else pos.trailing_stop_price - d_trail)
                if pos.status == "OPEN":
                    open_pos[pos.position_id] = pos
                else:
                    closed_pos.append(pos)

            # P5: cursor de funding = último FUNDING já lançado no ledger (não reaplica após restart)
            for h in history:
                if h.action == "FUNDING" and h.position_id in open_pos:
                    pp = open_pos[h.position_id]
                    pp.funding_cursor_ms = max(pp.funding_cursor_ms, self._ts_ms(h.timestamp))

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
            "paper_lot_usd": self.paper_lot_usd,
        }
        try:
            with self._get_connection() as con:
                cur = con.cursor()
                cur.execute("SELECT key, value FROM simulation_settings")
                for k, v in cur.fetchall():
                    if k in ("max_daily_golden_pots", "max_daily_scalps", "max_concurrent_positions"):
                        defaults[k] = int(v)
                    elif k in ("scalp_vwap_dev_pct", "scalp_rsi_oversold", "scalp_rsi_overbought", "scalp_volume_z", "max_margin_utilization_pct", "paper_lot_usd"):
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

    @staticmethod
    def _valid_paper_lot(v: Any) -> Optional[float]:
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        if x != x or not (PAPER_LOT_MIN_USD <= x <= PAPER_LOT_MAX_USD):
            return None
        return x

    def update_simulation_settings(self, settings: Dict[str, Any], reset_quotas: bool = False) -> Dict[str, Any]:
        """Atualiza configurações de simulação no SQLite e propaga aos parâmetros internos."""
        if "paper_lot_usd" in settings and self._valid_paper_lot(settings["paper_lot_usd"]) is None:
            raise ValueError(f"paper_lot_usd deve estar entre {PAPER_LOT_MIN_USD:g} e {PAPER_LOT_MAX_USD:g} USD.")
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
                    if "paper_lot_usd" in settings_dict:
                        lot = self._valid_paper_lot(settings_dict["paper_lot_usd"])
                        if lot is not None:
                            self.paper_lot_usd = lot

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
            return sum(p.margin for p in self.positions.values() if p.status == "OPEN")

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


    def can_execute_directional(self, timestamp: int, notional: float = 0.0) -> Tuple[bool, str]:
        """P6a: RollbackGuard/circuit breaker também no DIRECIONAL (sem cota diária: o day trade não tem cota)."""
        with self._lock:
            if getattr(self, "rollback_guard", None):
                allowed, reason = self.rollback_guard.can_open_new_position("DIRECTIONAL", strategy="DIRECTIONAL")
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

    def log_flow_gate(self, row: Dict[str, Any]) -> None:
        """Auditoria contrafactual do gate de fluxo (E8): TODO sinal que chega ao gate é registrado."""
        cols = ["ts_ms", "symbol", "strategy_type", "direction", "decision", "tape_vote", "tape_status",
                "tape_reason", "taker_imb_5m", "book_imb_10", "microprice_dev", "spread_bps", "entry_price",
                "stop_loss", "tp1", "tp2", "tp3", "atr", "confidence", "consensus_score", "leverage"]
        try:
            with self._get_connection() as con:
                con.execute(f"INSERT INTO flow_gate_log ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                            [row.get(c) for c in cols])
        except Exception as e:  # noqa: BLE001 - auditoria nunca derruba o ciclo
            logger.warning(f"flow_gate_log: falha ao registrar: {e}")

    def flow_gate_counts(self) -> Dict[str, int]:
        out = {"PASSED": 0, "BLOCKED": 0, "UNAVAILABLE_PASSED": 0}
        try:
            with self._get_connection() as con:
                for d, n in con.execute("SELECT decision, COUNT(*) FROM flow_gate_log GROUP BY decision"):
                    out[str(d)] = int(n)
        except Exception:  # noqa: BLE001
            pass
        return out

    # ------------------------------------------------------------------ Funil de sinais
    def funnel_inc(self, strategy: str, symbol: str, bar_ts: Union[int, float], stage: str) -> bool:
        """Conta `stage` no máximo UMA vez por (strategy, symbol, bar_ts). Retorna True se contou.

        Buffer em memória, flush (UPSERT somando) a cada ~30 s. Nunca levanta exceção (telemetria não
        pode interromper o trading). Thread-safe.
        """
        try:
            now = float(self._funnel_clock())
            key = (str(strategy), str(symbol), int(bar_ts), str(stage))
            with self._funnel_lock:
                if key in self._funnel_seen:
                    return False
                self._funnel_seen[key] = now
                hour_ms = int(now // 3600) * 3600 * 1000
                bkey = (hour_ms, key[0], key[3])
                self._funnel_buffer[bkey] = self._funnel_buffer.get(bkey, 0) + 1
                due = (now - self._funnel_last_flush) >= FUNNEL_FLUSH_INTERVAL_S
            if due:
                self.funnel_flush()
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"signal_funnel: falha em funnel_inc: {e}")
            return False

    def funnel_reject(self, strategy: str, symbol: str, bar_ts: Union[int, float], reason: str) -> bool:
        """Grava `reject:<motivo>` (motivo fora de FUNNEL_REJECT_REASONS vira 'other')."""
        if reason not in FUNNEL_REJECT_REASONS:
            reason = "other"
        return self.funnel_inc(strategy, symbol, bar_ts, f"reject:{reason}")

    def funnel_flush(self) -> None:
        """UPSERT somando o buffer em signal_funnel. Falha de DB => warning e mantém o buffer (retry no próximo flush)."""
        now = float(self._funnel_clock())
        with self._funnel_lock:
            self._funnel_last_flush = now
            cutoff = now - FUNNEL_DEDUPE_MAX_AGE_S
            if self._funnel_seen:
                self._funnel_seen = {k: t for k, t in self._funnel_seen.items() if t >= cutoff}
            pending = dict(self._funnel_buffer)
            self._funnel_buffer.clear()
        if not pending:
            return
        try:
            with self._get_connection() as con:
                con.executemany(
                    "INSERT INTO signal_funnel (hour_ms, strategy, stage, count) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(hour_ms, strategy, stage) DO UPDATE SET count = count + excluded.count",
                    [(h, st, sg, n) for (h, st, sg), n in pending.items()],
                )
        except Exception as e:  # noqa: BLE001 - telemetria nunca derruba o ciclo
            logger.warning(f"signal_funnel: falha ao gravar ({len(pending)} contadores): {e}")
            with self._funnel_lock:
                for k, n in pending.items():
                    self._funnel_buffer[k] = self._funnel_buffer.get(k, 0) + n

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
                "total_trades_count": len(pnl_by_pos),
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

    @staticmethod
    def entry_timestamp_for(decision_ts: Union[int, float], strategy_type: str) -> Union[int, float]:
        """P2: ts(t) da vela de decisão -> ts(t+1) = instante da entrada (preserva a unidade s/ms do chamador)."""
        bar_ms = STRATEGY_PROFILES[strategy_type].bar_ms
        if decision_ts > 1e11:
            return int(decision_ts + bar_ms)
        return int(decision_ts + bar_ms // 1000)

    def _resolve_leverage(self, side: str, exec_price: float, sl_price: float, requested: float) -> float:
        """Clamp 1..MAX_LEVERAGE (passos de 0.5) e garante SL ANTES da liquidação (senão reduz a alavancagem)."""
        lev = clamp_leverage(requested, MAX_LEVERAGE)
        while lev > 1.0:
            liq = PaperPosition.calc_liquidation_price(side, exec_price, lev)
            sl_ok = (sl_price > liq) if side == "LONG" else (sl_price < liq)
            if sl_ok:
                break
            lev -= 0.5
        return max(1.0, lev)

    def _lot_margin(self, signal: TradeSignal, lev_req: float, balance_cap_pct: float) -> Tuple[float, float]:
        """Margem do trade = min(lote, margem livre*0.95, cap do saldo, position_size(notional)/alavancagem)."""
        max_margin = self.balance * self.max_margin_utilization_pct
        free_margin = max(0.0, max_margin - self.margin_used)
        margin = min(self.paper_lot_usd, free_margin * 0.95, self.balance * balance_cap_pct,
                     signal.position_size / lev_req)
        return margin, free_margin

    # Metadados de entrada por tipo de oportunidade. A MECÂNICA de saída (ladder) é idêntica para todos.
    _ENTRY_META: Dict[str, Dict[str, Any]] = {
        # DIRECTIONAL (day trade 1h): entrada a mercado (taker + slippage), como antes; saídas = ladder comum.
        "DIRECTIONAL": {"label": "Ordem Direcional", "cap": 0.5, "maker_entry": False, "atr_fallback_pct": 0.01},
        "SCALP": {"label": "Scalp", "cap": 0.25, "maker_entry": True, "atr_fallback_pct": 0.005},
        "GOLDEN_POT": {"label": "Golden Pot", "cap": 0.20, "maker_entry": True, "atr_fallback_pct": 0.008},
    }

    def _open_ladder_position(
        self,
        signal: TradeSignal,
        current_price: float,
        timestamp: int,
        strategy_type: str,
        leverage: float = 1.0,
        tp1_atr_mult: Optional[float] = None,
        tp2_atr_mult: Optional[float] = None,
        tp3_atr_mult: Optional[float] = None,
        sl_atr_mult: Optional[float] = None,
        trail_atr_mult: Optional[float] = None,
    ) -> Optional[PaperPosition]:
        """Abre posição LONG/SHORT de qualquer estratégia com a Smart Trade Exit Ladder comum.

        Multiplicadores não informados vêm de STRATEGY_PROFILES[strategy_type]. Para DIRECTIONAL o SL é o
        do comitê (signal.stop_loss) quando válido; TP1/TP2/TP3 sempre derivam de ATR x multiplicador
        (signal.take_profit do SignalGenerator não é mais usado como alvo único).
        """
        prof = STRATEGY_PROFILES[strategy_type]
        meta = self._ENTRY_META[strategy_type]
        label = meta["label"]
        with self._lock:
            self.last_open_reject = "other"
            if signal.direction not in ["BUY", "SELL"] or signal.is_vetoed:
                self.last_open_reject = "veto"
                return None

            symbol = signal.symbol
            if any(p.symbol == symbol and p.strategy_type == strategy_type and p.status == "OPEN"
                   for p in self.positions.values()):
                self.last_open_reject = "margin"  # posição já aberta no par/estratégia
                return None

            if self.max_concurrent_positions is not None and self.max_concurrent_positions > 0:
                open_count = len([p for p in self.positions.values() if p.status == "OPEN"])
                if open_count >= self.max_concurrent_positions:
                    logger.info(f"[{symbol}] {label} Rejeitado: Teto de posições simultâneas atingido ({open_count}/{self.max_concurrent_positions}).")
                    self.last_open_reject = "margin"
                    return None

            # P2: `timestamp` = abertura da vela de decisão t (fechada). Entrada a mercado em close(t) == open(t+1);
            # o registro usa ts(t+1) = ts(t) + bar_ms (mesma convenção do BacktestEngine).
            decision_ts = timestamp
            timestamp = self.entry_timestamp_for(decision_ts, strategy_type)

            # Dimensionamento: lote = margem (limitada pela margem livre); notional = margem x alavancagem
            max_margin = self.balance * self.max_margin_utilization_pct
            lev_req = clamp_leverage(leverage, MAX_LEVERAGE)
            margin, free_margin = self._lot_margin(signal, lev_req, meta["cap"])
            if margin < 10.0:  # Mínimo de $10 USD de margem
                logger.info(f"[{symbol}] {label} Rejeitado: Margem livre insuficiente (${free_margin:,.2f}).")
                self.last_open_reject = "margin"
                return None

            if strategy_type == "SCALP":
                can_exec, reason = self.can_execute_scalp(timestamp, notional=margin)
            elif strategy_type == "GOLDEN_POT":
                can_exec, reason = self.can_execute_golden_pot(timestamp, notional=margin)
            else:
                can_exec, reason = self.can_execute_directional(timestamp, notional=margin)   # P6a
            if not can_exec:
                logger.info(f"[{symbol}] {label} Rejeitado: {reason}")
                self.last_open_reject = (
                    "breaker" if "Circuit Breaker" in reason else "quota" if reason.startswith("Limite diário") else "margin")
                return None

            side = "LONG" if signal.direction == "BUY" else "SHORT"
            if meta["maker_entry"]:
                slip, fee_rate = MAKER_SLIPPAGE_PCT, self.maker_fee   # limit order: slippage mínimo + taxa maker
            else:
                slip, fee_rate = self.slippage_market, self.taker_fee   # mercado: slippage simétrico long/short
            exec_price = current_price * (1.0 + slip) if side == "LONG" else current_price * (1.0 - slip)

            atr = signal.atr if signal.atr and signal.atr > 0 else current_price * meta["atr_fallback_pct"]
            m_tp1 = prof.tp1_atr_mult if tp1_atr_mult is None else tp1_atr_mult
            m_tp2 = prof.tp2_atr_mult if tp2_atr_mult is None else tp2_atr_mult
            m_tp3 = prof.tp3_atr_mult if tp3_atr_mult is None else tp3_atr_mult
            m_trail = prof.trail_atr_mult if trail_atr_mult is None else trail_atr_mult
            sgn = 1.0 if side == "LONG" else -1.0
            sl_price = exec_price - sgn * (prof.sl_atr_mult if sl_atr_mult is None else sl_atr_mult) * atr
            if strategy_type == "DIRECTIONAL" and sl_atr_mult is None:
                sig_sl = float(signal.stop_loss or 0.0)
                if sig_sl > 0 and ((side == "LONG" and sig_sl < exec_price) or (side == "SHORT" and sig_sl > exec_price)):
                    sl_price = sig_sl
            tp1_p = exec_price + sgn * m_tp1 * atr
            tp2_p = exec_price + sgn * m_tp2 * atr
            tp3_p = exec_price + sgn * m_tp3 * atr

            lev = self._resolve_leverage(side, exec_price, sl_price, lev_req)   # SL sempre antes da liquidação
            notional = margin * lev
            qty = notional / exec_price
            fee = notional * fee_rate

            if (self.margin_used + margin + fee) > max_margin:
                logger.info(f"[{symbol}] {label} Rejeitado: Excede margem máxima permitida.")
                self.last_open_reject = "margin"
                return None

            pos_id = str(uuid.uuid4())[:8]
            pos = PaperPosition(
                position_id=pos_id, symbol=symbol, side=side, entry_price=exec_price, current_price=exec_price,
                quantity=qty, notional=notional, stop_loss=sl_price, take_profit=tp3_p,
                highest_price=exec_price, lowest_price=exec_price, fees_paid=fee, entry_timestamp=timestamp,
                strategy_type=strategy_type, status="OPEN",
                tp1_price=tp1_p, tp1_hit=False, tp2_price=tp2_p, initial_quantity=qty, leverage=lev,
                tp3_price=tp3_p, tp2_hit=False, tp1_fraction=EXIT_TP1_FRACTION, tp2_fraction=EXIT_TP2_FRACTION,
                trailing_active=False, trailing_stop_price=0.0, trailing_atr_mult=m_trail, entry_atr=atr,
                funding_cursor_ms=self._ts_ms(timestamp), last_bar_close=current_price,
            )

            if strategy_type == "SCALP":
                action = f"OPEN_SCALP_{side}"
                rationale = f"Scalp Diário ({self.daily_scalps_count + 1}/{self.max_daily_scalps}) | {signal.rationale}"
            elif strategy_type == "GOLDEN_POT":
                action = f"OPEN_GOLDEN_POT_{side}"
                rationale = f"Golden Pot ({self.daily_golden_pots_count + 1}/{self.max_daily_golden_pots}) | {signal.rationale}"
            else:
                action = f"OPEN_{side}"
                rationale = signal.rationale
            trade = PaperTrade(
                trade_id=str(uuid.uuid4())[:8], position_id=pos_id, symbol=symbol, action=action,
                price=exec_price, quantity=qty, fee=fee, pnl=0.0, timestamp=timestamp, rationale=rationale,
            )

            # Persistência atômica no SQLite com integridade e rollback garantidos
            try:
                self._save_position(pos)
                self._save_trade(trade)
            except Exception as exc:
                logger.error(f"Falha ao persistir {label} no SQLite: {exc}. Rollback do estado em memória.")
                return None

            self.balance -= fee
            self.total_fees_paid += fee
            if strategy_type == "SCALP":
                self.daily_scalps_count += 1
            elif strategy_type == "GOLDEN_POT":
                self.daily_golden_pots_count += 1
            self.positions[pos_id] = pos
            self.trade_history.append(trade)

            logger.info(
                f"[{symbol}] {label} {side} ABERTO @ ${exec_price:,.2f} | Lev: {lev:g}x | Margem: ${margin:,.2f} | "
                f"SL: ${sl_price:,.2f} | TP1: ${tp1_p:,.2f} | TP2: ${tp2_p:,.2f} | TP3: ${tp3_p:,.2f} | "
                f"Trail: {m_trail:g}xATR | Risco: ${signal.risk_dollars:.2f}"
            )
            return pos

    def execute_directional_signal(
        self,
        signal: TradeSignal,
        current_price: float,
        timestamp: int,
        leverage: float = 1.0,
        **ladder_overrides: Optional[float],
    ) -> Optional[PaperPosition]:
        """Abre posição DIRECTIONAL (day trade 1h) a partir de um sinal do comitê, com a ladder comum.

        `leverage` é decidida pelo Smart Trader (leverage_policy); aqui só é clampada (1..5) e
        reduzida se o SL ficar além do preço de liquidação. margem = lote; notional = margem x lev.
        Entrada taker (+slippage); TP1/TP2/TP3 maker; SL/BE/trailing taker.
        """
        return self._open_ladder_position(signal, current_price, timestamp, "DIRECTIONAL", leverage, **ladder_overrides)

    def execute_scalp_signal(
        self,
        signal: TradeSignal,
        current_price: float,
        timestamp: int,
        tp1_atr_mult: Optional[float] = None,
        tp2_atr_mult: Optional[float] = None,
        sl_atr_mult: Optional[float] = None,
        leverage: float = 1.0,
        tp3_atr_mult: Optional[float] = None,
        trail_atr_mult: Optional[float] = None,
    ) -> Optional[PaperPosition]:
        """Scalp intradiário (15m) - cota diária estrita; entrada maker; saídas = ladder comum (perfil SCALP).

        Só deve ser chamado com sinal APROVADO pelo comitê (ver run_paper_simulation.evaluate_scalp_opportunities).
        """
        return self._open_ladder_position(
            signal, current_price, timestamp, "SCALP", leverage,
            tp1_atr_mult=tp1_atr_mult, tp2_atr_mult=tp2_atr_mult, tp3_atr_mult=tp3_atr_mult,
            sl_atr_mult=sl_atr_mult, trail_atr_mult=trail_atr_mult,
        )

    def execute_golden_pot_signal(
        self,
        signal: TradeSignal,
        current_price: float,
        timestamp: int,
        tp1_atr_mult: Optional[float] = None,
        tp2_atr_mult: Optional[float] = None,
        sl_atr_mult: Optional[float] = None,
        leverage: float = 1.0,
        tp3_atr_mult: Optional[float] = None,
        trail_atr_mult: Optional[float] = None,
    ) -> Optional[PaperPosition]:
        """Golden Pot (momentum + volatilidade extraordinária) - cota própria (máx 3/dia); ladder comum (perfil GOLDEN_POT)."""
        return self._open_ladder_position(
            signal, current_price, timestamp, "GOLDEN_POT", leverage,
            tp1_atr_mult=tp1_atr_mult, tp2_atr_mult=tp2_atr_mult, tp3_atr_mult=tp3_atr_mult,
            sl_atr_mult=sl_atr_mult, trail_atr_mult=trail_atr_mult,
        )

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

            # A1/E7: apenas TP1/TP2/TP3 (ordens limite) são maker, em QUALQUER estratégia de ladder. SL,
            # BREAKEVEN_STOP, TRAILING_STOP, MANUAL_CLOSE e LIQUIDATION são a mercado: taker + slippage.
            # (reason legado "TAKE_PROFIT" de posições direcionais pré-E7 continua taker.)
            is_maker_exit = reason in MAKER_EXIT_REASONS and pos.side in ("LONG", "SHORT")
            fee_rate = self.maker_fee if is_maker_exit else self.taker_fee
            # slippage único e simétrico (long == short) para ordens a mercado; limite = maker
            slip = MAKER_SLIPPAGE_PCT if is_maker_exit else self.slippage_market

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

    # =========================================================================
    # SMART TRADE EXIT LADDER (E7): TP1 -> BE, TP2 -> trailing, TP3/trailing no runner
    # =========================================================================

    @staticmethod
    def _uses_ladder(pos: PaperPosition) -> bool:
        """Posições SCALP/GOLDEN_POT/DIRECTIONAL LONG/SHORT usam a ladder. Direcional pré-E7 (sem tp1) segue legado."""
        if pos.side not in ("LONG", "SHORT"):
            return False
        if pos.strategy_type in ("SCALP", "GOLDEN_POT"):
            return True
        return pos.strategy_type == "DIRECTIONAL" and pos.tp1_price > 0.0

    def _ladder_partial(self, pos: PaperPosition, price: float, fraction: float, action: str,
                        timestamp: int, label: str, note: str, triggered: List[PaperTrade]) -> PaperTrade:
        """Realiza uma saída parcial a limite (maker, sem slippage no preço do alvo). PnL/fee/ledger por parcial."""
        sgn = 1.0 if pos.side == "LONG" else -1.0
        base = pos.initial_quantity if pos.initial_quantity > 0 else pos.quantity
        qty = min(base * fraction, pos.quantity)
        gross = sgn * (price - pos.entry_price) * qty
        fee = price * qty * self.maker_fee
        net = gross - fee
        pos.quantity -= qty
        pos.realized_pnl += net
        pos.fees_paid += fee
        self.balance += net                     # net já deduz a taxa da parcial
        self.total_fees_paid += fee
        self.total_realized_pnl += net
        trade = PaperTrade(
            trade_id=str(uuid.uuid4())[:8], position_id=pos.position_id, symbol=pos.symbol, action=action,
            price=price, quantity=qty, fee=fee, pnl=net, timestamp=timestamp,
            rationale=f"{label} Parcial ({fraction:.0%}) atingido @ ${price:,.2f} | {note}",
        )
        self.trade_history.append(trade)
        self._save_trade(trade)
        triggered.append(trade)
        logger.info(f"[{pos.symbol}] {action} ({label}, {fraction:.0%}) @ ${price:,.2f} | PnL: ${net:+.2f} | {note}")
        return trade

    def _ratchet_trailing(self, pos: PaperPosition, bar_high: float, bar_low: float) -> None:
        """Incorpora extremos de uma barra FECHADA ao trailing. Nunca afrouxa (só sobe p/ LONG, só desce p/ SHORT)."""
        dist = pos.trailing_atr_mult * pos.entry_atr
        if dist <= 0.0:
            return
        if pos.side == "LONG":
            pos.trail_ref_extreme = max(pos.trail_ref_extreme, bar_high)
            cand = pos.trail_ref_extreme - dist
            if cand > pos.trailing_stop_price:
                pos.trailing_stop_price = cand
            if pos.trailing_stop_price > pos.stop_loss:
                pos.stop_loss = pos.trailing_stop_price
        else:
            pos.trail_ref_extreme = bar_low if pos.trail_ref_extreme <= 0.0 else min(pos.trail_ref_extreme, bar_low)
            cand = pos.trail_ref_extreme + dist
            if pos.trailing_stop_price <= 0.0 or cand < pos.trailing_stop_price:
                pos.trailing_stop_price = cand
            if pos.trailing_stop_price > 0.0 and (pos.stop_loss <= 0.0 or pos.trailing_stop_price < pos.stop_loss):
                pos.stop_loss = pos.trailing_stop_price

    def _close_ladder_on_stop(self, pos: PaperPosition, high: float, low: float, timestamp: int,
                              triggered: List[PaperTrade], open_price: Optional[float] = None) -> None:
        if not pos.tp1_hit:
            reason = "STOP_LOSS"
        elif pos.trailing_active and pos.trailing_stop_price > 0.0 and pos.stop_loss == pos.trailing_stop_price:
            reason = "TRAILING_STOP"
        else:
            reason = "BREAKEVEN_STOP"
        # um stop não pode preencher fora do range da barra (gap): LONG vende <= high, SHORT compra >= low.
        # Gap de abertura (igual ao BacktestEngine): se a barra ABRE além do stop vigente, o fill é o PIOR entre
        # stop e open (LONG: min(sl, open); SHORT: max(sl, open)). `open_price` só é informado para o stop que já
        # existia ANTES da barra (stop pós-TP1 na mesma barra não sofre o gap do open).
        if pos.side == "LONG":
            fill = min(pos.stop_loss, high)
            if open_price is not None and open_price > 0.0:
                fill = min(fill, open_price)
        else:
            fill = max(pos.stop_loss, low)
            if open_price is not None and open_price > 0.0:
                fill = max(fill, open_price)
        t = self.close_position(pos.position_id, fill, timestamp, reason=reason)
        if t:
            triggered.append(t)

    def _process_ladder_bar(self, pos: PaperPosition, high: float, low: float, close: float,
                            timestamp: int, triggered: List[PaperTrade], open_price: Optional[float] = None) -> None:
        """Processa UMA barra para posição com ladder. Ordem conservadora e sem lookahead:

        0. barra anterior (já fechada) é incorporada ao trailing -> stop efetivo desta barra;
        1. stop vigente (SL/BE/trailing, definido ANTES desta barra) tem prioridade sobre qualquer TP;
        2. TP1 (maker) -> SL = breakeven; se a mesma barra toca o BE, assume BE depois do TP1 (conservador);
        3. TP2 (maker) -> ativa trailing (efetivo só a partir da próxima barra);
        4. TP3 (maker) fecha o runner. Extremos desta barra só alimentam o trailing na barra seguinte.
        Ticks repetidos da mesma barra em formação (daemon, 15s) apenas acumulam extremos.
        """
        long = pos.side == "LONG"
        sgn = 1.0 if long else -1.0
        ts = self._ts_ms(timestamp)
        pos.unrealized_pnl = sgn * (close - pos.entry_price) * pos.quantity

        if pos.trailing_active and pos.trail_bar_ts > 0.0 and ts > pos.trail_bar_ts:
            self._ratchet_trailing(pos, pos.trail_bar_high, pos.trail_bar_low)   # barra anterior fechou
            pos.trail_bar_ts = 0.0

        def stop_hit() -> bool:
            return (low <= pos.stop_loss) if long else (high >= pos.stop_loss)

        def reached(level: float) -> bool:
            return level > 0.0 and ((high >= level) if long else (low <= level))

        if stop_hit():
            self._close_ladder_on_stop(pos, high, low, timestamp, triggered, open_price=open_price)
            return

        label = {"GOLDEN_POT": "Golden Pot", "SCALP": "Scalp"}.get(pos.strategy_type, "Direcional")
        if not pos.tp1_hit and reached(pos.tp1_price):
            pos.tp1_hit = True
            be = pos.entry_price * (1.0 + sgn * BREAKEVEN_FEE_BUFFER_PCT)
            pos.stop_loss = max(pos.stop_loss, be) if long else min(pos.stop_loss, be)   # nunca afrouxa
            self._ladder_partial(pos, pos.tp1_price, pos.tp1_fraction, "TP1_PARTIAL", timestamp,
                                 f"TP1 {label}", "SL elevado para Breakeven", triggered)
            pos.unrealized_pnl = sgn * (close - pos.entry_price) * pos.quantity
            if stop_hit():
                self._close_ladder_on_stop(pos, high, low, timestamp, triggered)
                return

        if pos.tp1_hit and not pos.tp2_hit and reached(pos.tp2_price):
            if pos.tp3_price <= 0.0:   # posição legada (pré-E7): TP2 encerra o restante
                t = self.close_position(pos.position_id, pos.tp2_price, timestamp, reason="TAKE_PROFIT_2")
                if t:
                    triggered.append(t)
                return
            pos.tp2_hit = True
            self._ladder_partial(pos, pos.tp2_price, pos.tp2_fraction, "TP2_PARTIAL", timestamp,
                                 f"TP2 {label}", "trailing stop ativado no runner", triggered)
            if pos.trailing_atr_mult > 0.0 and pos.entry_atr > 0.0:
                pos.trailing_active = True
                pos.trail_ref_extreme = pos.tp2_price   # TP2 foi negociado -> extremo já observado
                pos.trail_bar_ts = 0.0
                self._ratchet_trailing(pos, pos.tp2_price, pos.tp2_price)
            pos.unrealized_pnl = sgn * (close - pos.entry_price) * pos.quantity

        if pos.tp2_hit and reached(pos.tp3_price):
            t = self.close_position(pos.position_id, pos.tp3_price, timestamp, reason="TAKE_PROFIT_3")
            if t:
                triggered.append(t)
            return

        if pos.trailing_active:   # acumula esta barra; só vira stop quando a barra seguinte chegar
            if pos.trail_bar_ts <= 0.0 or ts > pos.trail_bar_ts:
                pos.trail_bar_ts, pos.trail_bar_high, pos.trail_bar_low = ts, high, low
            elif ts == pos.trail_bar_ts:
                pos.trail_bar_high = max(pos.trail_bar_high, high)
                pos.trail_bar_low = min(pos.trail_bar_low, low)
        self._save_position(pos)

    # =========================================================================
    # P5: FUNDING real em posições direcionais/scalp/GP
    # =========================================================================

    def _funding_rate_at(self, symbol: str, at_ms: float) -> Tuple[float, str]:
        """Taxa de funding vigente em `at_ms`: último registro (<= at_ms) de `funding_rates` com <= 8h de idade
        (SQLite, SEM rede). Retorna (taxa, fonte). Sem registro -> (default_funding_rate_8h, 'FALLBACK')."""
        try:
            with self._get_connection() as con:
                row = con.execute(
                    "SELECT funding_rate FROM funding_rates WHERE exchange='binance' AND symbol=? "
                    "AND timestamp <= ? AND timestamp >= ? ORDER BY timestamp DESC LIMIT 1",
                    (symbol, int(at_ms + FUNDING_RATE_MATCH_TOLERANCE_MS),
                     int(at_ms - FUNDING_INTERVAL_MS - FUNDING_RATE_MATCH_TOLERANCE_MS))).fetchone()
            if row is not None and row[0] is not None:
                return float(row[0]), "CACHE"
        except Exception:
            pass   # tabela ausente/DB indisponível -> fallback documentado
        return float(self.default_funding_rate_8h), "FALLBACK"

    def has_funding_rate(self, symbol: str, at_ms: float) -> bool:
        return self._funding_rate_at(symbol, at_ms)[1] == "CACHE"

    def _apply_funding(self, pos: PaperPosition, timestamp: int, mark: float, triggered: List[PaperTrade]) -> None:
        """Aplica as liquidações de funding (entry + k*8h) em (cursor, ts_da_barra]. Lança linha FUNDING no
        ledger (pnl<0 = paga) -> saldo/restore consistentes. notional = quantidade restante x mark."""
        now_ms = self._ts_ms(timestamp)
        cursor = pos.funding_cursor_ms if pos.funding_cursor_ms > 0.0 else self._ts_ms(pos.entry_timestamp)
        settle = int(cursor) + FUNDING_INTERVAL_MS
        n = 0
        last_applied = cursor
        sgn = 1.0 if pos.side == "LONG" else -1.0
        while settle <= now_ms and n < FUNDING_MAX_SETTLEMENTS_PER_TICK:
            rate, src = self._funding_rate_at(pos.symbol, settle)
            amount = -sgn * rate * pos.quantity * mark
            pos.funding_collected += amount
            pos.realized_pnl += amount
            self.balance += amount
            self.total_realized_pnl += amount
            trade = PaperTrade(
                trade_id=str(uuid.uuid4())[:8], position_id=pos.position_id, symbol=pos.symbol, action="FUNDING",
                price=mark, quantity=pos.quantity, fee=0.0, pnl=amount, timestamp=int(settle),
                rationale=f"Funding {rate:+.4%} ({src}) {'paga' if amount < 0 else 'recebe'} ${abs(amount):.4f}",
            )
            self.trade_history.append(trade)
            self._save_trade(trade)
            last_applied = float(settle)
            n += 1
            settle += FUNDING_INTERVAL_MS
        pos.funding_cursor_ms = last_applied
        if settle <= now_ms:                    # gap enorme (dados faltantes): descarta o excedente
            pos.funding_cursor_ms = now_ms

    def mark_to_market(self, symbol: str, price: float) -> None:
        """Marca a mercado (UI) com o preço da barra em formação SEM disparar SL/TP/trailing/funding/timeout
        (gatilhos só em barras FECHADAS - paridade com o backtest)."""
        if not price or price <= 0.0:
            return
        with self._lock:
            for pos in self.positions.values():
                if pos.symbol != symbol or pos.status != "OPEN" or pos.side not in ("LONG", "SHORT"):
                    continue
                sgn = 1.0 if pos.side == "LONG" else -1.0
                pos.current_price = price
                pos.unrealized_pnl = sgn * (price - pos.entry_price) * pos.quantity

    def update_market_tick(
        self,
        symbol: str,
        high: float,
        low: float,
        close: float,
        timestamp: int,
        funding_rate_period: float = 0.0,
        open_price: Optional[float] = None,
    ) -> List[PaperTrade]:
        # Todo o ciclo de leitura/mutação roda sob RLock (reentrante com close_position),
        # evitando "dict changed size during iteration" e leituras inconsistentes de
        # equity/margin em threads concorrentes (Flask, supervisor).
        with self._lock:
            return self._update_market_tick_locked(symbol, high, low, close, timestamp, funding_rate_period,
                                                   open_price=open_price)

    def _update_market_tick_locked(
        self,
        symbol: str,
        bar_high: float,
        bar_low: float,
        close: float,
        timestamp: int,
        funding_rate_period: float = 0.0,
        open_price: Optional[float] = None,
    ) -> List[PaperTrade]:
        """
        Atualiza posições com a vela de mercado e verifica SL/TP intra-barra.
        Aplica o funding rate se for momento de liquidação (a cada 8h).
        Ladder comum (TP1/TP2 parciais + BE + trailing + TP3) para SCALP/GOLDEN_POT/DIRECTIONAL: ver _process_ladder_bar.
        Thread-safe com proteção de lock.

        `timestamp` = abertura da barra; a barra DEVE estar fechada (o daemon só alimenta barras fechadas; use
        `mark_to_market` para o preço da barra em formação). `open_price` (opcional) habilita o fill de SL com gap
        de abertura e o preço do timeout; sem ele usa-se o close da barra anterior (== open em cripto 24/7).
        Ordem por barra (igual ao BacktestEngine): funding -> timeout -> stops/TPs.
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

            # A3 (convenção P2): entry_timestamp = ts(t+1) = abertura da 1ª barra PÓS-entrada, que é processada com
            # high/low completos (como o BacktestEngine). Barras que abriram ANTES da entrada (ts < entry) não
            # podem disparar SL/TP com extremos anteriores à posição -> usa apenas o close.
            pre_entry = self._ts_ms(timestamp) < self._ts_ms(pos.entry_timestamp)
            if pre_entry:
                high = low = close
                bar_open = close
            else:
                high, low = bar_high, bar_low
                bar_open = open_price if (open_price is not None and open_price > 0.0) else None
            prev_bar_close = pos.last_bar_close if pos.last_bar_close > 0.0 else pos.current_price

            pos.current_price = close
            pos.last_bar_close = close
            pos.highest_price = max(pos.highest_price, high)
            pos.lowest_price = min(pos.lowest_price, low)

            if pos.side in ("LONG", "SHORT") and not pre_entry:
                # P5: funding real a cada 8h (long paga / short recebe quando funding > 0)
                self._apply_funding(pos, timestamp, close, triggered_trades)
                # P4: timeout por nº de barras do timeframe de decisão -> fecha a mercado (taker + slippage) no
                # open da barra (== close da anterior), antes de qualquer SL/TP desta barra (ordem do backtest).
                prof = STRATEGY_PROFILES.get(pos.strategy_type)
                if prof is not None and prof.max_holding_bars > 0 and \
                        self._ts_ms(timestamp) >= self._ts_ms(pos.entry_timestamp) + prof.max_holding_bars * prof.bar_ms:
                    px = bar_open if bar_open else prev_bar_close
                    t = self.close_position(pos_id, px, timestamp, reason="HORIZON_TIMEOUT")
                    if t:
                        triggered_trades.append(t)
                    continue

            # Liquidação (alavancagem > 1x): só ocorre antes do SL se o SL estiver além da liquidação
            # (o engine garante SL antes da liquidação na abertura; defesa em profundidade).
            liq = pos.liquidation_price
            if liq > 0.0:
                if (pos.side == "LONG" and low <= liq and liq >= pos.stop_loss) or \
                   (pos.side == "SHORT" and high >= liq and liq <= pos.stop_loss):
                    t = self.close_position(pos_id, liq, timestamp, reason="LIQUIDATION")
                    if t:
                        triggered_trades.append(t)
                    continue

            # 1. LADDER comum (SCALP / GOLDEN_POT / DIRECTIONAL): TP1 40% + BE, TP2 30% + trailing, TP3/trailing no runner
            if self._uses_ladder(pos):
                self._process_ladder_bar(pos, high, low, close, timestamp, triggered_trades, open_price=bar_open)
                continue

            # 2. DIRECTIONAL LEGADO (posição aberta antes do E7, sem tp1_price): SL/TP único, taker
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
                    tp1_price, tp1_hit, tp2_price, initial_quantity, leverage,
                    tp3_price, tp2_hit, tp1_fraction, tp2_fraction, trailing_active,
                    trailing_stop_price, trailing_atr_mult, entry_atr
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pos.position_id, pos.symbol, pos.side, pos.entry_price, pos.current_price,
                pos.quantity, pos.notional, pos.stop_loss, pos.take_profit,
                pos.unrealized_pnl, pos.realized_pnl, pos.funding_collected,
                pos.fees_paid, pos.entry_timestamp, pos.strategy_type, pos.status,
                pos.tp1_price, 1 if pos.tp1_hit else 0, pos.tp2_price, pos.initial_quantity, pos.leverage,
                pos.tp3_price, 1 if pos.tp2_hit else 0, pos.tp1_fraction, pos.tp2_fraction,
                1 if pos.trailing_active else 0, pos.trailing_stop_price, pos.trailing_atr_mult, pos.entry_atr,
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
