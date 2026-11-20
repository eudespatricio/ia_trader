"""
performance_tracker.py - Módulo Unificado de Telemetria Financeira e Rastreio de Performance.
Fase 7 do Projeto IA Trader.

Responsabilidades Principais:
1. Fonte Única da Verdade para PnL, Drawdown, Curva de Equity e Métricas de Risco.
2. Cálculo matemático vetorizado de Sharpe, Sortino, Calmar e Deflated Sharpe Ratio (DSR).
3. Decomposição de performance por estratégia (SCALP, GOLDEN_POT, CARRY).
4. Métricas de microestrutura: taxas acumuladas (Maker vs Taker), slippage e turnover.
5. Persistência e leitura sem locks do banco SQLite (market_cache.db) em WAL mode.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from db_conn import connect as _db_connect

import numpy as np
import pandas as pd

logger = logging.getLogger("ia_trader.performance_tracker")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [performance_tracker] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class PerformanceMetrics:
    """Estrutura formal e imutável de telemetria financeira."""
    initial_capital: float = 10000.0
    current_equity: float = 10000.0
    current_balance: float = 10000.0
    net_pnl: float = 0.0
    net_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_fees_paid: float = 0.0
    peak_equity: float = 10000.0
    max_drawdown_dollars: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    
    # Métricas de Amostra e Assertividade
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    payoff_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    rolling_window_size: int = 15
    rolling_trades_count: int = 0
    rolling_win_rate: float = 0.0
    rolling_profit_factor: float = 0.0
    
    # Sequências e Risco Comportamental
    current_streak_type: str = "NONE"   # "WIN", "LOSS", "NONE"
    current_streak_count: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Risco Ajustado ao Retorno
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Breakdown por Estratégia
    strategy_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    timestamp: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceTracker:
    """
    Rastreador institucional de métricas de trading para paper trading e live.
    Lê o SQLite em modo read-only e processa séries temporais de equity e histórico de trades.
    """

    def __init__(
        self,
        db_path: str = "datasets/market_cache.db",
        initial_capital: float = 10000.0,
        read_only: bool = False,
    ):
        self.db_path = db_path
        self.initial_capital = float(initial_capital)
        self.read_only = read_only
        if not self.read_only:
            self._ensure_telemetry_tables()

    def _get_connection(self) -> sqlite3.Connection:
        if self.read_only and os.path.exists(self.db_path):
            conn = _db_connect(f"file:{os.path.abspath(self.db_path)}?mode=ro", uri=True, timeout=15.0)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA query_only = ON;")
            except Exception:
                pass
            return conn
        conn = _db_connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _ensure_telemetry_tables(self):
        """Garante a criação de tabelas dedicadas a telemetria caso não existam."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp INTEGER NOT NULL,
                        equity REAL NOT NULL,
                        balance REAL NOT NULL,
                        net_pnl REAL NOT NULL,
                        net_pnl_pct REAL NOT NULL,
                        max_drawdown_pct REAL NOT NULL,
                        win_rate REAL NOT NULL,
                        profit_factor REAL NOT NULL,
                        total_trades INTEGER NOT NULL,
                        open_positions INTEGER NOT NULL,
                        total_fees REAL NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_snapshots_ts ON performance_snapshots(timestamp)")
                conn.commit()
        except Exception as e:
            logger.warning(f"Aviso ao inicializar tabelas de telemetria: {e}")

    def compute_metrics_from_db(self) -> PerformanceMetrics:
        """
        Calcula as métricas financeiras completas e consolidadas lendo
        as tabelas `paper_portfolio`, `paper_trades` e `paper_positions`.
        """
        now_ms = int(time.time() * 1000)
        
        try:
            with self._get_connection() as conn:
                # 0. Lê marcos de epoch de reset manual (P0 C-07 / 14)
                trade_epoch = 0
                has_sim = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='simulation_settings'").fetchone()
                has_paper = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_settings'").fetchone()
                if has_sim:
                    epoch_row = conn.execute("SELECT value FROM simulation_settings WHERE key='ledger_epoch_trade_rowid'").fetchone()
                    if epoch_row and epoch_row[0]:
                        try:
                            trade_epoch = int(float(epoch_row[0]))
                        except Exception:
                            trade_epoch = 0
                elif has_paper:
                    epoch_row = conn.execute("SELECT value FROM paper_settings WHERE key='ledger_epoch_trade_rowid'").fetchone()
                    if epoch_row and epoch_row[0]:
                        try:
                            trade_epoch = int(float(epoch_row[0]))
                        except Exception:
                            trade_epoch = 0

                # 1. Posições e Portfolio atual
                port_row = conn.execute(
                    "SELECT * FROM paper_portfolio ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                
                # 2. Histórico de Trades respeitando o epoch do reset
                if trade_epoch > 0:
                    trades_rows = conn.execute(
                        "SELECT rowid, * FROM paper_trades WHERE rowid > ? ORDER BY timestamp ASC", (trade_epoch,)
                    ).fetchall()
                else:
                    trades_rows = conn.execute(
                        "SELECT rowid, * FROM paper_trades ORDER BY timestamp ASC"
                    ).fetchall()
                
                # 3. Série temporal de Equity para cálculo de Sharpe e Drawdown
                first_trade_ts = trades_rows[0]["timestamp"] if trades_rows else 0
                if trade_epoch > 0:
                    if first_trade_ts > 0:
                        equity_rows = conn.execute(
                            "SELECT timestamp, equity FROM paper_portfolio WHERE timestamp >= ? ORDER BY timestamp ASC",
                            (first_trade_ts,)
                        ).fetchall()
                    else:
                        equity_rows = [port_row] if port_row else []
                else:
                    equity_rows = conn.execute(
                        "SELECT timestamp, equity FROM paper_portfolio ORDER BY timestamp ASC"
                    ).fetchall()

                # P0 (10): Mapeamento estruturado de estratégias diretamente de paper_positions
                pos_strategies = {}
                has_pos = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='paper_positions'").fetchone()
                if has_pos:
                    p_rows = conn.execute("SELECT position_id, strategy_type FROM paper_positions").fetchall()
                    pos_strategies = {str(r["position_id"]): str(r["strategy_type"]) for r in p_rows if r["position_id"]}

            # Caso ainda não haja dados no banco, retorna métricas neutras iniciais
            if not port_row and not trades_rows:
                return PerformanceMetrics(
                    initial_capital=self.initial_capital,
                    current_equity=self.initial_capital,
                    current_balance=self.initial_capital,
                    timestamp=now_ms
                )

            current_equity = float(port_row["equity"]) if port_row else self.initial_capital
            current_balance = float(port_row["balance"]) if port_row else self.initial_capital
            unrealized_pnl = float(port_row["unrealized_pnl"]) if port_row else 0.0
            total_fees = float(port_row["total_fees_paid"]) if port_row else 0.0
            realized_pnl = float(port_row["realized_pnl_cum"]) if port_row else 0.0

            net_pnl = current_equity - self.initial_capital
            net_pnl_pct = (net_pnl / self.initial_capital) * 100.0

            # ------------------------------------------------------------------
            # Análise de Trades Fechados Consolidados por Posição (position_id)
            # Evita contar BREAKEVEN_STOP (-$0.05) de posição vencedora (TP1+$1.12)
            # ou liquidações de FUNDING como derrotas independentes em consecutive_losses.
            # ------------------------------------------------------------------
            trades = [dict(r) for r in trades_rows]
            open_pos_ids: set[str] = set()
            try:
                with self._get_connection() as conn:
                    has_pos = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='paper_positions'"
                    ).fetchone()
                    if has_pos:
                        open_pos_ids = {
                            str(r["position_id"])
                            for r in conn.execute("SELECT position_id FROM paper_positions WHERE status='OPEN'").fetchall()
                            if r["position_id"]
                        }
            except Exception:
                open_pos_ids = set()

            raw_closing = [
                t for t in trades
                if not str(t.get("action", "")).startswith("OPEN_")
                and (
                    str(t.get("action", "")).startswith("CLOSE_")
                    or t.get("action") in (
                        "STOP_LOSS", "TAKE_PROFIT", "TP1_PARTIAL", "TP2_PARTIAL",
                        "TAKE_PROFIT_2", "TAKE_PROFIT_3", "BREAKEVEN_STOP",
                        "TRAILING_STOP", "HORIZON_TIMEOUT", "LIQUIDATION", "TP2_FULL"
                    )
                    or float(t.get("pnl", 0.0)) != 0.0
                )
            ]

            grouped_by_pos: Dict[str, List[Dict[str, Any]]] = {}
            for idx, t in enumerate(raw_closing):
                pid = str(t.get("position_id") or f"_leg_{idx}")
                if pid in open_pos_ids:
                    continue
                grouped_by_pos.setdefault(pid, []).append(t)

            closing_trades: List[Dict[str, Any]] = []
            for pid, legs in grouped_by_pos.items():
                # Se só há linhas de FUNDING sem nenhum evento de saída/parcial, ignora
                if all(str(l.get("action", "")) == "FUNDING" for l in legs):
                    continue
                last_leg = legs[-1]
                closing_trades.append({
                    **last_leg,
                    "position_id": pid,
                    "pnl": sum(float(l.get("pnl", 0.0)) for l in legs),
                    "fee": sum(float(l.get("fee", 0.0)) for l in legs),
                    "rationale": " | ".join(str(l.get("rationale", "")) for l in legs if l.get("rationale")),
                })

            total_trades = len(closing_trades)
            pnls = np.array([float(t.get("pnl", 0.0)) for t in closing_trades], dtype=np.float64) if closing_trades else np.array([])
            
            winning_trades = int(np.sum(pnls > 0))
            losing_trades = int(np.sum(pnls < 0))
            win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

            gross_profit = float(np.sum(pnls[pnls > 0])) if winning_trades > 0 else 0.0
            gross_loss = float(abs(np.sum(pnls[pnls < 0]))) if losing_trades > 0 else 0.0

            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
            avg_win = (gross_profit / winning_trades) if winning_trades > 0 else 0.0
            avg_loss = (gross_loss / losing_trades) if losing_trades > 0 else 0.0
            payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

            r_pnls = pnls[-15:]
            rolling_window_size = 15
            rolling_trades_count = int(len(r_pnls))
            r_wins = int(np.sum(r_pnls > 0)) if rolling_trades_count > 0 else 0
            r_losses = int(np.sum(r_pnls < 0)) if rolling_trades_count > 0 else 0
            rolling_win_rate = (r_wins / rolling_trades_count * 100.0) if rolling_trades_count > 0 else 0.0
            r_gp = float(np.sum(r_pnls[r_pnls > 0])) if r_wins > 0 else 0.0
            r_gl = float(abs(np.sum(r_pnls[r_pnls < 0]))) if r_losses > 0 else 0.0
            rolling_profit_factor = (r_gp / r_gl) if r_gl > 0 else (99.0 if r_gp > 0 else 0.0)

            # Sequências (Streaks)
            current_streak_type = "NONE"
            current_streak_count = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0

            cur_w = 0
            cur_l = 0
            for p in pnls:
                if p > 0:
                    cur_w += 1
                    cur_l = 0
                    if cur_w > max_consecutive_wins:
                        max_consecutive_wins = cur_w
                    current_streak_type = "WIN"
                    current_streak_count = cur_w
                elif p < 0:
                    cur_l += 1
                    cur_w = 0
                    if cur_l > max_consecutive_losses:
                        max_consecutive_losses = cur_l
                    current_streak_type = "LOSS"
                    current_streak_count = cur_l

            # ------------------------------------------------------------------
            # Curva de Equity e Drawdown
            # ------------------------------------------------------------------
            if equity_rows:
                eq_vals = np.array([float(r["equity"]) for r in equity_rows], dtype=np.float64)
            else:
                eq_vals = np.array([self.initial_capital, current_equity], dtype=np.float64)

            cummax_equity = np.maximum.accumulate(eq_vals)
            dd_dollars_series = cummax_equity - eq_vals
            dd_pct_series = (dd_dollars_series / (cummax_equity + 1e-12)) * 100.0

            peak_equity = float(np.max(cummax_equity)) if len(cummax_equity) > 0 else self.initial_capital
            max_drawdown_dollars = float(np.max(dd_dollars_series)) if len(dd_dollars_series) > 0 else 0.0
            max_drawdown_pct = float(np.max(dd_pct_series)) if len(dd_pct_series) > 0 else 0.0
            current_drawdown_pct = float(dd_pct_series[-1]) if len(dd_pct_series) > 0 else 0.0

            # ------------------------------------------------------------------
            # Índices de Sharpe, Sortino e Calmar
            # ------------------------------------------------------------------
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            calmar_ratio = 0.0

            if len(eq_vals) >= 5:
                returns = np.diff(eq_vals) / (eq_vals[:-1] + 1e-12)
                mean_r = np.mean(returns)
                std_r = np.std(returns) + 1e-12

                # Anualização dinâmica baseada na cadência real observada nos dados
                days_elapsed = 1.0
                if equity_rows and len(equity_rows) >= 2:
                    try:
                        first_row = dict(equity_rows[0])
                        last_row = dict(equity_rows[-1])
                        t0 = float(first_row.get("timestamp", 0.0))
                        t1 = float(last_row.get("timestamp", 0.0))
                        if t0 > 1e11:
                            t0 /= 1000.0
                            t1 /= 1000.0
                        if t1 > t0:
                            days_elapsed = max((t1 - t0) / 86400.0, 0.1)
                            snapshots_per_day = len(returns) / days_elapsed
                            periods_per_year = snapshots_per_day * 365.25
                        else:
                            periods_per_year = 365.25 * 24 * 4  # 15 min default
                    except Exception:
                        periods_per_year = 365.25 * 24 * 4
                else:
                    periods_per_year = 365.25 * 24 * 4

                sharpe_ratio = float((mean_r / std_r) * np.sqrt(periods_per_year))

                # Sortino (apenas desvio padrão dos retornos negativos)
                neg_returns = returns[returns < 0]
                if len(neg_returns) > 0:
                    downside_std = np.std(neg_returns) + 1e-12
                    sortino_ratio = float((mean_r / downside_std) * np.sqrt(periods_per_year))
                else:
                    sortino_ratio = sharpe_ratio

                # Calmar (Retorno Anualizado / Max Drawdown %)
                if max_drawdown_pct > 0.01:
                    tot_ret = (eq_vals[-1] / (eq_vals[0] + 1e-12)) - 1.0
                    if days_elapsed < 30.0:
                        annualized_return_pct = (tot_ret * (365.25 / max(days_elapsed, 0.5))) * 100.0
                    else:
                        annualized_return_pct = (((1.0 + max(tot_ret, -0.99)) ** (365.25 / days_elapsed)) - 1.0) * 100.0
                    calmar_ratio = float(annualized_return_pct / max_drawdown_pct)

            # ------------------------------------------------------------------
            # Breakdown por Estratégia
            # ------------------------------------------------------------------
            strategy_metrics = self._compute_strategy_breakdown(closing_trades, pos_strategies=pos_strategies)

            return PerformanceMetrics(
                initial_capital=self.initial_capital,
                current_equity=current_equity,
                current_balance=current_balance,
                net_pnl=net_pnl,
                net_pnl_pct=net_pnl_pct,
                realized_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                total_fees_paid=total_fees,
                peak_equity=peak_equity,
                max_drawdown_dollars=max_drawdown_dollars,
                max_drawdown_pct=max_drawdown_pct,
                current_drawdown_pct=current_drawdown_pct,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                payoff_ratio=payoff_ratio,
                avg_win=avg_win,
                avg_loss=avg_loss,
                rolling_window_size=rolling_window_size,
                rolling_trades_count=rolling_trades_count,
                rolling_win_rate=rolling_win_rate,
                rolling_profit_factor=rolling_profit_factor,
                current_streak_type=current_streak_type,
                current_streak_count=current_streak_count,
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                strategy_metrics=strategy_metrics,
                timestamp=now_ms,
            )

        except Exception as e:
            logger.error(f"Erro ao computar métricas de performance do banco: {e}")
            return PerformanceMetrics(initial_capital=self.initial_capital, timestamp=now_ms)

    def _compute_strategy_breakdown(
        self,
        closing_trades: List[Dict[str, Any]],
        pos_strategies: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Decompõe métricas agrupando por coluna estruturada strategy_type de paper_positions."""
        breakdown = {
            "SCALP": {"trades": 0, "wins": 0, "pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0},
            "GOLDEN_POT": {"trades": 0, "wins": 0, "pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0},
            "DIRECTIONAL": {"trades": 0, "wins": 0, "pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0},
            "CARRY": {"trades": 0, "wins": 0, "pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0},
        }

        pos_strategies = pos_strategies or {}
        gp_gl: Dict[str, List[float]] = {k: [0.0, 0.0] for k in breakdown}
        for t in closing_trades:
            pid = str(t.get("position_id", ""))
            stype = pos_strategies.get(pid)
            if not stype:
                rationale = str(t.get("rationale", "")).upper()
                action = str(t.get("action", "")).upper()
                if "GOLDEN_POT" in rationale or "GOLDEN_POT" in action:
                    stype = "GOLDEN_POT"
                elif "CARRY" in rationale or "CARRY" in action:
                    stype = "CARRY"
                elif "DIRECTIONAL" in rationale or "DIRECTIONAL" in action:
                    stype = "DIRECTIONAL"
                else:
                    stype = "SCALP"

            if stype not in breakdown:
                breakdown[stype] = {"trades": 0, "wins": 0, "pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0}
                gp_gl[stype] = [0.0, 0.0]

            pnl = float(t.get("pnl", 0.0))
            entry = breakdown[stype]
            entry["trades"] += 1
            entry["pnl"] += pnl
            if pnl > 0:
                entry["wins"] += 1
                gp_gl[stype][0] += pnl
            elif pnl < 0:
                gp_gl[stype][1] += abs(pnl)

        for stype, data in breakdown.items():
            if data["trades"] > 0:
                data["win_rate"] = (data["wins"] / data["trades"]) * 100.0
                gp, gl = gp_gl.get(stype, [0.0, 0.0])
                data["profit_factor"] = round(gp / gl, 2) if gl > 0 else (99.0 if gp > 0 else 0.0)

        return breakdown

    def record_snapshot(self, metrics: Optional[PerformanceMetrics] = None, open_positions: Optional[int] = None) -> None:
        """Persiste um snapshot periódico de telemetria na tabela `performance_snapshots`."""
        if metrics is None:
            metrics = self.compute_metrics_from_db()

        try:
            with self._get_connection() as conn:
                if open_positions is None:
                    try:
                        row = conn.execute(
                            "SELECT count(*) FROM paper_positions WHERE status = 'OPEN'"
                        ).fetchone()
                        open_positions = int(row[0]) if row else 0
                    except Exception:
                        open_positions = 0

                conn.execute("""
                    INSERT INTO performance_snapshots (
                        timestamp, equity, balance, net_pnl, net_pnl_pct,
                        max_drawdown_pct, win_rate, profit_factor,
                        total_trades, open_positions, total_fees
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp,
                    metrics.current_equity,
                    metrics.current_balance,
                    metrics.net_pnl,
                    metrics.net_pnl_pct,
                    metrics.max_drawdown_pct,
                    metrics.win_rate,
                    metrics.profit_factor,
                    metrics.total_trades,
                    int(open_positions),
                    metrics.total_fees_paid
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Erro ao salvar snapshot de performance: {e}")

    # ------------------------------------------------------------------
    # Evolução Histórica desde Semana 0 & Impacto de Patches Semanais
    # ------------------------------------------------------------------
    _TZ_BRT = datetime.timezone(datetime.timedelta(hours=-3))
    _WEEK_1_START = datetime.date(2026, 10, 19)

    PATCH_MILESTONES: List[Dict[str, Any]] = [
        {
            "id": "S0",
            "short": "S0",
            "label": "S0 — Piloto Limpo",
            "version": "v1.0.0",
            "start_date": "2026-10-15",
            "end_date": "2026-10-18",
            "marker_date": "2026-10-16",
            "summary": "Baseline E6–E9 após reset pré-piloto; lote fixo $100.",
        },
        {
            "id": "S1",
            "short": "S1",
            "label": "S1 — Funil, OMS & Risco 1%",
            "version": "v1.0.1–v1.0.3",
            "start_date": "2026-10-19",
            "end_date": "2026-10-26",
            "marker_date": "2026-10-19",
            "summary": "FlowScore contínuo, sizing 1% ATR, OMS 12 estados, quórum min 2 e Daily DD 5%.",
        },
        {
            "id": "S2_SETUP",
            "short": "S2-Setup",
            "label": "S2-Setup — Scalp 15m & Reset $10k",
            "version": "S2-Setup",
            "start_date": "2026-10-27",
            "end_date": "2026-11-02",
            "marker_date": "2026-10-27",
            "summary": "Scalp migrado p/ 15m, ensemble 50k no TechnicalAgent (0.55) e reset epoch $10k.",
        },
        {
            "id": "S2_PATCH",
            "short": "Patch S2",
            "label": "Patch S2 — Calibração Ridge",
            "version": "v1.0.4",
            "start_date": "2026-11-03",
            "end_date": "2026-11-04",
            "marker_date": "2026-11-03",
            "summary": "Pesos L2/Ridge por regime, dampening em SIDEWAYS_RANGE (conf <= 0.55) e retreino 50k.",
        },
        {
            "id": "S3_PATCH",
            "short": "Patch S3",
            "label": "Patch S3 — BE Intra-Barra",
            "version": "v1.0.5",
            "start_date": "2026-11-05",
            "end_date": None,
            "marker_date": "2026-11-05",
            "summary": "Proteção same_bar_safe do Breakeven contra pavios iniciais e threshold stale 15m/5m.",
        },
    ]

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0, decimals: int = 4) -> float:
        """Converte valor para float finito arredondado, evitando NaN/Inf no JSON."""
        try:
            f = float(val)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(f):
            return default
        return round(f, decimals)

    @classmethod
    def _ts_to_local_dt(cls, ts_ms: int) -> datetime.datetime:
        """Converte timestamp em ms para datetime em UTC-3."""
        return datetime.datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=cls._TZ_BRT)

    @classmethod
    def _resolve_patch_for_date(cls, d: datetime.date) -> Dict[str, Any]:
        """Retorna o marco de patch ativo para uma data calendário local."""
        iso = d.isoformat()
        for p in cls.PATCH_MILESTONES:
            if iso >= p["start_date"] and (p["end_date"] is None or iso <= p["end_date"]):
                return p
        return cls.PATCH_MILESTONES[0]

    @classmethod
    def _resolve_week_bucket(cls, d: datetime.date) -> Tuple[str, str, str]:
        """Retorna (sort_key, label, date_range) ancorado na Semana 0 (16–18/Out) e Semana 1 (19–25/Out)."""
        if d < cls._WEEK_1_START:
            return ("W00", "Sem 0", "16–18/Out")
        w_num = (d - cls._WEEK_1_START).days // 7 + 1
        w0 = cls._WEEK_1_START + datetime.timedelta(days=(w_num - 1) * 7)
        w1 = w0 + datetime.timedelta(days=6)
        return (f"W{w_num:02d}", f"Sem {w_num}", f"{w0:%d/%m}–{w1:%d/%m}")

    @staticmethod
    def _calc_max_dd_pct(eq_vals: List[float]) -> float:
        """Calcula Max Drawdown (%) sobre uma série cronológica de equity."""
        if not eq_vals:
            return 0.0
        arr = np.asarray(eq_vals, dtype=np.float64)
        cummax = np.maximum.accumulate(arr)
        dd = (cummax - arr) / (cummax + 1e-12) * 100.0
        val = float(np.max(dd))
        return round(val, 2) if math.isfinite(val) else 0.0

    @classmethod
    def _stitch_portfolio_series(
        cls, port_rows: List[Dict[str, Any]], initial_capital: float
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Constrói série contínua de equity ajustada através de resets de banca,
        preservando também raw_equity e drawdown subaquático contínuo.
        """
        if not port_rows:
            return [], []

        offset = 0.0
        resets: List[Dict[str, Any]] = []
        series: List[Dict[str, Any]] = []
        peak = -float("inf")

        for i, row in enumerate(port_rows):
            ts = int(row["timestamp"])
            raw_eq = float(row["equity"])
            bal = float(row.get("balance", raw_eq))
            realized_cum = float(row.get("realized_pnl_cum", 0.0))

            if i > 0:
                prev = port_rows[i - 1]
                prev_realized = float(prev.get("realized_pnl_cum", 0.0))
                if (
                    abs(prev_realized) > 1.0
                    and abs(realized_cum) < 1e-6
                    and abs(bal - initial_capital) < 5.0
                ):
                    prev_eq = float(prev["equity"])
                    offset += prev_eq - raw_eq
                    dt_local = cls._ts_to_local_dt(ts)
                    resets.append({
                        "ts": ts,
                        "dt": dt_local.strftime("%Y-%m-%d %H:%M"),
                        "date": dt_local.date().isoformat(),
                        "prev_equity": round(prev_eq, 2),
                        "new_equity": round(raw_eq, 2),
                        "offset_applied": round(offset, 2),
                    })

            adj_eq = raw_eq + offset
            if adj_eq > peak:
                peak = adj_eq
            dd_pct = ((peak - adj_eq) / (peak + 1e-12) * 100.0) if peak > 0 else 0.0
            dt_local = cls._ts_to_local_dt(ts)
            series.append({
                "ts": ts,
                "dt": dt_local.strftime("%Y-%m-%d %H:%M"),
                "date": dt_local.date().isoformat(),
                "adj_equity": cls._safe_float(adj_eq, initial_capital, 2),
                "raw_equity": cls._safe_float(raw_eq, initial_capital, 2),
                "cum_pnl": cls._safe_float(adj_eq - initial_capital, 0.0, 2),
                "dd_pct": cls._safe_float(dd_pct, 0.0, 2),
            })

        return series, resets

    @staticmethod
    def _downsample_series(points: List[Dict[str, Any]], max_points: int = 500) -> List[Dict[str, Any]]:
        """Reduz pontos preservando picos, vales, primeiro e último registro."""
        n = len(points)
        if n <= max_points:
            return points
        buckets = max(1, max_points // 2)
        step = n / buckets
        selected_idx = {0, n - 1}
        for b in range(buckets):
            start = int(b * step)
            end = min(n, int((b + 1) * step))
            if end <= start:
                continue
            seg = points[start:end]
            min_i = start + min(range(len(seg)), key=lambda j: seg[j]["adj_equity"])
            max_i = start + max(range(len(seg)), key=lambda j: seg[j]["adj_equity"])
            selected_idx.add(min_i)
            selected_idx.add(max_i)
        return [points[i] for i in sorted(selected_idx)]

    def _compute_bucket_metrics(
        self,
        trades: List[Dict[str, Any]],
        eq_vals: List[float],
        alm_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calcula indicadores consolidados (Acurácia, PnL, Max DD, PF, Payoff, Sharpe, Comitê) para um período."""
        n_trades = len(trades)
        pnls = [float(t["pnl"]) for t in trades]
        fees = sum(float(t.get("fee", 0.0)) for t in trades)
        wins = sum(1 for p in pnls if p > 0)
        losses = sum(1 for p in pnls if p < 0)
        acc = (wins / n_trades * 100.0) if n_trades > 0 else 0.0
        net_pnl = sum(pnls)
        gp = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p < 0))
        pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
        avg_win = (gp / wins) if wins > 0 else 0.0
        avg_loss = (gl / losses) if losses > 0 else 0.0
        payoff = (avg_win / avg_loss) if avg_loss > 0 else 0.0
        expectancy = (net_pnl / n_trades) if n_trades > 0 else 0.0

        # Max DD prioritariamente na série de equity do portfólio; fallback na curva de PnL por trade
        if len(eq_vals) >= 2:
            max_dd = self._calc_max_dd_pct(eq_vals)
            base_cap = eq_vals[0] if eq_vals[0] > 0 else self.initial_capital
        else:
            curve = [self.initial_capital]
            for p in pnls:
                curve.append(curve[-1] + p)
            max_dd = self._calc_max_dd_pct(curve)
            base_cap = self.initial_capital

        pnl_pct = (net_pnl / base_cap * 100.0) if base_cap > 0 else 0.0

        sharpe = 0.0
        if len(eq_vals) >= 5:
            arr = np.asarray(eq_vals, dtype=np.float64)
            rets = np.diff(arr) / (arr[:-1] + 1e-12)
            std_r = float(np.std(rets))
            if std_r > 1e-12:
                sharpe = float((np.mean(rets) / std_r) * np.sqrt(365.25 * 24))

        # Breakdown por estratégia reaproveitando método existente
        strat_raw = self._compute_strategy_breakdown(trades)
        strategies = {
            k: {
                "trades": int(v["trades"]),
                "wins": int(v["wins"]),
                "accuracy_pct": self._safe_float(v["win_rate"], 0.0, 1),
                "pnl_usd": self._safe_float(v["pnl"], 0.0, 2),
                "profit_factor": self._safe_float(v["profit_factor"], 0.0, 2),
            }
            for k, v in strat_raw.items()
            if v["trades"] > 0
        }

        # Métricas de Acurácia do Comitê & Agentes (agent_learning_memory)
        alm_n = len(alm_items)
        alm_wins = sum(1 for a in alm_items if str(a.get("outcome", "")).upper() == "WIN")
        alm_acc = (alm_wins / alm_n * 100.0) if alm_n > 0 else None
        avg_conf = (
            sum(float(a.get("consensus_confidence", 0.0)) for a in alm_items) / alm_n * 100.0
            if alm_n > 0 else None
        )
        cal_gap = (avg_conf - alm_acc) if (avg_conf is not None and alm_acc is not None) else None

        agent_stats: Dict[str, Dict[str, int]] = {}
        for item in alm_items:
            is_win = str(item.get("outcome", "")).upper() == "WIN"
            cons_dir = str(item.get("consensus_direction", "")).upper()
            raw_sigs = item.get("agent_signals_json")
            if not raw_sigs:
                continue
            try:
                sigs = json.loads(raw_sigs) if isinstance(raw_sigs, str) else raw_sigs
            except Exception:
                continue
            for s in sigs:
                aname = str(s.get("agent_name", ""))
                adir = str(s.get("direction", "HOLD")).upper()
                if not aname or adir in ("HOLD", "ABSTAIN", "NEUTRAL", ""):
                    continue
                bucket = agent_stats.setdefault(aname, {"n": 0, "wins": 0})
                bucket["n"] += 1
                # Agente acerta se votou a favor do consenso vencedor ou contra o consenso perdedor
                if (adir == cons_dir and is_win) or (adir != cons_dir and not is_win):
                    bucket["wins"] += 1

        agents_out = {
            name: {
                "n": st["n"],
                "wins": st["wins"],
                "accuracy_pct": round(st["wins"] / st["n"] * 100.0, 1) if st["n"] > 0 else 0.0,
            }
            for name, st in agent_stats.items()
        }

        return {
            "trades": n_trades,
            "wins": wins,
            "losses": losses,
            "accuracy_pct": self._safe_float(acc, 0.0, 1),
            "pnl_usd": self._safe_float(net_pnl, 0.0, 2),
            "pnl_pct": self._safe_float(pnl_pct, 0.0, 2),
            "max_drawdown_pct": self._safe_float(max_dd, 0.0, 2),
            "profit_factor": self._safe_float(pf, 0.0, 2),
            "payoff_ratio": self._safe_float(payoff, 0.0, 2),
            "expectancy_usd": self._safe_float(expectancy, 0.0, 2),
            "avg_win_usd": self._safe_float(avg_win, 0.0, 2),
            "avg_loss_usd": self._safe_float(avg_loss, 0.0, 2),
            "fees_usd": self._safe_float(fees, 0.0, 2),
            "sharpe_ratio": self._safe_float(sharpe, 0.0, 2),
            "committee_n": alm_n,
            "committee_accuracy_pct": self._safe_float(alm_acc, 0.0, 1) if alm_acc is not None else None,
            "avg_confidence_pct": self._safe_float(avg_conf, 0.0, 1) if avg_conf is not None else None,
            "calibration_gap_pct": self._safe_float(cal_gap, 0.0, 1) if cal_gap is not None else None,
            "agents": agents_out,
            "strategies": strategies,
        }

    def _group_evolution_buckets(
        self,
        closed_trades: List[Dict[str, Any]],
        portfolio_series: List[Dict[str, Any]],
        alm_rows: List[Dict[str, Any]],
        key_fn: Callable[[datetime.date], Tuple[str, str, str, str]],
    ) -> List[Dict[str, Any]]:
        """Agrupa trades, equity e memória de aprendizado em coortes ordenadas (semanas, patches ou dias)."""
        buckets: Dict[str, Dict[str, Any]] = {}

        def _ensure(d: datetime.date) -> Dict[str, Any]:
            sort_k, label, date_range, meta = key_fn(d)
            if sort_k not in buckets:
                buckets[sort_k] = {
                    "key": sort_k,
                    "label": label,
                    "date_range": date_range,
                    "meta": meta,
                    "trades": [],
                    "eq_vals": [],
                    "alm": [],
                }
            return buckets[sort_k]

        for pt in portfolio_series:
            d = datetime.date.fromisoformat(pt["date"])
            _ensure(d)["eq_vals"].append(pt["adj_equity"])
        for t in closed_trades:
            d = datetime.date.fromisoformat(t["date"])
            _ensure(d)["trades"].append(t)
        for a in alm_rows:
            d = self._ts_to_local_dt(int(a["timestamp"])).date()
            _ensure(d)["alm"].append(a)

        out: List[Dict[str, Any]] = []
        cum_pnl = 0.0
        for sort_k in sorted(buckets.keys()):
            b = buckets[sort_k]
            kpis = self._compute_bucket_metrics(b["trades"], b["eq_vals"], b["alm"])
            cum_pnl += kpis["pnl_usd"]
            out.append({
                "key": b["key"],
                "label": b["label"],
                "date_range": b["date_range"],
                "meta": b["meta"],
                "cum_pnl_usd": self._safe_float(cum_pnl, 0.0, 2),
                **kpis,
            })
        return out

    def compute_evolution_from_db(self) -> Dict[str, Any]:
        """
        Computa evolução histórica completa do IA Trader desde a Semana 0 (sem corte de epoch),
        agrupada por Semana, por Patch e por Dia, incluindo acurácia, PnL contínuo e Max Drawdown.
        """
        empty_payload = {
            "has_data": False,
            "initial_capital": self.initial_capital,
            "summary": {
                "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
                "accuracy_pct": 0.0, "cum_pnl_usd": 0.0, "cum_pnl_pct": 0.0,
                "current_epoch_pnl_usd": 0.0, "current_epoch_equity": self.initial_capital,
                "max_drawdown_pct": 0.0, "profit_factor": 0.0, "expectancy_usd": 0.0,
                "payoff_ratio": 0.0, "total_fees_usd": 0.0,
                "latest_week_label": "--", "latest_week_accuracy_pct": 0.0,
                "baseline_accuracy_pct": 0.0, "accuracy_delta_pp": 0.0,
                "latest_week_pf": 0.0, "baseline_pf": 0.0,
                "committee_accuracy_pct": None, "calibration_gap_pct": None,
            },
            "patches": self.PATCH_MILESTONES,
            "resets": [],
            "portfolio_series": [],
            "trade_series": [],
            "by_week": [],
            "by_patch": [],
            "by_day": [],
            "agent_calibration": [],
        }

        if not os.path.exists(self.db_path):
            return empty_payload

        try:
            conn = _db_connect(f"file:{os.path.abspath(self.db_path)}?mode=ro", uri=True, timeout=10.0)
            conn.row_factory = sqlite3.Row
        except Exception as e:
            logger.warning(f"Erro ao abrir banco em modo leitura para evolução: {e}")
            return empty_payload

        try:
            tables = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }

            port_rows: List[Dict[str, Any]] = []
            if "paper_portfolio" in tables:
                port_rows = [
                    dict(r) for r in conn.execute(
                        "SELECT timestamp, equity, balance, realized_pnl_cum FROM paper_portfolio ORDER BY timestamp ASC"
                    ).fetchall()
                ]

            pos_strategies: Dict[str, str] = {}
            pos_sides: Dict[str, str] = {}
            open_pos_ids: set[str] = set()
            if "paper_positions" in tables:
                for r in conn.execute("SELECT position_id, strategy_type, side, status FROM paper_positions").fetchall():
                    pid = str(r["position_id"])
                    if r["strategy_type"]:
                        pos_strategies[pid] = str(r["strategy_type"])
                    if r["side"]:
                        pos_sides[pid] = str(r["side"])
                    if str(r["status"]).upper() == "OPEN":
                        open_pos_ids.add(pid)

            trades_rows: List[Dict[str, Any]] = []
            if "paper_trades" in tables:
                trades_rows = [
                    dict(r) for r in conn.execute(
                        "SELECT position_id, symbol, action, pnl, fee, timestamp, rationale FROM paper_trades ORDER BY timestamp ASC"
                    ).fetchall()
                ]

            alm_rows: List[Dict[str, Any]] = []
            if "agent_learning_memory" in tables:
                alm_rows = [
                    dict(r) for r in conn.execute(
                        "SELECT timestamp, symbol, regime, consensus_direction, consensus_confidence, agent_signals_json, outcome, realized_pnl "
                        "FROM agent_learning_memory WHERE outcome IN ('WIN', 'LOSS') ORDER BY timestamp ASC"
                    ).fetchall()
                ]

            cal_rows: List[Dict[str, Any]] = []
            if "agent_regime_calibration" in tables:
                cal_rows = [
                    dict(r) for r in conn.execute(
                        "SELECT regime, agent_name, ece, brier_score, updated_at FROM agent_regime_calibration ORDER BY regime, agent_name"
                    ).fetchall()
                ]
        finally:
            conn.close()

        # Consolidar trades fechados por position_id (descontando taxa de abertura para reconciliar com equity)
        closing_actions = {
            "STOP_LOSS", "TAKE_PROFIT", "TP1_PARTIAL", "TP2_PARTIAL",
            "TAKE_PROFIT_2", "TAKE_PROFIT_3", "BREAKEVEN_STOP",
            "TRAILING_STOP", "HORIZON_TIMEOUT", "LIQUIDATION", "TP2_FULL",
        }
        open_fees_by_pos: Dict[str, float] = {}
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for idx, t in enumerate(trades_rows):
            act = str(t.get("action", ""))
            pid = str(t.get("position_id") or f"_leg_{idx}")
            if act.startswith("OPEN_"):
                open_fees_by_pos[pid] = open_fees_by_pos.get(pid, 0.0) + float(t.get("fee", 0.0) or 0.0)
                continue
            if not (act.startswith("CLOSE_") or act in closing_actions or float(t.get("pnl", 0.0) or 0.0) != 0.0):
                continue
            if pid in open_pos_ids:
                continue
            grouped.setdefault(pid, []).append(t)

        closed_trades: List[Dict[str, Any]] = []
        for pid, legs in grouped.items():
            if all(str(l.get("action", "")) == "FUNDING" for l in legs):
                continue
            last = legs[-1]
            close_ts = int(last["timestamp"])
            dt_local = self._ts_to_local_dt(close_ts)
            open_fee = open_fees_by_pos.get(pid, 0.0)
            close_fee = sum(float(l.get("fee", 0.0) or 0.0) for l in legs)
            net_pnl = sum(float(l.get("pnl", 0.0) or 0.0) for l in legs) - open_fee
            closed_trades.append({
                "position_id": pid,
                "symbol": str(last.get("symbol", "")),
                "strategy_type": pos_strategies.get(pid, ""),
                "side": pos_sides.get(pid, ""),
                "action": str(last.get("action", "")),
                "pnl": net_pnl,
                "fee": open_fee + close_fee,
                "rationale": " | ".join(str(l.get("rationale", "")) for l in legs if l.get("rationale")),
                "timestamp": close_ts,
                "dt": dt_local.strftime("%Y-%m-%d %H:%M"),
                "date": dt_local.date().isoformat(),
            })
        closed_trades.sort(key=lambda x: x["timestamp"])

        if not port_rows and not closed_trades and not alm_rows:
            return empty_payload

        portfolio_series, resets = self._stitch_portfolio_series(port_rows, self.initial_capital)

        # Série trade-a-trade com PnL acumulado e acurácia móvel (janela 15 trades)
        trade_series: List[Dict[str, Any]] = []
        cum_pnl = 0.0
        wins_list: List[int] = []
        for idx, t in enumerate(closed_trades):
            p = float(t["pnl"])
            cum_pnl += p
            wins_list.append(1 if p > 0 else 0)
            w_win = wins_list[max(0, len(wins_list) - 15):]
            roll_acc = (sum(w_win) / len(w_win)) * 100.0
            cum_acc = (sum(wins_list) / len(wins_list)) * 100.0
            trade_series.append({
                "idx": idx + 1,
                "ts": t["timestamp"],
                "dt": t["dt"],
                "date": t["date"],
                "symbol": t["symbol"],
                "strategy": t["strategy_type"] or "SCALP",
                "pnl": self._safe_float(p, 0.0, 2),
                "cum_pnl": self._safe_float(cum_pnl, 0.0, 2),
                "rolling_acc_pct": self._safe_float(roll_acc, 0.0, 1),
                "cum_acc_pct": self._safe_float(cum_acc, 0.0, 1),
            })

        def _week_key(d: datetime.date) -> Tuple[str, str, str, str]:
            sk, lbl, dr = self._resolve_week_bucket(d)
            patch = self._resolve_patch_for_date(d)
            return (sk, f"{lbl} ({dr})", dr, patch["short"])

        def _patch_key(d: datetime.date) -> Tuple[str, str, str, str]:
            p = self._resolve_patch_for_date(d)
            idx = next((i for i, m in enumerate(self.PATCH_MILESTONES) if m["id"] == p["id"]), 0)
            end_lbl = p["end_date"][5:] if p["end_date"] else "Atual"
            dr = f"{p['start_date'][5:]} → {end_lbl}"
            return (f"P{idx:02d}_{p['id']}", p["short"], dr, p["version"])

        def _day_key(d: datetime.date) -> Tuple[str, str, str, str]:
            p = self._resolve_patch_for_date(d)
            return (d.isoformat(), d.strftime("%d/%m"), d.isoformat(), p["short"])

        by_week = self._group_evolution_buckets(closed_trades, portfolio_series, alm_rows, _week_key)
        by_patch = self._group_evolution_buckets(closed_trades, portfolio_series, alm_rows, _patch_key)
        by_day = self._group_evolution_buckets(closed_trades, portfolio_series, alm_rows, _day_key)

        global_kpis = self._compute_bucket_metrics(
            closed_trades,
            [pt["adj_equity"] for pt in portfolio_series],
            alm_rows,
        )
        # Curva contínua final de PnL (usa portfolio ajustado quando existir, senão soma dos trades)
        if portfolio_series:
            cont_pnl = portfolio_series[-1]["cum_pnl"]
            cur_epoch_eq = portfolio_series[-1]["raw_equity"]
            cur_epoch_pnl = cur_epoch_eq - self.initial_capital
        else:
            cont_pnl = global_kpis["pnl_usd"]
            cur_epoch_eq = self.initial_capital + cont_pnl
            cur_epoch_pnl = cont_pnl

        # Comparativo Semana/Patch mais recente vs Baseline S0
        ref_list = by_week if len(by_week) >= 2 else by_patch
        base_bucket = next((b for b in ref_list if b["trades"] > 0), None)
        last_bucket = next((b for b in reversed(ref_list) if b["trades"] > 0), None)
        base_acc = base_bucket["accuracy_pct"] if base_bucket else 0.0
        last_acc = last_bucket["accuracy_pct"] if last_bucket else 0.0
        base_pf = base_bucket["profit_factor"] if base_bucket else 0.0
        last_pf = last_bucket["profit_factor"] if last_bucket else 0.0

        summary = {
            "total_trades": global_kpis["trades"],
            "winning_trades": global_kpis["wins"],
            "losing_trades": global_kpis["losses"],
            "accuracy_pct": global_kpis["accuracy_pct"],
            "cum_pnl_usd": self._safe_float(cont_pnl, 0.0, 2),
            "cum_pnl_pct": self._safe_float(cont_pnl / self.initial_capital * 100.0, 0.0, 2),
            "trade_sum_pnl_usd": global_kpis["pnl_usd"],
            "current_epoch_equity": self._safe_float(cur_epoch_eq, self.initial_capital, 2),
            "current_epoch_pnl_usd": self._safe_float(cur_epoch_pnl, 0.0, 2),
            "max_drawdown_pct": global_kpis["max_drawdown_pct"],
            "profit_factor": global_kpis["profit_factor"],
            "expectancy_usd": global_kpis["expectancy_usd"],
            "payoff_ratio": global_kpis["payoff_ratio"],
            "total_fees_usd": global_kpis["fees_usd"],
            "latest_week_label": last_bucket["label"] if last_bucket else "--",
            "latest_week_accuracy_pct": last_acc,
            "baseline_accuracy_pct": base_acc,
            "accuracy_delta_pp": self._safe_float(last_acc - base_acc, 0.0, 1),
            "latest_week_pf": last_pf,
            "baseline_pf": base_pf,
            "committee_accuracy_pct": global_kpis["committee_accuracy_pct"],
            "calibration_gap_pct": global_kpis["calibration_gap_pct"],
        }

        return {
            "has_data": True,
            "initial_capital": self.initial_capital,
            "summary": summary,
            "patches": self.PATCH_MILESTONES,
            "resets": resets,
            "portfolio_series": self._downsample_series(portfolio_series, max_points=500),
            "trade_series": trade_series,
            "by_week": by_week,
            "by_patch": by_patch,
            "by_day": by_day,
            "agent_calibration": [
                {
                    "regime": str(r["regime"]),
                    "agent_name": str(r["agent_name"]),
                    "ece": self._safe_float(r["ece"], 0.0, 4),
                    "brier_score": self._safe_float(r["brier_score"], 0.0, 4),
                }
                for r in cal_rows
            ],
        }
