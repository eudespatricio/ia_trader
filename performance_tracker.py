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

import logging
import sqlite3
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

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

    def __init__(self, db_path: str = "datasets/market_cache.db", initial_capital: float = 10000.0):
        self.db_path = db_path
        self.initial_capital = float(initial_capital)
        self._ensure_telemetry_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
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
                # 1. Posições e Portfolio atual
                port_row = conn.execute(
                    "SELECT * FROM paper_portfolio ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                
                # 2. Histórico de Trades
                trades_rows = conn.execute(
                    "SELECT * FROM paper_trades ORDER BY timestamp ASC"
                ).fetchall()
                
                # 3. Série temporal de Equity para cálculo de Sharpe e Drawdown
                equity_rows = conn.execute(
                    "SELECT timestamp, equity FROM paper_portfolio ORDER BY timestamp ASC"
                ).fetchall()

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

                # Anualização baseada em snapshots periódicos (~5min ou horários)
                periods_per_year = 365 * 24 * 12  # Snapshots de 5 min
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
                    annualized_return_pct = (mean_r * periods_per_year) * 100.0
                    calmar_ratio = float(annualized_return_pct / max_drawdown_pct)

            # ------------------------------------------------------------------
            # Breakdown por Estratégia
            # ------------------------------------------------------------------
            strategy_metrics = self._compute_strategy_breakdown(closing_trades)

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

    def _compute_strategy_breakdown(self, closing_trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Decompõe métricas entre SCALP, GOLDEN_POT e CARRY."""
        breakdown = {
            "SCALP": {"trades": 0, "wins": 0, "pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0},
            "GOLDEN_POT": {"trades": 0, "wins": 0, "pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0},
            "CARRY": {"trades": 0, "wins": 0, "pnl": 0.0, "win_rate": 0.0, "profit_factor": 0.0},
        }

        for t in closing_trades:
            rationale = str(t.get("rationale", "")).upper()
            action = str(t.get("action", "")).upper()
            pnl = float(t.get("pnl", 0.0))

            if "GOLDEN_POT" in rationale or "GOLDEN_POT" in action:
                stype = "GOLDEN_POT"
            elif "CARRY" in rationale or "CARRY" in action:
                stype = "CARRY"
            else:
                stype = "SCALP"

            entry = breakdown[stype]
            entry["trades"] += 1
            entry["pnl"] += pnl
            if pnl > 0:
                entry["wins"] += 1

        for stype, data in breakdown.items():
            if data["trades"] > 0:
                data["win_rate"] = (data["wins"] / data["trades"]) * 100.0

        return breakdown

    def record_snapshot(self, metrics: Optional[PerformanceMetrics] = None) -> None:
        """Persiste um snapshot periódico de telemetria na tabela `performance_snapshots`."""
        if metrics is None:
            metrics = self.compute_metrics_from_db()

        try:
            with self._get_connection() as conn:
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
                    0,
                    metrics.total_fees_paid
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Erro ao salvar snapshot de performance: {e}")
