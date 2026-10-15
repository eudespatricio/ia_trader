"""
rollback_guard.py - Monitor de Estabilidade, Circuit Breakers e Rollback Automático.
Fase 7 do Projeto IA Trader.

Responsabilidades Principais:
1. Circuit Breakers Financeiros:
   - Trava por Drawdown Diário Máximo (padrão: 5.0%).
   - Trava por Drawdown Total Máximo (padrão: 10.0%).
   - Trava de Cooldown por Sequência de Perdas Consecutivas (padrão: 4 stops seguidos -> 2h de pausa).
2. Detecção de Degradação de Desempenho e Reversão Automática (Auto-Rollback):
   - Se o Win Rate dos últimos N trades cair criticamente (< 30%) ou o Profit Factor cair < 0.60,
     ativa automaticamente a configuração defensiva (pesos ultraconservadores no Comitê e threshold mais rígido).
3. Persistência de Estado e Auditoria:
   - Tabela SQLite `circuit_breaker_events` com histórico de disparos, motivos e resets.
4. Interface Unificada de Consulta:
   - `can_open_new_position(symbol, strategy)`: Retorna (True, "") ou (False, motivo).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from db_conn import connect as _db_connect
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ia_trader.rollback_guard")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [rollback_guard] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class CircuitBreakerStatus:
    """Estado do Guardião de Circuit Breakers e Rollback."""
    state: str = "CLEAR"                   # "CLEAR", "COOLDOWN", "TRIPPED", "EMERGENCY_HALT"
    is_trading_allowed: bool = True
    reason: str = "Sistema operacional dentro dos limites normais de risco."
    current_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    cooldown_until: int = 0               # Timestamp ms
    is_rollback_active: bool = False
    tripped_timestamp: int = 0
    actions_taken: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RollbackGuard:
    """
    Guardião de sobrevivência financeira, mitigação de ruína e controle de falhas.
    Monitora métricas em tempo real e bloqueia novas operações se anomalias forem detectadas.
    """

    def __init__(
        self,
        db_path: str = "datasets/market_cache.db",
        max_daily_drawdown_pct: float = 5.0,
        max_total_drawdown_pct: float = 10.0,
        max_consecutive_losses: int = 4,
        cooldown_seconds: int = 7200,      # 2 horas de pausa
        min_win_rate_threshold: float = 30.0,
        min_trades_for_degradation: int = 12,
        auto_rollback_enabled: bool = True,
    ):
        self.db_path = db_path
        self.max_daily_drawdown_pct = float(max_daily_drawdown_pct)
        self.max_total_drawdown_pct = float(max_total_drawdown_pct)
        self.max_consecutive_losses = int(max_consecutive_losses)
        self.cooldown_seconds = int(cooldown_seconds)
        self.min_win_rate_threshold = float(min_win_rate_threshold)
        self.min_trades_for_degradation = int(min_trades_for_degradation)
        self.auto_rollback_enabled = auto_rollback_enabled

        self._ensure_tables()
        self._load_persisted_state()

    def _get_connection(self) -> sqlite3.Connection:
        conn = _db_connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _ensure_tables(self):
        """Cria tabelas de circuit breaker e estado de rollback."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS circuit_breaker_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        state TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        drawdown_pct REAL NOT NULL,
                        consecutive_losses INTEGER NOT NULL,
                        cooldown_until INTEGER,
                        actions_json TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS circuit_breaker_state (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        state TEXT NOT NULL,
                        is_trading_allowed INTEGER NOT NULL,
                        reason TEXT NOT NULL,
                        cooldown_until INTEGER NOT NULL,
                        is_rollback_active INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.warning(f"Aviso ao inicializar tabelas do RollbackGuard: {e}")

    def _load_persisted_state(self):
        """Carrega o último estado persistido do circuit breaker para evitar amnésia após reinicialização."""
        self.status = CircuitBreakerStatus()
        try:
            with self._get_connection() as conn:
                row = conn.execute("SELECT * FROM circuit_breaker_state WHERE id = 1").fetchone()
                if row:
                    now_ms = int(time.time() * 1000)
                    cd_until = int(row["cooldown_until"])
                    
                    # Se estava em cooldown e o tempo já expirou, restaura para CLEAR
                    if row["state"] == "COOLDOWN" and now_ms >= cd_until:
                        self.status.state = "CLEAR"
                        self.status.is_trading_allowed = True
                        self.status.reason = "Cooldown expirado. Sistema reativado normalmente."
                        self.status.cooldown_until = 0
                    else:
                        self.status.state = str(row["state"])
                        self.status.is_trading_allowed = bool(row["is_trading_allowed"])
                        self.status.reason = str(row["reason"])
                        self.status.cooldown_until = cd_until
                        self.status.is_rollback_active = bool(row["is_rollback_active"])
        except Exception as e:
            logger.warning(f"Aviso ao carregar estado do RollbackGuard: {e}")

    def _persist_state(self):
        """Salva o estado atual no SQLite."""
        now_ms = int(time.time() * 1000)
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO circuit_breaker_state (
                        id, state, is_trading_allowed, reason, cooldown_until, is_rollback_active, updated_at
                    ) VALUES (1, ?, ?, ?, ?, ?, ?)
                """, (
                    self.status.state,
                    1 if self.status.is_trading_allowed else 0,
                    self.status.reason,
                    self.status.cooldown_until,
                    1 if self.status.is_rollback_active else 0,
                    now_ms
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Erro ao persistir estado do circuit breaker: {e}")

    def _log_event(self, event_type: str, reason: str, dd_pct: float, losses: int, cd_until: int = 0, actions: List[str] = None):
        """Registra auditoria formal na tabela de eventos."""
        now_ms = int(time.time() * 1000)
        import json
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO circuit_breaker_events (
                        timestamp, event_type, state, reason, drawdown_pct, consecutive_losses, cooldown_until, actions_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    now_ms, event_type, self.status.state, reason, dd_pct, losses, cd_until, json.dumps(actions or [])
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"Erro ao registrar log de circuit breaker: {e}")

    def evaluate_telemetry(self, metrics: Any) -> CircuitBreakerStatus:
        """
        Avalia o objeto PerformanceMetrics ou dict e atualiza o estado dos circuit breakers.
        """
        now_ms = int(time.time() * 1000)
        m = metrics if isinstance(metrics, dict) else metrics.to_dict()

        current_dd = float(m.get("current_drawdown_pct", 0.0))
        max_dd = float(m.get("max_drawdown_pct", 0.0))
        consec_losses = int(m.get("max_consecutive_losses", 0))
        cur_streak_type = str(m.get("current_streak_type", "NONE"))
        cur_streak_count = int(m.get("current_streak_count", 0))
        active_losses = cur_streak_count if cur_streak_type == "LOSS" else 0
        
        total_trades = int(m.get("total_trades", 0))
        win_rate = float(m.get("win_rate", 0.0))
        profit_factor = float(m.get("profit_factor", 0.0))

        self.status.current_drawdown_pct = current_dd
        self.status.consecutive_losses = active_losses

        # Se estiver em parada de emergência manual, mantém travado
        if self.status.state == "EMERGENCY_HALT":
            return self.status

        # -------------------------------------------------------------
        # 1. Checagem de Cooldown Ativo
        # -------------------------------------------------------------
        if self.status.state == "COOLDOWN":
            if now_ms < self.status.cooldown_until:
                remaining_s = (self.status.cooldown_until - now_ms) / 1000.0
                self.status.reason = f"Em período de Cooldown de proteção contra perdas consecutivas ({remaining_s/60.0:.1f} minutos restantes)."
                self.status.is_trading_allowed = False
                return self.status
            else:
                # Cooldown expirou com sucesso!
                logger.info("🟢 Período de Cooldown encerrado com sucesso. Sistema retornando ao estado CLEAR.")
                self.status.state = "CLEAR"
                self.status.is_trading_allowed = True
                self.status.reason = "Cooldown expirado. Sistema pronto para operar."
                self.status.cooldown_until = 0
                self._persist_state()

        # -------------------------------------------------------------
        # 2. Circuit Breaker de Drawdown Total (> 10.0%)
        # -------------------------------------------------------------
        if current_dd >= self.max_total_drawdown_pct:
            reason = f"CIRCUIT BREAKER DISPARADO: Drawdown atual ({current_dd:.2f}%) atingiu o limite crítico ({self.max_total_drawdown_pct:.1f}%). Novas operações bloqueadas compulsoriamente."
            self.status.state = "TRIPPED"
            self.status.is_trading_allowed = False
            self.status.reason = reason
            self.status.tripped_timestamp = now_ms
            self.status.actions_taken = ["HALT_NEW_POSITIONS", "NOTIFY_OPERATOR", "PRESERVE_CAPITAL"]
            self._log_event("TOTAL_DRAWDOWN_BREAKER", reason, current_dd, active_losses, actions=self.status.actions_taken)
            self._persist_state()
            logger.error(f"🚨 {reason}")
            return self.status

        # -------------------------------------------------------------
        # 3. Circuit Breaker de Perdas Consecutivas (>= 4 stops)
        # -------------------------------------------------------------
        if active_losses >= self.max_consecutive_losses:
            cooldown_until = now_ms + (self.cooldown_seconds * 1000)
            reason = f"COOLDOWN ACIONADO: {active_losses} perdas consecutivas detectadas. Bloqueio temporário de {self.cooldown_seconds/60:.0f}min para evitar overfitting a regime desfavorável."
            self.status.state = "COOLDOWN"
            self.status.is_trading_allowed = False
            self.status.reason = reason
            self.status.cooldown_until = cooldown_until
            self.status.actions_taken = ["PAUSE_TRADING_2H", "PREVENT_REVENGE_TRADING"]
            self._log_event("CONSECUTIVE_LOSS_COOLDOWN", reason, current_dd, active_losses, cd_until=cooldown_until, actions=self.status.actions_taken)
            self._persist_state()
            logger.warning(f"⚠️ {reason}")
            return self.status

        # -------------------------------------------------------------
        # 4. Detecção de Degradação & Rollback Automático de Configuração
        # -------------------------------------------------------------
        if self.auto_rollback_enabled and total_trades >= self.min_trades_for_degradation:
            if win_rate < self.min_win_rate_threshold or (profit_factor < 0.60 and profit_factor > 0):
                if not self.status.is_rollback_active:
                    reason = f"DEGRADAÇÃO DETECTADA: Win Rate={win_rate:.1f}% e Profit Factor={profit_factor:.2f} abaixo do limiar de segurança. Ativando rollback para modo defensivo ultra-conservador."
                    self.status.is_rollback_active = True
                    self.status.actions_taken.append("ROLLBACK_TO_DEFENSIVE_MODE")
                    self._log_event("DEGRADATION_ROLLBACK_ACTIVATED", reason, current_dd, active_losses, actions=["ROLLBACK_TO_DEFENSIVE_MODE"])
                    self._persist_state()
                    logger.warning(f"🛡️ {reason}")

        # Se passou por todas as barreiras sem disparar travas
        if self.status.state != "TRIPPED" and self.status.state != "COOLDOWN":
            self.status.state = "CLEAR"
            self.status.is_trading_allowed = True
            self.status.reason = "Condições operacionais normais. Sem travas ativas."
            self._persist_state()

        return self.status

    def can_open_new_position(self, symbol: str, strategy: str = "SCALP") -> Tuple[bool, str]:
        """
        Consulta instantânea para o motor de execução antes de enviar ordens.
        Retorna (True, "") ou (False, justificativa_do_bloqueio).
        """
        # Checa se o cooldown expirou dinamicamente
        now_ms = int(time.time() * 1000)
        if self.status.state == "COOLDOWN":
            if now_ms >= self.status.cooldown_until:
                self.status.state = "CLEAR"
                self.status.is_trading_allowed = True
                self.status.reason = "Cooldown expirado. Sistema liberado."
                self._persist_state()
            else:
                remaining_m = (self.status.cooldown_until - now_ms) / 60000.0
                return False, f"Trading bloqueado por Cooldown de perdas consecutivas ({remaining_m:.1f} min restantes)."

        if not self.status.is_trading_allowed or self.status.state in ("TRIPPED", "EMERGENCY_HALT"):
            return False, f"Trading bloqueado pelo Circuit Breaker: {self.status.reason}"

        return True, "Trading autorizado pelo Guardião de Risco."

    def trigger_emergency_stop(self, reason: str = "Parada manual de emergência disparada pelo operador."):
        """Trava total e imediata de segurança."""
        now_ms = int(time.time() * 1000)
        self.status.state = "EMERGENCY_HALT"
        self.status.is_trading_allowed = False
        self.status.reason = reason
        self.status.tripped_timestamp = now_ms
        self.status.actions_taken = ["EMERGENCY_HALT_ALL"]
        self._log_event("EMERGENCY_STOP", reason, self.status.current_drawdown_pct, self.status.consecutive_losses)
        self._persist_state()
        logger.error(f"🛑 EMERGENCY STOP ACIONADO: {reason}")

    def reset_circuit_breaker(self, operator_note: str = "Reset manual autorizado"):
        """Restaura os circuit breakers manualmente após revisão humana."""
        self.status = CircuitBreakerStatus(
            state="CLEAR",
            is_trading_allowed=True,
            reason=f"Circuit Breaker resetado manualmente. Nota: {operator_note}",
            actions_taken=["MANUAL_RESET"]
        )
        self._log_event("MANUAL_RESET", operator_note, 0.0, 0)
        self._persist_state()
        logger.info(f"🔄 Circuit Breaker resetado com sucesso: {operator_note}")
