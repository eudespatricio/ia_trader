"""
learning_tracker.py - Motor de Memória Cognitiva, Aprendizado Adaptativo e Evolução de Modelos.
Fase 7 / Governança do IA Trader.

Premissa Fundamental:
Não somos um robô estático com regras congeladas no tempo, mas uma inteligência quantitativa
capaz de assimilar o contexto de mercado, registrar a eficácia histórica de cada agente especialista,
adaptar pesos dinamicamente por regime de mercado e evoluir modelos de Machine Learning (Champion-Challenger).

Pilares:
1. Memória Episódica de Deliberação (Decision Memory):
   - Grava cada sinal do comitê com o regime macro vigente e o voto de cada agente.
   - Associa o desfecho do trade (Win, Loss, PnL) aos votos individuais de cada agente.
2. Refinamento Adaptativo de Pesos (Dynamic Weight Calibration):
   - Computa a acurácia empírica e o Brier Score de cada agente especialista segregado por Regime:
     * Ex: Em SIDEWAYS_RANGE, valoriza OrderFlow e POI; em BULL_TREND, valoriza Momentum/Macro.
   - Alimenta o `ConsensusJudge` com `weights_override` calibrados em tempo real.
3. Protocolo de Evolução Contínua e Retreinamento (Champion-Challenger):
   - Monitora o envelhecimento dos modelos e o fluxo de novos dados.
   - Valida novos modelos fora da amostra antes de promovê-los a produção (Hot-Swap seguro).
"""

from __future__ import annotations

from contextlib import closing
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from db_conn import connect as _db_connect

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger("ia_trader.learning_tracker")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [learning_tracker] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class AgentPerformanceScore:
    """Pontuação de assertividade de um agente especialista em determinado regime."""
    agent_name: str
    regime: str
    total_calls: int = 0
    correct_calls: int = 0
    accuracy: float = 0.50
    brier_score: float = 0.25           # Menor é melhor (0 = perfeição, 0.25 = aleatório)
    suggested_weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LearningTracker:
    """
    Rastreador de aprendizado contínuo e memória adaptativa do comitê de agentes.
    Persiste deliberações e desfechos no SQLite para calibração de pesos orientada a dados.
    """

    DEFAULT_BASE_WEIGHTS = {
        "RiskSentinelAgent": 1.5,
        "TechnicalAgent": 1.0,
        "OrderFlowAgent": 0.8,
        "SentimentAgent": 0.6,
        "MacroAgent": 0.7,
        "OnChainAgent": 0.5,
    }

    CALIBRATION_STALE_TTL_MS: int = 72 * 3_600_000  # 72h anti-deadlock cooldown (OOS-proven S6-3)

    @staticmethod
    def _get_pilot_epoch_ms(conn: sqlite3.Connection) -> Optional[int]:
        try:
            row = conn.execute(
                "SELECT value FROM simulation_settings WHERE key = 'pilot_start_utc' LIMIT 1"
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        if not row or row[0] is None:
            return None
        raw = str(row[0]).strip()
        if not raw:
            return None
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except (ValueError, TypeError):
            return None

    def _is_regime_calibration_stale_or_empty(
        self, conn: sqlite3.Connection, regime: str, now_ms: int
    ) -> Tuple[bool, Optional[int]]:
        pilot_ms = self._get_pilot_epoch_ms(conn)
        try:
            if pilot_ms is not None:
                row = conn.execute(
                    "SELECT MAX(timestamp) FROM agent_learning_memory "
                    "WHERE regime = ? AND outcome IN ('WIN', 'LOSS') AND timestamp >= ?",
                    (regime, pilot_ms),
                ).fetchone()
                if not row or row[0] is None:
                    return (True, None)
                last_ts = int(row[0])
                return ((now_ms - last_ts) >= self.CALIBRATION_STALE_TTL_MS, last_ts)
            else:
                row = conn.execute(
                    "SELECT MAX(timestamp) FROM agent_learning_memory "
                    "WHERE regime = ? AND outcome IN ('WIN', 'LOSS')",
                    (regime,),
                ).fetchone()
                if not row or row[0] is None:
                    return (False, None)
                last_ts = int(row[0])
                return ((now_ms - last_ts) >= self.CALIBRATION_STALE_TTL_MS, last_ts)
        except sqlite3.OperationalError:
            return (False, None)

    def __init__(self, db_path: str = "datasets/market_cache.db"):
        self.db_path = db_path
        self._ensure_tables()
        self._bootstrap_structural_sub_rules_if_empty()

    def _get_connection(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = _db_connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _ensure_tables(self):
        """Cria tabelas de memória episódica e calibração de pesos."""
        try:
            with self._get_connection() as conn:
                # 1. Memória de decisões
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_learning_memory (
                        decision_id TEXT PRIMARY KEY,
                        timestamp INTEGER NOT NULL,
                        symbol TEXT NOT NULL,
                        regime TEXT NOT NULL,
                        consensus_direction TEXT NOT NULL,
                        consensus_confidence REAL NOT NULL,
                        agent_signals_json TEXT NOT NULL,
                        outcome TEXT DEFAULT 'PENDING',  -- 'WIN', 'LOSS', 'BREAKEVEN', 'PENDING'
                        realized_pnl REAL DEFAULT 0.0,
                        position_id TEXT
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_learning_ts ON agent_learning_memory(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_learning_regime ON agent_learning_memory(regime)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_learning_pos ON agent_learning_memory(position_id)")

                # 2. Histórico de calibração de pesos por regime
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_regime_weights (
                        regime TEXT NOT NULL,
                        agent_name TEXT NOT NULL,
                        weight REAL NOT NULL,
                        accuracy REAL NOT NULL,
                        total_samples INTEGER NOT NULL,
                        updated_at INTEGER NOT NULL,
                        PRIMARY KEY (regime, agent_name)
                    )
                """)
                # 3. Métricas avançadas de calibração probabilística (ECE / Brier)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS agent_regime_calibration (
                        regime TEXT NOT NULL,
                        agent_name TEXT NOT NULL,
                        ece REAL DEFAULT 0.0,
                        brier_score REAL DEFAULT 0.0,
                        updated_at INTEGER NOT NULL,
                        PRIMARY KEY (regime, agent_name)
                    )
                """)
                # 4. Auto-calibração Bayesiana de sub-regras estruturais (Living Sub-Rule Learner - S5)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS structural_sub_rule_calibration (
                        sub_rule TEXT PRIMARY KEY,
                        total_samples INTEGER NOT NULL,
                        wins INTEGER NOT NULL,
                        posterior_wr REAL NOT NULL,
                        cumulative_pnl REAL NOT NULL,
                        multiplier REAL NOT NULL,
                        is_muted INTEGER NOT NULL DEFAULT 0,
                        updated_at INTEGER NOT NULL
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.warning(f"Aviso ao inicializar tabelas do LearningTracker: {e}")

    def record_decision(
        self,
        decision_id: str,
        symbol: str,
        regime: str,
        consensus_direction: str,
        consensus_confidence: float,
        agent_signals: List[Any],
        position_id: Optional[str] = None,
        timestamp: Optional[int] = None,
    ) -> None:
        """
        Registra uma deliberação do comitê com os votos individuais e o contexto de regime.
        """
        ts = timestamp or int(time.time() * 1000)
        
        # Converte sinais dos agentes para formato serializável
        signals_data = []
        for s in agent_signals:
            if hasattr(s, "to_dict"):
                signals_data.append(s.to_dict())
            elif isinstance(s, dict):
                signals_data.append(s)

        signals_json = json.dumps(signals_data)

        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO agent_learning_memory (
                        decision_id, timestamp, symbol, regime,
                        consensus_direction, consensus_confidence, agent_signals_json,
                        outcome, position_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)
                """, (
                    decision_id, ts, symbol, regime,
                    consensus_direction, consensus_confidence, signals_json, position_id
                ))
                conn.commit()
            logger.debug(f"Memória de decisão {decision_id} ({symbol}, {regime}) gravada com sucesso.")
        except Exception as e:
            logger.warning(f"Erro ao salvar decisão no LearningTracker: {e}")

    def update_trade_outcome(
        self,
        position_id: str,
        pnl: float,
        outcome_label: Optional[str] = None,
    ) -> None:
        """
        Atualiza o desfecho financeiro (Win/Loss) associado a uma posição e ativa o aprendizado.
        """
        if outcome_label is None:
            if pnl > 0.0001:
                outcome = "WIN"
            elif pnl < -0.0001:
                outcome = "LOSS"
            else:
                outcome = "BREAKEVEN"
        else:
            outcome = outcome_label

        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE agent_learning_memory
                    SET outcome = ?, realized_pnl = ?
                    WHERE position_id = ?
                """, (outcome, pnl, position_id))
                conn.commit()
                
            # Dispara recalibração de pesos e sub-regras estruturais com a nova evidência
            self.recalibrate_agent_weights()
            self.recalibrate_structural_sub_rules()
            logger.info(f"Desfecho {outcome} (${pnl:+.2f}) registrado para posição {position_id}. Pesos e sub-regras atualizados.")
        except Exception as e:
            logger.warning(f"Erro ao atualizar desfecho de trade no LearningTracker: {e}")

    def recalibrate_agent_weights(self, now_ms: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Nível 1 & Nível 2 do Protocolo S2 (docs/S2_CALIBRACAO_AGENTES_BACKTEST.md):
        Calcula os pesos dos agentes por regime usando Otimização Convexa Restrita
        (Ridge / L2 com prior bayesiano DEFAULT_BASE_WEIGHTS), minimizando o erro quadrático direcional,
        garantindo soma de pesos normalizada, pesos não-negativos e preservação estrita do veto
        do RiskSentinelAgent (w = 1.5).
        Calcula também a acurácia empírica e o ECE (Expected Calibration Error).
        Retorna dicionário: {regime: {agent_name: weight}}
        """
        raw_now_ms = now_ms
        now_ms = int(now_ms) if now_ms is not None else int(time.time() * 1000)
        regimes = ["BULL_TREND", "BEAR_TREND", "SIDEWAYS_RANGE", "HIGH_VOLATILITY_PANIC"]
        all_weights: Dict[str, Dict[str, float]] = {}

        try:
            # Patch S4 (v1.0.6-s4) + Patch S6-3: Janela deslizante pós-S2/S3 (ou pilot_start_utc se ativo,
            # últimas 30 decisões concluídas), com TTL de 72h anti-deadlock por regime.
            s3_epoch_ms = 1793577600000
            with closing(self._get_connection()) as conn:
                pilot_ms = self._get_pilot_epoch_ms(conn)
                if pilot_ms is not None:
                    effective_epoch_ms = max(s3_epoch_ms, pilot_ms)
                    rows = conn.execute("""
                        SELECT timestamp, regime, consensus_direction, agent_signals_json, outcome, realized_pnl
                        FROM agent_learning_memory
                        WHERE outcome IN ('WIN', 'LOSS') AND timestamp >= ?
                        ORDER BY timestamp DESC
                        LIMIT 30
                    """, (effective_epoch_ms,)).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT timestamp, regime, consensus_direction, agent_signals_json, outcome, realized_pnl
                        FROM agent_learning_memory
                        WHERE outcome IN ('WIN', 'LOSS') AND timestamp >= ?
                        ORDER BY timestamp DESC
                        LIMIT 30
                    """, (s3_epoch_ms,)).fetchall()
                    if len(rows) < 5:
                        rows = conn.execute("""
                            SELECT timestamp, regime, consensus_direction, agent_signals_json, outcome, realized_pnl
                            FROM agent_learning_memory
                            WHERE outcome IN ('WIN', 'LOSS')
                            ORDER BY timestamp DESC
                            LIMIT 30
                        """).fetchall()

            if not rows and pilot_ms is None:
                # Retorna pesos padrão caso ainda não haja desfechos acumulados
                for r in regimes:
                    all_weights[r] = dict(self.DEFAULT_BASE_WEIGHTS)
                return all_weights

            ttl_ref_now_ms = now_ms
            if raw_now_ms is None and pilot_ms is None and rows and int(rows[0]["timestamp"]) < s3_epoch_ms:
                ttl_ref_now_ms = int(rows[0]["timestamp"])

            # Agrupa por regime (mantendo timestamp DESC) e aplica TTL 72h + silence-gap clustering
            rows_by_regime: Dict[str, List[sqlite3.Row]] = {r: [] for r in regimes}
            for row in rows:
                rows_by_regime.setdefault(row["regime"], []).append(row)

            active_all_rows: List[sqlite3.Row] = []
            for r, r_rows in rows_by_regime.items():
                if not r_rows or (ttl_ref_now_ms - int(r_rows[0]["timestamp"])) >= self.CALIBRATION_STALE_TTL_MS:
                    active_rows: List[sqlite3.Row] = []
                else:
                    active_rows = [r_rows[0]]
                    prev_ts = int(r_rows[0]["timestamp"])
                    for row in r_rows[1:]:
                        cur_ts = int(row["timestamp"])
                        if (prev_ts - cur_ts) >= self.CALIBRATION_STALE_TTL_MS:
                            break
                        active_rows.append(row)
                        prev_ts = cur_ts
                active_all_rows.extend(active_rows)

            # Agrupa estatísticas por (regime, agent_name) e amostras por regime
            stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
            regime_samples: Dict[str, List[Dict[str, Any]]] = {r: [] for r in regimes}

            for row in active_all_rows:
                regime = row["regime"]
                outcome = row["outcome"]
                cons_dir = row["consensus_direction"]
                try:
                    signals = json.loads(row["agent_signals_json"])
                except Exception:
                    continue

                # Direção alvo realizada: +1.0 para BUY vencedor ou SELL perdedor, -1.0 para o inverso
                y_target = 1.0 if (
                    (outcome == "WIN" and cons_dir == "BUY") or (outcome == "LOSS" and cons_dir == "SELL")
                ) else -1.0

                agent_votes: Dict[str, float] = {}
                for sig in signals:
                    name = sig.get("agent_name", "")
                    direction = sig.get("direction", "HOLD")
                    if direction not in ("BUY", "SELL"):
                        continue
                    conf = float(sig.get("confidence", 0.5))
                    s_vote = conf if direction == "BUY" else -conf
                    agent_votes[name] = s_vote

                    key = (regime, name)
                    if key not in stats:
                        stats[key] = {"total": 0, "correct": 0, "brier_sum": 0.0, "predictions": []}

                    stats[key]["total"] += 1
                    is_correct = (s_vote * y_target > 0)
                    if is_correct:
                        stats[key]["correct"] += 1

                    stats[key]["predictions"].append((conf, is_correct))
                    target_brier = 1.0 if is_correct else 0.0
                    stats[key]["brier_sum"] += (conf - target_brier) ** 2

                if regime in regime_samples and agent_votes:
                    regime_samples[regime].append({
                        "y": y_target,
                        "votes": agent_votes,
                    })

            # Calcula ECE (Expected Calibration Error) para cada (regime, agente)
            # Patch S4 (v1.0.6-s4): Para amostras pequenas (5 <= tot < 20), bins finos de 0.10 com 1-2 trades
            # geram ruído extremo (ex.: 1 LOSS em bin 0.90 produz |0.0 - 0.90| = 0.90). Usa o gap agregado
            # |mean_acc - mean_conf| quando tot < 20, e ECE particionado em 5 bins quando tot >= 20.
            for key, s in stats.items():
                preds = s.get("predictions", [])
                tot = len(preds)
                if tot >= 20:
                    bins = [[] for _ in range(5)]
                    for conf, corr in preds:
                        b_idx = min(4, max(0, int((conf - 0.50) / 0.10)))
                        bins[b_idx].append((conf, corr))
                    ece = 0.0
                    for b in bins:
                        if b:
                            b_acc = sum(1 for _, c in b if c) / len(b)
                            b_conf = sum(c for c, _ in b) / len(b)
                            ece += (len(b) / tot) * abs(b_acc - b_conf)
                    s["ece"] = round(ece, 4)
                elif tot >= 5:
                    overall_acc = sum(1 for _, c in preds if c) / tot
                    overall_conf = sum(c for c, _ in preds) / tot
                    s["ece"] = round(abs(overall_acc - overall_conf), 4)
                else:
                    s["ece"] = 0.0

            # Otimização Convexa Restrita (Ridge / L2 com prior bayesiano) por regime
            with closing(self._get_connection()) as conn:
                for r in regimes:
                    all_weights[r] = dict(self.DEFAULT_BASE_WEIGHTS)
                    active_agents = [
                        name for name in self.DEFAULT_BASE_WEIGHTS
                        if (r, name) in stats and stats[(r, name)]["total"] >= 5
                    ]

                    calibrated_weights: Dict[str, float] = {}

                    if active_agents and regime_samples[r]:
                        samples_r = regime_samples[r]
                        valid_samples = [
                            s for s in samples_r if any(ag in s["votes"] for ag in active_agents)
                        ]
                        if valid_samples:
                            S = np.array([
                                [s["votes"].get(ag, 0.0) for ag in active_agents]
                                for s in valid_samples
                            ], dtype=np.float64)
                            y = np.array([s["y"] for s in valid_samples], dtype=np.float64)
                            w_prior = np.array([self.DEFAULT_BASE_WEIGHTS[ag] for ag in active_agents], dtype=np.float64)

                            lam = 0.20
                            def loss(w: np.ndarray) -> float:
                                pred = S @ w
                                mse = float(np.mean((pred - y) ** 2))
                                reg = float(lam * np.sum((w - w_prior) ** 2))
                                return mse + reg

                            bounds = [(0.10, 2.50) for _ in active_agents]
                            res = minimize(loss, w_prior, bounds=bounds, method="L-BFGS-B")
                            w_opt = res.x if res.success else w_prior

                            if len(active_agents) > 1:
                                # Normalização da soma de pesos preservando a massa total dos priors ativos
                                prior_sum = float(np.sum(w_prior))
                                opt_sum = float(np.sum(w_opt))
                                w_final = w_opt * (prior_sum / opt_sum) if opt_sum > 0 else w_prior
                            else:
                                w_final = w_opt

                            for idx, ag in enumerate(active_agents):
                                calibrated_weights[ag] = round(float(w_final[idx]), 3)

                    # Persiste e popula all_weights
                    for agent_name, base_w in self.DEFAULT_BASE_WEIGHTS.items():
                        key = (r, agent_name)
                        if agent_name in calibrated_weights:
                            cal_w = calibrated_weights[agent_name]
                            acc = stats[key]["correct"] / stats[key]["total"]
                            tot = stats[key]["total"]
                            ece = stats[key].get("ece", 0.0)
                        else:
                            cal_w = base_w
                            acc = stats[key]["correct"] / stats[key]["total"] if (key in stats and stats[key]["total"] > 0) else 0.50
                            tot = stats[key]["total"] if key in stats else 0
                            ece = stats[key].get("ece", 0.0) if key in stats else 0.0

                        brier = round(stats[key]["brier_sum"] / stats[key]["total"], 4) if (key in stats and stats[key]["total"] > 0) else 0.0
                        all_weights[r][agent_name] = cal_w
                        conn.execute("""
                            INSERT OR REPLACE INTO agent_regime_weights (
                                regime, agent_name, weight, accuracy, total_samples, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?)
                        """, (r, agent_name, cal_w, acc, tot, now_ms))
                        conn.execute("""
                            INSERT OR REPLACE INTO agent_regime_calibration (
                                regime, agent_name, ece, brier_score, updated_at
                            ) VALUES (?, ?, ?, ?, ?)
                        """, (r, agent_name, ece, brier, now_ms))

                conn.commit()

            return all_weights

        except Exception as e:
            logger.error(f"Erro ao recalibrar pesos dos agentes: {e}")
            for r in regimes:
                all_weights[r] = dict(self.DEFAULT_BASE_WEIGHTS)
            return all_weights

    def _resolve_effective_now_ms(
        self, conn: sqlite3.Connection, now_ms: Optional[int]
    ) -> int:
        if now_ms is not None:
            return int(now_ms)
        wall_now_ms = int(time.time() * 1000)
        if self._get_pilot_epoch_ms(conn) is not None:
            return wall_now_ms
        try:
            row = conn.execute(
                "SELECT MAX(timestamp) FROM agent_learning_memory WHERE outcome IN ('WIN', 'LOSS')"
            ).fetchone()
            if row and row[0] is not None and int(row[0]) < 1793577600000:
                return int(row[0])
        except sqlite3.OperationalError:
            pass
        return wall_now_ms

    def get_calibrated_weights_for_regime(
        self, regime: str, now_ms: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Retorna o dicionário de pesos atualizado para um determinado regime de mercado.
        Pode ser passado diretamente como `weights_override` no `ConsensusJudge`.
        """
        weights = dict(self.DEFAULT_BASE_WEIGHTS)
        try:
            with closing(self._get_connection()) as conn:
                eff_now_ms = self._resolve_effective_now_ms(conn, now_ms)
                is_stale, _ = self._is_regime_calibration_stale_or_empty(conn, regime, eff_now_ms)
                if is_stale:
                    weights["RiskSentinelAgent"] = 1.5
                    return weights

                rows = conn.execute("""
                    SELECT agent_name, weight, total_samples FROM agent_regime_weights
                    WHERE regime = ?
                """, (regime,)).fetchall()

                for row in rows:
                    if row["total_samples"] >= 5:
                        weights[row["agent_name"]] = float(row["weight"])
        except Exception as e:
            logger.debug(f"Aviso ao buscar pesos calibrados: {e}")

        # Sentinela de risco SEMPRE mantém peso prioritário
        weights["RiskSentinelAgent"] = 1.5
        return weights

    def get_agent_calibration_for_regime(
        self, regime: str, now_ms: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Retorna mapa de calibração empírica por agente para um determinado regime:
        {agent_name: {"weight": float, "accuracy": float, "total_samples": int, "ece": float}}
        """
        calib: Dict[str, Dict[str, float]] = {}
        try:
            with closing(self._get_connection()) as conn:
                eff_now_ms = self._resolve_effective_now_ms(conn, now_ms)
                is_stale, _ = self._is_regime_calibration_stale_or_empty(conn, regime, eff_now_ms)
                if is_stale:
                    return {
                        ag: {
                            "weight": float(w),
                            "accuracy": 0.5,
                            "total_samples": 0,
                            "ece": 0.0,
                            "stale_expired": True,
                        }
                        for ag, w in self.DEFAULT_BASE_WEIGHTS.items()
                    }

                rows = conn.execute("""
                    SELECT w.agent_name, w.weight, w.accuracy, w.total_samples, COALESCE(c.ece, 0.0) as ece
                    FROM agent_regime_weights w
                    LEFT JOIN agent_regime_calibration c ON w.regime = c.regime AND w.agent_name = c.agent_name
                    WHERE w.regime = ?
                """, (regime,)).fetchall()
                for row in rows:
                    calib[row["agent_name"]] = {
                        "weight": float(row["weight"]),
                        "accuracy": float(row["accuracy"]),
                        "total_samples": int(row["total_samples"]),
                        "ece": float(row["ece"]),
                    }
        except Exception as e:
            logger.debug(f"Aviso ao buscar calibração de agentes: {e}")
        return calib

    @staticmethod
    def extract_structural_sub_rule(signals_list: List[Dict[str, Any]]) -> str:
        """Extrai `structural_sub_rule` do TechnicalAgent com fallback determinístico pré-S5."""
        if not signals_list:
            return "normal_no_sr"
        tech_sig: Optional[Dict[str, Any]] = None
        for raw_sig in signals_list:
            sig = raw_sig.to_dict() if hasattr(raw_sig, "to_dict") else (raw_sig if isinstance(raw_sig, dict) else None)
            if sig is not None and str(sig.get("agent_name", "")) == "TechnicalAgent":
                tech_sig = sig
                break
        if tech_sig is None:
            return "normal_no_sr"

        metrics = tech_sig.get("metrics")
        metrics_dict = metrics if isinstance(metrics, dict) else {}
        explicit_sub = metrics_dict.get("structural_sub_rule")
        if isinstance(explicit_sub, str) and explicit_sub.strip():
            return explicit_sub.strip()

        rationale = str(tech_sig.get("rationale") or "")
        if "Gatilho Estrutural" in rationale:
            return "sweep_trigger"
        direction = str(tech_sig.get("direction") or "").upper()
        poi_sweep_true = any(
            bool(metrics_dict.get(k))
            for k in ("at_bullish_poi", "at_bearish_poi", "liquidity_sweep_bull", "liquidity_sweep_bear")
        )
        if (
            "CONFLUÊNCIA POI/SWEEP" in rationale
            or "CONFLUENCIA POI/SWEEP" in rationale.upper()
            or (poi_sweep_true and direction != "HOLD")
        ):
            return "poi_confluence"
        return "normal_no_sr"

    def recalibrate_structural_sub_rules(
        self, window: int = 20, alpha: float = 4.0, beta: float = 4.0,
        min_samples: int = 5, mute_wr_threshold: float = 0.44,
    ) -> Dict[str, Dict[str, Any]]:
        """Auto-calibração Bayesiana Beta(alpha, beta) em janela deslizante por `sub_rule`."""
        now_ms = int(time.time() * 1000)
        eff_window = max(1, int(window))
        stats_out: Dict[str, Dict[str, Any]] = {}
        try:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT agent_signals_json, outcome, realized_pnl FROM agent_learning_memory "
                    "WHERE outcome IN ('WIN', 'LOSS') ORDER BY timestamp DESC, rowid DESC"
                ).fetchall()
                buckets: Dict[str, List[Tuple[str, float]]] = {}
                for row in rows:
                    try:
                        parsed = json.loads(row["agent_signals_json"])
                        signals_list = parsed if isinstance(parsed, list) else []
                    except Exception:
                        signals_list = []
                    sub_rule = self.extract_structural_sub_rule(signals_list)
                    bucket = buckets.setdefault(sub_rule, [])
                    if len(bucket) < eff_window:
                        bucket.append((str(row["outcome"]), float(row["realized_pnl"] or 0.0)))

                for sub_rule, trades in buckets.items():
                    n = len(trades)
                    wins = sum(1 for outcome, _ in trades if outcome == "WIN")
                    cum_pnl = float(sum(pnl for _, pnl in trades))
                    post_wr = float((wins + alpha) / (n + alpha + beta))
                    if n >= min_samples and post_wr < mute_wr_threshold and cum_pnl < 0.0:
                        is_muted, multiplier = True, 0.0
                    elif n < min_samples:
                        is_muted, multiplier = False, 1.0
                    else:
                        is_muted, multiplier = False, round(min(1.25, max(0.25, post_wr / 0.50)), 4)

                    conn.execute(
                        "INSERT OR REPLACE INTO structural_sub_rule_calibration ("
                        "sub_rule, total_samples, wins, posterior_wr, cumulative_pnl, multiplier, is_muted, updated_at"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (sub_rule, int(n), int(wins), post_wr, cum_pnl, float(multiplier), 1 if is_muted else 0, now_ms),
                    )
                    stats_out[sub_rule] = {
                        "sub_rule": sub_rule, "total_samples": int(n), "wins": int(wins),
                        "posterior_wr": post_wr, "cumulative_pnl": cum_pnl,
                        "multiplier": float(multiplier), "is_muted": bool(is_muted), "updated_at": now_ms,
                    }
                conn.commit()
        except Exception as e:
            logger.error(f"Erro ao recalibrar sub-regras estruturais: {e}")
        return stats_out

    def get_sub_rule_multipliers(self) -> Dict[str, float]:
        """Retorna mapa `{sub_rule: float(multiplier)}` persistido no SQLite (vazio se tabela vazia)."""
        mults: Dict[str, float] = {}
        try:
            with self._get_connection() as conn:
                rows = conn.execute("SELECT sub_rule, multiplier FROM structural_sub_rule_calibration").fetchall()
                for row in rows:
                    mults[str(row["sub_rule"])] = float(row["multiplier"])
        except Exception as e:
            logger.debug(f"Aviso ao buscar multiplicadores de sub-regra: {e}")
        return mults

    def _bootstrap_structural_sub_rules_if_empty(self) -> None:
        """Auto-calibra `structural_sub_rule_calibration` no startup se vazia e houver histórico WIN/LOSS."""
        needs_bootstrap = False
        try:
            with closing(self._get_connection()) as conn:
                has_cal = conn.execute(
                    "SELECT 1 FROM structural_sub_rule_calibration LIMIT 1"
                ).fetchone()
                if has_cal is not None:
                    return
                has_closed = conn.execute(
                    "SELECT 1 FROM agent_learning_memory WHERE outcome IN ('WIN', 'LOSS') LIMIT 1"
                ).fetchone()
                needs_bootstrap = has_closed is not None
        except Exception as e:
            logger.warning(f"Aviso ao verificar bootstrap de sub-regras estruturais: {e}")
            return

        if needs_bootstrap:
            self.recalibrate_structural_sub_rules()

    def retrain_daily_models(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        max_candles: int = 50000,
        model_type: str = "lgbm",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Executa ciclo diário de retreinamento e verificação Champion-Challenger para o modelo especificado
        usando dados novos acumulados no SQLite (BTCUSDT 1h, max_candles=50000).
        """
        mem = ModelEvolutionManager(db_path=self.db_path)
        return mem.run_retraining_cycle(
            symbol=symbol,
            timeframe=timeframe,
            max_candles=max_candles,
            model_type=model_type,
            **kwargs,
        )


def retrain_daily_models(
    symbol: str = "BTCUSDT",
    timeframe: str = "1h",
    max_candles: int = 50000,
    model_type: str = "lgbm",
    db_path: str = "datasets/market_cache.db",
    **kwargs,
) -> Dict[str, Any]:
    """Função de conveniência no módulo para execução direta do ciclo diário de retreinamento."""
    mem = ModelEvolutionManager(db_path=db_path)
    return mem.run_retraining_cycle(
        symbol=symbol,
        timeframe=timeframe,
        max_candles=max_candles,
        model_type=model_type,
        **kwargs,
    )


class ModelEvolutionManager:
    """
    Gerenciador de Evolução e Retreinamento Contínuo dos Modelos de Machine Learning.
    Opera sob o protocolo Champion-Challenger:
    - Retreina o modelo LightGBM quando novos dados se acumulam.
    - Valida o desafiante em janela out-of-sample estrita.
    - Promove o modelo desafiante a Champion apenas se comprovadamente superior.
    """

    def __init__(
        self,
        db_path: str = "datasets/market_cache.db",
        model_dir: str = "models",
        retrain_cadence_candles: int = 1000,
        min_improvement_pct: float = 0.5,     # Exige ao menos +0.5% de melhoria em AUC/Accuracy
    ):
        self.db_path = db_path
        self.model_dir = model_dir
        self.retrain_cadence_candles = retrain_cadence_candles
        self.min_improvement_pct = min_improvement_pct
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "archive"), exist_ok=True)

    def should_trigger_retraining(self, last_trained_timestamp: int, current_timestamp: int, new_candles_count: int) -> bool:
        """Determina se as condições para re-treinamento foram satisfeitas."""
        if new_candles_count >= self.retrain_cadence_candles:
            return True
        # Se mais de 14 dias se passaram desde o último treinamento
        elapsed_days = (current_timestamp - last_trained_timestamp) / (1000 * 86400)
        return elapsed_days >= 14.0

    def evaluate_challenger(
        self,
        champion_metric: float,
        challenger_metric: float,
        metric_name: str = "roc_auc"
    ) -> Tuple[bool, str]:
        """
        Compara o modelo Champion contra o Challenger.
        Retorna (should_promote, rationale).
        """
        delta = challenger_metric - champion_metric
        delta_pct = (delta / (champion_metric + 1e-12)) * 100.0

        if delta_pct >= self.min_improvement_pct:
            return True, f"PROMOÇÃO APROVADA: Challenger superou Champion em {delta_pct:+.2f}% na métrica {metric_name} ({champion_metric:.4f} -> {challenger_metric:.4f})."
        else:
            return False, f"PROMOÇÃO REJEITADA: Challenger obteve {delta_pct:+.2f}% de variação (abaixo da melhoria mínima de +{self.min_improvement_pct}%). Champion mantido em produção."

    @staticmethod
    def _extract_finite_metric(metrics: Optional[Dict[str, Any]], key: str) -> Optional[float]:
        if not isinstance(metrics, dict) or key not in metrics or metrics[key] is None:
            return None
        try:
            val = float(metrics[key])
        except (TypeError, ValueError):
            return None
        return val if np.isfinite(val) else None

    def evaluate_challenger_joint(
        self,
        champion_metrics: Dict[str, Any],
        challenger_metrics: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """
        Gate conjunto Champion-Challenger orientado a calibração probabilística e discriminação OOS:
        - Se `log_loss > 0` finito e `roc_auc_ovr` finito existirem em ambos:
          * Rejeita se `ch_ll > cp_ll + 0.002` ou `ch_auc < cp_auc - 0.005`;
          * Aprova se `(cp_ll - ch_ll) >= 0.001` ou `(ch_auc - cp_auc) >= 0.005` (sem regressão fora da tolerância);
          * Caso contrário, rejeita por melhoria insuficiente.
        - Fallback limpo para `evaluate_challenger` sobre `emitted_signals_accuracy` quando métricas conjuntas ausentes.
        """
        cp_ll = self._extract_finite_metric(champion_metrics, "log_loss")
        ch_ll = self._extract_finite_metric(challenger_metrics, "log_loss")
        cp_auc = self._extract_finite_metric(champion_metrics, "roc_auc_ovr")
        ch_auc = self._extract_finite_metric(challenger_metrics, "roc_auc_ovr")

        if (
            cp_ll is not None
            and ch_ll is not None
            and cp_ll > 0.0
            and ch_ll > 0.0
            and cp_auc is not None
            and ch_auc is not None
        ):
            eps = 1e-12
            if (ch_ll - cp_ll) > (0.002 + eps) or (cp_auc - ch_auc) > (0.005 + eps):
                return (
                    False,
                    f"PROMOÇÃO REJEITADA: regressão conjunta detectada "
                    f"(log_loss {cp_ll:.4f} -> {ch_ll:.4f}, roc_auc_ovr {cp_auc:.4f} -> {ch_auc:.4f}).",
                )
            if (cp_ll - ch_ll) >= (0.001 - eps) or (ch_auc - cp_auc) >= (0.005 - eps):
                return (
                    True,
                    f"PROMOÇÃO APROVADA: melhoria conjunta OOS validada "
                    f"(log_loss {cp_ll:.4f} -> {ch_ll:.4f}, roc_auc_ovr {cp_auc:.4f} -> {ch_auc:.4f}).",
                )
            return (
                False,
                f"PROMOÇÃO REJEITADA: melhoria conjunta insuficiente "
                f"(log_loss {cp_ll:.4f} -> {ch_ll:.4f}, roc_auc_ovr {cp_auc:.4f} -> {ch_auc:.4f}).",
            )

        cp_acc = self._extract_finite_metric(champion_metrics, "emitted_signals_accuracy") or 0.0
        ch_acc = self._extract_finite_metric(challenger_metrics, "emitted_signals_accuracy") or 0.0
        if ch_acc <= 0.0:
            return False, "PROMOÇÃO REJEITADA: challenger sem acurácia positiva válida."
        eff_cp_acc = cp_acc if cp_acc > 0.0 else ch_acc * 0.95
        return self.evaluate_challenger(
            champion_metric=eff_cp_acc,
            challenger_metric=ch_acc,
            metric_name="emitted_signals_accuracy",
        )

    @staticmethod
    def _load_symbol_candles_from_conn(
        conn: sqlite3.Connection,
        sym: str,
        timeframe: str,
        max_candles: int,
    ) -> pd.DataFrame:
        """Carrega candles de `sym` tolerando esquemas sintéticos sem `quote_volume` ou `exchange`."""
        tbl_info = conn.execute("PRAGMA table_info(candles)").fetchall()
        if not tbl_info:
            return pd.DataFrame()
        col_names = {str(r[1]) for r in tbl_info}
        sel_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        if "quote_volume" in col_names:
            sel_cols.append("quote_volume")
        cols_sql = ", ".join(sel_cols)

        if "exchange" in col_names:
            df_sym = pd.read_sql(
                f"SELECT {cols_sql} FROM candles "
                "WHERE symbol = ? AND timeframe = ? AND (exchange = 'binance' OR exchange IS NULL) "
                "ORDER BY timestamp DESC LIMIT ?",
                conn,
                params=(sym, timeframe, int(max_candles)),
            )
            if len(df_sym) < min(500, int(max_candles)):
                df_sym = pd.read_sql(
                    f"SELECT {cols_sql} FROM candles "
                    "WHERE symbol = ? AND timeframe = ? ORDER BY timestamp DESC LIMIT ?",
                    conn,
                    params=(sym, timeframe, int(max_candles)),
                )
        else:
            df_sym = pd.read_sql(
                f"SELECT {cols_sql} FROM candles "
                "WHERE symbol = ? AND timeframe = ? ORDER BY timestamp DESC LIMIT ?",
                conn,
                params=(sym, timeframe, int(max_candles)),
            )

        if df_sym.empty:
            return df_sym
        return df_sym.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    def run_retraining_cycle(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        max_candles: int = 50000,
        model_type: str = "lgbm",
        horizon: int = 5,
        multiple_tp: float = 1.2,
        multiple_sl: float = 1.2,
        confidence_threshold: float = 0.50,
        force_promote: bool = False,
        symbols: Optional[Tuple[str, ...]] = None,
        calibration_method: str = "platt",
    ) -> Dict[str, Any]:
        """
        Executa uma rodada completa de treinamento Walk-Forward + avaliação Champion-Challenger
        usando os candles acumulados no SQLite local (custo zero de API), suportando painel
        multi-símbolo F0_TechOnly com isolamento de rótulos Triple-Barrier por ativo.
        """
        import shutil
        from datetime import datetime, timezone
        from data.feature_engineering import FeatureEngineer
        from model_trainer import ModelTrainer

        model_type_norm = model_type.lower()
        cal_method_norm = (calibration_method or "none").lower()
        filename = f"{model_type}_model_v1.joblib"
        filepath = os.path.join(self.model_dir, filename)
        champ_metrics: Dict[str, Any] = {}
        champion_acc = 0.0
        champion_global = 0.0
        if os.path.exists(filepath):
            try:
                champ_payload = ModelTrainer.load_model(filepath)
                if isinstance(champ_payload, dict):
                    fp = champ_payload.get("feature_pipeline")
                    sv = champ_payload.get("schema_version")
                    if fp == "ml_features_v2" or (sv is not None and str(sv).startswith("2.")):
                        return {
                            "status": "SKIPPED_V2_GUARD",
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "model_type": model_type,
                            "rationale": (
                                f"Retreinamento v1 bloqueado (SKIPPED_V2_GUARD): Champion atual ({filename}) "
                                f"opera com schema v2 (feature_pipeline={fp!r}, schema_version={sv!r})."
                            ),
                        }
                    loaded_metrics = champ_payload.get("metrics", {})
                    if isinstance(loaded_metrics, dict):
                        champ_metrics = dict(loaded_metrics)
                        champion_acc = float(champ_metrics.get("emitted_signals_accuracy", 0.0))
                        champion_global = float(champ_metrics.get("global_accuracy", 0.0))
            except Exception as exc:
                logger.warning(f"Aviso ao ler Champion atual ({filepath}): {exc}")

        if symbols is not None:
            candidate_symbols = tuple(dict.fromkeys(str(s) for s in symbols if s))
        elif symbol == "BTCUSDT":
            candidate_symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
        else:
            candidate_symbols = (symbol,)

        symbol_raw_dfs: Dict[str, pd.DataFrame] = {}
        with closing(_db_connect(self.db_path, timeout=15.0)) as conn:
            with conn:
                for sym in candidate_symbols:
                    df_sym = self._load_symbol_candles_from_conn(conn, sym, timeframe, int(max_candles))
                    if len(df_sym) >= 500:
                        symbol_raw_dfs[sym] = df_sym

        if not symbol_raw_dfs:
            return {
                "status": "SKIPPED",
                "reason": f"Candles insuficientes (< 500) para {candidate_symbols} {timeframe}.",
            }

        symbols_used = list(symbol_raw_dfs.keys())
        fe = FeatureEngineer()
        per_symbol_feats: Dict[str, pd.DataFrame] = {}
        for sym in symbols_used:
            # F0_TechOnly estrito: sem merge macro/on-chain
            df_feat = fe.compute_all_features(symbol_raw_dfs[sym], drop_na=False)
            # Restaura NaN nativo onde não há BSL/SSL ativo (evitando falso 0.0% de distância S/R)
            if "nearest_bsl" in df_feat.columns and "dist_resistance_pct" in df_feat.columns:
                no_bsl = df_feat["nearest_bsl"].isna() | (df_feat["nearest_bsl"] <= 0.0)
                df_feat.loc[no_bsl, "dist_resistance_pct"] = np.nan
            if "nearest_ssl" in df_feat.columns and "dist_support_pct" in df_feat.columns:
                no_ssl = df_feat["nearest_ssl"].isna() | (df_feat["nearest_ssl"] <= 0.0)
                df_feat.loc[no_ssl, "dist_support_pct"] = np.nan
            if len(df_feat) > 210:
                df_feat = df_feat.iloc[210:].reset_index(drop=True)
            per_symbol_feats[sym] = df_feat

        # Seleciona apenas features estacionárias/normalizadas para evitar overfitting de nível de preço
        non_stationary = {
            "timestamp", "open", "high", "low", "close", "volume", "quote_volume",
            "sma_20", "sma_50", "sma_200", "ema_9", "ema_21", "bb_high", "bb_mid", "bb_low",
            "obv", "vol_sma_20", "atr_14", "nearest_bsl", "nearest_ssl",
        }
        structural_optional_nan = {"dist_resistance_pct", "dist_support_pct"}
        ref_df = per_symbol_feats[symbols_used[0]]
        feature_cols = [
            c for c in ref_df.select_dtypes(include=["number"]).columns
            if c not in non_stationary
            and all(c in d.columns for d in per_symbol_feats.values())
            and (
                c in structural_optional_nan
                or all(d[c].notna().mean() > 0.95 for d in per_symbol_feats.values())
            )
        ]

        trainer = ModelTrainer(
            model_type=model_type,
            horizon=horizon,
            multiple_tp=multiple_tp,
            multiple_sl=multiple_sl,
            confidence_threshold=confidence_threshold,
            models_dir=self.model_dir,
            calibration_method=cal_method_norm,
        )

        is_lgbm = (model_type_norm == "lgbm")
        base_cols = ["timestamp", "open", "high", "low", "close", "atr_14"]
        X_parts: List[pd.DataFrame] = []
        y_parts: List[pd.Series] = []
        ts_parts: List[np.ndarray] = []

        for sym in symbols_used:
            df_sub = per_symbol_feats[sym][base_cols + feature_cols].copy()
            df_sub = df_sub.replace([np.inf, -np.inf], np.nan)
            if is_lgbm:
                fill_zero_cols = [c for c in df_sub.columns if c not in structural_optional_nan]
                df_sub[fill_zero_cols] = df_sub[fill_zero_cols].fillna(0.0)
            else:
                df_sub = df_sub.fillna(0.0)

            # Gera rótulos Triple-Barrier isoladamente por símbolo (zero contaminação cruzada entre ativos)
            X_sym, y_sym = trainer.prepare_dataset(df_sub, feature_columns=feature_cols)
            ts_sym = df_sub.loc[X_sym.index, "timestamp"].to_numpy(dtype=np.int64)
            X_parts.append(X_sym)
            y_parts.append(y_sym)
            ts_parts.append(ts_sym)

        X_cat = pd.concat(X_parts, axis=0, ignore_index=True)
        y_cat = pd.concat(y_parts, axis=0, ignore_index=True)
        ts_cat = np.concatenate(ts_parts)

        sort_order = np.argsort(ts_cat, kind="mergesort")
        X = X_cat.iloc[sort_order].reset_index(drop=True)
        y = y_cat.iloc[sort_order].reset_index(drop=True)
        ts_arr = ts_cat[sort_order]

        wf_res = trainer.train_walk_forward(X, y, n_splits=4, timestamps=ts_arr)

        challenger_acc = float(wf_res.get("emitted_signals_accuracy", 0.0))
        challenger_global = float(wf_res.get("global_accuracy", 0.0))
        challenger_f1 = float(wf_res.get("f1_macro", 0.0))
        challenger_ll = float(wf_res.get("log_loss", 0.0))
        challenger_auc = float(wf_res.get("roc_auc_ovr", 0.0))
        challenger_ece = float(wf_res.get("ece", 0.0))

        if force_promote:
            promoted = True
            rationale = (
                f"PROMOÇÃO FORÇADA (Saneamento de Base Desduplicada): Challenger promovido "
                f"(acc={challenger_acc:.4f}, global={challenger_global:.4f}, "
                f"log_loss={challenger_ll:.4f}, roc_auc_ovr={challenger_auc:.4f})."
            )
        else:
            promoted, rationale = self.evaluate_challenger_joint(
                champion_metrics=champ_metrics,
                challenger_metrics=wf_res,
            )

        if promoted:
            if os.path.exists(filepath):
                ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                arch_path = os.path.join(self.model_dir, "archive", f"{model_type}_model_v1_{ts_tag}.joblib")
                shutil.copy2(filepath, arch_path)
            trainer.train_full_model(
                X,
                y,
                save_name=filename,
                timestamps=ts_arr,
                extra_metadata={
                    "symbols_used": list(symbols_used),
                    "calibration_method": cal_method_norm,
                },
            )
            try:
                from agents.technical_agent import TechnicalAgent
                TechnicalAgent.invalidate_default_model_cache()
            except Exception as exc:
                logger.debug(f"Aviso ao invalidar cache de modelo do TechnicalAgent: {exc}")

        return {
            "status": "PROMOTED" if promoted else "RETAINED",
            "symbol": symbol,
            "symbols_used": list(symbols_used),
            "timeframe": timeframe,
            "model_type": model_type,
            "calibration_method": cal_method_norm,
            "samples_evaluated": int(wf_res.get("samples_evaluated", 0)),
            "features_count": len(feature_cols),
            "champion_emitted_accuracy": round(champion_acc, 4),
            "challenger_emitted_accuracy": round(challenger_acc, 4),
            "champion_global_accuracy": round(champion_global, 4),
            "challenger_global_accuracy": round(challenger_global, 4),
            "challenger_f1_macro": round(challenger_f1, 4),
            "challenger_log_loss": round(challenger_ll, 6),
            "challenger_roc_auc_ovr": round(challenger_auc, 6),
            "challenger_ece": round(challenger_ece, 6),
            "emitted_signals_count": int(wf_res.get("emitted_signals_count", 0)),
            "rationale": rationale,
        }
