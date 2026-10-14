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

import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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

    def __init__(self, db_path: str = "datasets/market_cache.db"):
        self.db_path = db_path
        self._ensure_tables()

    def _get_connection(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=15.0)
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
                
            # Dispara recalibração de pesos com a nova evidência
            self.recalibrate_agent_weights()
            logger.info(f"Desfecho {outcome} (${pnl:+.2f}) registrado para posição {position_id}. Pesos atualizados.")
        except Exception as e:
            logger.warning(f"Erro ao atualizar desfecho de trade no LearningTracker: {e}")

    def recalibrate_agent_weights(self) -> Dict[str, Dict[str, float]]:
        """
        Analisa o histórico de decisões e desfechos para recalibrar
        os pesos de voto de cada agente em cada regime de mercado.
        Retorna dicionário: {regime: {agent_name: weight}}
        """
        now_ms = int(time.time() * 1000)
        regimes = ["BULL_TREND", "BEAR_TREND", "SIDEWAYS_RANGE", "HIGH_VOLATILITY_PANIC"]
        all_weights: Dict[str, Dict[str, float]] = {}

        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT regime, consensus_direction, agent_signals_json, outcome, realized_pnl
                    FROM agent_learning_memory
                    WHERE outcome IN ('WIN', 'LOSS')
                """).fetchall()

            if not rows:
                # Retorna pesos padrão caso ainda não haja desfechos acumulados
                for r in regimes:
                    all_weights[r] = dict(self.DEFAULT_BASE_WEIGHTS)
                return all_weights

            # Agrupa estatísticas por (regime, agent_name)
            stats: Dict[Tuple[str, str], Dict[str, Any]] = {}

            for row in rows:
                regime = row["regime"]
                outcome = row["outcome"]
                try:
                    signals = json.loads(row["agent_signals_json"])
                except Exception:
                    continue

                for sig in signals:
                    name = sig.get("agent_name", "")
                    direction = sig.get("direction", "HOLD")
                    conf = float(sig.get("confidence", 0.5))

                    key = (regime, name)
                    if key not in stats:
                        stats[key] = {"total": 0, "correct": 0, "brier_sum": 0.0}

                    stats[key]["total"] += 1

                    # O agente acertou se votou na mesma direção que resultou em WIN,
                    # ou se votou na direção oposta ao que resultou em LOSS.
                    is_correct = False
                    if outcome == "WIN" and direction == row["consensus_direction"]:
                        is_correct = True
                    elif outcome == "LOSS" and direction != row["consensus_direction"] and direction != "HOLD":
                        is_correct = True

                    if is_correct:
                        stats[key]["correct"] += 1

                    # Brier score do voto direcional (p - 1)^2 se correto, p^2 se incorreto
                    target = 1.0 if is_correct else 0.0
                    brier = (conf - target) ** 2
                    stats[key]["brier_sum"] += brier

            # Atualiza pesos ponderados por performance empírica
            with self._get_connection() as conn:
                for r in regimes:
                    all_weights[r] = dict(self.DEFAULT_BASE_WEIGHTS)
                    for agent_name in self.DEFAULT_BASE_WEIGHTS:
                        key = (r, agent_name)
                        base_w = self.DEFAULT_BASE_WEIGHTS[agent_name]

                        if key in stats and stats[key]["total"] >= 5:
                            s = stats[key]
                            acc = s["correct"] / s["total"]
                            # Fator multiplicador baseado em vantagem sobre o acaso (50%)
                            # Ex: 70% acerto -> 1.0 + (0.7 - 0.5)*1.5 = 1.3x do peso base
                            # Ex: 30% acerto -> 1.0 + (0.3 - 0.5)*1.5 = 0.7x do peso base
                            multiplier = max(0.40, min(1.80, 1.0 + (acc - 0.50) * 1.5))
                            calibrated_w = round(base_w * multiplier, 3)
                            all_weights[r][agent_name] = calibrated_w

                            conn.execute("""
                                INSERT OR REPLACE INTO agent_regime_weights (
                                    regime, agent_name, weight, accuracy, total_samples, updated_at
                                ) VALUES (?, ?, ?, ?, ?, ?)
                            """, (r, agent_name, calibrated_w, acc, s["total"], now_ms))
                        else:
                            # Sem amostra suficiente, preserva peso base a priori
                            conn.execute("""
                                INSERT OR REPLACE INTO agent_regime_weights (
                                    regime, agent_name, weight, accuracy, total_samples, updated_at
                                ) VALUES (?, ?, ?, ?, ?, ?)
                            """, (r, agent_name, base_w, 0.50, 0, now_ms))

                conn.commit()

            return all_weights

        except Exception as e:
            logger.error(f"Erro ao recalibrar pesos dos agentes: {e}")
            for r in regimes:
                all_weights[r] = dict(self.DEFAULT_BASE_WEIGHTS)
            return all_weights

    def get_calibrated_weights_for_regime(self, regime: str) -> Dict[str, float]:
        """
        Retorna o dicionário de pesos atualizado para um determinado regime de mercado.
        Pode ser passado diretamente como `weights_override` no `ConsensusJudge`.
        """
        weights = dict(self.DEFAULT_BASE_WEIGHTS)
        try:
            with self._get_connection() as conn:
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
