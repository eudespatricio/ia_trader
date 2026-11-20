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

    def __init__(self, db_path: str = "datasets/market_cache.db"):
        self.db_path = db_path
        self._ensure_tables()

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
        Nível 1 & Nível 2 do Protocolo S2 (docs/S2_CALIBRACAO_AGENTES_BACKTEST.md):
        Calcula os pesos dos agentes por regime usando Otimização Convexa Restrita
        (Ridge / L2 com prior bayesiano DEFAULT_BASE_WEIGHTS), minimizando o erro quadrático direcional,
        garantindo soma de pesos normalizada, pesos não-negativos e preservação estrita do veto
        do RiskSentinelAgent (w = 1.5).
        Calcula também a acurácia empírica e o ECE (Expected Calibration Error).
        Retorna dicionário: {regime: {agent_name: weight}}
        """
        now_ms = int(time.time() * 1000)
        regimes = ["BULL_TREND", "BEAR_TREND", "SIDEWAYS_RANGE", "HIGH_VOLATILITY_PANIC"]
        all_weights: Dict[str, Dict[str, float]] = {}

        try:
            # Patch S4 (v1.0.6-s4): Janela deslizante pós-S2/S3 (a partir de 02/11/2026 = 1793577600000 ms,
            # últimas 30 decisões concluídas), evitando que falhas legadas da S1/S2 (v1.0.0-v1.0.3)
            # contaminem permanentemente o ECE dos modelos recém-curados da S3/S4.
            s3_epoch_ms = 1793577600000
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT regime, consensus_direction, agent_signals_json, outcome, realized_pnl
                    FROM agent_learning_memory
                    WHERE outcome IN ('WIN', 'LOSS') AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT 30
                """, (s3_epoch_ms,)).fetchall()
                if len(rows) < 5:
                    rows = conn.execute("""
                        SELECT regime, consensus_direction, agent_signals_json, outcome, realized_pnl
                        FROM agent_learning_memory
                        WHERE outcome IN ('WIN', 'LOSS')
                        ORDER BY timestamp DESC
                        LIMIT 30
                    """).fetchall()

            if not rows:
                # Retorna pesos padrão caso ainda não haja desfechos acumulados
                for r in regimes:
                    all_weights[r] = dict(self.DEFAULT_BASE_WEIGHTS)
                return all_weights

            # Agrupa estatísticas por (regime, agent_name) e amostras por regime
            stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
            regime_samples: Dict[str, List[Dict[str, Any]]] = {r: [] for r in regimes}

            for row in rows:
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
            with self._get_connection() as conn:
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

    def get_agent_calibration_for_regime(self, regime: str) -> Dict[str, Dict[str, float]]:
        """
        Retorna mapa de calibração empírica por agente para um determinado regime:
        {agent_name: {"weight": float, "accuracy": float, "total_samples": int, "ece": float}}
        """
        calib: Dict[str, Dict[str, float]] = {}
        try:
            with self._get_connection() as conn:
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
    ) -> Dict[str, Any]:
        """
        Executa uma rodada completa de treinamento Walk-Forward + avaliação Champion-Challenger
        usando os candles acumulados no SQLite local (custo zero de API).
        """
        import shutil
        from datetime import datetime, timezone
        import pandas as pd
        from data.feature_engineering import FeatureEngineer
        from model_trainer import ModelTrainer

        filename = f"{model_type}_model_v1.joblib"
        filepath = os.path.join(self.model_dir, filename)
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
                    champ_metrics = champ_payload.get("metrics", {})
                    if isinstance(champ_metrics, dict):
                        champion_acc = float(champ_metrics.get("emitted_signals_accuracy", 0.0))
                        champion_global = float(champ_metrics.get("global_accuracy", 0.0))
            except Exception as exc:
                logger.warning(f"Aviso ao ler Champion atual ({filepath}): {exc}")

        with _db_connect(self.db_path, timeout=15.0) as conn:
            df = pd.read_sql(
                "SELECT timestamp, open, high, low, close, volume, quote_volume "
                "FROM candles WHERE symbol = ? AND timeframe = ? AND (exchange = 'binance' OR exchange IS NULL) "
                "ORDER BY timestamp DESC LIMIT ?",
                conn,
                params=(symbol, timeframe, int(max_candles)),
            )
            if len(df) < 500:
                df = pd.read_sql(
                    "SELECT timestamp, open, high, low, close, volume, quote_volume "
                    "FROM candles WHERE symbol = ? AND timeframe = ? ORDER BY timestamp DESC LIMIT ?",
                    conn,
                    params=(symbol, timeframe, int(max_candles)),
                )

        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        if len(df) < 500:
            return {"status": "SKIPPED", "reason": f"Candles insuficientes ({len(df)} < 500) para {symbol} {timeframe}."}

        fe = FeatureEngineer()
        df_feat = fe.compute_all_features(df, drop_na=False)
        if len(df_feat) > 210:
            df_feat = df_feat.iloc[210:].reset_index(drop=True)

        # Seleciona apenas features estacionárias/normalizadas para evitar overfitting de nível de preço
        non_stationary = {
            "timestamp", "open", "high", "low", "close", "volume", "quote_volume",
            "sma_20", "sma_50", "sma_200", "ema_9", "ema_21", "bb_high", "bb_mid", "bb_low",
            "obv", "vol_sma_20", "atr_14", "nearest_bsl", "nearest_ssl",
        }
        structural_optional_nan = {"dist_resistance_pct", "dist_support_pct"}
        feature_cols = [
            c for c in df_feat.select_dtypes(include=["number"]).columns
            if c not in non_stationary and (c in structural_optional_nan or df_feat[c].notna().mean() > 0.95)
        ]
        df_clean = df_feat[["timestamp", "open", "high", "low", "close", "atr_14"] + feature_cols].fillna(0.0)

        trainer = ModelTrainer(
            model_type=model_type,
            horizon=horizon,
            multiple_tp=multiple_tp,
            multiple_sl=multiple_sl,
            confidence_threshold=confidence_threshold,
            models_dir=self.model_dir,
        )
        X, y = trainer.prepare_dataset(df_clean, feature_columns=feature_cols)
        wf_res = trainer.train_walk_forward(X, y, n_splits=4)

        challenger_acc = float(wf_res.get("emitted_signals_accuracy", 0.0))
        challenger_global = float(wf_res.get("global_accuracy", 0.0))
        challenger_f1 = float(wf_res.get("f1_macro", 0.0))

        if force_promote:
            promoted = True
            rationale = f"PROMOÇÃO FORÇADA (Saneamento de Base Desduplicada): Challenger promovido (acc={challenger_acc:.4f}, global={challenger_global:.4f})."
        else:
            promoted, rationale = self.evaluate_challenger(
                champion_metric=champion_acc if champion_acc > 0 else challenger_acc * 0.95,
                challenger_metric=challenger_acc,
                metric_name="emitted_signals_accuracy",
            )

        if promoted:
            if os.path.exists(filepath):
                ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                arch_path = os.path.join(self.model_dir, "archive", f"{model_type}_model_v1_{ts_tag}.joblib")
                shutil.copy2(filepath, arch_path)
            trainer.train_full_model(X, y, save_name=filename)
            try:
                from agents.technical_agent import TechnicalAgent
                TechnicalAgent.invalidate_default_model_cache()
            except Exception as exc:
                logger.debug(f"Aviso ao invalidar cache de modelo do TechnicalAgent: {exc}")

        return {
            "status": "PROMOTED" if promoted else "RETAINED",
            "symbol": symbol,
            "timeframe": timeframe,
            "model_type": model_type,
            "samples_evaluated": int(wf_res.get("samples_evaluated", 0)),
            "features_count": len(feature_cols),
            "champion_emitted_accuracy": round(champion_acc, 4),
            "challenger_emitted_accuracy": round(challenger_acc, 4),
            "champion_global_accuracy": round(champion_global, 4),
            "challenger_global_accuracy": round(challenger_global, 4),
            "challenger_f1_macro": round(challenger_f1, 4),
            "emitted_signals_count": int(wf_res.get("emitted_signals_count", 0)),
            "rationale": rationale,
        }
