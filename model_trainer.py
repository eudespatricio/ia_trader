"""
Módulo de Treinamento e Validação de Modelos Preditivos (Fase 3).
Implementa:
1. Rotulagem (Triple-Barrier Simplificada) estritamente SEM lookahead bias:
   - Alvo baseado no retorno futuro em N velas (horizonte configurável).
   - Barreiras de Take Profit (TP) e Stop Loss (SL) proporcionais ao ATR local.
   - Descarte estrito das últimas N velas (sem horizonte completo disponível).
2. Modelos Baseline: Random Forest e LightGBM.
3. Detecção em runtime de suporte a GPU/CUDA com fallback automático e resiliente para CPU (i7-11800H, n_jobs≈12).
4. Validação temporal estrita: Walk-Forward com gap de embargo/purge igual ao horizonte de rotulagem.
5. Ponderação de classes (class_weight='balanced') para tratar desbalanceamento.
6. Métricas out-of-sample: acurácia, precisão, F1-macro e taxa de acerto SOMENTE nos sinais emitidos (com cobertura).
7. Persistência e reuso de modelos com joblib em /models acompanhado de metadados completos.
"""

from __future__ import annotations

import os
import sys
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

# Setup de logger
logger = logging.getLogger("ia_trader.model_trainer")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [model_trainer] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =====================================================================
# 1. HARDWARE & DEVICE RESOLUTION
# =====================================================================

def resolve_compute_device(requested_device: str = "auto", n_jobs: int = 12) -> Tuple[str, Dict[str, Any]]:
    """
    Detecta dinamicamente a disponibilidade de aceleração por hardware (GPU/CUDA).
    Se o build do LightGBM ou drivers não suportarem CUDA/OpenCL, aplica fallback
    automático para CPU multinúcleo (i7-11800H, n_jobs≈12).
    """
    device_info = {
        "device": "cpu",
        "n_jobs": n_jobs,
        "gpu_supported": False,
        "detail": "CPU (i7-11800H / Multi-thread)"
    }

    if not HAS_LIGHTGBM or requested_device.lower() == "cpu":
        logger.info(f"Dispositivo selecionado para treinamento: CPU (n_jobs={n_jobs}).")
        return "cpu", device_info

    # Teste em runtime para verificar suporte a CUDA ou GPU no LightGBM
    for candidate in ["cuda", "gpu"]:
        if requested_device.lower() in ["auto", candidate]:
            try:
                test_clf = lgb.LGBMClassifier(
                    device=candidate,
                    n_estimators=2,
                    max_depth=2,
                    verbosity=-1,
                )
                X_dummy = np.random.randn(20, 4)
                y_dummy = np.array([0, 1] * 10)
                test_clf.fit(X_dummy, y_dummy)

                device_info["device"] = candidate
                device_info["gpu_supported"] = True
                device_info["detail"] = f"Aceleração {candidate.upper()} ativa"
                logger.info(f"Dispositivo selecionado para treinamento: {candidate.upper()} com sucesso.")
                return candidate, device_info
            except Exception as exc:
                logger.debug(f"LightGBM com device='{candidate}' indisponível: {exc}")

    logger.info(f"Fallback automático para CPU multinúcleo (n_jobs={n_jobs}).")
    return "cpu", device_info


# =====================================================================
# 2. ROTULAGEM TRIPLE-BARRIER SEM LOOKAHEAD
# =====================================================================

def create_triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    multiple_tp: float = 1.5,
    multiple_sl: float = 1.0,
    atr_col: str = "atr_14",
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.Series:
    """
    Rotulagem por Barreira Tripla Simplificada (Triple-Barrier):
    - Alvo: Previsão do movimento futuro nas próximas N velas (horizonte).
    - Barreira Superior (TP) = Close_t + multiple_tp * ATR_t
    - Barreira Inferior (SL) = Close_t - multiple_sl * ATR_t
    - Classes:
      * 1 (BUY): Atinge Barreira Superior antes da Inferior.
      * 2 (SELL): Atinge Barreira Inferior antes da Superior.
      * 0 (HOLD): Nenhuma barreira tocada até o término do horizonte N.
    - RIGOR ANTI-LOOKAHEAD:
      * As últimas N linhas da série temporal NÃO possuem horizonte futuro completo.
        Elas são rotuladas estritamente com np.nan para posterior descarte.
    """
    n_rows = len(df)
    labels = np.full(n_rows, np.nan, dtype=np.float64)

    closes = df[close_col].values
    highs = df[high_col].values
    lows = df[low_col].values
    atrs = df[atr_col].values if atr_col in df.columns else (closes * 0.01)

    for i in range(n_rows - horizon):
        p0 = closes[i]
        atr = atrs[i]
        if np.isnan(atr) or atr <= 0.0:
            atr = p0 * 0.01

        upper_barrier = p0 + (multiple_tp * atr)
        lower_barrier = p0 - (multiple_sl * atr)

        hit_upper_idx = None
        hit_lower_idx = None

        for k in range(i + 1, i + horizon + 1):
            if highs[k] >= upper_barrier and hit_upper_idx is None:
                hit_upper_idx = k
            if lows[k] <= lower_barrier and hit_lower_idx is None:
                hit_lower_idx = k

            if hit_upper_idx is not None and hit_lower_idx is not None:
                break

        if hit_upper_idx is not None and hit_lower_idx is None:
            labels[i] = 1.0  # BUY
        elif hit_lower_idx is not None and hit_upper_idx is None:
            labels[i] = 2.0  # SELL
        elif hit_upper_idx is not None and hit_lower_idx is not None:
            if hit_upper_idx < hit_lower_idx:
                labels[i] = 1.0  # BUY primeiro
            elif hit_lower_idx < hit_upper_idx:
                labels[i] = 2.0  # SELL primeiro
            else:
                labels[i] = 0.0  # Ambas tocadas na mesma vela -> HOLD
        else:
            labels[i] = 0.0  # Nenhuma barreira atingida -> HOLD

    return pd.Series(labels, index=df.index, name="target")


# =====================================================================
# 3. VALIDAÇÃO TEMPORAL PURGADA COM EMBARGO
# =====================================================================

class PurgedTimeSeriesSplit:
    """
    Gerador de splits temporais com purge/embargo para evitar contaminação
    entre conjuntos de treino e teste causada pelo horizonte futuro dos rótulos.
    """

    def __init__(self, n_splits: int = 4, horizon_embargo: int = 5, min_train_ratio: float = 0.4):
        self.n_splits = n_splits
        self.horizon_embargo = horizon_embargo
        self.min_train_ratio = min_train_ratio

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        indices = np.arange(n_samples)

        min_train_samples = int(n_samples * self.min_train_ratio)
        remaining = n_samples - min_train_samples
        test_size = remaining // self.n_splits

        splits = []
        for i in range(self.n_splits):
            train_end = min_train_samples + (i * test_size)
            # Aplica o embargo (gap temporal) para anular contaminação do rótulo
            test_start = train_end + self.horizon_embargo
            test_end = min_train_samples + ((i + 1) * test_size)

            if test_start >= test_end or test_end > n_samples:
                continue

            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]
            splits.append((train_idx, test_idx))

        return splits


# =====================================================================
# 4. MOTOR ORQUESTRADOR: ModelTrainer
# =====================================================================

class ModelTrainer:
    """
    Gerenciador de treinamento, calibração e validação de modelos preditivos.
    """

    DEFAULT_MODELS_DIR = "/home/eudespatricio/ia_trader/models"

    def __init__(
        self,
        model_type: str = "lgbm",
        horizon: int = 5,
        multiple_tp: float = 1.5,
        multiple_sl: float = 1.0,
        confidence_threshold: float = 0.55,
        n_jobs: int = 12,
        models_dir: Optional[str] = None,
    ):
        self.model_type = model_type.lower()
        self.horizon = horizon
        self.multiple_tp = multiple_tp
        self.multiple_sl = multiple_sl
        self.confidence_threshold = confidence_threshold
        self.n_jobs = n_jobs
        self.models_dir = models_dir or self.DEFAULT_MODELS_DIR
        os.makedirs(self.models_dir, exist_ok=True)

        self.device, self.device_info = resolve_compute_device(requested_device="auto", n_jobs=self.n_jobs)

    def prepare_dataset(
        self,
        df_features: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara a matriz de features X e o vetor alvo y:
        1. Gera os rótulos Triple-Barrier sem lookahead.
        2. Remove estritamente as últimas N velas (rótulos NaN).
        3. Isola apenas as colunas preditoras numéricas.
        """
        df = df_features.copy()

        # Adiciona rótulos futuros
        df["target"] = create_triple_barrier_labels(
            df=df,
            horizon=self.horizon,
            multiple_tp=self.multiple_tp,
            multiple_sl=self.multiple_sl,
        )

        # Remove as linhas sem rótulo futuro (últimas N velas)
        df_clean = df.dropna(subset=["target"]).copy()
        df_clean["target"] = df_clean["target"].astype(int)

        # Seleção de colunas de features
        if feature_columns is not None:
            feature_cols = [c for c in feature_columns if c in df_clean.columns]
        else:
            num_cols = df_clean.select_dtypes(include=[np.number]).columns
            exclude_cols = {"timestamp", "target"}
            feature_cols = [c for c in num_cols if c not in exclude_cols]

        X = df_clean[feature_cols].copy()
        y = df_clean["target"].copy()

        return X, y

    def _build_estimator(self, random_state: int = 42) -> Any:
        """Instancia o modelo preditivo configurado."""
        if self.model_type == "lgbm" and HAS_LIGHTGBM:
            return lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.03,
                max_depth=5,
                num_leaves=31,
                class_weight="balanced",
                device=self.device,
                n_jobs=self.n_jobs if self.device == "cpu" else -1,
                random_state=random_state,
                verbosity=-1,
            )
        else:
            return RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                min_samples_split=10,
                class_weight="balanced",
                n_jobs=self.n_jobs,
                random_state=random_state,
            )

    def train_walk_forward(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 4,
    ) -> Dict[str, Any]:
        """
        Validação Walk-Forward temporal estrita com embargo anti-contaminação.
        Calcula métricas agregadas out-of-sample honestas.
        """
        splitter = PurgedTimeSeriesSplit(n_splits=n_splits, horizon_embargo=self.horizon)
        splits = splitter.split(X, y)

        if not splits:
            raise ValueError("Amostras insuficientes para gerar splits temporais com embargo.")

        all_test_indices: List[int] = []
        all_y_true: List[int] = []
        all_y_pred: List[int] = []
        all_y_proba: List[np.ndarray] = []

        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            # ASSERT ANTI-VAZAMENTO: Prova matemática que não há interseção e o embargo foi respeitado
            assert len(set(train_idx).intersection(set(test_idx))) == 0, (
                f"Vazamento detectado no Fold {fold+1}: interseção não-nula entre treino e teste!"
            )
            assert min(test_idx) >= max(train_idx) + self.horizon, (
                f"Vazamento detectado no Fold {fold+1}: gap ({min(test_idx) - max(train_idx)}) "
                f"inferior ao horizonte de embargo ({self.horizon})!"
            )

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            # Valida classes presentes no treino
            if len(np.unique(y_train)) < 2:
                continue

            model = self._build_estimator(random_state=42 + fold)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            all_test_indices.extend(test_idx.tolist())
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(y_pred.tolist())
            all_y_proba.append(y_proba)

            fold_acc = float(accuracy_score(y_test, y_pred))
            fold_metrics.append({"fold": fold + 1, "accuracy": fold_acc, "train_size": len(train_idx), "test_size": len(test_idx)})

        if not all_y_true:
            raise RuntimeError("Falha na validação walk-forward: nenhum fold completado.")

        y_true_arr = np.array(all_y_true)
        y_pred_arr = np.array(all_y_pred)
        y_proba_arr = np.vstack(all_y_proba)

        # Classes inferidas
        classes = getattr(model, "classes_", np.array([0, 1, 2]))

        # Métricas globais
        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        f1_mac = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

        # Precision por classe: 0 (HOLD), 1 (BUY), 2 (SELL)
        precisions = precision_score(y_true_arr, y_pred_arr, average=None, zero_division=0)
        class_precisions = {f"class_{cls}_precision": float(prec) for cls, prec in zip(classes, precisions)}

        # =====================================================================
        # MÉTRICA CRÍTICA: Taxa de acerto SOMENTE nos sinais emitidos
        # =====================================================================
        # Encontra a probabilidade máxima prevista para classes ativas (BUY=1 ou SELL=2)
        buy_col_idx = np.where(classes == 1)[0]
        sell_col_idx = np.where(classes == 2)[0]

        prob_buy = y_proba_arr[:, buy_col_idx[0]] if len(buy_col_idx) > 0 else np.zeros(len(y_true_arr))
        prob_sell = y_proba_arr[:, sell_col_idx[0]] if len(sell_col_idx) > 0 else np.zeros(len(y_true_arr))

        active_signals_mask = (prob_buy >= self.confidence_threshold) | (prob_sell >= self.confidence_threshold)
        total_signals = int(np.sum(active_signals_mask))
        coverage_pct = float(total_signals / len(y_true_arr)) if len(y_true_arr) > 0 else 0.0

        if total_signals > 0:
            active_y_true = y_true_arr[active_signals_mask]
            # Sinal emitido: 1 se prob_buy >= prob_sell senão 2
            emitted_signals = np.where(prob_buy[active_signals_mask] >= prob_sell[active_signals_mask], 1, 2)
            emitted_accuracy = float(np.mean(active_y_true == emitted_signals))
        else:
            emitted_accuracy = 0.0

        results = {
            "model_type": self.model_type,
            "device": self.device,
            "samples_evaluated": len(y_true_arr),
            "global_accuracy": acc,
            "f1_macro": f1_mac,
            "f1_weighted": f1_weighted,
            "class_precisions": class_precisions,
            "emitted_signals_accuracy": emitted_accuracy,
            "emitted_signals_count": total_signals,
            "coverage_pct": coverage_pct,
            "confidence_threshold": self.confidence_threshold,
            "fold_details": fold_metrics,
            "test_indices": all_test_indices,
            "y_true": y_true_arr,
            "y_pred": y_pred_arr,
            "y_proba": y_proba_arr,
        }

        logger.info(
            f"Resultados Walk-Forward [{self.model_type.upper()}]: "
            f"Global Acc={acc:.2%}, Sinais Emitidos Acc={emitted_accuracy:.2%}, "
            f"Cobertura={coverage_pct:.2%} ({total_signals}/{len(y_true_arr)} trades)."
        )

        return results

    def train_full_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Treina o estimador com todo o conjunto histórico disponível,
        avalia walk-forward para documentação e persiste o artefato com joblib.
        """
        # 1. Avaliação out-of-sample formal
        eval_metrics = self.train_walk_forward(X, y)

        # 2. Treino final na base completa
        final_model = self._build_estimator(random_state=42)
        final_model.fit(X, y)

        # 3. Metadados e Persistência
        filename = save_name or f"{self.model_type}_model_v1.joblib"
        filepath = os.path.join(self.models_dir, filename)

        payload = {
            "model": final_model,
            "model_type": self.model_type,
            "features": list(X.columns),
            "feature_names": list(X.columns),
            "horizon": self.horizon,
            "multiple_tp": self.multiple_tp,
            "multiple_sl": self.multiple_sl,
            "confidence_threshold": self.confidence_threshold,
            "device_info": self.device_info,
            "metrics": eval_metrics,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
        }

        joblib.dump(payload, filepath, compress=3)
        logger.info(f"Modelo salvo com sucesso em: {filepath}")

        return payload

    @classmethod
    def load_model(cls, filepath: str) -> Dict[str, Any]:
        """Carrega modelo persistido e seus metadados estruturados."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {filepath}")
        return joblib.load(filepath)
