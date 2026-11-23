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
import tempfile
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

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


def _atomic_joblib_dump(payload: Any, filepath: str, compress: int = 3) -> None:
    """Persist joblib artifact atomically via temporary file in destination dir + os.replace."""
    target = os.path.abspath(filepath)
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=parent, prefix=".tmp_", suffix=".joblib")
    os.close(fd)
    try:
        joblib.dump(payload, tmp_path, compress=compress)
        os.replace(tmp_path, target)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


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
    Suporta split por índice (série única) ou por timestamp real (painel multi-ativo).
    """

    def __init__(self, n_splits: int = 4, horizon_embargo: int = 5, min_train_ratio: float = 0.4):
        self.n_splits = n_splits
        self.horizon_embargo = horizon_embargo
        self.min_train_ratio = min_train_ratio

    def resolve_embargo_ms(
        self,
        timestamps: Any,
        embargo_ms: Optional[Union[int, float]] = None,
    ) -> float:
        """Resolve o gap temporal de embargo em ms a partir dos timestamps únicos."""
        if embargo_ms is not None:
            return float(embargo_ms)
        ts = np.asarray(timestamps)
        uniq_ts = np.unique(ts)
        if len(uniq_ts) > 1:
            diffs = np.diff(uniq_ts)
            pos_diffs = diffs[diffs > 0]
            median_positive_step_ms = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 1.0
        else:
            median_positive_step_ms = 1.0
        return float(self.horizon_embargo) * median_positive_step_ms

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        timestamps: Optional[Any] = None,
        embargo_ms: Optional[Union[int, float]] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        indices = np.arange(n_samples)

        if timestamps is None:
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

        ts = np.asarray(timestamps)
        if len(ts) != n_samples:
            raise ValueError(
                f"Comprimento de timestamps ({len(ts)}) difere do número de linhas de X ({n_samples})."
            )

        uniq_ts = np.unique(ts)
        n_uniq = len(uniq_ts)
        min_train_uniq = int(n_uniq * self.min_train_ratio)
        remaining_uniq = n_uniq - min_train_uniq
        test_size_uniq = remaining_uniq // self.n_splits

        if min_train_uniq <= 0 or test_size_uniq <= 0:
            return []

        eff_embargo_ms = self.resolve_embargo_ms(ts, embargo_ms=embargo_ms)
        splits = []
        for i in range(self.n_splits):
            train_end_pos = min_train_uniq + (i * test_size_uniq)
            test_end_pos = min_train_uniq + ((i + 1) * test_size_uniq)

            if train_end_pos <= 0 or train_end_pos >= n_uniq or test_end_pos > n_uniq:
                continue

            train_ts_end = uniq_ts[train_end_pos - 1]
            test_ts_start = train_ts_end + eff_embargo_ms
            test_ts_end = uniq_ts[test_end_pos - 1]

            if test_ts_start > test_ts_end:
                continue

            train_idx = indices[ts <= train_ts_end]
            test_idx = indices[(ts >= test_ts_start) & (ts > train_ts_end) & (ts <= test_ts_end)]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            splits.append((train_idx, test_idx))

        return splits


# =====================================================================
# 3.5 CALIBRAÇÃO MULTICLASSE OOS & ECE
# =====================================================================

def compute_multiclass_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Calcula o Expected Calibration Error (ECE) multiclasse top-label em `n_bins` faixas uniformes.
    Alinha `argmax(y_proba)` ao vetor `classes` para suportar subconjuntos arbitrários de rótulos.
    """
    y_true_arr = np.asarray(y_true)
    proba_arr = np.asarray(y_proba, dtype=np.float64)
    classes_arr = np.asarray(classes)
    if len(y_true_arr) == 0 or proba_arr.size == 0 or len(classes_arr) == 0:
        return 0.0
    if proba_arr.ndim == 1:
        proba_arr = proba_arr.reshape(-1, 1)

    n_bins_eff = max(1, int(n_bins))
    pred_col = np.argmax(proba_arr, axis=1)
    pred_cls = classes_arr[np.clip(pred_col, 0, len(classes_arr) - 1)]
    confidences = np.clip(np.max(proba_arr, axis=1), 0.0, 1.0)
    accuracies = (y_true_arr == pred_cls).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins_eff + 1)
    ece = 0.0
    n_total = float(len(y_true_arr))
    for b in range(n_bins_eff):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = ((confidences >= lo) & (confidences <= hi)) if b == 0 else ((confidences > lo) & (confidences <= hi))
        cnt = int(np.sum(mask))
        if cnt > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            ece += (cnt / n_total) * abs(bin_acc - bin_conf)
    return float(ece)


class CalibratedMulticlassModel:
    """
    Wrapper multiclasse serializável (picklable) para calibração pós-hoc One-Vs-Rest ('platt' ou 'isotonic')
    sobre estimadores já ajustados em partição temporal disjunta (sem vazamento in-sample).
    """

    VALID_METHODS = ("platt", "isotonic")

    def __init__(self, base_estimator: Any, method: str = "platt"):
        method_norm = (method or "platt").lower()
        if method_norm not in self.VALID_METHODS:
            raise ValueError(f"Método de calibração inválido: {method!r}. Use 'platt' ou 'isotonic'.")
        self.base_estimator = base_estimator
        self.method = method_norm
        self.classes_: np.ndarray = np.asarray(getattr(base_estimator, "classes_", [0, 1, 2]))
        self.calibrators_: List[Optional[Any]] = []

    @property
    def feature_names_in_(self) -> np.ndarray:
        if hasattr(self.base_estimator, "feature_names_in_"):
            return getattr(self.base_estimator, "feature_names_in_")
        raise AttributeError(f"{type(self).__name__!r} base_estimator has no attribute 'feature_names_in_'")

    @staticmethod
    def _to_logit(p: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)

    def fit(self, X_cal: Any, y_cal: Any) -> "CalibratedMulticlassModel":
        raw_p = np.asarray(self.base_estimator.predict_proba(X_cal), dtype=np.float64)
        if raw_p.ndim == 1:
            raw_p = raw_p.reshape(-1, 1)
        self.classes_ = np.asarray(self.base_estimator.classes_)
        y_arr = np.asarray(y_cal)
        n_samples = len(y_arr)
        self.calibrators_ = []

        for col_idx, cls_val in enumerate(self.classes_):
            y_bin = (y_arr == cls_val).astype(int)
            p_col = raw_p[:, col_idx]
            # Fallback para identidade quando alvo binário tem <2 classes únicas ou <5 amostras
            if n_samples < 5 or len(np.unique(y_bin)) < 2:
                self.calibrators_.append(None)
                continue

            if self.method == "platt":
                cal = LogisticRegression(C=1.0, max_iter=500)
                cal.fit(self._to_logit(p_col), y_bin)
                self.calibrators_.append(cal)
            else:
                cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                cal.fit(np.clip(p_col, 0.0, 1.0), y_bin)
                self.calibrators_.append(cal)

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        raw_p = np.asarray(self.base_estimator.predict_proba(X), dtype=np.float64)
        if raw_p.ndim == 1:
            raw_p = raw_p.reshape(1, -1)
        n_rows = raw_p.shape[0]
        n_classes = len(self.classes_)

        if not self.calibrators_:
            cal_p = raw_p.copy()
        else:
            cal_p = np.empty((n_rows, n_classes), dtype=np.float64)
            for col_idx in range(n_classes):
                p_col = raw_p[:, col_idx]
                cal = self.calibrators_[col_idx] if col_idx < len(self.calibrators_) else None
                if cal is None:
                    cal_p[:, col_idx] = p_col
                elif self.method == "platt":
                    cal_p[:, col_idx] = cal.predict_proba(self._to_logit(p_col))[:, 1]
                else:
                    cal_p[:, col_idx] = cal.predict(np.clip(p_col, 0.0, 1.0))

        cal_p = np.nan_to_num(cal_p, nan=1.0 / max(n_classes, 1), posinf=1.0, neginf=1e-7)
        cal_p = np.clip(cal_p, 1e-7, 1.0)
        row_sums = cal_p.sum(axis=1, keepdims=True)
        proba = cal_p / np.where(row_sums > 0.0, row_sums, 1.0)
        return proba

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


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
        calibration_method: str = "none",
    ):
        self.model_type = model_type.lower()
        self.horizon = horizon
        self.multiple_tp = multiple_tp
        self.multiple_sl = multiple_sl
        self.confidence_threshold = confidence_threshold
        self.n_jobs = n_jobs
        self.models_dir = models_dir or self.DEFAULT_MODELS_DIR
        self.calibration_method = (calibration_method or "none").lower()
        if self.calibration_method not in ("none", "platt", "isotonic"):
            raise ValueError(
                f"calibration_method inválido: {calibration_method!r}. Use 'none', 'platt' ou 'isotonic'."
            )
        os.makedirs(self.models_dir, exist_ok=True)

        req_dev = "auto" if self.model_type == "lgbm" else "cpu"
        self.device, self.device_info = resolve_compute_device(requested_device=req_dev, n_jobs=self.n_jobs)

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
        """Instancia o modelo preditivo configurado (otimizado para i7-11800H 8C/16T)."""
        if self.model_type == "lgbm" and HAS_LIGHTGBM:
            # No i7-11800H (8 cores físicos / 16 threads), limitar LGBM a <= 8-12 threads evita contenção de OpenMP/HT
            eff_jobs = min(int(self.n_jobs), 8) if self.n_jobs and self.n_jobs > 0 else 8
            return lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=15,
                min_child_samples=100,
                reg_alpha=1.0,
                reg_lambda=5.0,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                device=self.device,
                n_jobs=eff_jobs if self.device == "cpu" else -1,
                random_state=random_state,
                verbosity=-1,
            )
        elif self.model_type in ("extratrees", "et"):
            return ExtraTreesClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=self.n_jobs,
                random_state=random_state,
            )
        else:
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=self.n_jobs,
                random_state=random_state,
            )

    def _split_inner_calibration(
        self,
        idx_arr: np.ndarray,
        ts_arr: Optional[np.ndarray] = None,
        eff_embargo_ms: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Particiona índices cronológicos em `inner_fit` (primeiros 80%) e `inner_cal` (últimos 20%)
        com embargo estrito anti-vazamento (gap >= self.horizon em índice E >= eff_embargo_ms em tempo).
        """
        idx_sorted = np.sort(np.asarray(idx_arr, dtype=int))
        if len(idx_sorted) < 200:
            return None

        if ts_arr is not None and eff_embargo_ms is not None:
            ts_sub = ts_arr[idx_sorted]
            uniq_ts = np.unique(ts_sub)
            if len(uniq_ts) < 10:
                return None
            fit_ts_pos = max(1, int(len(uniq_ts) * 0.80))
            fit_end_ts = uniq_ts[fit_ts_pos - 1]
            cal_start_ts = fit_end_ts + float(eff_embargo_ms)
            inner_fit = idx_sorted[ts_sub <= fit_end_ts]
            if len(inner_fit) == 0:
                return None
            min_cal_idx = int(np.max(inner_fit)) + int(self.horizon)
            inner_cal = idx_sorted[
                (ts_sub >= cal_start_ts) & (ts_sub > fit_end_ts) & (idx_sorted >= min_cal_idx)
            ]
            if len(inner_cal) < 5:
                return None
            assert int(np.max(inner_fit)) + int(self.horizon) <= int(np.min(inner_cal)), (
                "Vazamento de índice entre inner_fit e inner_cal na calibração!"
            )
            assert float(np.min(ts_arr[inner_cal]) - np.max(ts_arr[inner_fit])) >= float(eff_embargo_ms), (
                "Vazamento temporal entre inner_fit e inner_cal na calibração!"
            )
            return inner_fit, inner_cal

        split_pos = int(len(idx_sorted) * 0.80)
        inner_fit = idx_sorted[:split_pos]
        if len(inner_fit) == 0:
            return None
        min_cal_idx = int(np.max(inner_fit)) + max(1, int(self.horizon))
        inner_cal = idx_sorted[idx_sorted >= min_cal_idx]
        if len(inner_cal) < 5:
            return None
        assert int(np.max(inner_fit)) + max(1, int(self.horizon)) <= int(np.min(inner_cal)), (
            "Vazamento de índice entre inner_fit e inner_cal na calibração!"
        )
        return inner_fit, inner_cal

    def _fit_with_optional_calibration(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        idx_arr: np.ndarray,
        ts_arr: Optional[np.ndarray] = None,
        eff_embargo_ms: Optional[float] = None,
        random_state: int = 42,
    ) -> Any:
        """Ajusta estimador base ou wrapper calibrado OOF respeitando embargo temporal."""
        if self.calibration_method in ("platt", "isotonic") and len(idx_arr) >= 200:
            cal_split = self._split_inner_calibration(
                idx_arr, ts_arr=ts_arr, eff_embargo_ms=eff_embargo_ms
            )
            if cal_split is not None:
                inner_fit, inner_cal = cal_split
                if set(np.unique(y.iloc[inner_fit])) != set(np.unique(y.iloc[idx_arr])):
                    cal_split = None
            if cal_split is not None:
                inner_fit, inner_cal = cal_split
                X_fit, y_fit = X.iloc[inner_fit], y.iloc[inner_fit]
                X_cal, y_cal = X.iloc[inner_cal], y.iloc[inner_cal]
                if len(np.unique(y_fit)) >= 2 and len(inner_cal) >= 5:
                    base_est = self._build_estimator(random_state=random_state)
                    base_est.fit(X_fit, y_fit)
                    cal_model = CalibratedMulticlassModel(base_est, method=self.calibration_method)
                    cal_model.fit(X_cal, y_cal)
                    return cal_model

        est = self._build_estimator(random_state=random_state)
        est.fit(X.iloc[idx_arr], y.iloc[idx_arr])
        return est

    @staticmethod
    def _compute_probabilistic_metrics(
        y_true_arr: np.ndarray,
        y_proba_arr: np.ndarray,
        classes: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Calcula (log_loss, roc_auc_ovr, ece) de forma numericamente estável em OOS."""
        n_classes = max(len(classes), 1)
        proba_clean = np.nan_to_num(
            np.asarray(y_proba_arr, dtype=np.float64),
            nan=1.0 / n_classes,
            posinf=1.0,
            neginf=1e-7,
        )
        proba_clipped = np.clip(proba_clean, 1e-7, 1.0)
        row_sums = proba_clipped.sum(axis=1, keepdims=True)
        proba_norm = proba_clipped / np.where(row_sums > 0.0, row_sums, 1.0)

        if len(classes) >= 2 and len(y_true_arr) > 0:
            ll = float(log_loss(y_true_arr, proba_norm, labels=classes))
        else:
            ll = 0.0

        ovr_aucs: List[float] = []
        for col_i, cls_val in enumerate(classes):
            if col_i >= proba_norm.shape[1]:
                continue
            y_bin = (y_true_arr == cls_val).astype(int)
            n_pos = int(np.sum(y_bin))
            if 0 < n_pos < len(y_bin):
                ovr_aucs.append(float(roc_auc_score(y_bin, proba_norm[:, col_i])))
        auc_ovr = float(np.mean(ovr_aucs)) if ovr_aucs else 0.5

        ece = compute_multiclass_ece(y_true_arr, proba_norm, classes=classes, n_bins=10)
        return ll, auc_ovr, ece

    def train_walk_forward(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 4,
        timestamps: Optional[Any] = None,
        embargo_ms: Optional[Union[int, float]] = None,
    ) -> Dict[str, Any]:
        """
        Validação Walk-Forward temporal estrita com embargo anti-contaminação.
        Calcula métricas agregadas out-of-sample honestas.
        """
        splitter = PurgedTimeSeriesSplit(n_splits=n_splits, horizon_embargo=self.horizon)
        splits = splitter.split(X, y, timestamps=timestamps, embargo_ms=embargo_ms)

        if not splits:
            raise ValueError("Amostras insuficientes para gerar splits temporais com embargo.")

        ts_arr = np.asarray(timestamps) if timestamps is not None else None
        eff_embargo_ms = (
            splitter.resolve_embargo_ms(ts_arr, embargo_ms=embargo_ms)
            if ts_arr is not None
            else None
        )

        canonical_classes = np.sort(np.unique(np.asarray(y)))
        if len(canonical_classes) == 0:
            canonical_classes = np.array([0, 1, 2])

        all_test_indices: List[int] = []
        all_y_true: List[int] = []
        all_y_pred: List[int] = []
        all_y_proba: List[np.ndarray] = []

        fold_metrics = []
        last_model: Optional[Any] = None

        for fold, (train_idx, test_idx) in enumerate(splits):
            # ASSERT ANTI-VAZAMENTO: Prova matemática que não há interseção e o embargo foi respeitado
            assert len(set(train_idx).intersection(set(test_idx))) == 0, (
                f"Vazamento detectado no Fold {fold+1}: interseção não-nula entre treino e teste!"
            )
            if ts_arr is not None and eff_embargo_ms is not None:
                observed_gap_ms = float(np.min(ts_arr[test_idx]) - np.max(ts_arr[train_idx]))
                assert observed_gap_ms >= eff_embargo_ms, (
                    f"Vazamento temporal detectado no Fold {fold+1}: gap ({observed_gap_ms}) "
                    f"inferior ao horizonte de embargo ({eff_embargo_ms} ms)!"
                )
            else:
                assert min(test_idx) >= max(train_idx) + self.horizon, (
                    f"Vazamento detectado no Fold {fold+1}: gap ({min(test_idx) - max(train_idx)}) "
                    f"inferior ao horizonte de embargo ({self.horizon})!"
                )

            y_train = y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            # Valida classes presentes no treino
            if len(np.unique(y_train)) < 2:
                continue

            model = self._fit_with_optional_calibration(
                X,
                y,
                train_idx,
                ts_arr=ts_arr,
                eff_embargo_ms=eff_embargo_ms,
                random_state=42 + fold,
            )
            last_model = model

            y_pred = model.predict(X_test)
            raw_fold_proba = np.asarray(model.predict_proba(X_test), dtype=np.float64)
            model_classes = np.asarray(getattr(model, "classes_", canonical_classes))
            if np.array_equal(model_classes, canonical_classes):
                y_proba = raw_fold_proba
            else:
                y_proba = np.full((len(X_test), len(canonical_classes)), 1e-7, dtype=np.float64)
                for col_i, cls_val in enumerate(model_classes):
                    target_idx = np.where(canonical_classes == cls_val)[0]
                    if len(target_idx) > 0 and col_i < raw_fold_proba.shape[1]:
                        y_proba[:, target_idx[0]] = raw_fold_proba[:, col_i]
                row_sums = y_proba.sum(axis=1, keepdims=True)
                y_proba = y_proba / np.where(row_sums > 0.0, row_sums, 1.0)

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

        # Classes inferidas alinhadas às colunas de y_proba_arr
        classes = canonical_classes if len(canonical_classes) == y_proba_arr.shape[1] else getattr(last_model, "classes_", np.array([0, 1, 2]))

        # Métricas globais
        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        f1_mac = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))

        # Brier score multiclasse
        y_onehot = np.zeros_like(y_proba_arr, dtype=np.float64)
        for col_i, cls_val in enumerate(classes):
            if col_i < y_onehot.shape[1]:
                y_onehot[:, col_i] = (y_true_arr == cls_val).astype(np.float64)
        brier_score = float(np.mean((y_proba_arr - y_onehot) ** 2))

        # Log-loss, ROC-AUC OvR e ECE multiclasse OOS
        ll_oos, roc_auc_ovr, ece_oos = self._compute_probabilistic_metrics(
            y_true_arr, y_proba_arr, classes
        )

        # Precision por classe: 0 (HOLD), 1 (BUY), 2 (SELL)
        precisions = precision_score(y_true_arr, y_pred_arr, labels=classes, average=None, zero_division=0)
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
            "calibration_method": self.calibration_method,
            "device": self.device,
            "samples_evaluated": len(y_true_arr),
            "global_accuracy": acc,
            "f1_macro": f1_mac,
            "f1_weighted": f1_weighted,
            "brier_score": brier_score,
            "log_loss": ll_oos,
            "roc_auc_ovr": roc_auc_ovr,
            "ece": ece_oos,
            "class_precisions": class_precisions,
            "emitted_signals_accuracy": emitted_accuracy,
            "emitted_signals_count": total_signals,
            "coverage_pct": coverage_pct,
            "confidence_threshold": self.confidence_threshold,
            "embargo_ms": float(eff_embargo_ms) if eff_embargo_ms is not None else None,
            "fold_details": fold_metrics,
            "test_indices": all_test_indices,
            "y_true": y_true_arr,
            "y_pred": y_pred_arr,
            "y_proba": y_proba_arr,
        }

        logger.info(
            f"Resultados Walk-Forward [{self.model_type.upper()}|cal={self.calibration_method}]: "
            f"Global Acc={acc:.2%}, Sinais Emitidos Acc={emitted_accuracy:.2%}, "
            f"LogLoss={ll_oos:.4f}, AUC={roc_auc_ovr:.4f}, ECE={ece_oos:.4f}, "
            f"Cobertura={coverage_pct:.2%} ({total_signals}/{len(y_true_arr)} trades)."
        )

        return results

    def train_full_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_name: Optional[str] = None,
        timestamps: Optional[Any] = None,
        embargo_ms: Optional[Union[int, float]] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Treina o estimador com todo o conjunto histórico disponível,
        avalia walk-forward para documentação e persiste o artefato com joblib.
        """
        # 1. Avaliação out-of-sample formal
        eval_metrics = self.train_walk_forward(
            X,
            y,
            timestamps=timestamps,
            embargo_ms=embargo_ms,
        )

        # 2. Treino final na base completa (ou split cronológico 80/20 com embargo quando calibrado)
        ts_arr = np.asarray(timestamps) if timestamps is not None else None
        eff_embargo_ms = (
            PurgedTimeSeriesSplit(horizon_embargo=self.horizon).resolve_embargo_ms(
                ts_arr, embargo_ms=embargo_ms
            )
            if ts_arr is not None
            else None
        )
        if self.calibration_method in ("platt", "isotonic") and len(X) >= 200:
            full_indices = np.arange(len(X), dtype=int)
            final_model = self._fit_with_optional_calibration(
                X,
                y,
                full_indices,
                ts_arr=ts_arr,
                eff_embargo_ms=eff_embargo_ms,
                random_state=42,
            )
        else:
            final_model = self._build_estimator(random_state=42)
            final_model.fit(X, y)

        # 3. Metadados e Persistência
        filename = save_name or f"{self.model_type}_model_v1.joblib"
        filepath = os.path.join(self.models_dir, filename)

        payload: Dict[str, Any] = {
            "model": final_model,
            "model_type": self.model_type,
            "calibration_method": self.calibration_method,
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
        if isinstance(extra_metadata, dict):
            for k, v in extra_metadata.items():
                if k != "model":
                    payload[k] = v

        _atomic_joblib_dump(payload, filepath, compress=3)
        logger.info(f"Modelo salvo com sucesso em: {filepath}")

        return payload

    @classmethod
    def load_model(cls, filepath: str) -> Dict[str, Any]:
        """Carrega modelo persistido e seus metadados estruturados."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {filepath}")
        return joblib.load(filepath)
