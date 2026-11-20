"""
Agente Técnico da Fase 4.
Utiliza os modelos estatísticos e preditivos de Machine Learning da Fase 3 (LightGBM / Random Forest)
treinados em indicadores técnicos (RSI, MACD, Bollinger, EMA, ATR).
"""

from __future__ import annotations

import logging
import os
import threading
import warnings

_pw = os.environ.get("PYTHONWARNINGS", "")
if "sklearn.utils.parallel" not in _pw:
    os.environ["PYTHONWARNINGS"] = (
        f"{_pw},ignore::UserWarning:sklearn.utils.parallel"
        if _pw
        else "ignore::UserWarning:sklearn.utils.parallel"
    )
warnings.filterwarnings("ignore", message=r".*sklearn\.utils\.parallel.*", category=UserWarning)

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent, AgentSignal

logger = logging.getLogger("ia_trader.technical_agent")


class _DefaultMLEnsemble:
    """Ensemble leve em memória combinando LightGBM + Random Forest (suporta múltiplos schemas v1/v2)."""

    def __init__(
        self,
        models: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        *,
        models_with_features: Optional[List[Tuple[Any, List[str]]]] = None,
    ):
        if models_with_features is not None:
            pairs = [(m, list(fnames or [])) for m, fnames in models_with_features if m is not None]
        elif (
            models is not None
            and feature_names is None
            and isinstance(models, (list, tuple))
            and len(models) > 0
            and isinstance(models[0], (tuple, list))
            and len(models[0]) == 2
            and isinstance(models[0][1], (list, tuple))
        ):
            pairs = [(m, list(fnames or [])) for m, fnames in models if m is not None]
        else:
            raw_models = list(models or [])
            shared_feats = list(feature_names or [])
            pairs = []
            for m in raw_models:
                if m is None:
                    continue
                if not shared_feats and hasattr(m, "feature_names_in_"):
                    m_feats = [str(x) for x in getattr(m, "feature_names_in_")]
                else:
                    m_feats = list(shared_feats)
                pairs.append((m, m_feats))

        self.models_with_features: List[Tuple[Any, List[str]]] = pairs
        self.models: List[Any] = [m for m, _ in pairs]
        union_feats: List[str] = []
        seen_feats = set()
        for _, fnames in pairs:
            for f in fnames:
                if f not in seen_feats:
                    seen_feats.add(f)
                    union_feats.append(f)
        self.feature_names: List[str] = union_feats
        self.classes_ = np.array([0, 1, 2])

    @staticmethod
    def _safe_float(v: Any) -> float:
        if v is None:
            return 0.0
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return 0.0
        return fv if np.isfinite(fv) else 0.0

    def predict_proba(self, X: Any) -> np.ndarray:
        X_df: Optional[pd.DataFrame] = None
        arr: Optional[np.ndarray] = None

        if isinstance(X, pd.DataFrame):
            X_df = X
        elif isinstance(X, dict):
            all_keys = set(self.feature_names) | set(X.keys())
            row = {k: self._safe_float(X.get(k, 0.0)) for k in all_keys}
            X_df = pd.DataFrame([row], dtype=np.float64)
        elif isinstance(X, pd.Series):
            if any(isinstance(idx, str) for idx in X.index):
                row = {str(k): self._safe_float(v) for k, v in X.items()}
                X_df = pd.DataFrame([row], dtype=np.float64)
            else:
                arr = np.asarray(X.values, dtype=np.float64).reshape(1, -1)
        else:
            arr = np.asarray(X, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        if X_df is None and arr is not None and self.feature_names and arr.shape[1] == len(self.feature_names):
            X_df = pd.DataFrame(arr, columns=self.feature_names)

        probs = []
        for m, fnames in self.models_with_features:
            if X_df is not None:
                if fnames:
                    X_sub = X_df.reindex(columns=fnames, fill_value=0.0)
                    if X_sub.isna().any().any():
                        X_sub = X_sub.fillna(0.0)
                else:
                    X_sub = X_df
            elif arr is not None:
                if fnames and arr.shape[1] == len(fnames):
                    X_sub = pd.DataFrame(arr, columns=fnames)
                elif fnames and arr.shape[1] > len(fnames):
                    X_sub = pd.DataFrame(arr[:, : len(fnames)], columns=fnames)
                elif fnames:
                    pad = np.zeros((arr.shape[0], len(fnames)), dtype=np.float64)
                    pad[:, : arr.shape[1]] = arr
                    X_sub = pd.DataFrame(pad, columns=fnames)
                else:
                    X_sub = arr
            else:
                X_sub = X

            p = np.asarray(m.predict_proba(X_sub), dtype=np.float64)
            if p.ndim == 1:
                p = p.reshape(1, -1)
            m_classes = getattr(m, "classes_", None)
            if m_classes is not None and p.shape[1] != len(self.classes_):
                aligned = np.zeros((p.shape[0], len(self.classes_)), dtype=np.float64)
                for col_i, cls_val in enumerate(m_classes):
                    target_idx = np.where(self.classes_ == cls_val)[0]
                    if len(target_idx) > 0 and col_i < p.shape[1]:
                        aligned[:, target_idx[0]] = p[:, col_i]
                p = aligned
            probs.append(p)

        if not probs:
            return np.zeros((1, len(self.classes_)), dtype=np.float64)
        return np.mean(probs, axis=0)


def _disable_estimator_parallelism(estimator: Any, _seen: Optional[set] = None) -> Any:
    if _seen is None:
        _seen = set()
    if estimator is None:
        return estimator
    obj_id = id(estimator)
    if obj_id in _seen:
        return estimator
    _seen.add(obj_id)

    if hasattr(estimator, "n_jobs"):
        try:
            setattr(estimator, "n_jobs", 1)
        except (AttributeError, TypeError):
            pass

    # Recurse into sub-estimators
    for attr_name in ("estimator", "base_estimator"):
        try:
            sub_est = getattr(estimator, attr_name, None)
        except Exception:
            sub_est = None
        if sub_est is not None:
            _disable_estimator_parallelism(sub_est, _seen)

    try:
        estimators = getattr(estimator, "estimators_", None)
    except Exception:
        estimators = None
    if isinstance(estimators, (list, tuple)):
        for sub_est in estimators:
            _disable_estimator_parallelism(sub_est, _seen)

    try:
        calibrated = getattr(estimator, "calibrated_classifiers_", None)
    except Exception:
        calibrated = None
    if isinstance(calibrated, (list, tuple)):
        for sub_est in calibrated:
            _disable_estimator_parallelism(sub_est, _seen)

    try:
        steps = getattr(estimator, "steps", None)
    except Exception:
        steps = None
    if isinstance(steps, (list, tuple)):
        for name_step in steps:
            if isinstance(name_step, (list, tuple)) and len(name_step) >= 2:
                step_est = name_step[1]
                _disable_estimator_parallelism(step_est, _seen)

    return estimator


class TechnicalAgent(BaseAgent):
    """
    Agente especialista em análise quantitativa e técnica tabular.
    """

    _cached_default_model: Optional[Any] = None
    _cached_load_attempted: bool = False
    _cached_model_signature: Optional[Tuple[Tuple[str, int, int], ...]] = None
    _cache_lock = threading.RLock()

    @classmethod
    def _get_model_dir_paths(cls) -> List[str]:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return [
            os.path.join(base_dir, "models", "lgbm_model_v1.joblib"),
            os.path.join(base_dir, "models", "rf_model_v1.joblib"),
        ]

    @classmethod
    def _current_model_signature(cls) -> Tuple[Tuple[str, int, int], ...]:
        sig: List[Tuple[str, int, int]] = []
        for p in cls._get_model_dir_paths():
            try:
                if os.path.exists(p):
                    st = os.stat(p)
                    sig.append((p, int(st.st_mtime_ns), int(st.st_size)))
                else:
                    sig.append(("missing", 0, 0))
            except OSError:
                sig.append(("missing", 0, 0))
        return tuple(sig)

    @classmethod
    def invalidate_default_model_cache(cls) -> None:
        with cls._cache_lock:
            cls._cached_default_model = None
            cls._cached_load_attempted = False
            cls._cached_model_signature = None

    @classmethod
    def _get_default_model(cls) -> Optional[Any]:
        with cls._cache_lock:
            current_sig = cls._current_model_signature()
            if (
                cls._cached_load_attempted
                and cls._cached_model_signature is not None
                and cls._cached_model_signature == current_sig
            ):
                return cls._cached_default_model

            cls._cached_load_attempted = True
            cls._cached_model_signature = current_sig
            try:
                import joblib
                paths = cls._get_model_dir_paths()
                models_with_features: List[Tuple[Any, List[str]]] = []
                for p in paths:
                    if os.path.exists(p):
                        bundle = joblib.load(p)
                        if isinstance(bundle, dict) and "model" in bundle and bundle["model"] is not None:
                            est = bundle["model"]
                            _disable_estimator_parallelism(est)
                            fnames = list(bundle.get("feature_names") or bundle.get("features") or [])
                            if not fnames and hasattr(est, "feature_names_in_"):
                                fnames = [str(x) for x in getattr(est, "feature_names_in_")]
                            if fnames:
                                models_with_features.append((est, fnames))
                if models_with_features:
                    cls._cached_default_model = _DefaultMLEnsemble(models_with_features=models_with_features)
                else:
                    cls._cached_default_model = None
            except Exception as exc:
                logger.warning(f"Falha ao carregar modelo ML padrao: {exc}")
                cls._cached_default_model = None
            return cls._cached_default_model

    def __init__(
        self,
        name: str = "TechnicalAgent",
        weight: float = 1.0,
        confidence_threshold: float = 0.55,
        model: Optional[Any] = None,
    ):
        super().__init__(name=name, weight=weight, is_backtestable=True)
        self.confidence_threshold = confidence_threshold
        self._uses_default_model: bool = model is None
        self.model = model if model is not None else self._get_default_model()

    def evaluate(self, context: Dict[str, Any]) -> AgentSignal:
        """
        Avalia o contexto de features.
        Suporta:
        1. Predição direta via `model.predict_proba(features)`
        2. Probabilidades pré-computadas out-of-sample via `proba_buy`, `proba_sell`
        """
        if self._uses_default_model:
            self.model = self._get_default_model()

        proba_buy = context.get("proba_buy")
        proba_sell = context.get("proba_sell")
        implicit_ml_buy: float = 0.0
        implicit_ml_sell: float = 0.0

        # Se features brutas forem passadas e houver modelo instanciado
        if proba_buy is None and self.model is not None and "features" in context:
            features = context["features"]
            if isinstance(self.model, _DefaultMLEnsemble) and isinstance(features, (pd.DataFrame, dict)):
                probas = self.model.predict_proba(features)[0]
            elif hasattr(features, "values"):
                feat_arr = features.values.reshape(1, -1)
                probas = self.model.predict_proba(feat_arr)[0]
            else:
                feat_arr = np.asarray(features).reshape(1, -1)
                probas = self.model.predict_proba(feat_arr)[0]
            classes = getattr(self.model, "classes_", np.array([0, 1, 2]))
            buy_idx = np.where(classes == 1)[0]
            sell_idx = np.where(classes == 2)[0]
            proba_buy = float(probas[buy_idx[0]]) if len(buy_idx) > 0 else 0.0
            proba_sell = float(probas[sell_idx[0]]) if len(sell_idx) > 0 else 0.0
        elif proba_buy is None and proba_sell is None and self.model is not None:
            models_with_feats = getattr(self.model, "models_with_features", None)
            if isinstance(models_with_feats, list) and models_with_feats:
                has_matching_schema = any(
                    bool(fnames) and all(k in context for k in fnames)
                    for _, fnames in models_with_feats
                )
                if has_matching_schema:
                    try:
                        probas = self.model.predict_proba(context)[0]
                        classes = getattr(self.model, "classes_", np.array([0, 1, 2]))
                        buy_idx = np.where(classes == 1)[0]
                        sell_idx = np.where(classes == 2)[0]
                        implicit_ml_buy = float(probas[buy_idx[0]]) if len(buy_idx) > 0 else 0.0
                        implicit_ml_sell = float(probas[sell_idx[0]]) if len(sell_idx) > 0 else 0.0
                    except Exception:
                        implicit_ml_buy, implicit_ml_sell = 0.0, 0.0
            else:
                feat_names = getattr(self.model, "feature_names", None)
                if feat_names and all(k in context for k in feat_names):
                    try:
                        row_vals = []
                        for k in feat_names:
                            v = float(context.get(k, 0.0) or 0.0)
                            row_vals.append(v if np.isfinite(v) else 0.0)
                        feat_arr = np.asarray(row_vals, dtype=np.float64).reshape(1, -1)
                        probas = self.model.predict_proba(feat_arr)[0]
                        classes = getattr(self.model, "classes_", np.array([0, 1, 2]))
                        buy_idx = np.where(classes == 1)[0]
                        sell_idx = np.where(classes == 2)[0]
                        implicit_ml_buy = float(probas[buy_idx[0]]) if len(buy_idx) > 0 else 0.0
                        implicit_ml_sell = float(probas[sell_idx[0]]) if len(sell_idx) > 0 else 0.0
                    except Exception:
                        implicit_ml_buy, implicit_ml_sell = 0.0, 0.0

        def _clean(v: Any) -> Optional[float]:
            """None/NaN/inf -> None; caso contrário float."""
            if v is None:
                return None
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return None
            return fv if np.isfinite(fv) else None

        proba_buy = _clean(proba_buy) or 0.0
        proba_sell = _clean(proba_sell) or 0.0

        # Filtro de Regime Estrutural Macro (SMA 200)
        close = _clean(context.get("close"))
        sma_200 = _clean(context.get("sma_200"))
        dist_sma_200 = _clean(context.get("dist_sma_200"))
        rsi_14 = _clean(context.get("rsi_14"))
        dist_vwap = _clean(context.get("dist_vwap_pct"))

        macro_bull = None
        if close is not None and sma_200 is not None:
            macro_bull = close >= sma_200
        elif dist_sma_200 is not None:
            macro_bull = dist_sma_200 >= 0.0

        direction = "HOLD"
        confidence = max(proba_buy, proba_sell, 1.0 - proba_buy - proba_sell)
        rationale = "Sinais técnicos neutros ou abaixo do limiar de convicção."

        # Penalização contra a tendência macro dominante (Regime Filter)
        eff_buy_thresh = self.confidence_threshold if (macro_bull is not False) else self.confidence_threshold + 0.10
        eff_sell_thresh = self.confidence_threshold if (macro_bull is not True) else self.confidence_threshold + 0.10

        # Se probabilidades de ML explícitas foram geradas (ou se o Ensemble 50k atingir o limiar de alta convicção)
        if proba_buy > 0.0 or proba_sell > 0.0:
            if proba_buy >= eff_buy_thresh and proba_buy > proba_sell:
                direction = "BUY"
                confidence = proba_buy
                regime_str = " (Alinhado com regime macro > SMA200)" if macro_bull else " (Contra-tendência macro)"
                rationale = f"Modelo ML sinaliza expansão de alta com convicção de {proba_buy:.1%}{regime_str}."
            elif proba_sell >= eff_sell_thresh and proba_sell > proba_buy:
                direction = "SELL"
                confidence = proba_sell
                regime_str = " (Alinhado com regime macro < SMA200)" if macro_bull is False else " (Contra-tendência macro)"
                rationale = f"Modelo ML sinaliza pressão vendedora com convicção de {proba_sell:.1%}{regime_str}."
        elif implicit_ml_buy >= eff_buy_thresh and implicit_ml_buy > implicit_ml_sell:
            proba_buy, proba_sell = implicit_ml_buy, implicit_ml_sell
            direction = "BUY"
            confidence = implicit_ml_buy
            regime_str = " (Alinhado com regime macro > SMA200)" if macro_bull else " (Contra-tendência macro)"
            rationale = f"Ensemble ML (50k) sinaliza expansão de alta com convicção de {implicit_ml_buy:.1%}{regime_str}."
        elif implicit_ml_sell >= eff_sell_thresh and implicit_ml_sell > implicit_ml_buy:
            proba_buy, proba_sell = implicit_ml_buy, implicit_ml_sell
            direction = "SELL"
            confidence = implicit_ml_sell
            regime_str = " (Alinhado com regime macro < SMA200)" if macro_bull is False else " (Contra-tendência macro)"
            rationale = f"Ensemble ML (50k) sinaliza pressão vendedora com convicção de {implicit_ml_sell:.1%}{regime_str}."

        # Fallback Quantitativo Institucional + Modulação pelo Ensemble ML (Neurônio Técnico Ativo)
        elif macro_bull is not None and rsi_14 is not None:
            if implicit_ml_buy > 0.0 or implicit_ml_sell > 0.0:
                proba_buy, proba_sell = implicit_ml_buy, implicit_ml_sell
            rsi_val = float(rsi_14)
            vwap_val = float(dist_vwap) if dist_vwap is not None else 0.0

            # 1. Pullback institucional em tendência de alta
            if macro_bull and rsi_val <= 42.0 and vwap_val <= 0.5:
                direction = "BUY"
                confidence = min(0.88, 0.60 + (45.0 - rsi_val) * 0.01)
                rationale = f"Pullback institucional em alta (Preço > SMA200, RSI={rsi_val:.1f} em retração e VWAP equilibrado)."
            # 2. Repique corretivo em tendência de baixa
            elif not macro_bull and rsi_val >= 58.0 and vwap_val >= -0.5:
                direction = "SELL"
                confidence = min(0.88, 0.60 + (rsi_val - 55.0) * 0.01)
                rationale = f"Repique corretivo em baixa (Preço < SMA200, RSI={rsi_val:.1f} em exaustão e VWAP resistente)."
            # 3. Impulso e Momentum de Alta Ativo (com teto de exaustão em RSI > 75.0)
            elif macro_bull and 48.0 <= rsi_val <= 75.0:
                direction = "BUY"
                confidence = min(0.84, 0.62 + (rsi_val - 48.0) * 0.008)
                rationale = f"Neurônio de Momentum Ativo: Estrutura compradora alinhada (Preço > SMA200, RSI={rsi_val:.1f} sustentando aceleração)."
            elif macro_bull and rsi_val > 75.0:
                direction = "HOLD"
                confidence = 0.55
                rationale = f"Exaustão de sobrecompra: RSI={rsi_val:.1f} > 75.0 desaconselha compras no topo."
            # 4. Impulso e Pressão Vendedora Ativa (com piso de exaustão em RSI < 25.0)
            elif not macro_bull and 25.0 <= rsi_val <= 52.0:
                direction = "SELL"
                confidence = min(0.84, 0.62 + (52.0 - rsi_val) * 0.008)
                rationale = f"Neurônio de Momentum Ativo: Pressão vendedora dominante (Preço < SMA200, RSI={rsi_val:.1f} em enfraquecimento)."
            elif not macro_bull and rsi_val < 25.0:
                direction = "HOLD"
                confidence = 0.55
                rationale = f"Exaustão de sobrevenda: RSI={rsi_val:.1f} < 25.0 desaconselha vendas no fundo."
            # 5. Neutralidade temporária no centro
            else:
                direction = "HOLD"
                confidence = 0.55
                rationale = f"Compressão de volatilidade entre médias (RSI={rsi_val:.1f}). Neurônio em calibração sináptica."

            # Modulação de convicção quando o Ensemble ML de 50k velas converge com o gatilho técnico
            if direction == "BUY" and implicit_ml_buy > implicit_ml_sell and implicit_ml_buy >= 0.35:
                confidence = min(0.92, confidence + (implicit_ml_buy - 0.35) * 0.25)
                rationale += f" [ML 50k: {implicit_ml_buy:.1%} BUY]"
            elif direction == "SELL" and implicit_ml_sell > implicit_ml_buy and implicit_ml_sell >= 0.35:
                confidence = min(0.92, confidence + (implicit_ml_sell - 0.35) * 0.25)
                rationale += f" [ML 50k: {implicit_ml_sell:.1%} SELL]"

        # Integração de Estrutura de Mercado, Zonas de Liquidez e POI (Fase 4 Avançada)
        at_bullish_poi = bool(context.get("at_bullish_poi", 0))
        at_bearish_poi = bool(context.get("at_bearish_poi", 0))
        sweep_bull = bool(context.get("liquidity_sweep_bull", 0))
        sweep_bear = bool(context.get("liquidity_sweep_bear", 0))
        dist_supp = context.get("dist_support_pct")
        dist_res = context.get("dist_resistance_pct")
        nearest_bsl = context.get("nearest_bsl")
        nearest_ssl = context.get("nearest_ssl")
        has_valid_bsl = True if nearest_bsl is None else float(nearest_bsl or 0.0) > 0.0
        has_valid_ssl = True if nearest_ssl is None else float(nearest_ssl or 0.0) > 0.0
        dist_supp_val = float(dist_supp) if dist_supp is not None and not np.isnan(float(dist_supp)) else None
        dist_res_val = float(dist_res) if dist_res is not None and not np.isnan(float(dist_res)) else None

        # Ajuste de convicção e direção por Confluência Estrutural
        if direction == "BUY":
            # Confluência positiva em POI ou varredura de fundo (Sweep de SSL)
            if sweep_bull or at_bullish_poi:
                bonus = 0.15 if sweep_bull else 0.08
                confidence = min(0.98, confidence + bonus)
                rationale += " [CONFLUÊNCIA POI/SWEEP: Alta convicção compradora em zona de liquidez mitigada]."
            # Penalidade se comprando direto contra resistência imediata (< 0.25%)
            if dist_res_val is not None and 0.0 < dist_res_val < 0.25 and has_valid_bsl and not sweep_bull:
                confidence = max(0.40, confidence - 0.20)
                if confidence < self.confidence_threshold:
                    direction = "HOLD"
                    rationale = f"Sinal de compra neutralizado: preço muito próximo da resistência imediata ({dist_res_val:.2f}% de distância)."
        elif direction == "SELL":
            # Confluência positiva em POI vendedor ou varredura de topo (Sweep de BSL)
            if sweep_bear or at_bearish_poi:
                bonus = 0.15 if sweep_bear else 0.08
                confidence = min(0.98, confidence + bonus)
                rationale += " [CONFLUÊNCIA POI/SWEEP: Alta convicção vendedora em zona de liquidez mitigada]."
            # Penalidade se vendendo direto contra suporte imediato (< 0.25%)
            if dist_supp_val is not None and 0.0 < dist_supp_val < 0.25 and has_valid_ssl and not sweep_bear:
                confidence = max(0.40, confidence - 0.20)
                if confidence < self.confidence_threshold:
                    direction = "HOLD"
                    rationale = f"Sinal de venda neutralizado: preço muito próximo do suporte imediato ({dist_supp_val:.2f}% de distância)."
        elif direction == "HOLD":
            vol_z = _clean(context.get("volume_zscore_20"))
            # Gatilho Estrutural Puro: Reversão em Liquidity Sweep mesmo sem sinal do modelo tradicional
            if sweep_bull and (macro_bull is not False) and (dist_supp_val is None or dist_supp_val <= 0.8) and (vol_z is None or vol_z >= 0.8):
                direction = "BUY"
                confidence = 0.72 if (vol_z is not None and vol_z >= 1.5) else 0.68
                rationale = "Gatilho Estrutural: Liquidity Sweep comprador detectado com rejeição de mínima (SSL varrido)."
            elif sweep_bear and (macro_bull is not True) and (dist_res_val is None or dist_res_val <= 0.8) and (vol_z is None or vol_z >= 0.8):
                direction = "SELL"
                confidence = 0.72 if (vol_z is not None and vol_z >= 1.5) else 0.68
                rationale = "Gatilho Estrutural: Liquidity Sweep vendedor detectado com rejeição de máxima (BSL varrido)."

        return AgentSignal(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            weight=self.weight,
            rationale=rationale,
            metrics={
                "proba_buy": proba_buy,
                "proba_sell": proba_sell,
                "threshold": self.confidence_threshold,
                "macro_bull": macro_bull,
                "rsi_14": float(rsi_14) if rsi_14 is not None else None,
                "at_bullish_poi": at_bullish_poi,
                "at_bearish_poi": at_bearish_poi,
                "liquidity_sweep_bull": sweep_bull,
                "liquidity_sweep_bear": sweep_bear,
                "dist_support_pct": dist_supp_val,
                "dist_resistance_pct": dist_res_val,
            },
            timestamp=context.get("timestamp"),
        )

