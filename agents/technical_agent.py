"""
Agente Técnico da Fase 4.
Utiliza os modelos estatísticos e preditivos de Machine Learning da Fase 3 (LightGBM / Random Forest)
treinados em indicadores técnicos (RSI, MACD, Bollinger, EMA, ATR).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np

from agents.base_agent import BaseAgent, AgentSignal


class TechnicalAgent(BaseAgent):
    """
    Agente especialista em análise quantitativa e técnica tabular.
    """

    def __init__(
        self,
        name: str = "TechnicalAgent",
        weight: float = 1.0,
        confidence_threshold: float = 0.60,
        model: Optional[Any] = None,
    ):
        super().__init__(name=name, weight=weight, is_backtestable=True)
        self.confidence_threshold = confidence_threshold
        self.model = model

    def evaluate(self, context: Dict[str, Any]) -> AgentSignal:
        """
        Avalia o contexto de features.
        Suporta:
        1. Predição direta via `model.predict_proba(features)`
        2. Probabilidades pré-computadas out-of-sample via `proba_buy`, `proba_sell`
        """
        proba_buy = context.get("proba_buy")
        proba_sell = context.get("proba_sell")

        # Se features brutas forem passadas e houver modelo instanciado
        if proba_buy is None and self.model is not None and "features" in context:
            features = context["features"]
            # Garante formato 2D
            if hasattr(features, "values"):
                feat_arr = features.values.reshape(1, -1)
            else:
                feat_arr = np.asarray(features).reshape(1, -1)
            probas = self.model.predict_proba(feat_arr)[0]
            classes = getattr(self.model, "classes_", np.array([0, 1, 2]))
            buy_idx = np.where(classes == 1)[0]
            sell_idx = np.where(classes == 2)[0]
            proba_buy = float(probas[buy_idx[0]]) if len(buy_idx) > 0 else 0.0
            proba_sell = float(probas[sell_idx[0]]) if len(sell_idx) > 0 else 0.0

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

        # Se probabilidades de ML foram geradas
        if proba_buy > 0.0 or proba_sell > 0.0:
            # Penalização contra a tendência macro dominante (Regime Filter)
            eff_buy_thresh = self.confidence_threshold if (macro_bull is not False) else self.confidence_threshold + 0.10
            eff_sell_thresh = self.confidence_threshold if (macro_bull is not True) else self.confidence_threshold + 0.10

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

        # Fallback Quantitativo Institucional quando ML não estiver instanciado (Neurônio Técnico Ativo)
        elif macro_bull is not None and rsi_14 is not None:
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
            # 3. Impulso e Momentum de Alta Ativo
            elif macro_bull and rsi_val >= 48.0:
                direction = "BUY"
                confidence = min(0.84, 0.62 + (rsi_val - 48.0) * 0.008)
                rationale = f"Neurônio de Momentum Ativo: Estrutura compradora alinhada (Preço > SMA200, RSI={rsi_val:.1f} sustentando aceleração)."
            # 4. Impulso e Pressão Vendedora Ativa
            elif not macro_bull and rsi_val <= 52.0:
                direction = "SELL"
                confidence = min(0.84, 0.62 + (52.0 - rsi_val) * 0.008)
                rationale = f"Neurônio de Momentum Ativo: Pressão vendedora dominante (Preço < SMA200, RSI={rsi_val:.1f} em enfraquecimento)."
            # 5. Neutralidade temporária no centro
            else:
                direction = "HOLD"
                confidence = 0.55
                rationale = f"Compressão de volatilidade entre médias (RSI={rsi_val:.1f}). Neurônio em calibração sináptica."

        # Integração de Estrutura de Mercado, Zonas de Liquidez e POI (Fase 4 Avançada)
        at_bullish_poi = bool(context.get("at_bullish_poi", 0))
        at_bearish_poi = bool(context.get("at_bearish_poi", 0))
        sweep_bull = bool(context.get("liquidity_sweep_bull", 0))
        sweep_bear = bool(context.get("liquidity_sweep_bear", 0))
        dist_supp = context.get("dist_support_pct")
        dist_res = context.get("dist_resistance_pct")
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
            if dist_res_val is not None and dist_res_val < 0.25 and not sweep_bull:
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
            if dist_supp_val is not None and dist_supp_val < 0.25 and not sweep_bear:
                confidence = max(0.40, confidence - 0.20)
                if confidence < self.confidence_threshold:
                    direction = "HOLD"
                    rationale = f"Sinal de venda neutralizado: preço muito próximo do suporte imediato ({dist_supp_val:.2f}% de distância)."
        elif direction == "HOLD":
            # Gatilho Estrutural Puro: Reversão em Liquidity Sweep mesmo sem sinal do modelo tradicional
            if sweep_bull and (dist_supp_val is None or dist_supp_val <= 0.8):
                direction = "BUY"
                confidence = 0.72
                rationale = "Gatilho Estrutural: Liquidity Sweep comprador detectado com rejeição de mínima (SSL varrido)."
            elif sweep_bear and (dist_res_val is None or dist_res_val <= 0.8):
                direction = "SELL"
                confidence = 0.72
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

