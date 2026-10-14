"""
Agente de Fluxo de Ordens e Microestrutura da Fase 4.
Avalia desbalanceamento de livro (OrderBook Imbalance), volume delta e pressão compradora/vendedora.
Suporta tanto execução em tempo real (livro L2) quanto backtesting histórico (proxy de fluxo e volume).
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np

from agents.base_agent import BaseAgent, AgentSignal


class OrderFlowAgent(BaseAgent):
    """
    Agente especialista em microestrutura e dinâmica de fluxo de liquidez.
    """

    def __init__(
        self,
        name: str = "OrderFlowAgent",
        weight: float = 1.0,
        imbalance_threshold: float = 0.20,
    ):
        super().__init__(name=name, weight=weight, is_backtestable=True)
        self.imbalance_threshold = imbalance_threshold

    def evaluate(self, context: Dict[str, Any]) -> AgentSignal:
        """
        Avalia o fluxo:
        1. Se `order_book` estiver presente (modo ao vivo): calcula L2 bid-ask imbalance.
        2. Se `flow_metrics` ou velas históricas forem passadas (modo backtest):
           calcula proxy de fluxo via posição intra-vela do fechamento e volume relativo.
        """
        direction = "HOLD"
        confidence = 0.50
        rationale = "Fluxo equilibrado entre compradores e vendedores."
        metrics = {}

        # 1. Modo L2 Live: Profundidade do Livro de Ofertas
        if "order_book" in context and context["order_book"] is not None:
            ob = context["order_book"]
            bid_vol = getattr(ob, "bids_volume", None) or sum(b[1] for b in ob.get("bids", []))
            ask_vol = getattr(ob, "asks_volume", None) or sum(a[1] for a in ob.get("asks", []))
            total_vol = bid_vol + ask_vol

            if total_vol > 0:
                imbalance = (bid_vol - ask_vol) / total_vol
                metrics["imbalance"] = imbalance
                if imbalance >= self.imbalance_threshold:
                    direction = "BUY"
                    confidence = min(0.95, 0.50 + abs(imbalance))
                    rationale = f"Livro L2 com forte pressão compradora (desbalanceamento={imbalance:+.1%})."
                elif imbalance <= -self.imbalance_threshold:
                    direction = "SELL"
                    confidence = min(0.95, 0.50 + abs(imbalance))
                    rationale = f"Livro L2 com forte pressão vendedora (desbalanceamento={imbalance:+.1%})."

        # 2. Modo Histórico / Backtest: Proxy de Volume Delta Intra-Vela
        elif "close" in context and "open" in context and "high" in context and "low" in context:
            close = float(context["close"])
            open_p = float(context["open"])
            high = float(context["high"])
            low = float(context["low"])
            vol = float(context.get("volume", 1.0))
            vol_mean = float(context.get("volume_sma", vol))

            bar_range = max(1e-8, high - low)
            # Posição relativa do fechamento na amplitude da vela: [-1, +1]
            close_position = 2.0 * ((close - low) / bar_range) - 1.0
            vol_ratio = vol / (vol_mean + 1e-8)

            metrics["close_position"] = close_position
            metrics["vol_ratio"] = vol_ratio

            # Funding rate e Z-score
            funding = float(context.get("funding_rate", 0.0))
            metrics["funding_rate"] = funding
            vol_z = context.get("volume_zscore_20")
            dist_vwap = context.get("dist_vwap_pct")
            funding_z = context.get("funding_zscore")

            if vol_z is not None:
                metrics["volume_zscore_20"] = float(vol_z)
            if dist_vwap is not None:
                metrics["dist_vwap_pct"] = float(dist_vwap)
            if funding_z is not None:
                metrics["funding_zscore"] = float(funding_z)

            # Prioridade 1: Volume Institucional Anômalo (Z-score >= 1.5)
            if vol_z is not None and float(vol_z) >= 1.5 and close_position >= 0.35:
                direction = "BUY"
                confidence = min(0.92, 0.55 + min(0.25, float(vol_z) * 0.08))
                rationale = f"Fluxo comprador institucional confirmado (Z-Score Volume={float(vol_z):.2f}, fechamento={close_position:+.2f})."
            elif vol_z is not None and float(vol_z) >= 1.5 and close_position <= -0.35:
                direction = "SELL"
                confidence = min(0.92, 0.55 + min(0.25, float(vol_z) * 0.08))
                rationale = f"Fluxo vendedor institucional confirmado (Z-Score Volume={float(vol_z):.2f}, fechamento={close_position:+.2f})."
            # Prioridade 2: Reversão Elástica / Exaustão em relação ao Rolling VWAP
            elif dist_vwap is not None and float(dist_vwap) <= -2.5 and close_position >= 0.0:
                direction = "BUY"
                confidence = min(0.85, 0.60 + min(0.20, abs(float(dist_vwap)) * 0.05))
                rationale = f"Reversão elástica de VWAP (Preço {float(dist_vwap):.2f}% abaixo do VWAP móvel com suporte)."
            elif dist_vwap is not None and float(dist_vwap) >= 2.5 and close_position <= 0.0:
                direction = "SELL"
                confidence = min(0.85, 0.60 + min(0.20, float(dist_vwap) * 0.05))
                rationale = f"Exaustão sobreestendida de VWAP (Preço {float(dist_vwap):.2f}% acima do VWAP móvel com resistência)."
            # Prioridade 3: Anomalias de Funding Rate Z-score
            elif funding_z is not None and float(funding_z) > 2.0:
                direction = "SELL"
                confidence = min(0.85, 0.65 + min(0.15, (float(funding_z) - 2.0) * 0.08))
                rationale = f"Funding Rate anômalo (Z-Score={float(funding_z):.2f} > +2.0): sobrealavancagem long e risco de cascata."
            elif funding_z is not None and float(funding_z) < -2.0:
                direction = "BUY"
                confidence = min(0.85, 0.65 + min(0.15, (abs(float(funding_z)) - 2.0) * 0.08))
                rationale = f"Funding Rate anômalo (Z-Score={float(funding_z):.2f} < -2.0): pessimismo extremo e risco de short squeeze."
            # Prioridade 4: Sinais Clássicos de Volume Ratio e Fechamento Extremo
            elif vol_ratio >= 1.2 and close_position >= 0.50:
                direction = "BUY"
                confidence = min(0.90, 0.50 + (close_position * 0.3) + min(0.15, (vol_ratio - 1.0) * 0.1))
                rationale = f"Fluxo comprador dominante (fechamento a {close_position:+.2f} do range, volume {vol_ratio:.1f}x da média)."
            elif vol_ratio >= 1.2 and close_position <= -0.50:
                direction = "SELL"
                confidence = min(0.90, 0.50 + (abs(close_position) * 0.3) + min(0.15, (vol_ratio - 1.0) * 0.1))
                rationale = f"Fluxo vendedor agressivo (fechamento a {close_position:+.2f} do range, volume {vol_ratio:.1f}x da média)."
            elif funding > 0.0003:
                direction = "SELL"
                confidence = 0.65
                rationale = f"Funding rate excessivo ({funding:+.4%}), sugerindo exaustão de longs."
            elif funding < -0.0003:
                direction = "BUY"
                confidence = 0.65
                rationale = f"Funding rate negativo ({funding:+.4%}), propício para short squeeze."
            # Prioridade 5: Neurônio de Fluxo Ativo (Microestrutura Intra-Bar)
            elif close_position >= 0.20:
                direction = "BUY"
                confidence = min(0.82, 0.58 + close_position * 0.22)
                rationale = f"Neurônio de Fluxo Ativo: Absorção compradora no fechamento da vela (posição={close_position:+.2f}, range={bar_range:.2f})."
            elif close_position <= -0.20:
                direction = "SELL"
                confidence = min(0.82, 0.58 + abs(close_position) * 0.22)
                rationale = f"Neurônio de Fluxo Ativo: Rejeição vendedora no fechamento da vela (posição={close_position:+.2f}, range={bar_range:.2f})."

        return AgentSignal(
            agent_name=self.name,
            direction=direction,
            confidence=confidence,
            weight=self.weight,
            rationale=rationale,
            metrics=metrics,
            timestamp=context.get("timestamp"),
        )
