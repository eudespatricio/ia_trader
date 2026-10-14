"""
Agente Sentinela de Risco da Fase 4.
Possui PODER DE VETO ABSOLUTO sobre o Comitê de Decisão.
Se qualquer parâmetro de risco ou volatilidade for violado, o Sentinela veta sumariamente
qualquer nova operação, forçando o sinal consolidado para 'HOLD'.
"""

from __future__ import annotations

from typing import Any, Dict
from agents.base_agent import BaseAgent, AgentSignal


class RiskSentinelAgent(BaseAgent):
    """
    Guardião de sobrevivência e mitigação de risco de cauda (Tail Risk / Black Swan).
    Executa regras reproduzíveis e determinísticas tanto em live quanto no backtesting.
    """

    def __init__(
        self,
        name: str = "RiskSentinelAgent",
        weight: float = 1.0,
        max_atr_volatility_multiplier: float = 2.5,
        min_relative_volume: float = 0.15,
        max_spread_pct: float = 0.0015,  # 0.15% de spread máximo no livro
    ):
        super().__init__(name=name, weight=weight, is_backtestable=True)
        self.max_atr_volatility_multiplier = max_atr_volatility_multiplier
        self.min_relative_volume = min_relative_volume
        self.max_spread_pct = max_spread_pct

    def evaluate(self, context: Dict[str, Any]) -> AgentSignal:
        """
        Verifica os critérios de proteção e veto:
        1. Veto por Explosão de Volatilidade: ATR atual > 2.5x a mediana histórica do ATR.
        2. Veto por Iliquidez Súbita: Volume da barra < 15% do volume médio (risco de derrapagem massiva).
        3. Veto por Spread Excessivo: Spread bid-ask > 0.15% no livro de ofertas.
        4. Veto por Janela de Evento Crítico: Divulgação agendada de CPI / decisão de juros FOMC.
        """
        # 1. Checagem de Volatilidade Anômala (ATR Spike)
        current_atr = context.get("atr_14")
        baseline_atr = context.get("atr_median") or context.get("atr_sma")
        if current_atr is not None and baseline_atr is not None:
            if float(baseline_atr) > 0 and (float(current_atr) / float(baseline_atr)) > self.max_atr_volatility_multiplier:
                ratio = float(current_atr) / float(baseline_atr)
                return AgentSignal(
                    agent_name=self.name,
                    direction="HOLD",
                    confidence=1.0,
                    weight=self.weight,
                    rationale=f"VETO DE RISCO: Volatilidade anômala detectada (ATR {ratio:.2f}x acima da linha de base). Risco de cauda iminente.",
                    is_veto=True,
                    metrics={"veto_type": "VOLATILITY_SPIKE", "atr_ratio": ratio},
                    timestamp=context.get("timestamp"),
                )

        # 2. Checagem de Iliquidez Súbita (Volume Crash)
        volume = context.get("volume")
        vol_mean = context.get("volume_sma")
        if volume is not None and vol_mean is not None:
            if float(vol_mean) > 0 and (float(volume) / float(vol_mean)) < self.min_relative_volume:
                v_ratio = float(volume) / float(vol_mean)
                return AgentSignal(
                    agent_name=self.name,
                    direction="HOLD",
                    confidence=1.0,
                    weight=self.weight,
                    rationale=f"VETO DE RISCO: Volume crítico residual ({v_ratio:.1%} da média). Risco severo de slippage e falta de contraparte.",
                    is_veto=True,
                    metrics={"veto_type": "ILLIQUIDITY", "vol_ratio": v_ratio},
                    timestamp=context.get("timestamp"),
                )

        # 3. Checagem de Spread no Livro L2 (Live)
        spread = context.get("spread_pct")
        if spread is not None and float(spread) > self.max_spread_pct:
            return AgentSignal(
                agent_name=self.name,
                direction="HOLD",
                confidence=1.0,
                weight=self.weight,
                rationale=f"VETO DE RISCO: Spread do livro ({float(spread):.3%}) excede o limite tolerável ({self.max_spread_pct:.3%}).",
                is_veto=True,
                metrics={"veto_type": "WIDE_SPREAD", "spread_pct": float(spread)},
                timestamp=context.get("timestamp"),
            )

        # 4. Checagem de Janela de Evento Macroeconômico de Alto Impacto
        if context.get("is_cpi_fomc_window", False):
            return AgentSignal(
                agent_name=self.name,
                direction="HOLD",
                confidence=1.0,
                weight=self.weight,
                rationale="VETO DE RISCO: Janela temporal de divulgação de dados macro de altíssimo impacto (FOMC / CPI). Operações bloqueadas.",
                is_veto=True,
                metrics={"veto_type": "SCHEDULED_MACRO_EVENT"},
                timestamp=context.get("timestamp"),
            )

        # 5. Checagem de Volatilidade Extrema Normalizada (NATR > 12%)
        natr = context.get("natr_14")
        if natr is not None and float(natr) > 12.0:
            return AgentSignal(
                agent_name=self.name,
                direction="HOLD",
                confidence=1.0,
                weight=self.weight,
                rationale=f"VETO DE RISCO: Volatilidade normalizada em nível de colapso/flash crash (NATR={float(natr):.2f}% > 12.0%).",
                is_veto=True,
                metrics={"veto_type": "EXTREME_NATR", "natr_14": float(natr)},
                timestamp=context.get("timestamp"),
            )

        # 6. Checagem de Desequilíbrio Crítico de Funding (|Z| > 4.0)
        funding_z = context.get("funding_zscore")
        if funding_z is not None and abs(float(funding_z)) > 4.0:
            return AgentSignal(
                agent_name=self.name,
                direction="HOLD",
                confidence=1.0,
                weight=self.weight,
                rationale=f"VETO DE RISCO: Anomalia extrema no mercado de derivativos (Funding Z-Score={float(funding_z):.2f}). Risco crítico de liquidação em cascata.",
                is_veto=True,
                metrics={"veto_type": "EXTREME_FUNDING_DISLOCATION", "funding_zscore": float(funding_z)},
                timestamp=context.get("timestamp"),
            )

        # 7. Checagem de Risco Estrutural de Entrada (S/R Imediato sem Absorção)
        intended_dir = context.get("intended_direction")
        dist_res = context.get("dist_resistance_pct")
        dist_supp = context.get("dist_support_pct")
        sweep_bull = bool(context.get("liquidity_sweep_bull", 0))
        sweep_bear = bool(context.get("liquidity_sweep_bear", 0))

        if intended_dir == "BUY" and dist_res is not None and not sweep_bull:
            if float(dist_res) < 0.15:
                return AgentSignal(
                    agent_name=self.name,
                    direction="HOLD",
                    confidence=1.0,
                    weight=self.weight,
                    rationale=f"VETO DE RISCO ESTRUTURAL: Compra bloqueada imediatamente abaixo de resistência maior ({float(dist_res):.2f}%). Relação R:R assimétrica desfavorável.",
                    is_veto=True,
                    metrics={"veto_type": "RESISTANCE_PROXIMITY", "dist_resistance_pct": float(dist_res)},
                    timestamp=context.get("timestamp"),
                )
        elif intended_dir == "SELL" and dist_supp is not None and not sweep_bear:
            if float(dist_supp) < 0.15:
                return AgentSignal(
                    agent_name=self.name,
                    direction="HOLD",
                    confidence=1.0,
                    weight=self.weight,
                    rationale=f"VETO DE RISCO ESTRUTURAL: Venda bloqueada imediatamente acima de suporte maior ({float(dist_supp):.2f}%). Risco crítico de armadilha de urso.",
                    is_veto=True,
                    metrics={"veto_type": "SUPPORT_PROXIMITY", "dist_support_pct": float(dist_supp)},
                    timestamp=context.get("timestamp"),
                )

        # Se nenhum critério de risco foi violado, o Sentinela libera a operação
        return AgentSignal(
            agent_name=self.name,
            direction="HOLD",
            confidence=0.50,
            weight=self.weight,
            rationale="Parâmetros de risco, volatilidade e liquidez dentro dos limites operacionais seguros.",
            is_veto=False,
            metrics={"status": "CLEAR"},
            timestamp=context.get("timestamp"),
        )

