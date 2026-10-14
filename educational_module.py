"""
educational_module.py - Módulo Educacional Interativo e Explicabilidade Didática de Decisões.
Fase 9 do Projeto IA Trader (Páginas 3 e 6 da Documentação Oficial).

Objetivo:
1. Tradução de Métricas Complexas para Linguagem Humana Acessível:
   - Desmistifica sinais quantitativos complexos (ex: "Consenso BUY +0.65", "Sweep de SSL", "Fair Value Gap")
     em narrativas didáticas e compreensíveis ("Por que a IA decidiu comprar aqui?").
2. Auditoria e Racional de Risco:
   - Explica o porquê do Stop Loss ter sido colocado naquele nível específico.
   - Detalha a relação Risco:Retorno e como as taxas foram consideradas.
3. Glossário Educativo Integrado:
   - Termos quantitativos explicados de forma clara (Order Blocks, Fair Value Gaps, Liquidity Sweeps, ATR, Funding Carry, Circuit Breakers).
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ia_trader.educational_module")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [educational_module] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class EducationalExplainer:
    """
    Gerador de explicações pedagógicas sobre as decisões do robô e conceitos de mercado.
    """

    GLOSSARY: Dict[str, str] = {
        "ORDER_BLOCK": (
            "Order Block (OB) é uma região de preço onde grandes instituições financeiras (bancos, fundos e formadores de mercado) "
            "acumularam ordens volumosas antes de um movimento expressivo de expansão. Quando o preço retorna a essa região, "
            "costuma encontrar forte suporte ou resistência institucional."
        ),
        "FAIR_VALUE_GAP": (
            "Fair Value Gap (FVG) ou desbalanceamento de preço ocorre quando há um movimento muito rápido e agressivo em 3 velas consecutivas, "
            "deixando uma faixa de preço onde apenas uma das pontas (compradora ou vendedora) foi preenchida. O mercado tende a retornar "
            "para reequilibrar essa faixa (mitigação)."
        ),
        "LIQUIDITY_SWEEP": (
            "Liquidity Sweep (Varredura de Liquidez ou 'Stop Hunt') ocorre quando o preço rapidamente fura um topo ou fundo anterior "
            "para capturar as ordens de stop dos traders de varejo, mas fecha imediatamente de volta dentro da região com uma rejeição expressiva (pavio longo). "
            "É um dos sinais de maior probabilidade de reversão."
        ),
        "ATR": (
            "Average True Range (ATR) é a medida matemática da volatilidade real do ativo ao longo de 14 períodos. "
            "Nossa IA utiliza o ATR para calibrar dinamicamente o Stop Loss e o Take Profit, garantindo que o stop não seja curto demais a ponto "
            "de ser pego por ruído, nem longo demais a ponto de gerar risco desproporcional."
        ),
        "FUNDING_RATE": (
            "Taxa de Financiamento periódica cobrada ou paga entre posições compradas e vendidas no mercado futuro perpétuo. "
            "Quando a taxa está muito alta, os comprados pagam aos vendidos a cada 8 horas, sinalizando euforia e sobrealavancagem."
        ),
        "CIRCUIT_BREAKER": (
            "Mecanismo de segurança automática que bloqueia novas operações em caso de perdas consecutivas (cooldown) ou quando o drawdown "
            "atinge o limite de tolerância pré-estabelecido, protegendo a carteira contra falhas emocionais ou eventos extremos de mercado."
        ),
        "RISK_REWARD": (
            "Relação Risco:Retorno (R:R). Indica quantos dólares o sistema busca ganhar para cada dólar arriscado no Stop Loss. "
            "Operações com R:R de 2.5:1 ou 3:1 permitem que o sistema seja altamente lucrativo mesmo com taxas de acerto moderadas."
        ),
    }

    def explain_trade_decision(self, trade_or_signal: Dict[str, Any]) -> Dict[str, str]:
        """
        Gera uma explicação em linguagem natural detalhada sobre uma decisão operacional.
        """
        direction = str(trade_or_signal.get("direction", trade_or_signal.get("action", "HOLD"))).upper()
        symbol = trade_or_signal.get("symbol", "Ativo")
        price = trade_or_signal.get("entry_price", trade_or_signal.get("price", 0.0))
        sl = trade_or_signal.get("stop_loss", 0.0)
        tp = trade_or_signal.get("take_profit", 0.0)
        rationale_raw = str(trade_or_signal.get("rationale", ""))
        regime = trade_or_signal.get("regime", "Neutro")

        if "BUY" in direction or "LONG" in direction:
            action_desc = f"Compra (Long) em {symbol} cotado a ${price:,.2f}"
            why_bullets = [
                f"O Comitê de Agentes identificou confluência de fatores técnicos favoráveis à alta.",
                f"Regime de mercado alinhado com o contexto estrutural vigente ({regime}).",
            ]
            if "SWEEP" in rationale_raw or "POI" in rationale_raw:
                why_bullets.append("Identificamos uma zona de liquidez mitigada (Smart Money) onde os stops de venda foram varridos, gerando uma entrada assimétrica com risco controlado.")
            if sl > 0 and tp > 0:
                risk_dist = abs(price - sl)
                reward_dist = abs(tp - price)
                rr_ratio = (reward_dist / (risk_dist + 1e-12))
                risk_desc = f"Stop Loss posicionado em ${sl:,.2f} e Take Profit em ${tp:,.2f}, resultando em uma relação de retorno de {rr_ratio:.1f}x o risco assumido."
            else:
                risk_desc = "Gestão de risco baseada em dimensionamento de volatilidade e proteção de capital."

        elif "SELL" in direction or "SHORT" in direction:
            action_desc = f"Venda (Short) em {symbol} cotado a ${price:,.2f}"
            why_bullets = [
                f"O Comitê detectou exaustão compradora e pressão vendedora institucional.",
                f"Estrutura de topos descendentes ou rejeição em resistência importante.",
            ]
            if sl > 0 and tp > 0:
                risk_dist = abs(sl - price)
                reward_dist = abs(price - tp)
                rr_ratio = (reward_dist / (risk_dist + 1e-12))
                risk_desc = f"Stop Loss protegido em ${sl:,.2f} e Take Profit em ${tp:,.2f} (relação R:R={rr_ratio:.1f}:1)."
            else:
                risk_desc = "Stop Loss posicionado estrategicamente acima da máxima de rejeição."

        else:
            action_desc = f"Aguardar (Hold / Neutro) em {symbol}"
            why_bullets = [
                "Nenhuma confluência de alta probabilidade foi encontrada no momento.",
                "O Sentinela de Risco recomendou cautela para preservar capital contra ruído de mercado."
            ]
            risk_desc = "Preservação de capital é a prioridade número um quando não há vantagem estatística clara."

        narrative = (
            f"### 🎓 O que a IA está fazendo?\n"
            f"**Operação:** {action_desc}.\n\n"
            f"**Por que entramos nesse trade?**\n"
            + "\n".join([f"- {b}" for b in why_bullets]) + "\n\n"
            f"**Como o risco está sendo controlado?**\n"
            f"{risk_desc}\n\n"
            f"**Lição de Trading Institucional:**\n"
            f"Nunca operamos por intuição. Cada ordem é fruto de um consenso matemático que exige convergência entre fluxo, estrutura e controle de custos."
        )

        return {
            "title": action_desc,
            "action": action_desc,
            "narrative": narrative,
            "risk_analysis": risk_desc,
            "raw_rationale": rationale_raw
        }

    def get_term_definition(self, term_key: str) -> Optional[str]:
        """Consulta o significado didático de um conceito."""
        return self.GLOSSARY.get(term_key.upper())

    def list_all_terms(self) -> Dict[str, str]:
        """Retorna o glossário completo."""
        return dict(self.GLOSSARY)

    def get_glossary(self) -> Dict[str, str]:
        """Alias para list_all_terms."""
        return self.list_all_terms()
