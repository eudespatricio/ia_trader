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

    def explain_trade_decision(self, trade_or_signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gera uma explicação em linguagem natural detalhada sobre uma decisão operacional
        ou estado de uma posição (aberta, fechada ou deliberação neutra).
        """
        direction = str(
            trade_or_signal.get("direction")
            or trade_or_signal.get("side")
            or trade_or_signal.get("action")
            or "HOLD"
        ).upper()
        symbol = str(trade_or_signal.get("symbol", "Ativo")).upper()
        price = float(trade_or_signal.get("entry_price") or trade_or_signal.get("price") or 0.0)
        cur_price = float(trade_or_signal.get("current_price") or price)
        sl = float(trade_or_signal.get("stop_loss") or 0.0)
        tp = float(trade_or_signal.get("take_profit") or trade_or_signal.get("tp3_price") or 0.0)
        tp1 = float(trade_or_signal.get("tp1_price") or trade_or_signal.get("tp1") or 0.0)
        tp2 = float(trade_or_signal.get("tp2_price") or trade_or_signal.get("tp2") or 0.0)
        tp3 = float(trade_or_signal.get("tp3_price") or trade_or_signal.get("tp3") or 0.0)
        tp1_hit = bool(trade_or_signal.get("tp1_hit"))
        tp2_hit = bool(trade_or_signal.get("tp2_hit"))
        status = str(trade_or_signal.get("status") or "").upper()
        strat = str(trade_or_signal.get("strategy_type") or "Smart Trade").upper()
        lev = float(trade_or_signal.get("leverage") or 1.0)
        rec_pnl = float(trade_or_signal.get("realized_pnl") or 0.0)
        rationale_raw = str(trade_or_signal.get("rationale") or "")
        regime = str(trade_or_signal.get("regime") or "Neutro")
        is_open = status == "OPEN"
        is_closed = status in ("CLOSED", "STOPPED", "LIQUIDATED")

        is_buy = any(k in direction for k in ("BUY", "LONG"))
        is_sell = any(k in direction for k in ("SELL", "SHORT"))

        status_tag = ""
        if is_open:
            if tp2_hit:
                status_tag = " · TP2 Atingido (Trailing Ativo)"
            elif tp1_hit:
                status_tag = " · TP1 Atingido ✔ (Risco Zero)"
            else:
                status_tag = " · Em Andamento"
        elif is_closed:
            pnl_sign = "+$" if rec_pnl >= 0 else "-$"
            status_tag = f" · Encerrada ({pnl_sign}{abs(rec_pnl):.2f})"

        lev_str = f" ({lev:.0f}x)" if lev > 1.0 else ""
        px_fmt = f"${price:,.2f}" if price >= 1.0 else f"${price:.4f}"

        if is_buy:
            action_desc = f"Compra (Long) em {symbol}{lev_str} cotado a {px_fmt}{status_tag}"
            why_bullets = [
                f"O Comitê de Agentes identificou confluência de fatores técnicos favoráveis à alta ({strat}).",
                f"Regime de mercado alinhado com o contexto estrutural vigente ({regime}).",
            ]
            if "SWEEP" in rationale_raw or "POI" in rationale_raw:
                why_bullets.append("Identificamos uma zona de liquidez mitigada (Smart Money) onde os stops de venda foram varridos, gerando uma entrada assimétrica com risco controlado.")
            elif rationale_raw:
                why_bullets.append(rationale_raw)

            if is_open and tp1_hit:
                tp1_str = f"${tp1:,.2f}" if tp1 else "atingido"
                sl_str = f"${sl:,.2f}" if sl else "breakeven"
                tp2_str = f"${tp2:,.2f}" if tp2 else "alvo 2"
                tp3_str = f"${tp3:,.2f}" if tp3 else "alvo 3"
                risk_desc = (
                    f"Alvo parcial TP1 ({tp1_str}) já foi executado (+${rec_pnl:,.2f} no saldo). "
                    f"Stop Loss elevado para Breakeven ({sl_str}), eliminando o risco da operação (risco zero). "
                    f"Fração remanescente busca TP2 ({tp2_str}) e TP3 ({tp3_str})."
                )
            elif sl > 0 and tp > 0:
                risk_dist = abs(price - sl)
                reward_dist = abs(tp - price)
                rr_ratio = (reward_dist / (risk_dist + 1e-12))
                risk_desc = f"Stop Loss posicionado em ${sl:,.2f} e Take Profit em ${tp:,.2f}, resultando em uma relação de retorno de {rr_ratio:.1f}x o risco assumido."
            else:
                risk_desc = "Gestão de risco quantitativa baseada em dimensionamento de volatilidade ATR."

        elif is_sell:
            action_desc = f"Venda (Short) em {symbol}{lev_str} cotado a {px_fmt}{status_tag}"
            why_bullets = [
                f"O Comitê detectou exaustão compradora e pressão vendedora institucional ({strat}).",
                f"Estrutura de topos descendentes ou rejeição alinhada com o regime vigente ({regime}).",
            ]
            if "SWEEP" in rationale_raw or "POI" in rationale_raw:
                why_bullets.append("Rejeição em topo institucional com varredura prévia de liquidez compradora.")
            elif rationale_raw:
                why_bullets.append(rationale_raw)

            if is_open and tp1_hit:
                tp1_str = f"${tp1:,.2f}" if tp1 else "atingido"
                sl_str = f"${sl:,.2f}" if sl else "breakeven"
                tp2_str = f"${tp2:,.2f}" if tp2 else "alvo 2"
                tp3_str = f"${tp3:,.2f}" if tp3 else "alvo 3"
                risk_desc = (
                    f"Alvo parcial TP1 ({tp1_str}) já foi executado (+${rec_pnl:,.2f} no saldo). "
                    f"Stop Loss ajustado para Breakeven ({sl_str}), eliminando o risco da operação (risco zero). "
                    f"Fração remanescente busca TP2 ({tp2_str}) e TP3 ({tp3_str})."
                )
            elif sl > 0 and tp > 0:
                risk_dist = abs(sl - price)
                reward_dist = abs(price - tp)
                rr_ratio = (reward_dist / (risk_dist + 1e-12))
                risk_desc = f"Stop Loss protegido em ${sl:,.2f} e Take Profit em ${tp:,.2f} (relação R:R={rr_ratio:.1f}:1)."
            else:
                risk_desc = "Stop Loss posicionado estrategicamente acima da máxima de rejeição via ATR."

        else:
            action_desc = f"Aguardar (Hold / Neutro) em {symbol} cotado a {px_fmt}"
            why_bullets = [
                f"Nenhuma confluência assimétrica encontrada para {symbol} nas condições atuais.",
                f"Regime de mercado ({regime}): robô opera apenas com confluência de fluxo e sem vetos de risco.",
            ]
            if rationale_raw:
                why_bullets.append(rationale_raw)
            risk_desc = "Preservação de capital é a prioridade número um: sem confluência estatística, o robô permanece líquido."

        if is_open:
            summary = (
                f"Operação {strat} em andamento. "
                + (f"TP1 já executado (+${rec_pnl:,.2f} garantidos) com stop em Breakeven (risco zero). " if tp1_hit else "")
                + f"O Comitê segue monitorando o fluxo de {symbol} rumo aos alvos da escada."
            )
        elif is_closed:
            pnl_sign = "+$" if rec_pnl >= 0 else "-$"
            summary = (
                f"Trade {strat} encerrado em {symbol}. Resultado consolidado: {pnl_sign}{abs(rec_pnl):.2f}. "
                f"Execução seguiu estritamente o protocolo institucional de gestão de risco."
            )
        else:
            summary = (
                f"O Comitê de Agentes mantém postura neutra para {symbol} cotado a {px_fmt} no regime {regime}. "
                f"Aguardando confluência matemática entre fluxo, order book e tendência antes de expor margem."
            )

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
            "regime": regime,
            "is_open": is_open,
            "direction": "BUY" if is_buy else ("SELL" if is_sell else "HOLD"),
            "status": status or ("OPEN" if is_open else "DELIBERATION"),
            "summary": summary,
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
