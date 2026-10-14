"""
Stub do Assistente LLM (ChatGPT / Claude / Local LLM) para a Fase 4.
IMPORTANTE:
- Desativado por padrão (`enabled=False`).
- NUNCA é invocado ou utilizado em backtest histórico (não-determinístico e sem série histórica).
- Utilizado estritamente como árbitro opcional em live trading com timeout de 2.5s.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("ia_trader.chatgpt_assistant")


class ChatGPTAssistant:
    """
    Interface controlada para chamadas LLM de alto nível.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", enabled: bool = False):
        self.api_key = api_key
        self.model = model
        self.enabled = enabled

    async def analyze_async(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Execução assíncrona com verificação de habilitação estrita.
        Se desativado, retorna None imediatamente sem realizar chamadas de rede.
        """
        if not self.enabled or not self.api_key:
            logger.debug("Assistente LLM desativado por governança (stub ativo).")
            return None

        # Stub de fallback para quando habilitado em ambiente live
        return {
            "direction": "HOLD",
            "confidence": 0.50,
            "rationale": "Arbitragem LLM stub (desativada no ambiente padrão de simulação).",
        }
