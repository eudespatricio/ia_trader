"""
Stub do Assistente LLM (ChatGPT / Claude / Local LLM) para a Fase 4.
IMPORTANTE:
- Desativado por padrão (`enabled=False`).
- NUNCA é invocado ou utilizado em backtest histórico (não-determinístico e sem série histórica).
- Utilizado estritamente como árbitro opcional em live trading com timeout de 2.5s.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger("ia_trader.chatgpt_assistant")


def _get_default_ollama_url() -> str:
    """Resolve o endpoint do Ollama (variável de ambiente, localhost ou gateway WSL)."""
    env_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")
    if env_url:
        return env_url.rstrip("/")

    # Tenta descobrir o gateway Windows a partir do WSL2
    try:
        host_ip = (
            subprocess.check_output("ip route show default 2>/dev/null | awk '{print $3}'", shell=True)
            .decode()
            .strip()
        )
        if host_ip:
            return f"http://{host_ip}:11434"
    except Exception:
        pass

    return "http://127.0.0.1:11434"


class ChatGPTAssistant:
    """
    Interface controlada para chamadas LLM de alto nível (Local Ollama ou OpenAI).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama3.1:latest",
        provider: str = "ollama",
        base_url: Optional[str] = None,
        enabled: bool = False,
        timeout: float = 5.0,
    ):
        self.api_key = api_key
        self.model = model
        self.provider = provider.lower()
        self.base_url = (base_url.rstrip("/") if base_url else None) or _get_default_ollama_url()
        self.enabled = enabled
        self.timeout = timeout

    async def analyze_async(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Execução assíncrona com verificação de habilitação estrita.
        Se desativado, retorna None imediatamente sem realizar chamadas de rede.
        """
        if not self.enabled:
            logger.debug("Assistente LLM desativado por governança (stub ativo).")
            return None

        if self.provider == "openai" and not self.api_key:
            logger.warning("Assistente OpenAI habilitado mas sem chave de API.")
            return None

        if self.provider == "ollama":
            return await self._call_ollama(prompt)

        # Fallback stub para outros provedores
        return {
            "direction": "HOLD",
            "confidence": 0.50,
            "rationale": "Arbitragem LLM stub (desativada no ambiente padrão de simulação).",
        }

    async def _call_ollama(self, prompt: str) -> Dict[str, Any]:
        """Chama o Ollama local com formatação JSON forçada e timeout controlado."""
        def _request_sync() -> Dict[str, Any]:
            payload = json.dumps({
                "model": self.model,
                "prompt": (
                    f"{prompt}\n\n"
                    "Responda EXCLUSIVAMENTE em formato JSON com o seguinte schema:\n"
                    '{"direction": "BUY"|"SELL"|"HOLD", "confidence": float_entre_0_e_1, "rationale": "motivo conciso"}'
                ),
                "format": "json",
                "stream": False,
                "options": {
                    "num_predict": 80,
                    "temperature": 0.1,
                },
            }).encode("utf-8")

            url = f"{self.base_url}/api/generate"
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                parsed = json.loads(data.get("response", "{}"))
                direction = str(parsed.get("direction", "HOLD")).upper()
                if direction not in ("BUY", "SELL", "HOLD"):
                    direction = "HOLD"
                confidence = float(parsed.get("confidence", 0.50))
                confidence = max(0.0, min(1.0, confidence))
                rationale = str(parsed.get("rationale", "Inferência local Ollama."))
                return {
                    "direction": direction,
                    "confidence": confidence,
                    "rationale": rationale,
                }

        try:
            return await asyncio.to_thread(_request_sync)
        except Exception as e:
            logger.warning("Falha na chamada ao Ollama local: %s. Retornando HOLD seguro.", e)
            return {
                "direction": "HOLD",
                "confidence": 0.50,
                "rationale": f"Falha/Timeout no Ollama: {e}",
            }

