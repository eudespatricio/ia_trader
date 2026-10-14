"""
Contrato Base e Interfaces dos Agentes Especialistas da Fase 4.
Define o formato padronizado do sinal (AgentSignal) e a classe abstrata BaseAgent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional


@dataclass
class AgentSignal:
    """
    Contrato canônico de decisão emitido por cada agente especialista.
    """
    agent_name: str
    direction: str          # 'BUY', 'SELL', 'HOLD'
    confidence: float       # [0.0, 1.0]
    weight: float           # Peso ponderado no comitê
    rationale: str          # Explicação técnica/fundamentalista sintética
    is_veto: bool = False   # Se True, aciona veto incondicional (ex: RiskSentinel)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[int] = None

    def __post_init__(self):
        self.direction = self.direction.upper()
        if self.direction not in ["BUY", "SELL", "HOLD"]:
            raise ValueError(f"Direção inválida no AgentSignal: '{self.direction}'. Deve ser BUY, SELL ou HOLD.")
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        self.weight = float(max(0.0, self.weight))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseAgent(ABC):
    """
    Classe base abstrata para todos os agentes especialistas do ecossistema IA Trader.
    """

    def __init__(self, name: str, weight: float = 1.0, is_backtestable: bool = True):
        self.name = name
        self.weight = weight
        self.is_backtestable = is_backtestable

    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> AgentSignal:
        """
        Avalia o contexto de mercado recebido e emite um AgentSignal determinístico.
        - `context` pode conter: candles (DataFrame), index atual, indicadores, macro, orderbook, etc.
        """
        pass

    async def evaluate_async(self, context: Dict[str, Any]) -> AgentSignal:
        """
        Execução assíncrona padrão para orquestração paralela sem bloqueio de I/O.
        """
        return self.evaluate(context)
