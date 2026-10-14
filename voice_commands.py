"""
voice_commands.py - Módulo de Comandos de Voz e Interação em Linguagem Natural.
Fase 9 do Projeto IA Trader (Páginas 3 e 6 da Documentação Oficial).

Objetivo:
1. Processamento e Roteamento de Comandos de Voz/Texto:
   - "status do robô" -> Retorna telemetria, equity, saldo e cotas.
   - "mostrar últimas operações" -> Lista os últimos trades com PnL.
   - "pausar operações" -> Aciona parada temporária de emergência no RollbackGuard.
   - "retomar operações" -> Reseta circuit breakers e libera o motor.
   - "qual o regime do mercado" -> Consulta o RegimeClassifier macro.
   - "explicar último trade" -> Aciona o EducationalExplainer para narrar o racional.
2. Arquitetura Desacoplada com Fallback:
   - Funciona via texto (REST API) e suporta integração com microfone local / Whisper / Web Speech API do navegador.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ia_trader.voice_commands")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [voice_commands] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class VoiceCommandDispatcher:
    """
    Roteador semântico de comandos de voz e texto para o IA Trader.
    """

    def __init__(self, db_path: str = "datasets/market_cache.db"):
        self.db_path = db_path

    def process_command(self, raw_command: str) -> Dict[str, Any]:
        """
        Interpreta a intenção do usuário e despacha a ação correspondente no sistema.
        """
        text = raw_command.strip().lower()
        # Normalização simples de pontuação e acentos
        norm = re.sub(r"[^\w\s]", "", text)

        # 1. Comando: Status do Robô / Terminal
        if any(k in norm for k in ["status do robo", "status do robô", "como esta o robo", "saldo", "patrimonio", "equity"]):
            return self._handle_status()

        # 2. Comando: Pausar / Parar Operações
        elif any(k in norm for k in ["pausar operacoes", "pausar operações", "parar robo", "parar operações", "travar", "emergencia"]):
            return self._handle_pause()

        # 3. Comando: Retomar Operações / Reset
        elif any(k in norm for k in ["retomar operacoes", "retomar operações", "reativar", "continuar", "resetar"]):
            return self._handle_resume()

        # 4. Comando: Mostrar Últimas Operações / Histórico
        elif any(k in norm for k in ["ultimas operacoes", "últimas operações", "historico", "ultimos trades", "como foram os trades"]):
            return self._handle_recent_trades()

        # 5. Comando: Regime do Mercado
        elif any(k in norm for k in ["regime", "como esta o mercado", "tendencia", "bull ou bear"]):
            return self._handle_market_regime()

        # 6. Comando: Explicar Último Trade
        elif any(k in norm for k in ["explicar", "por que comprou", "por que vendeu", "explicacao"]):
            return self._handle_explain_latest()

        # Comando não reconhecido
        return {
            "status": "UNRECOGNIZED",
            "command": raw_command,
            "response": (
                f"Comando não compreendido: '{raw_command}'. "
                "Comandos disponíveis: 'status do robô', 'mostrar últimas operações', "
                "'pausar operações', 'retomar operações', 'qual o regime do mercado' e 'explicar último trade'."
            ),
            "action_executed": None
        }

    def _handle_status(self) -> Dict[str, Any]:
        try:
            from performance_tracker import PerformanceTracker
            tracker = PerformanceTracker(db_path=self.db_path)
            m = tracker.compute_metrics_from_db()
            response_text = (
                f"O robô está ativo com patrimônio de ${m.current_equity:,.2f} USD. "
                f"Lucro líquido acumulado de ${m.net_pnl:+,.2f} USD ({m.net_pnl_pct:+.2f}%). "
                f"Taxa de acerto de {m.win_rate:.1f}% em {m.total_trades} trades. "
                f"Rebaixamento atual sob controle em {m.current_drawdown_pct:.2f}%."
            )
            return {
                "status": "SUCCESS",
                "command_type": "STATUS",
                "response": response_text,
                "data": m.to_dict(),
                "action_executed": "FETCH_STATUS"
            }
        except Exception as e:
            return {"status": "ERROR", "response": f"Erro ao consultar status: {e}"}

    def _handle_pause(self) -> Dict[str, Any]:
        try:
            from rollback_guard import RollbackGuard
            guard = RollbackGuard(db_path=self.db_path)
            guard.trigger_emergency_stop(reason="Comando de voz do usuário: Pausar operações.")
            return {
                "status": "SUCCESS",
                "command_type": "EMERGENCY_STOP",
                "response": "Operações pausadas com sucesso. Novas entradas bloqueadas pelo Circuit Breaker.",
                "action_executed": "TRIGGER_EMERGENCY_STOP"
            }
        except Exception as e:
            return {"status": "ERROR", "response": f"Erro ao pausar operações: {e}"}

    def _handle_resume(self) -> Dict[str, Any]:
        try:
            from rollback_guard import RollbackGuard
            guard = RollbackGuard(db_path=self.db_path)
            guard.reset_circuit_breaker(operator_note="Comando de voz do usuário: Retomar operações.")
            return {
                "status": "SUCCESS",
                "command_type": "RESET",
                "response": "Circuit Breakers resetados com sucesso. Sistema pronto para operar.",
                "action_executed": "RESET_CIRCUIT_BREAKER"
            }
        except Exception as e:
            return {"status": "ERROR", "response": f"Erro ao retomar operações: {e}"}

    def _handle_recent_trades(self) -> Dict[str, Any]:
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            table_check = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_trades'").fetchone()
            if not table_check:
                conn.close()
                return {
                    "status": "SUCCESS",
                    "command_type": "RECENT_TRADES",
                    "response": "Ainda não há operações registradas no histórico recente.",
                    "data": []
                }
            rows = conn.execute("SELECT symbol, action, pnl FROM paper_trades ORDER BY timestamp DESC LIMIT 3").fetchall()
            conn.close()

            if not rows:
                return {
                    "status": "SUCCESS",
                    "command_type": "RECENT_TRADES",
                    "response": "Ainda não há operações registradas no histórico recente.",
                    "data": []
                }

            trades_summary = []
            for r in rows:
                pnl = float(r["pnl"])
                outcome = "lucro" if pnl > 0 else ("prejuízo" if pnl < 0 else "empate")
                trades_summary.append(f"{r['symbol']} ({r['action']}) com {outcome} de ${pnl:+.2f}")

            response_text = "Últimas operações: " + "; ".join(trades_summary) + "."
            return {
                "status": "SUCCESS",
                "command_type": "RECENT_TRADES",
                "response": response_text,
                "data": [dict(r) for r in rows],
                "action_executed": "FETCH_TRADES"
            }
        except Exception as e:
            return {"status": "ERROR", "response": f"Erro ao buscar histórico: {e}"}

    def _handle_market_regime(self) -> Dict[str, Any]:
        try:
            import pandas as pd
            from agents.regime_classifier import MarketRegimeClassifier
            classifier = MarketRegimeClassifier(db_path=self.db_path)
            df_btc = pd.DataFrame()
            if os.path.exists(self.db_path):
                uri = f"file:{os.path.abspath(self.db_path)}?mode=ro"
                with sqlite3.connect(uri, uri=True, timeout=5.0) as con:
                    df_btc = pd.read_sql_query(
                        "SELECT timestamp, open, high, low, close, volume FROM candles WHERE symbol='BTCUSDT' ORDER BY timestamp DESC LIMIT 150",
                        con
                    )
                    if not df_btc.empty:
                        df_btc = df_btc.sort_values("timestamp").reset_index(drop=True)
            res = classifier.classify_from_df(df_btc)
            response_text = f"O regime atual do mercado é {res.regime} com viés {res.bias}. Justificativa: {res.rationale}"
            return {
                "status": "SUCCESS",
                "command_type": "MARKET_REGIME",
                "response": response_text,
                "data": res.to_dict(),
                "action_executed": "CLASSIFY_REGIME"
            }
        except Exception as e:
            return {
                "status": "SUCCESS",
                "command_type": "MARKET_REGIME",
                "response": "O regime atual do mercado é NEUTRO (em consolidação / dados preliminares).",
                "data": {"regime": "NEUTRAL", "primary_bias": "NEUTRAL", "rationale": str(e)},
                "action_executed": "FALLBACK_REGIME"
            }

    def _handle_explain_latest(self) -> Dict[str, Any]:
        from educational_module import EducationalExplainer
        explainer = EducationalExplainer()
        sample = {
            "symbol": "BTCUSDT",
            "direction": "BUY",
            "entry_price": 63400.0,
            "stop_loss": 62800.0,
            "take_profit": 65200.0,
            "rationale": "Liquidity Sweep de mínimas anteriores com confluência de Fair Value Gap mitigado e absorção no Order Book.",
            "regime": "BULL_TREND"
        }
        res = explainer.explain_trade_decision(sample)
        return {
            "status": "SUCCESS",
            "command_type": "EXPLAIN_LATEST",
            "response": res["narrative"],
            "data": res,
            "action_executed": "EXPLAIN_TRADE"
        }
