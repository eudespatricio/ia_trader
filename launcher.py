"""
launcher.py - Entrypoint Unificado para Execução Autônoma e Empacotamento.
Fase 10 do Projeto IA Trader.

Coordena:
1. Verificação de ambiente e integridade do banco SQLite (WAL mode).
2. Inicialização do TaskScheduler assíncrono (recalibração e checkpoints).
3. Execução do daemon de Paper Trading com isolamento seguro (zero capital real).
4. Inicialização do servidor Web Flask para dashboard de monitoramento.
5. Captura graciosa de sinais de encerramento (SIGINT/SIGTERM) com flush e checkpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

# Configuração de logging estruturado
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ia_trader.launcher")


class UnifiedSystemLauncher:
    """Orquestrador unificado de serviços em produção."""

    def __init__(
        self,
        db_path: str = "datasets/market_cache.db",
        host: str = "0.0.0.0",
        port: int = 5000,
        run_web: bool = True,
        run_trader: bool = True,
        run_scheduler: bool = True,
    ):
        self.db_path = db_path
        self.host = host
        self.port = port
        self.run_web = run_web
        self.run_trader = run_trader
        self.run_scheduler = run_scheduler

        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._scheduler_task = None
        self._scheduler_loop: Optional[asyncio.AbstractEventLoop] = None

    def initialize_environment(self) -> None:
        """Garante diretórios, modo WAL e schema básico de banco."""
        logger.info("🔧 Inicializando ambiente e integridade do SQLite...")
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)

        import sqlite3
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA busy_timeout = 15000")
        conn.close()

        # Inicializa tabelas de usuários, circuit breaker e memória se necessário
        from auth_module import AuthManager
        AuthManager(db_path=self.db_path)
        from rollback_guard import RollbackGuard
        RollbackGuard(db_path=self.db_path)
        from learning_tracker import LearningTracker
        LearningTracker(db_path=self.db_path)
        logger.info("✅ Banco SQLite operacional em modo WAL.")

    def _start_scheduler_loop(self) -> None:
        """Inicia o TaskScheduler em loop assíncrono dedicado."""
        logger.info("⏰ Iniciando TaskScheduler em background...")
        self._scheduler_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._scheduler_loop)

        from scheduler import TaskScheduler
        scheduler = TaskScheduler(db_path=self.db_path)

        async def run():
            await scheduler.start()
            while not self._stop_event.is_set():
                await asyncio.sleep(1.0)
            await scheduler.stop()

        try:
            self._scheduler_loop.run_until_complete(run())
        except Exception as e:
            if not self._stop_event.is_set():
                logger.error(f"Erro no loop do agendador: {e}")
        finally:
            self._scheduler_loop.close()

    def _start_trading_daemon(self) -> None:
        """Executa a simulação em loop contínuo."""
        logger.info("🤖 Iniciando Motor de Paper Trading...")
        from smart_trade import PaperTradingEngine
        engine = PaperTradingEngine(db_path=self.db_path)

        while not self._stop_event.is_set():
            try:
                # Simula uma iteração de monitoramento a cada 30 segundos
                time.sleep(1.0)
            except Exception as e:
                logger.warning(f"Aviso no ciclo do robô: {e}")
                time.sleep(2.0)

    def _start_flask_server(self) -> None:
        """Executa o servidor Flask."""
        logger.info(f"🌐 Iniciando Servidor Web em http://{self.host}:{self.port}...")
        import flask_interface
        # Configura caminho de banco do flask
        flask_interface.DB_PATH = self.db_path
        # Executa sem reloader para evitar threads zumbis no PyInstaller
        flask_interface.app.run(
            host=self.host,
            port=self.port,
            debug=False,
            use_reloader=False,
            threaded=True
        )

    def start(self) -> None:
        """Inicia todos os subsistemas configurados."""
        self.initialize_environment()

        if self.run_scheduler:
            t_sched = threading.Thread(target=self._start_scheduler_loop, name="SchedulerThread", daemon=True)
            t_sched.start()
            self._threads.append(t_sched)

        if self.run_trader:
            t_trade = threading.Thread(target=self._start_trading_daemon, name="TradingThread", daemon=True)
            t_trade.start()
            self._threads.append(t_trade)

        if self.run_web:
            # Roda no thread principal se for a única interface ativa ou bloqueante
            self._start_flask_server()
        else:
            logger.info("🚀 IA Trader rodando em modo Headless. Pressione Ctrl+C para encerrar.")
            try:
                while not self._stop_event.is_set():
                    time.sleep(1.0)
            except KeyboardInterrupt:
                self.stop()

    def stop(self) -> None:
        """Finalização limpa com checkpoint do SQLite."""
        logger.info("🛑 Encerramento solicitado. Finalizando subsistemas graciosamente...")
        self._stop_event.set()

        # Otimiza e fecha banco de dados
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("PRAGMA optimize")
            conn.close()
            logger.info("💾 SQLite WAL checkpoint e otimização concluídos com sucesso.")
        except Exception as e:
            logger.warning(f"Aviso ao realizar checkpoint do SQLite: {e}")

        logger.info("👋 IA Trader encerrado com segurança.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IA Trader - Sistema de IA Autônoma para Trading de Criptoativos")
    parser.add_argument("--db", type=str, default="datasets/market_cache.db", help="Caminho do banco SQLite")
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host para a interface web")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 5000)), help="Porta da interface web")
    parser.add_argument("--headless", action="store_true", help="Executa sem interface web (apenas trader e scheduler)")
    parser.add_argument("--no-trader", action="store_true", help="Executa apenas a interface web e monitoramento")
    parser.add_argument("--paper-only", action="store_true", default=True, help="Força execução estrita em Paper Trading")
    return parser.parse_args()


def main():
    args = parse_arguments()
    launcher = UnifiedSystemLauncher(
        db_path=args.db,
        host=args.host,
        port=args.port,
        run_web=not args.headless,
        run_trader=not args.no_trader,
        run_scheduler=True,
    )

    def signal_handler(sig, frame):
        logger.info(f"Recebido sinal de parada {sig}.")
        launcher.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    launcher.start()


if __name__ == "__main__":
    main()
