#!/usr/bin/env python3
"""
scripts/system_supervisor.py
Gestor e Supervisor de Processos e Dependências do IA Trader.
Monitora continuamente os serviços críticos:
 1. Servidor Web Flask (porta 5000)
 2. Daemon de Simulação / Paper Trading (run_paper_simulation.py)
 3. Cloudflare Tunnel (cloudflared)
 4. Integridade do Banco SQLite (datasets/market_cache.db)

Realiza Auto-Healing imediato em caso de travamento, timeout HTTP ou queda de processo.
Gera telemetria em datasets/supervisor_state.json para a interface web.
"""

import os
import sys
import time
import json
import signal
import socket
import logging
import sqlite3
import os as _os_dbc, sys as _sys_dbc
_sys_dbc.path.insert(0, _os_dbc.path.dirname(_os_dbc.path.dirname(_os_dbc.path.abspath(__file__))))
from db_conn import connect as _db_connect
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_BIN = os.path.join(BASE_DIR, ".venv", "bin", "python")
if not os.path.exists(PYTHON_BIN):
    PYTHON_BIN = sys.executable

CLOUDFLARED_BIN = os.path.join(BASE_DIR, "bin", "cloudflared")
DB_PATH = os.path.join(BASE_DIR, "datasets", "market_cache.db")
STATE_FILE = os.path.join(BASE_DIR, "datasets", "supervisor_state.json")
LOG_FILE = os.path.join(BASE_DIR, "datasets", "supervisor.log")
FLASK_LOG_FILE = os.path.join(BASE_DIR, "logs", "flask.log")
SOCKET_TIMEOUT = 5.0
HEAL_FAILURE_THRESHOLD = 3

os.makedirs(os.path.join(BASE_DIR, "datasets"), exist_ok=True)

# Configuração de Logger
logger = logging.getLogger("system_supervisor")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [supervisor] %(message)s"))
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [supervisor] %(message)s"))
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class ServiceManager:
    """Gerencia o ciclo de vida e monitoramento de um serviço individual."""

    def __init__(self, name: str, cmd: list, env_vars: Optional[dict] = None, check_type: str = "process", port: Optional[int] = None, log_file: Optional[str] = None):
        self.log_file = log_file
        self.name = name
        self.cmd = cmd
        self.env_vars = env_vars or {}
        self.check_type = check_type  # 'process', 'http_port', 'sqlite'
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None
        self.start_time: Optional[float] = None
        self.restart_count: int = 0
        self.last_restart_time: Optional[float] = None
        self.last_error: Optional[str] = None
        self.consecutive_failures: int = 0

    def start(self):
        """Inicia o processo associado ao serviço."""
        if self.is_running():
            logger.info(f"[{self.name}] Já está em execução (PID={self.pid}).")
            return

        env = os.environ.copy()
        env.update(self.env_vars)
        env["PYTHONPATH"] = BASE_DIR

        log_fh = None
        try:
            out_target = subprocess.DEVNULL
            if self.log_file:
                os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
                log_fh = open(self.log_file, "ab")
                out_target = log_fh
            self.process = subprocess.Popen(
                self.cmd,
                cwd=BASE_DIR,
                env=env,
                stdout=out_target,
                stderr=subprocess.STDOUT if log_fh else subprocess.DEVNULL,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None
            )
            self.pid = self.process.pid
            self.start_time = time.time()
            self.consecutive_failures = 0
            logger.info(f"🟢 [{self.name}] Iniciado com sucesso (PID={self.pid}).")
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"❌ [{self.name}] Erro fatal ao iniciar: {e}")
        finally:
            if log_fh:
                log_fh.close()

    def stop(self, timeout: float = 3.0):
        """Finaliza o processo de forma graciosa."""
        if not self.process:
            return

        try:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()
            
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()
        except Exception as e:
            logger.debug(f"[{self.name}] Aviso ao finalizar: {e}")
        finally:
            self.process = None
            self.pid = None
            logger.info(f"🛑 [{self.name}] Processo encerrado.")

    def is_running(self) -> bool:
        """Verifica se o processo continua vivo no sistema operacional."""
        if self.process is not None:
            ret = self.process.poll()
            if ret is not None:
                self.process = None
                self.pid = None
                return False
            return True
        elif self.pid is not None:
            try:
                os.kill(self.pid, 0)
                return True
            except OSError:
                self.pid = None
                return False
        return False

    def check_health(self) -> bool:
        """Verifica a saúde funcional do serviço."""
        if not self.is_running():
            logger.warning(f"⚠️ [{self.name}] Processo não está em execução (PID={self.pid}).")
            return False

        if self.check_type == "http_port" and self.port:
            # Testa abertura e resposta rápida de socket na porta
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=SOCKET_TIMEOUT) as sock:
                    return True
            except (socket.timeout, ConnectionRefusedError, OSError) as e:
                logger.warning(f"⚠️ [{self.name}] Falha no socket porta {self.port} "
                               f"(PID={self.pid}): {type(e).__name__}: {e}")
                return False

        return True

    def find_port_pid(self) -> Optional[int]:
        """Retorna o PID que escuta na porta do serviço (lsof/ss/psutil), ou None."""
        if not self.port:
            return None
        try:
            out = subprocess.check_output(["lsof", "-t", f"-iTCP:{self.port}", "-sTCP:LISTEN"],
                                          text=True, stderr=subprocess.DEVNULL, timeout=5)
            pids = [int(p) for p in out.split() if p.isdigit()]
            if pids:
                return pids[0]
        except Exception:
            pass
        try:
            import re
            out = subprocess.check_output(["ss", "-ltnpH", f"sport = :{self.port}"],
                                          text=True, stderr=subprocess.DEVNULL, timeout=5)
            m = re.search(r"pid=(\d+)", out)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        try:
            import psutil  # opcional
            for c in psutil.net_connections(kind="tcp"):
                if c.status == "LISTEN" and c.laddr and c.laddr.port == self.port and c.pid:
                    return c.pid
        except Exception:
            pass
        return None

    def auto_heal(self):
        """Reinicia o serviço caso esteja inoperante ou travado.

        Se a porta já é atendida por outro PID vivo, re-adota esse PID em vez de reiniciar.
        """
        if self.check_type == "http_port" and self.port:
            port_pid = self.find_port_pid()
            if port_pid and port_pid != self.pid:
                self.process = None
                self.pid = port_pid
                self.start_time = time.time()
                self.consecutive_failures = 0
                logger.info(f"🔗 [{self.name}] Porta {self.port} atendida por PID={port_pid}; "
                            f"re-adotado sem reiniciar.")
                return
        self.restart_count += 1
        self.last_restart_time = time.time()
        logger.warning(f"⚠️ [{self.name}] Auto-Healing acionado! Reiniciando serviço (Reinício #{self.restart_count})...")
        self.stop()
        time.sleep(1.0)
        self.start()

    def get_status_dict(self) -> Dict[str, Any]:
        """Retorna telemetria estruturada do serviço."""
        running = self.is_running()
        uptime = round(time.time() - self.start_time, 1) if (running and self.start_time) else 0.0
        return {
            "name": self.name,
            "status": "HEALTHY" if running else "STOPPED",
            "pid": self.pid,
            "uptime_seconds": uptime,
            "restart_count": self.restart_count,
            "last_restart": datetime.fromtimestamp(self.last_restart_time, tz=timezone.utc).isoformat() if self.last_restart_time else None,
            "last_error": self.last_error
        }


class SystemSupervisor:
    """Orquestrador supervisor contínuo com verificação periódica e auto-recuperação."""

    def __init__(self, check_interval: float = 8.0):
        self.check_interval = check_interval
        self.running = False

        # 1. Serviço Flask
        self.flask_service = ServiceManager(
            name="flask_dashboard",
            cmd=[PYTHON_BIN, os.path.join(BASE_DIR, "flask_interface.py"), "5000"],
            check_type="http_port",
            port=5000,
            log_file=FLASK_LOG_FILE
        )

        # 2. Serviço de Simulação Paper Trading
        self.paper_service = ServiceManager(
            name="paper_trading_daemon",
            cmd=[PYTHON_BIN, os.path.join(BASE_DIR, "scripts", "run_paper_simulation.py"), "0", "15"],
            check_type="process"
        )

        # 3. Serviço Cloudflare Tunnel
        self.tunnel_service = ServiceManager(
            name="cloudflare_tunnel",
            cmd=[CLOUDFLARED_BIN, "tunnel", "--url", "http://127.0.0.1:5000"],
            check_type="process"
        )

        # 4. Coletor contínuo de microestrutura (dados públicos, sem chave de API)
        self.collector_service = ServiceManager(
            name="microstructure_collector",
            cmd=[PYTHON_BIN, os.path.join(BASE_DIR, "scripts", "microstructure_collector.py")],
            check_type="process",
            log_file=os.path.join(BASE_DIR, "logs", "collector.log")
        )

        self.services = [self.flask_service, self.paper_service, self.tunnel_service, self.collector_service]

    def verify_database_integrity(self) -> Dict[str, Any]:
        """Verifica a integridade e modo de operação do SQLite."""
        res = {"status": "HEALTHY", "wal_mode": False, "size_bytes": 0, "error": None}
        try:
            if not os.path.exists(DB_PATH):
                res["status"] = "MISSING"
                return res

            res["size_bytes"] = os.path.getsize(DB_PATH)
            with _db_connect(f"file:{os.path.abspath(DB_PATH)}?mode=ro", uri=True, timeout=3.0) as con:
                mode = con.execute("PRAGMA journal_mode;").fetchone()[0]
                res["wal_mode"] = (mode.lower() == "wal")
                check = con.execute("PRAGMA quick_check;").fetchone()[0]
                if check.lower() != "ok":
                    res["status"] = "CORRUPTED"
                    res["error"] = check
        except Exception as e:
            res["status"] = "ERROR"
            res["error"] = str(e)
        return res

    def dump_state_file(self):
        """Grava o estado atual consolidado em datasets/supervisor_state.json para a interface web."""
        db_health = self.verify_database_integrity()
        services_status = [s.get_status_dict() for s in self.services]
        all_healthy = all(s["status"] == "HEALTHY" for s in services_status) and (db_health["status"] == "HEALTHY")

        payload = {
            "supervisor_online": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "HEALTHY" if all_healthy else "DEGRADED",
            "services_healthy_count": sum(1 for s in services_status if s["status"] == "HEALTHY"),
            "services_total_count": len(services_status),
            "services": services_status,
            "database": db_health,
            "check_interval_seconds": self.check_interval,
            "auto_recovery_enabled": True
        }

        try:
            tmp_path = STATE_FILE + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, STATE_FILE)
        except Exception as e:
            logger.debug(f"Aviso ao persistir supervisor_state.json: {e}")

    def adopt_existing_processes(self):
        """Descobre processos já em execução no SO para não duplicar instâncias."""
        import subprocess
        try:
            # Procura Flask na porta 5000
            out = subprocess.check_output(["lsof", "-t", "-i:5000"], text=True, stderr=subprocess.DEVNULL)
            pids = [int(p) for p in out.strip().split() if p.isdigit()]
            if pids and not self.flask_service.is_running():
                # Adota o processo
                self.flask_service.pid = pids[0]
                self.flask_service.start_time = time.time()
                logger.info(f"🔗 [flask_dashboard] Adotou processo existente na porta 5000 (PID={pids[0]}).")
        except Exception:
            pass

        try:
            # Procura run_paper_simulation
            out = subprocess.check_output(["pgrep", "-f", "run_paper_simulation.py"], text=True, stderr=subprocess.DEVNULL)
            pids = [int(p) for p in out.strip().split() if p.isdigit()]
            if pids and not self.paper_service.is_running():
                self.paper_service.pid = pids[0]
                self.paper_service.start_time = time.time()
                logger.info(f"🔗 [paper_trading_daemon] Adotou processo existente (PID={pids[0]}).")
        except Exception:
            pass

        try:
            # Procura cloudflared
            out = subprocess.check_output(["pgrep", "-f", "cloudflared tunnel"], text=True, stderr=subprocess.DEVNULL)
            pids = [int(p) for p in out.strip().split() if p.isdigit()]
            if pids and not self.tunnel_service.is_running():
                self.tunnel_service.pid = pids[0]
                self.tunnel_service.start_time = time.time()
                logger.info(f"🔗 [cloudflare_tunnel] Adotou túnel existente (PID={pids[0]}).")
        except Exception:
            pass

        try:
            # Procura o coletor de microestrutura
            out = subprocess.check_output(["pgrep", "-f", "microstructure_collector.py"], text=True, stderr=subprocess.DEVNULL)
            pids = [int(p) for p in out.strip().split() if p.isdigit()]
            if pids and not self.collector_service.is_running():
                self.collector_service.pid = pids[0]
                self.collector_service.start_time = time.time()
                logger.info(f"🔗 [microstructure_collector] Adotou processo existente (PID={pids[0]}).")
        except Exception:
            pass

    def check_all_services(self):
        """Varredura de integridade e auto-recuperação de todos os serviços."""
        for svc in self.services:
            if not svc.check_health():
                svc.consecutive_failures += 1
                logger.warning(f"⚠️ [{svc.name}] Falha detectada (Contagem={svc.consecutive_failures}).")
                if svc.consecutive_failures >= HEAL_FAILURE_THRESHOLD:
                    svc.auto_heal()
            else:
                svc.consecutive_failures = 0

        self.maybe_checkpoint_wal()
        self.dump_state_file()

    def maybe_checkpoint_wal(self, interval_seconds: float = 300.0, threshold_bytes: int = 16 * 1024 * 1024) -> Optional[Dict[str, Any]]:
        """Checkpoint WAL (TRUNCATE→PASSIVE, tolerante a busy) quando o WAL excede o limiar."""
        now = time.time()
        if now - getattr(self, "_last_wal_checkpoint", 0.0) < interval_seconds:
            return None
        self._last_wal_checkpoint = now
        try:
            if BASE_DIR not in sys.path:
                sys.path.insert(0, BASE_DIR)
            from data.db_maintenance import wal_checkpoint
            res = wal_checkpoint(DB_PATH, busy_timeout_ms=500, truncate_threshold_bytes=threshold_bytes)
            if res.get("mode") not in (None, "SKIPPED"):
                logger.info(f"🧹 WAL checkpoint {res['mode']} busy={res['busy']} "
                            f"{res['wal_before']/1e6:.1f}MB -> {res['wal_after']/1e6:.1f}MB")
            return res
        except Exception as e:
            logger.debug(f"Aviso no checkpoint WAL: {e}")
            return None

    def run_forever(self):
        """Loop contínuo de supervisão."""
        self.running = True
        logger.info("🛡️ [SystemSupervisor] Iniciando Gestor de Sistema e Dependências (Piloto Resiliente)...")

        # Tenta adotar processos já em execução primeiro
        self.adopt_existing_processes()

        # Inicia quaisquer serviços ausentes
        for svc in self.services:
            if not svc.is_running():
                svc.start()

        self.dump_state_file()

        while self.running:
            try:
                time.sleep(self.check_interval)
                self.check_all_services()
            except KeyboardInterrupt:
                logger.info("Interrupção pelo usuário recebida.")
                break
            except Exception as e:
                logger.error(f"Erro inesperado no loop do supervisor: {e}")

        logger.info("Encerrando SystemSupervisor...")
        self.dump_state_file()


def main():
    action = sys.argv[1] if len(sys.argv) > 1 else "run"

    if action == "status":
        sup = SystemSupervisor()
        sup.adopt_existing_processes()
        sup.check_all_services()
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            print(f.read())
        sys.exit(0)

    sup = SystemSupervisor()

    def handle_sig(sig, frame):
        sup.running = False

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    sup.run_forever()


if __name__ == "__main__":
    main()
