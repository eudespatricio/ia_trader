#!/usr/bin/env python3
"""
scripts/reset_pilot.py - Reset TOTAL e seguro dos registros do piloto de paper trading.

DEVE ser executado com os daemons PARADOS (run_paper_simulation.py e system_supervisor.py,
que o relançaria). O script recusa se detectar esses processos (pgrep), salvo --force.

Passos:
  1. Backup consistente (sqlite3 backup API) -> datasets/backups/pre_reset_<ts>.db + integrity_check.
  2. UMA transação: DELETE nas tabelas do piloto, zera sqlite_sequence, remove marcadores
     ledger_epoch_*/quota_epoch_*/MANUAL_RESET/reset_daily_quotas, grava pilot_start_utc/pilot_reset_id.
  3. VACUUM + wal_checkpoint(TRUNCATE).
  4. Arquiva (não apaga) logs/flask.log e datasets/reports/pilot_report.json.
  5. Relatório (contagens antes/depois + snapshot das configurações) em datasets/backups/reset_report_<ts>.json.
  6. Verifica PaperTradingEngine(restore=True): saldo = capital inicial, 0 posições, cotas zeradas.

Decisões de escopo:
  * circuit_breaker_state: a linha única é REMOVIDA. RollbackGuard._load_persisted_state sem linha
    parte de CircuitBreakerStatus() default (CLEAR, trading permitido) - é o estado "limpo" esperado.
  * market_regime: escrita SOMENTE por agents/regime_classifier.py (_save_regime) e nunca lida via SQL por
    nenhum agente/dashboard/voz (o classificador recalcula do candles a cada chamada). É histórico derivado e
    recomputável -> LIMPO por inteiro (a tabela só contém o período do piloto). --keep-market-regime preserva.
  * NÃO tocados: candles, funding_rates, order_books, macro_*, low_freq_cache, users, sqlite_stat1,
    datasets/reports/trial_log.csv (N do DSR) e datasets/supervisor.log.

Uso:  .venv/bin/python scripts/reset_pilot.py [--db PATH] [--dry-run] [--force] [--keep-market-regime] [--yes]
"""
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

DB = os.path.join(BASE, "datasets", "market_cache.db")
BACKUP_DIR = os.path.join(BASE, "datasets", "backups")
FLASK_LOG = os.path.join(BASE, "logs", "flask.log")
PILOT_REPORT = os.path.join(BASE, "datasets", "reports", "pilot_report.json")

PILOT_TABLES: List[str] = [
    "paper_trades", "paper_positions", "paper_portfolio",
    "circuit_breaker_events", "circuit_breaker_state",
    "golden_pots", "performance_snapshots",
]
# Memória episódica / calibração de pesos por regime: PRESERVADAS por decisão do usuário.
# position_id referenciado é uuid4[:8] (sem colisão com posições novas) e o LearningTracker só o usa em
# "UPDATE ... WHERE position_id = ?" (sem JOIN) -> linhas órfãs são inertes; nada a tratar no engine.
PRESERVED_MEMORY_TABLES: List[str] = ["agent_learning_memory", "agent_regime_weights"]
DERIVED_TABLES: List[str] = ["market_regime"]
UNTOUCHED_TABLES: List[str] = [
    "candles", "funding_rates", "order_books", "macro_series", "macro_daily_history",
    "macro_events_calendar", "low_freq_cache",
]
# Marcadores baseados em rowid (recomeçam em 1 após o reset) e flags pendentes.
SETTINGS_TO_REMOVE: List[str] = [
    "ledger_epoch_trade_rowid", "ledger_epoch_pos_rowid", "quota_epoch_trade_rowid",
    "MANUAL_RESET", "reset_daily_quotas",
]
DAEMON_PATTERNS: List[str] = ["run_paper_simulation.py", "system_supervisor.py"]


class ResetRefused(RuntimeError):
    pass


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def detect_daemons() -> List[str]:
    """Linhas do pgrep -af para os daemons do piloto (exclui o próprio processo)."""
    found: List[str] = []
    me = str(os.getpid())
    for pat in DAEMON_PATTERNS:
        try:
            out = subprocess.run(["pgrep", "-af", pat], capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        for ln in out.splitlines():
            pid = ln.split(None, 1)[0] if ln.strip() else ""
            if pid and pid != me and "reset_pilot.py" not in ln and "pgrep" not in ln:
                found.append(ln.strip())
    return found


def _tables(con: sqlite3.Connection) -> set:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def count_tables(db_path: str, names: List[str]) -> Dict[str, Optional[int]]:
    con = sqlite3.connect(db_path, timeout=30.0)
    try:
        have = _tables(con)
        return {n: (con.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0] if n in have else None)
                for n in names}
    finally:
        con.close()


def make_backup(db_path: str, backup_dir: str, ts: str) -> str:
    os.makedirs(backup_dir, exist_ok=True)
    dst = os.path.join(backup_dir, f"pre_reset_{ts}.db")
    src = sqlite3.connect(db_path, timeout=30.0)
    bak = sqlite3.connect(dst)
    try:
        src.backup(bak)
    finally:
        bak.close()
        src.close()
    chk = sqlite3.connect(dst)
    try:
        res = chk.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        chk.close()
    if res != "ok":
        os.remove(dst)
        raise RuntimeError(f"Backup corrompido (integrity_check={res}); reset abortado")
    return dst


def archive_files(base_dir_paths: Dict[str, str], backup_dir: str, ts: str) -> Dict[str, str]:
    """Arquiva (copia) flask.log e pilot_report.json; log fica vazio, json é removido."""
    out: Dict[str, str] = {}
    for kind, path in base_dir_paths.items():
        if not os.path.exists(path):
            continue
        stem, ext = os.path.splitext(os.path.basename(path))
        dst = os.path.join(backup_dir, f"{stem}_{ts}{ext}")
        shutil.copy2(path, dst)
        if kind == "log":
            open(path, "w").close()      # mantém o arquivo (handle do Flask pode estar aberto)
        else:
            os.remove(path)
        out[path] = dst
    return out


def reset_database(db_path: str, ts: str, reset_id: str, keep_market_regime: bool = False) -> Dict[str, Any]:
    """Executa a limpeza em UMA transação e depois VACUUM/checkpoint. Retorna o dict do relatório parcial."""
    con = sqlite3.connect(db_path, timeout=60.0, isolation_level=None)
    try:
        have = _tables(con)
        targets = [t for t in PILOT_TABLES + ([] if keep_market_regime else DERIVED_TABLES) if t in have]
        before = {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in targets}
        settings_before: Dict[str, Any] = {}
        if "simulation_settings" in have:
            settings_before = {k: v for k, v in con.execute("SELECT key, value FROM simulation_settings")}

        start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        con.execute("BEGIN IMMEDIATE")
        try:
            for t in targets:
                con.execute(f'DELETE FROM "{t}"')
            if "sqlite_sequence" in have:
                con.executemany("DELETE FROM sqlite_sequence WHERE name = ?", [(t,) for t in targets])
            con.execute("""CREATE TABLE IF NOT EXISTS simulation_settings (
                key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            con.executemany("DELETE FROM simulation_settings WHERE key = ?", [(k,) for k in SETTINGS_TO_REMOVE])
            for k, v in (("pilot_start_utc", start_iso), ("pilot_reset_id", reset_id)):
                con.execute("INSERT OR REPLACE INTO simulation_settings (key, value, updated_at) VALUES (?, ?, ?)",
                            (k, v, start_iso))
            after = {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in targets}
            if any(after.values()):
                raise RuntimeError(f"DELETE incompleto: {after}")
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
        con.execute("VACUUM")
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        settings_after = {k: v for k, v in con.execute("SELECT key, value FROM simulation_settings")}
    finally:
        con.close()
    return {"tables_before": before, "tables_after": after, "pilot_start_utc": start_iso,
            "settings_before": settings_before, "settings_after": settings_after,
            "preserved_settings": {k: v for k, v in settings_after.items()
                                   if k not in ("pilot_start_utc", "pilot_reset_id")}}


def verify_engine(db_path: str) -> Dict[str, Any]:
    """PaperTradingEngine(restore=True) sobre o DB resetado deve iniciar limpo."""
    from smart_trade import PaperTradingEngine
    eng = PaperTradingEngine(db_path=db_path, restore=True)
    m = eng.get_summary_metrics()
    res = {
        "initial_capital": eng.initial_capital, "balance": eng.balance, "open_positions": len(eng.positions),
        "trade_history": len(eng.trade_history), "daily_scalps_count": eng.daily_scalps_count,
        "daily_golden_pots_count": eng.daily_golden_pots_count, "summary_metrics_ok": isinstance(m, dict),
    }
    res["ok"] = (res["balance"] == res["initial_capital"] and res["open_positions"] == 0
                 and res["trade_history"] == 0 and res["daily_scalps_count"] == 0
                 and res["daily_golden_pots_count"] == 0 and res["summary_metrics_ok"])
    return res


def run_reset(db_path: str = DB, backup_dir: str = BACKUP_DIR, flask_log: str = FLASK_LOG,
              pilot_report: str = PILOT_REPORT, force: bool = False, keep_market_regime: bool = False,
              check_daemons: bool = True, verify: bool = True) -> Dict[str, Any]:
    if check_daemons and not force:
        procs = detect_daemons()
        if procs:
            raise ResetRefused("Daemons do piloto em execução (pare-os ou use --force):\n  " + "\n  ".join(procs))
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)
    ts = _utc_ts()
    reset_id = f"reset_{ts}_{uuid.uuid4().hex[:8]}"
    untouched_before = count_tables(db_path, UNTOUCHED_TABLES + PRESERVED_MEMORY_TABLES)
    backup = make_backup(db_path, backup_dir, ts)
    part = reset_database(db_path, ts, reset_id, keep_market_regime)
    archived = archive_files({"log": flask_log, "json": pilot_report}, backup_dir, ts)
    untouched_after = count_tables(db_path, UNTOUCHED_TABLES + PRESERVED_MEMORY_TABLES)
    if untouched_before != untouched_after:
        raise RuntimeError(f"Dados de mercado/memória alterados! {untouched_before} -> {untouched_after}")
    report: Dict[str, Any] = {
        "reset_id": reset_id, "timestamp_utc": ts, "db_path": db_path, "backup": backup,
        "archived_files": archived, **part,
        "market_regime_cleared": not keep_market_regime,
        "untouched_tables": {"before": untouched_before, "after": untouched_after},
        "removed_settings": [k for k in SETTINGS_TO_REMOVE if k in part["settings_before"]],
        "settings_snapshot_pre_reset": part["settings_before"],
        "preserved_memory_tables": {
            "tables": PRESERVED_MEMORY_TABLES,
            "note": "Preservadas por decisão do usuário (memória de aprendizado permanece).",
            "counts": {t: untouched_after.get(t) for t in PRESERVED_MEMORY_TABLES}},
        "preserved_trial_log": "datasets/reports/trial_log.csv NÃO tocado (N do DSR)",
    }
    if verify:
        report["engine_verification"] = verify_engine(db_path)
    rp = os.path.join(backup_dir, f"reset_report_{ts}.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    report["report_path"] = rp
    return report


def print_report(r: Dict[str, Any]) -> None:
    print(f"== RESET DO PILOTO {r['reset_id']} ==")
    print(f"backup: {r['backup']}")
    print(f"{'tabela':28s}{'antes':>10s}{'depois':>10s}")
    for t, b in r["tables_before"].items():
        print(f"{t:28s}{b:>10d}{r['tables_after'][t]:>10d}")
    print("intactas:", r["untouched_tables"]["after"])
    print("configs removidas:", r["removed_settings"])
    print("configs preservadas:", json.dumps(r["preserved_settings"], ensure_ascii=False))
    print("arquivados:", r["archived_files"])
    if "engine_verification" in r:
        print("engine:", r["engine_verification"])
    print(f"pilot_start_utc={r['pilot_start_utc']}  relatório: {r['report_path']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--force", action="store_true", help="ignora a checagem de daemons (PERIGOSO)")
    ap.add_argument("--keep-market-regime", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="só mostra contagens e checa daemons")
    ap.add_argument("--yes", action="store_true", help="não pede confirmação")
    a = ap.parse_args()
    if a.dry_run:
        print("daemons:", detect_daemons() or "nenhum")
        for t, n in count_tables(a.db, PILOT_TABLES + DERIVED_TABLES).items():
            print(f"{t:28s}{n}")
        return 0
    if not a.yes and input(f"Resetar TODO o piloto em {a.db}? Digite RESET: ").strip() != "RESET":
        print("Cancelado.")
        return 1
    try:
        r = run_reset(a.db, force=a.force, keep_market_regime=a.keep_market_regime)
    except ResetRefused as e:
        print(f"RECUSADO: {e}")
        return 3
    print_report(r)
    return 0 if r.get("engine_verification", {"ok": True})["ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
