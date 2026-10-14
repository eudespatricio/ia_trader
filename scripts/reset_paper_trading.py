"""
scripts/reset_paper_trading.py - Script de Reset Limpo da Simulação de Paper Trading.
Restaura o estado inicial de demonstração com $10,000.00 USD e zera todas as tabelas transacionais.
"""

import os
import sys
import time
import sqlite3

DB_PATH = "datasets/market_cache.db"

def reset_simulation(initial_capital: float = 10000.0):
    print(f"🔄 Iniciando reset da simulação em {DB_PATH} com carteira de ${initial_capital:,.2f} USD...")
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    
    # 1. Limpa posições e histórico transacional
    tables_to_clear = [
        "paper_positions",
        "paper_trades",
        "paper_portfolio",
        "performance_snapshots",
        "circuit_breaker_events",
        "agent_learning_memory"
    ]
    
    for t in tables_to_clear:
        try:
            conn.execute(f"DELETE FROM {t}")
            print(f"   ✓ Tabela '{t}' limpa.")
        except sqlite3.OperationalError:
            print(f"   ℹ️ Tabela '{t}' ainda não existia.")

    # 2. Insere saldo inicial limpo de $10,000.00
    now_ms = int(time.time() * 1000)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_portfolio (
                timestamp INTEGER NOT NULL,
                equity REAL NOT NULL,
                balance REAL NOT NULL,
                margin_used REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                realized_pnl_cum REAL NOT NULL,
                total_fees_paid REAL NOT NULL,
                open_positions_count INTEGER NOT NULL
            )
        """)
        conn.execute("""
            INSERT INTO paper_portfolio (
                timestamp, equity, balance, margin_used, unrealized_pnl,
                realized_pnl_cum, total_fees_paid, open_positions_count
            ) VALUES (?, ?, ?, 0.0, 0.0, 0.0, 0.0, 0)
        """, (now_ms, initial_capital, initial_capital))
        print(f"   ✓ Portfólio inicializado: Equity = ${initial_capital:,.2f} USD.")
    except Exception as e:
        print(f"   ⚠️ Erro ao inicializar portfólio: {e}")

    conn.commit()
    conn.close()

    # 3. Reseta Circuit Breaker para CLEAR
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from rollback_guard import RollbackGuard
        guard = RollbackGuard(db_path=DB_PATH)
        guard.reset_circuit_breaker(operator_note="Reset para início de demonstração oficial")
        print("   ✓ Circuit Breakers resetados para estado CLEAR.")
    except Exception as e:
        print(f"   ⚠️ Aviso ao resetar RollbackGuard: {e}")

    # 4. Checkpoint do WAL
    try:
        chk_conn = sqlite3.connect(DB_PATH, timeout=5.0)
        chk_conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        chk_conn.execute("PRAGMA optimize")
        chk_conn.close()
    except Exception as e:
        print(f"   ℹ️ Checkpoint WAL postergado: {e}")

    print("✨ Reset de Paper Trading concluído com sucesso!")

if __name__ == "__main__":
    reset_simulation()
