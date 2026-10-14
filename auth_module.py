"""
auth_module.py - Gestão de Usuários, Planos e Controle de Acesso (RBAC).
Fase 9 do Projeto IA Trader (Páginas 6 e 12 da Documentação Oficial).

Objetivo:
1. Autenticação e Armazenamento Seguro de Credenciais:
   - Tabela SQLite `users` com senhas criptografadas via PBKDF2-HMAC-SHA256 e salt randômico.
2. Níveis de Assinatura e Permissões (Free, Pro, Premium):
   - FREE: Acesso básico ao BTC/USDT em 1h, sem Golden Pots, modo observador.
   - PRO: Acesso a múltiplos pares, Scalps e Golden Pots, limites padrão.
   - PREMIUM: Acesso total, comandos de voz irrestritos e telemetria avançada.
3. Rastreabilidade de Aceite de Termos de Uso (LGPD / GDPR).
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sqlite3
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ia_trader.auth_module")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [auth_module] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


PLAN_PERMISSIONS: Dict[str, Dict[str, Any]] = {
    "FREE": {
        "max_pairs": 1,
        "allowed_symbols": ["BTCUSDT"],
        "allowed_timeframes": ["1h"],
        "can_execute_golden_pots": False,
        "can_use_voice_commands": False,
        "daily_scalps_limit": 2,
    },
    "PRO": {
        "max_pairs": 10,
        "allowed_symbols": ["ALL"],
        "allowed_timeframes": ["15m", "1h", "4h", "1d"],
        "can_execute_golden_pots": True,
        "can_use_voice_commands": True,
        "daily_scalps_limit": 5,
    },
    "PREMIUM": {
        "max_pairs": 999,
        "allowed_symbols": ["ALL"],
        "allowed_timeframes": ["ALL"],
        "can_execute_golden_pots": True,
        "can_use_voice_commands": True,
        "daily_scalps_limit": 10,
        "priority_support": True,
    },
}


class AuthManager:
    """
    Gerenciador de autenticação, usuários e planos.
    """

    def __init__(self, db_path: str = "datasets/market_cache.db"):
        self.db_path = db_path
        self._ensure_tables()

    def _get_connection(self) -> sqlite3.Connection:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _ensure_tables(self):
        """Inicializa tabela de usuários no SQLite."""
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        salt TEXT NOT NULL,
                        plan TEXT DEFAULT 'FREE',  -- 'FREE', 'PRO', 'PREMIUM'
                        is_active INTEGER DEFAULT 1,
                        terms_accepted INTEGER DEFAULT 0,
                        created_at INTEGER NOT NULL
                    )
                """)
                # Cria usuário administrador padrão se tabela estiver vazia
                row = conn.execute("SELECT count(*) as cnt FROM users").fetchone()
                if row and row["cnt"] == 0:
                    salt = secrets.token_hex(16)
                    p_hash = self._hash_password("admin123", salt)
                    now_ms = int(time.time() * 1000)
                    conn.execute("""
                        INSERT INTO users (email, password_hash, salt, plan, is_active, terms_accepted, created_at)
                        VALUES ('admin@iatrader.local', ?, ?, 'PREMIUM', 1, 1, ?)
                    """, (p_hash, salt, now_ms))
                conn.commit()
        except Exception as e:
            logger.warning(f"Aviso ao inicializar tabela de usuários: {e}")

    def _hash_password(self, password: str, salt: str) -> str:
        """Derivação criptográfica robusta com 100.000 iterações de PBKDF2-HMAC-SHA256."""
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
        return dk.hex()

    def register_user(self, email: str, password: str, plan: str = "FREE") -> Tuple[bool, str]:
        """Cadastra um novo usuário no sistema."""
        email_clean = email.strip().lower()
        if "@" not in email_clean or len(password) < 6:
            return False, "Email inválido ou senha com menos de 6 caracteres."

        plan_valid = plan.upper() if plan.upper() in PLAN_PERMISSIONS else "FREE"
        salt = secrets.token_hex(16)
        p_hash = self._hash_password(password, salt)
        now_ms = int(time.time() * 1000)

        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT INTO users (email, password_hash, salt, plan, is_active, terms_accepted, created_at)
                    VALUES (?, ?, ?, ?, 1, 0, ?)
                """, (email_clean, p_hash, salt, plan_valid, now_ms))
                conn.commit()
            return True, f"Usuário {email_clean} cadastrado com sucesso no plano {plan_valid}."
        except sqlite3.IntegrityError:
            return False, "Este email já está cadastrado no sistema."
        except Exception as e:
            return False, f"Erro interno ao cadastrar usuário: {e}"

    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Valida credenciais e retorna o perfil do usuário caso correto."""
        email_clean = email.strip().lower()
        try:
            with self._get_connection() as conn:
                row = conn.execute("SELECT * FROM users WHERE email = ? AND is_active = 1", (email_clean,)).fetchone()
                if not row:
                    return None

                salt = row["salt"]
                expected_hash = row["password_hash"]
                actual_hash = self._hash_password(password, salt)

                if secrets.compare_digest(expected_hash, actual_hash):
                    return {
                        "id": row["id"],
                        "email": row["email"],
                        "plan": row["plan"],
                        "terms_accepted": bool(row["terms_accepted"]),
                        "permissions": PLAN_PERMISSIONS.get(row["plan"], PLAN_PERMISSIONS["FREE"])
                    }
        except Exception as e:
            logger.error(f"Erro na autenticação de {email}: {e}")
        return None

    def accept_terms(self, email: str) -> bool:
        """Registra o consentimento dos termos de uso."""
        email_clean = email.strip().lower()
        try:
            with self._get_connection() as conn:
                conn.execute("UPDATE users SET terms_accepted = 1 WHERE email = ?", (email_clean,))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Erro ao registrar aceite de termos: {e}")
            return False

    def can_access_feature(self, user_plan: str, feature_key: str) -> bool:
        """Verifica se determinado plano tem autorização para uma funcionalidade."""
        perms = PLAN_PERMISSIONS.get(user_plan.upper(), PLAN_PERMISSIONS["FREE"])
        return bool(perms.get(feature_key, False))
