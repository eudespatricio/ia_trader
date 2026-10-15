"""
db_conn.py — conexões SQLite que realmente FECHAM ao sair do bloco `with`.

Por quê: em `with sqlite3.connect(...) as con:` o context manager padrão só faz commit/rollback e NÃO fecha a
conexão. Em processos longos (Flask, daemon de paper trading, supervisor) isso acumulava descritores de arquivo
abertos (milhares em horas) até estourar o limite do SO ("Too many open files"). `connect()` devolve uma
conexão cujo `__exit__` faz commit/rollback e fecha.
"""
import sqlite3


class ClosingConnection(sqlite3.Connection):
    """Connection cujo `with` faz commit/rollback e fecha a conexão."""

    def __exit__(self, exc_type, exc, tb):
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            try:
                self.close()
            except Exception:  # noqa: BLE001
                pass


def connect(*args, **kwargs) -> sqlite3.Connection:
    kwargs.setdefault("factory", ClosingConnection)
    return sqlite3.connect(*args, **kwargs)
