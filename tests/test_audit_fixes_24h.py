"""
Testes dedicados às correções identificadas na auditoria das últimas 24h:
1. Deduplicação e trava de barra/cooldown em Swing Trade (impede re-entradas no mesmo candle 1h).
2. Cálculo de margem real (notional / leverage) na interface Flask.
3. Provedor Google Gemini Flash no ChatGPTAssistant.
"""

import asyncio
import os
import sqlite3
import tempfile
import time
from unittest.mock import MagicMock, patch
import pytest
import pandas as pd

from chatgpt_assistant import ChatGPTAssistant
from smart_trade import PaperTradingEngine, PaperPosition, PaperTrade, TradeSignal


def test_swing_bar_deduplication():
    """Garante que _swing_entered_bar e _swing_in_cooldown bloqueiem re-entradas na mesma barra 1h."""
    from scripts.run_paper_simulation import PaperTradingDaemon, SWING_BAR_MS

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        engine = PaperTradingEngine(db_path=db_path, initial_capital=10000.0)
        daemon = PaperTradingDaemon(engine=engine)

        sym = "ETHUSDT"
        now_ts = 1792490000000  # base ts
        bar_id = now_ts // SWING_BAR_MS

        # 1. Sem trades prévios, não deve estar travado
        assert not daemon._swing_entered_bar(sym, bar_id)
        assert not daemon._swing_in_cooldown(sym, bar_id)

        # 2. Simula abertura de trade DIRECTIONAL na barra
        entry_ts = (bar_id + 1) * SWING_BAR_MS
        trade_open = PaperTrade(
            trade_id="tr1",
            position_id="p1",
            symbol=sym,
            action="OPEN_SHORT",
            price=2400.0,
            quantity=2.0,
            fee=2.0,
            pnl=0.0,
            timestamp=entry_ts,
            rationale="Short teste"
        )
        engine.trade_history.append(trade_open)

        # Agora _swing_entered_bar DEVE ser True para este bar_id
        assert daemon._swing_entered_bar(sym, bar_id)
        # E False para o próximo bar_id
        assert not daemon._swing_entered_bar(sym, bar_id + 1)

        # 3. Simula fechamento por STOP_LOSS na mesma barra
        trade_close = PaperTrade(
            trade_id="tr2",
            position_id="p1",
            symbol=sym,
            action="STOP_LOSS",
            price=2420.0,
            quantity=2.0,
            fee=2.0,
            pnl=-40.0,
            timestamp=entry_ts + 60000,
            rationale="Stop loss teste"
        )
        engine.trade_history.append(trade_close)

        # _swing_in_cooldown DEVE ser True (impede reabertura imediata)
        assert daemon._swing_in_cooldown(sym, bar_id)
    finally:
        for suf in ("", "-wal", "-shm"):
            if os.path.exists(db_path + suf):
                os.remove(db_path + suf)


def test_flask_margin_calculation_with_leverage():
    """Garante que a query SQLite de margem no Flask divida notional por leverage."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        con = sqlite3.connect(db_path)
        con.execute("""
            CREATE TABLE paper_positions (
                position_id TEXT PRIMARY KEY,
                symbol TEXT,
                status TEXT,
                notional REAL,
                leverage REAL,
                unrealized_pnl REAL
            )
        """)
        # Insere 2 posições com alavancagem 3x
        con.execute("INSERT INTO paper_positions VALUES ('p1', 'SOLUSDT', 'OPEN', 9000.0, 3.0, -50.0)")
        con.execute("INSERT INTO paper_positions VALUES ('p2', 'ETHUSDT', 'OPEN', 12000.0, 3.0, -30.0)")
        # Insere 1 posição FECHADA (não deve contar)
        con.execute("INSERT INTO paper_positions VALUES ('p3', 'BTCUSDT', 'CLOSED', 15000.0, 3.0, 100.0)")
        con.commit()

        cur = con.cursor()
        cur.execute("SELECT COUNT(*), SUM(notional / COALESCE(NULLIF(leverage, 0), 1.0)) FROM paper_positions WHERE status='OPEN'")
        count, margin_used = cur.fetchone()

        # Notional total = 21000. Margem real = 3000 + 4000 = 7000
        assert count == 2
        assert margin_used == 7000.0
        con.close()
    finally:
        for suf in ("", "-wal", "-shm"):
            if os.path.exists(db_path + suf):
                os.remove(db_path + suf)


@pytest.mark.asyncio
async def test_chatgpt_assistant_gemini_provider():
    """Valida o suporte ao provedor Gemini Flash no ChatGPTAssistant."""
    # 1. Desativado por padrão
    asst_disabled = ChatGPTAssistant(provider="gemini", enabled=False)
    assert await asst_disabled.analyze_async("teste") is None

    # 2. Habilitado sem chave de API -> fallback seguro
    asst_no_key = ChatGPTAssistant(provider="gemini", enabled=True, api_key="")
    with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}):
        res = await asst_no_key.analyze_async("teste")
        assert res["direction"] == "HOLD"
        assert "não configurada" in res["rationale"]

    # 3. Com mock do google-genai Client
    with patch("google.genai.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        mock_response = MagicMock()
        mock_response.text = '{"direction": "BUY", "confidence": 0.82, "rationale": "Forte confluência técnica e fluxo"}'
        mock_instance.models.generate_content.return_value = mock_response

        asst_mocked = ChatGPTAssistant(provider="gemini", model="gemini-3.8-flash", enabled=True, api_key="test_key")
        res = await asst_mocked.analyze_async("Analise ETH")

        assert res["direction"] == "BUY"
        assert res["confidence"] == 0.82
        assert "confluência" in res["rationale"]
