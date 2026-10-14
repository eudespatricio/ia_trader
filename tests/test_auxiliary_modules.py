"""
test_auxiliary_modules.py - Testes de Homologação da Fase 9:
Módulos de Apoio, Scanner de Pares, Educacional, Comandos de Voz, Gestão de Usuários e Legal.
"""

import os
import time
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from pair_scanner import PairScanner, VolatilityOpportunity
from educational_module import EducationalExplainer
from voice_commands import VoiceCommandDispatcher
from auth_module import AuthManager, PLAN_PERMISSIONS
import flask_interface


@pytest.fixture
def temp_db(tmp_path):
    """Cria um banco SQLite temporário isolado para os testes."""
    db_file = tmp_path / "test_auxiliary.db"
    return str(db_file)


class TestPairScanner:
    """Valida o PairScanner e seus detectores de anomalia e risco."""

    def test_insufficient_candles(self, temp_db):
        scanner = PairScanner(db_path=temp_db)
        df_short = pd.DataFrame({"close": [100.0] * 10})
        res = scanner.analyze_pair_candles("BTCUSDT", df_short)
        assert res is None

    def test_delisting_risk_detection(self, temp_db):
        scanner = PairScanner(db_path=temp_db)
        # 30 velas simulando um colapso repentino de 50%
        n = 30
        closes = np.linspace(100.0, 50.0, n)
        highs = closes + 1.0
        lows = closes - 1.0
        volumes = np.full(n, 100.0)
        df = pd.DataFrame({"close": closes, "high": highs, "low": lows, "volume": volumes})

        opp = scanner.analyze_pair_candles("COLLAPSEUSDT", df)
        assert opp is not None
        assert opp.opportunity_type == "DELISTING_RISK"
        assert opp.risk_level == "CRITICAL"
        assert opp.allocated_capital_pct == 0.0

    def test_volatility_breakout(self, temp_db):
        scanner = PairScanner(db_path=temp_db, volatility_threshold=1.5, volume_surge_threshold=2.0)
        # 35 velas: 20 velas calmas com range 0.1 e volume 10, seguidas por 15 velas de alta volatilidade e volume
        n = 35
        closes = [100.0] * 20 + [100.0 + i * 2.0 for i in range(15)]
        highs = [100.1] * 20 + [c + 5.0 for c in closes[20:]]
        lows = [99.9] * 20 + [c - 5.0 for c in closes[20:]]
        volumes = [10.0] * 34 + [200.0]  # Spike de volume na última vela
        df = pd.DataFrame({"close": closes, "high": highs, "low": lows, "volume": volumes})

        opp = scanner.analyze_pair_candles("PUMPUSDT", df)
        assert opp is not None
        assert opp.opportunity_type in ["VOLATILITY_BREAKOUT", "PUMP_DETECTED", "VOLUME_SURGE"]
        assert opp.allocated_capital_pct > 0.0
        assert opp.allocated_capital_pct <= scanner.max_isolated_capital_pct

    def test_watchlist_scan(self, temp_db):
        scanner = PairScanner(db_path=temp_db)
        opps = scanner.scan_watchlist(["BTCUSDT", "ETHUSDT"])
        assert isinstance(opps, list)
        assert scanner.last_scan_time > 0


class TestEducationalExplainer:
    """Valida a geração didática de explicações e glossário."""

    def test_glossary_content(self):
        explainer = EducationalExplainer()
        glossary = explainer.get_glossary()
        assert "ORDER_BLOCK" in glossary
        assert "FAIR_VALUE_GAP" in glossary
        assert "ATR" in glossary
        assert "CIRCUIT_BREAKER" in glossary
        assert len(glossary["ORDER_BLOCK"]) > 30

    def test_explain_buy_trade(self):
        explainer = EducationalExplainer()
        trade = {
            "symbol": "BTCUSDT",
            "direction": "BUY",
            "entry_price": 60000.0,
            "stop_loss": 59000.0,
            "take_profit": 63000.0,
            "rationale": "SWEEP_SSL + POI Mitigation",
            "regime": "BULL_TREND"
        }
        res = explainer.explain_trade_decision(trade)
        assert "Compra" in res["action"]
        assert "Stop Loss" in res["risk_analysis"]
        assert "narrative" in res

    def test_explain_sell_trade(self):
        explainer = EducationalExplainer()
        trade = {
            "symbol": "ETHUSDT",
            "direction": "SELL",
            "entry_price": 3000.0,
            "stop_loss": 3100.0,
            "take_profit": 2700.0,
            "rationale": "SWEEP_BSL + Bearish FVG",
            "regime": "BEAR_TREND"
        }
        res = explainer.explain_trade_decision(trade)
        assert "Venda" in res["action"]
        assert "Stop Loss" in res["risk_analysis"]

    def test_explain_hold_trade(self):
        explainer = EducationalExplainer()
        trade = {
            "symbol": "SOLUSDT",
            "direction": "HOLD",
            "entry_price": 150.0,
            "rationale": "Low conviction",
            "regime": "CHOPPY"
        }
        res = explainer.explain_trade_decision(trade)
        assert "Aguardar" in res["action"]


class TestVoiceCommands:
    """Valida o despachante semântico de comandos de voz e texto."""

    def test_unrecognized_command(self, temp_db):
        dispatcher = VoiceCommandDispatcher(db_path=temp_db)
        res = dispatcher.process_command("abra uma ordem de 100x na shiba inu")
        assert res["status"] == "UNRECOGNIZED"
        assert res["action_executed"] is None

    def test_status_command(self, temp_db):
        dispatcher = VoiceCommandDispatcher(db_path=temp_db)
        res = dispatcher.process_command("qual o status do robô?")
        assert res["status"] == "SUCCESS"
        assert res["command_type"] == "STATUS"
        assert "patrimônio" in res["response"] or "ativo" in res["response"]

    def test_pause_and_resume_commands(self, temp_db):
        dispatcher = VoiceCommandDispatcher(db_path=temp_db)
        # Pausa
        res_pause = dispatcher.process_command("pausar operacoes imediatamente!")
        assert res_pause["status"] == "SUCCESS"
        assert res_pause["command_type"] == "EMERGENCY_STOP"

        # Retomada
        res_resume = dispatcher.process_command("retomar operações agora")
        assert res_resume["status"] == "SUCCESS"
        assert res_resume["command_type"] == "RESET"

    def test_market_regime_command(self, temp_db):
        dispatcher = VoiceCommandDispatcher(db_path=temp_db)
        res = dispatcher.process_command("qual o regime de mercado atual?")
        assert res["status"] == "SUCCESS"
        assert res["command_type"] == "MARKET_REGIME"

    def test_recent_trades_command(self, temp_db):
        dispatcher = VoiceCommandDispatcher(db_path=temp_db)
        res = dispatcher.process_command("mostrar últimas operações")
        assert res["status"] == "SUCCESS"
        assert res["command_type"] == "RECENT_TRADES"


class TestAuthModule:
    """Valida gestão de usuários, planos e aceite legal."""

    def test_admin_created_by_default(self, temp_db):
        auth = AuthManager(db_path=temp_db)
        admin = auth.authenticate_user("admin@iatrader.local", "admin123")
        assert admin is not None
        assert admin["plan"] == "PREMIUM"
        assert admin["terms_accepted"] is True

    def test_register_and_authenticate(self, temp_db):
        auth = AuthManager(db_path=temp_db)
        # Sucesso
        ok, msg = auth.register_user("trader@test.com", "senhaForte123", plan="PRO")
        assert ok is True

        # Duplicata
        ok2, msg2 = auth.register_user("trader@test.com", "senhaForte123")
        assert ok2 is False
        assert "já está cadastrado" in msg2

        # Autenticação correta
        user = auth.authenticate_user("trader@test.com", "senhaForte123")
        assert user is not None
        assert user["email"] == "trader@test.com"
        assert user["plan"] == "PRO"
        assert user["terms_accepted"] is False

        # Senha incorreta
        bad_auth = auth.authenticate_user("trader@test.com", "senhaErrada")
        assert bad_auth is None

    def test_accept_terms(self, temp_db):
        auth = AuthManager(db_path=temp_db)
        auth.register_user("legal@test.com", "segredo123", plan="FREE")
        user = auth.authenticate_user("legal@test.com", "segredo123")
        assert user["terms_accepted"] is False

        accepted = auth.accept_terms("legal@test.com")
        assert accepted is True

        user_after = auth.authenticate_user("legal@test.com", "segredo123")
        assert user_after["terms_accepted"] is True

    def test_plan_permissions_gating(self, temp_db):
        auth = AuthManager(db_path=temp_db)
        # FREE não pode golden pots nem voz
        assert auth.can_access_feature("FREE", "can_execute_golden_pots") is False
        assert auth.can_access_feature("FREE", "can_use_voice_commands") is False

        # PRO pode golden pots e voz
        assert auth.can_access_feature("PRO", "can_execute_golden_pots") is True
        assert auth.can_access_feature("PRO", "can_use_voice_commands") is True

        # PREMIUM tem suporte prioritário
        assert auth.can_access_feature("PREMIUM", "priority_support") is True


class TestFlaskAuxiliaryEndpoints:
    """Valida as novas rotas expostas no Flask."""

    @pytest.fixture
    def client(self):
        flask_interface.app.config["TESTING"] = True
        with flask_interface.app.test_client() as c:
            yield c

    def test_legal_routes(self, client):
        r_terms = client.get("/terms")
        assert r_terms.status_code == 200
        assert b"Termos de Uso" in r_terms.data

        r_privacy = client.get("/privacy")
        assert r_privacy.status_code == 200
        assert b"Pol" in r_privacy.data

    def test_educational_endpoints(self, client):
        r_glossary = client.get("/api/educational/glossary")
        assert r_glossary.status_code == 200
        data = r_glossary.get_json()
        assert "ORDER_BLOCK" in data

        r_explain = client.post("/api/educational/explain", json={
            "symbol": "BTCUSDT",
            "direction": "BUY",
            "entry_price": 60000.0,
            "stop_loss": 59000.0,
            "take_profit": 63000.0
        })
        assert r_explain.status_code == 200
        explain_data = r_explain.get_json()
        assert "title" in explain_data
        assert "Compra" in explain_data["title"]

    def test_voice_command_endpoint(self, client):
        r_voice = client.post("/api/voice/command", json={"command": "qual o regime do mercado?"})
        assert r_voice.status_code == 200
        res = r_voice.get_json()
        assert res["status"] == "SUCCESS"
        assert res["command_type"] == "MARKET_REGIME"

    def test_scanner_endpoint(self, client):
        r_scanner = client.get("/api/scanner/opportunities")
        assert r_scanner.status_code == 200
        data = r_scanner.get_json()
        assert isinstance(data, list)
