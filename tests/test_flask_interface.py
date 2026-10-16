"""
Testes unitários e de integração para o Flask Interface (Fase 8 / Dashboard & Telemetria).
Validação de:
1. Respostas HTTP e renderização de templates.
2. Contrato da API de telemetria (/api/status) e integridade do SQLite.
3. Rota de dados de candles para Plotly (/api/chart/<symbol>/<timeframe>).
4. Mascaramento e atualização segura de configurações (/api/config).
5. Health check (/health).
6. Garantia de modo READ-ONLY estrito.
"""

import os
import sys
import json
import sqlite3
import pytest

# Garante inclusão do diretório raiz no sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask_interface import app, get_db_connection, DB_PATH, mask_key_value


@pytest.fixture
def client():
    """Configura o client de testes da aplicação Flask."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestFlaskInterfaceEndpoints:
    """Testa os endpoints principais da interface web do IA Trader."""

    def test_health_check_endpoint(self, client):
        """Verifica se /health responde 200 e reporta status saudável, read-only e fases ativas."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert data["database_connected"] is True
        assert data["read_only_mode"] is True
        assert "version" in data
        assert "server_time_utc" in data
        assert "active_phase" in data
        # Fase 3 e Fase 2 com status de governança válidos
        assert data["active_phases"].get("phase_3") in ("EM_REVISAO", "CONCLUIDO", "HOMOLOGADO")
        assert data["active_phases"].get("phase_2") == "HOMOLOGADO"

    def test_dashboard_home_page_renders(self, client):
        """Verifica se a rota raiz (/) renderiza o template com sucesso e contém elementos-chave."""
        response = client.get("/")
        assert response.status_code == 200
        content = response.get_data(as_text=True)
        assert "IA TRADER" in content
        assert "READ-ONLY STRICT" in content
        assert "plotly" in content.lower()
        assert "settingsModal" in content

    def test_api_status_contract(self, client):
        """Verifica se /api/status retorna a estrutura completa de telemetria e governança."""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.get_json()

        # Validações de alto nível
        assert data["read_only_mode"] is True
        assert "project" in data
        assert "database" in data
        assert "phases" in data
        assert "connectors" in data
        assert "agents" in data

        # Projeto e testes
        assert data["project"]["tests_passed"] >= 23
        assert data["project"]["pass_rate_pct"] == 100.0
        assert data.get("agent_mode") in ("LIVE_EVALUATION_PAPER_TRADING", "SIMULATED_PENDING_PHASE_4")

        # Fases do projeto (11 fases da Regra de Ouro)
        assert len(data["phases"]) == 11
        phase3 = next(p for p in data["phases"] if p["id"] == 3)
        assert phase3["status"] in ("EM_REVISAO", "CONCLUIDO", "HOMOLOGADO")
        phase2 = next(p for p in data["phases"] if p["id"] == 2)
        assert phase2["status"] == "HOMOLOGADO"

        # Conectores gratuitos
        assert len(data["connectors"]) in (6, 7)
        connector_names = [c["name"] for c in data["connectors"]]
        assert any("Bitget" in name for name in connector_names)
        assert any("FRED" in name for name in connector_names)
        assert any("DefiLlama" in name for name in connector_names)

        # Agentes de deliberação (incluindo Sentinela de Risco com poder de veto)
        agent_ids = [a["id"] for a in data["agents"]]
        assert "technical" in agent_ids
        assert "risk_guard" in agent_ids

    def test_api_chart_endpoint_with_data(self, client):
        """Verifica se /api/chart retorna candles estruturados e indicadores calculados."""
        response = client.get("/api/chart/BTCUSDT/1h?limit=50")
        assert response.status_code == 200
        data = response.get_json()

        assert data["symbol"] == "BTCUSDT"
        assert data["timeframe"] == "1h"
        assert "timestamps" in data
        assert "open" in data
        assert "high" in data
        assert "low" in data
        assert "close" in data
        assert "volume" in data
        assert "indicators" in data
        assert "summary" in data

        # Se há dados, deve ter arrays com mesmo comprimento
        count = data["count"]
        if count > 0:
            assert len(data["timestamps"]) == count
            assert len(data["open"]) == count
            assert len(data["close"]) == count
            assert len(data["volume"]) == count
            assert "sma20" in data["indicators"]
            assert "rsi" in data["indicators"]

    def test_api_chart_empty_symbol_graceful_handling(self, client):
        """Verifica se um par inexistente retorna resposta JSON estruturada sem erro 500."""
        response = client.get("/api/chart/NONEXISTENT_PAIR/1h")
        assert response.status_code == 200
        data = response.get_json()
        assert data["count"] == 0
        assert data["symbol"] == "NONEXISTENT_PAIR"
        assert data["open"] == []

    def test_api_config_get_and_post(self, client):
        """Verifica obtenção mascarada e salvamento seguro de configurações."""
        # 1. GET /api/config
        get_res = client.get("/api/config")
        assert get_res.status_code == 200
        config_data = get_res.get_json()
        assert config_data["status"] == "success"
        assert "keys" in config_data
        assert "BITGET_API_KEY" in config_data["keys"]
        assert "masked" in config_data["keys"]["BITGET_API_KEY"]

        # 2. POST /api/config (atualização de preferência de trading sem sobrescrever senhas já mascaradas)
        post_payload = {
            "DEFAULT_SYMBOL": "ETHUSDT",
            "DEFAULT_TIMEFRAME": "15m",
            "BITGET_API_KEY": "••••••••",  # Deve ser ignorado pelo filtro anti-sobrescrita
        }
        post_res = client.post("/api/config", json=post_payload)
        assert post_res.status_code == 200
        save_data = post_res.get_json()
        assert save_data["status"] == "success"
        assert "DEFAULT_SYMBOL" in save_data["updated_keys"]
        assert "DEFAULT_TIMEFRAME" in save_data["updated_keys"]
        # BITGET_API_KEY não deve ter sido alterado pois era valor mascarado
        assert "BITGET_API_KEY" not in save_data["updated_keys"]

    def test_key_masking_utility(self):
        """Testa o comportamento da função utilitária de mascaramento de chaves."""
        empty_res = mask_key_value("")
        assert empty_res["configured"] is False
        assert empty_res["preview"] == "Não configurado"

        short_res = mask_key_value("abc")
        assert short_res["configured"] is True
        assert short_res["masked"] == "••••"

        long_res = mask_key_value("secret_token_123456789")
        assert long_res["configured"] is True
        assert long_res["masked"].endswith("6789")
        assert long_res["masked"].startswith("••••")

    def test_strict_read_only_mode(self):
        """Verifica se a conexão de leitura com PRAGMA query_only impede inserções."""
        conn = get_db_connection(read_only=True)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("INSERT INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume, quote_volume) VALUES ('TEST', '1h', 1, 1, 1, 1, 1, 1, 1);")
        finally:
            conn.close()

    def test_api_config_empty_string_does_not_wipe_key(self, client):
        """Garante que campo vazio não apaga chave existente nem grava CHAVE= vazia no .env."""
        # 1. Configura uma chave temporária
        test_val = "test_fred_key_secure_12345"
        client.post("/api/config", json={"FRED_API_KEY": test_val})

        # 2. Envia campo vazio
        res_empty = client.post("/api/config", json={"FRED_API_KEY": "   "})
        assert res_empty.status_code == 200
        data = res_empty.get_json()
        assert "FRED_API_KEY" not in data.get("updated_keys", [])
        assert "FRED_API_KEY" not in data.get("removed_keys", [])

        # 3. Verifica que a chave ainda existe e permanece intacta
        get_res = client.get("/api/config")
        keys = get_res.get_json()["keys"]
        assert keys["FRED_API_KEY"]["configured"] is True

    def test_api_config_explicit_remove_keys(self, client):
        """Garante que apenas ação explícita remove_keys remove a chave do .env."""
        # 1. Configura chave de teste
        client.post("/api/config", json={"FRED_API_KEY": "temporary_to_remove"})

        # 2. Remove explicitamente
        res_del = client.post("/api/config", json={"remove_keys": ["FRED_API_KEY"]})
        assert res_del.status_code == 200
        data = res_del.get_json()
        assert "FRED_API_KEY" in data.get("removed_keys", [])

        # 3. Verifica que foi desconfigurado
        get_res = client.get("/api/config")
        keys = get_res.get_json()["keys"]
        assert keys["FRED_API_KEY"]["configured"] is False

    def test_static_assets_fallback_routes(self, client):
        """Verifica se os assets locais de fallback são servidos com sucesso."""
        for asset in ["/static/lucide.min.js", "/static/plotly.min.js", "/static/dashboard.css"]:
            res = client.get(asset)
            assert res.status_code == 200, f"Falha ao carregar {asset}"
            assert len(res.data) > 50, f"Arquivo {asset} está vazio ou truncado"

    def test_api_paper_trading_endpoints(self, client):
        """Verifica os endpoints de telemetria e estado de Paper Trading da Fase 6."""
        # 1. Status da conta de simulação
        res_status = client.get("/api/paper-trading/status")
        assert res_status.status_code == 200
        data_status = res_status.get_json()
        assert data_status["simulation_mode"] is True
        assert data_status["initial_capital"] == 10000.0
        assert "current_equity" in data_status
        assert "win_rate_pct" in data_status

        # 2. Posições de simulação
        res_pos = client.get("/api/paper-trading/positions")
        assert res_pos.status_code == 200
        data_pos = res_pos.get_json()
        assert "positions" in data_pos
        assert isinstance(data_pos["positions"], list)

        # 3. Trades de simulação
        res_trades = client.get("/api/paper-trading/trades")
        assert res_trades.status_code == 200
        data_trades = res_trades.get_json()
        assert "trades" in data_trades
        assert isinstance(data_trades["trades"], list)

    def test_api_golden_pots_endpoint(self, client):
        """Verifica o endpoint do radar de Golden Pots para Scalps e Swings."""
        res = client.get("/api/golden-pots")
        assert res.status_code == 200
        data = res.get_json()
        assert "timestamp" in data
        assert "total_pots" in data
        assert "top_scalps" in data
        assert "top_swings" in data
        assert "all_pots" in data
        assert isinstance(data["top_scalps"], list)
        assert isinstance(data["top_swings"], list)


class TestSimulationLotSettings:
    """paper_lot_usd editável (10..1000); alavancagem nunca é aceita via POST."""

    @pytest.fixture
    def tmp_db(self, tmp_path, monkeypatch):
        import flask_interface
        db = str(tmp_path / "sim_settings.db")
        sqlite3.connect(db).close()
        monkeypatch.setattr(flask_interface, "DB_PATH", db)
        return db

    def test_get_default_lot(self, client, tmp_db):
        data = client.get("/api/simulation/settings").get_json()
        assert data["paper_lot_usd"] == 100.0
        assert "leverage" not in data

    def test_post_valid_lot_persisted_with_updated_at(self, client, tmp_db):
        res = client.post("/api/simulation/settings", json={"paper_lot_usd": 250})
        assert res.status_code == 200
        assert res.get_json()["paper_lot_usd"] == 250.0
        row = sqlite3.connect(tmp_db).execute(
            "SELECT value, updated_at FROM simulation_settings WHERE key='paper_lot_usd'").fetchone()
        assert float(row[0]) == 250.0 and row[1]

    @pytest.mark.parametrize("bad", [9.99, 0, -5, 1000.01, 5000, "abc", True, float("nan")])
    def test_post_invalid_lot_rejected(self, client, tmp_db, bad):
        res = client.post("/api/simulation/settings", json={"paper_lot_usd": bad})
        assert res.status_code == 400
        assert "lote" in res.get_json()["message"].lower()
        con = sqlite3.connect(tmp_db)
        try:
            n = con.execute("SELECT COUNT(*) FROM simulation_settings WHERE key='paper_lot_usd'").fetchone()[0]
        except sqlite3.OperationalError:
            n = 0
        assert n == 0

    @pytest.mark.parametrize("lot", [10, 1000])
    def test_lot_boundaries_accepted(self, client, tmp_db, lot):
        assert client.post("/api/simulation/settings", json={"paper_lot_usd": lot}).status_code == 200

    def test_leverage_ignored_on_post(self, client, tmp_db):
        res = client.post("/api/simulation/settings", json={"paper_lot_usd": 50, "leverage": 20, "max_leverage": 50})
        assert res.status_code == 200
        keys = [r[0] for r in sqlite3.connect(tmp_db).execute("SELECT key FROM simulation_settings")]
        assert "leverage" not in keys and "max_leverage" not in keys
        assert "leverage" not in res.get_json()

    def test_dashboard_has_lot_input_and_readonly_leverage(self, client):
        html = client.get("/").get_data(as_text=True)
        assert 'id="inputPaperLotUsd"' in html
        assert "máximo 5x (piloto)" in html
        assert "A11" in html
        assert 'id="inputLeverage' not in html


class TestOpenPositionsEndpoint:
    """Aba de abertos: somente status OPEN (inclui parcial pós-TP1); fechados nunca em /open."""

    @pytest.fixture
    def pos_db(self, tmp_path, monkeypatch):
        import flask_interface
        db = str(tmp_path / "pos.db")
        con = sqlite3.connect(db)
        con.execute("""CREATE TABLE paper_positions (position_id TEXT PRIMARY KEY, symbol TEXT, side TEXT,
            entry_price REAL, current_price REAL, quantity REAL, notional REAL, stop_loss REAL, take_profit REAL,
            unrealized_pnl REAL, entry_timestamp INTEGER, strategy_type TEXT, status TEXT,
            tp1_price REAL, tp1_hit INTEGER, tp2_price REAL, initial_quantity REAL)""")
        rows = [
            ("A", "BTCUSDT", "LONG", 100, 101, 1.0, 100, 95, 110, 1.0, 3, "SCALP", "OPEN", 105, 0, 110, 1.0),
            ("B", "ETHUSDT", "SHORT", 50, 49, 0.5, 25, 55, 40, 0.5, 2, "SWING", "CLOSED", 45, 1, 40, 1.0),
            ("C", "SOLUSDT", "LONG", 20, 21, 0.5, 10, 18, 25, 0.5, 1, "SCALP", "OPEN", 22, 1, 25, 1.0),
        ]
        con.executemany("INSERT INTO paper_positions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        con.commit(); con.close()
        monkeypatch.setattr(flask_interface, "DB_PATH", db)
        return db

    def test_open_excludes_closed(self, client, pos_db):
        for url in ("/api/paper-trading/positions/open", "/api/paper-trading/positions"):
            data = client.get(url).get_json()
            ids = [p["position_id"] for p in data["positions"]]
            assert sorted(ids) == ["A", "C"], url
            assert all(p["status"] == "OPEN" for p in data["positions"])
            assert len(ids) == len(set(ids))

    def test_partially_reduced_tp1_stays_open(self, client, pos_db):
        data = client.get("/api/paper-trading/positions/open").get_json()
        c = [p for p in data["positions"] if p["position_id"] == "C"][0]
        assert c["tp1_hit"] == 1 and c["quantity"] == 0.5 and c["initial_quantity"] == 1.0

    def test_closed_only_in_closed(self, client, pos_db):
        data = client.get("/api/paper-trading/positions/closed").get_json()
        assert [p["position_id"] for p in data["positions"]] == ["B"]
        allp = client.get("/api/paper-trading/positions?status=all").get_json()
        assert allp["total"] == 3



class TestPositionsEnrichment:
    @pytest.fixture
    def rich_db(self, tmp_path, monkeypatch):
        import flask_interface
        db = str(tmp_path / "rich.db")
        con = sqlite3.connect(db)
        con.execute("""CREATE TABLE paper_positions (position_id TEXT PRIMARY KEY, symbol TEXT, side TEXT,
            entry_price REAL, current_price REAL, quantity REAL, notional REAL, stop_loss REAL, take_profit REAL,
            unrealized_pnl REAL, realized_pnl REAL, entry_timestamp INTEGER, strategy_type TEXT, status TEXT,
            tp1_price REAL, tp1_hit INTEGER, tp2_price REAL, initial_quantity REAL, leverage REAL)""")
        con.execute("""CREATE TABLE paper_trades (trade_id TEXT PRIMARY KEY, position_id TEXT, symbol TEXT, action TEXT,
            price REAL, quantity REAL, fee REAL, pnl REAL, timestamp INTEGER, rationale TEXT)""")
        con.execute("INSERT INTO paper_positions VALUES ('O','BTCUSDT','LONG',100,102,1,102,95,110,2,0,1700000000,'SCALP','OPEN',105,0,110,1,4)")
        con.execute("INSERT INTO paper_positions VALUES ('S','ETHUSDT','SHORT',200,190,1,190,210,150,0,5,1700000000000,'SCALP','CLOSED',0,0,0,1,NULL)")
        con.execute("INSERT INTO paper_trades VALUES ('t1','S','ETHUSDT','OPEN_SCALP_SHORT',200,1,0,0,1700000000000,'')")
        con.execute("INSERT INTO paper_trades VALUES ('t2','S','ETHUSDT','STOP_LOSS',190,1,0,5,1700003600000,'')")
        con.commit(); con.close()
        monkeypatch.setattr(flask_interface, "DB_PATH", db)

    def test_open_enrichment_ms_roi_and_margin(self, client, rich_db):
        d = client.get("/api/paper-trading/positions/open").get_json()
        p = d["positions"][0]
        assert p["opened_at_ms"] == 1700000000000  # s -> ms
        assert p["leverage_effective"] == 4 and p["margin_effective"] == 25.0
        assert p["roi_pct"] == 8.0 and p["price_change_pct"] == 2.0 and p["roi_total_pct"] == 8.0
        assert d["summary"]["count"] == 1 and d["summary"]["win_rate_pct"] == 100.0
        assert d["summary"]["total_pnl_pct_initial"] == 0.02
        assert d["summary"]["unrealized_pnl"] == 2.0 and d["summary"]["realized_pnl"] == 0.0

    def test_open_enrichment_with_realized_partial_roi_total(self, client, rich_db):
        # simula posição com parciais já realizadas (TP1/TP2)
        import flask_interface, sqlite3
        con = sqlite3.connect(flask_interface.DB_PATH)
        con.execute("UPDATE paper_positions SET realized_pnl = 3.68, unrealized_pnl = 2.12, quantity = 0.3, initial_quantity = 1.0 WHERE position_id = 'O'")
        con.commit(); con.close()
        d = client.get("/api/paper-trading/positions/open").get_json()
        p = d["positions"][0]
        # margem inicial = 1.0 * 100 / 4 = 25.0; total pnl = 3.68 + 2.12 = 5.80; roi_total = 5.80 / 25.0 = 23.2%
        # margem restante = 0.3 * 100 / 4 = 7.5; roi_runner = 2.12 / 7.5 = 28.2667%
        assert round(p["roi_total_pct"], 2) == 23.2
        assert round(p["roi_pct"], 2) == 28.27

    def test_closed_enrichment_fallback_1x_and_close_time(self, client, rich_db):
        d = client.get("/api/paper-trading/positions/closed").get_json()
        p = d["positions"][0]
        assert p["leverage_effective"] == 1.0 and p["margin_effective"] == 200.0 and p["margin_estimated"] is True
        assert p["closed_at_ms"] == 1700003600000 and p["duration_s"] == 3600
        assert p["net_pnl"] == 5.0 and p["net_pnl_pct_margin"] == 2.5 and p["price_change_pct"] == 5.0
        assert d["summary"]["wins"] == 1

    def test_trades_endpoint_has_timestamp_ms(self, client, rich_db):
        t = client.get("/api/paper-trading/trades").get_json()["trades"]
        assert all(x["timestamp_ms"] >= 1700000000000 for x in t)


class TestExitLadderApi:
    @pytest.fixture
    def ladder_db(self, tmp_path, monkeypatch):
        import flask_interface
        db = str(tmp_path / "ladder.db")
        con = sqlite3.connect(db)
        con.execute("""CREATE TABLE paper_positions (position_id TEXT PRIMARY KEY, symbol TEXT, side TEXT,
            entry_price REAL, current_price REAL, quantity REAL, notional REAL, stop_loss REAL, take_profit REAL,
            unrealized_pnl REAL, realized_pnl REAL, entry_timestamp INTEGER, strategy_type TEXT, status TEXT,
            tp1_price REAL, tp1_hit INTEGER, tp2_price REAL, initial_quantity REAL, leverage REAL,
            tp3_price REAL, tp2_hit INTEGER, tp1_fraction REAL, tp2_fraction REAL, trailing_active INTEGER,
            trailing_stop_price REAL)""")
        con.execute("""CREATE TABLE paper_trades (trade_id TEXT PRIMARY KEY, position_id TEXT, symbol TEXT, action TEXT,
            price REAL, quantity REAL, fee REAL, pnl REAL, timestamp INTEGER, rationale TEXT)""")
        con.execute("INSERT INTO paper_positions VALUES ('L','BTCUSDT','LONG',100,112,0.3,30,100,120,1,5,1700000000000,'SCALP','OPEN',105,1,110,1,3,120,1,0.4,0.3,1,108)")
        con.execute("INSERT INTO paper_positions VALUES ('G','BTCUSDT','LONG',100,101,1,100,95,110,1,0,1700000000000,'SCALP','OPEN',0,0,0,1,3,0,0,0.5,0.0,0,0)")
        con.execute("INSERT INTO paper_positions VALUES ('X','ETHUSDT','LONG',100,118,0,0,100,120,0,9,1700000000000,'SCALP','CLOSED',105,1,110,1,3,120,1,0.4,0.3,1,108)")
        for i, (a, ts) in enumerate([("OPEN_SCALP_LONG", 0), ("TP1_PARTIAL", 1), ("TP2_PARTIAL", 2), ("TRAILING_STOP", 3)]):
            con.execute("INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?,?,?)", (f"t{i}", "X", "ETHUSDT", a, 100 + i, 0.3, 0, 1, 1700000000000 + ts * 1000, ""))
        con.commit(); con.close()
        monkeypatch.setattr(flask_interface, "DB_PATH", db)

    def test_ladder_fields_exposed(self, client, ladder_db):
        pos = {p["position_id"]: p for p in client.get("/api/paper-trading/positions/open").get_json()["positions"]}
        l = pos["L"]
        assert (l["tp1"], l["tp2"], l["tp3"]) == (105, 110, 120)
        assert (l["tp1_fraction"], l["tp2_fraction"]) == (0.4, 0.3) and abs(l["runner_fraction"] - 0.3) < 1e-9
        assert l["remaining_fraction"] == 0.3 and l["trailing_active"] is True and l["trailing_stop_price"] == 108
        assert l["tp1_hit"] is True and l["tp2_hit"] is True

    def test_legacy_missing_values_are_none(self, client, ladder_db):
        pos = {p["position_id"]: p for p in client.get("/api/paper-trading/positions/open").get_json()["positions"]}
        g = pos["G"]
        assert g["tp1"] is None and g["tp3"] is None and g["trailing_stop_price"] is None
        assert g["tp1_fraction"] is None and g["runner_fraction"] is None and g["trailing_active"] is False

    def test_ledger_labels_and_close_action(self, client, ladder_db):
        x = client.get("/api/paper-trading/positions/closed").get_json()
        assert x["total"] == 1 and x["summary"]["wins"] == 1 and x["summary"]["win_rate_pct"] == 100.0
        p = x["positions"][0]
        assert p["close_action"] == "TRAILING_STOP" and p["close_action_label"] == "Trailing stop"
        labels = [e["label"] for e in p["events"]]
        assert "TP1 parcial" in labels and "TP2 parcial (trailing ativado)" in labels


class TestTapeGateStatus:
    """/api/tape-gate/status e card do Agente Tape Reading (gate pós-comitê). Bancos e agente mockados."""

    @staticmethod
    def _gate_db(path, rows=None, create=True):
        con = sqlite3.connect(path)
        if create:
            con.execute("""CREATE TABLE flow_gate_log (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_ms INTEGER, symbol TEXT,
                strategy_type TEXT, direction TEXT, decision TEXT, tape_vote TEXT, tape_status TEXT, tape_reason TEXT)""")
            for r in rows or []:
                con.execute("INSERT INTO flow_gate_log (ts_ms,symbol,strategy_type,direction,decision,tape_vote,tape_status,tape_reason) VALUES (?,?,?,?,?,?,?,?)", r)
        con.commit()
        con.close()

    @staticmethod
    def _fake_agent(status="OK", vote="BUY"):
        class Fake:
            def evaluate(self, symbol, direction="", now_ms=None):
                if symbol != "BTCUSDT" and status == "OK":
                    return {"agent_name": "TapeReadingAgent", "symbol": symbol, "vote": "HOLD", "status": "UNAVAILABLE",
                            "confidence": 0.0, "features": {}, "reason": "no_book_data", "timestamp": 1}
                return {"agent_name": "TapeReadingAgent", "symbol": symbol, "vote": vote, "status": status,
                        "confidence": 0.62 if status == "OK" else 0.0,
                        "features": {"taker_imb_5m": 0.22, "book_imb_10": 0.14, "microprice_dev": 0.3, "spread_bps": 0.1} if status == "OK" else {},
                        "reason": "flow_buy_confirmed" if status == "OK" else "db_unavailable:FileNotFoundError", "timestamp": 1}
        return Fake()

    def test_empty_table_and_missing_collector_db(self, client, tmp_path, monkeypatch):
        import flask_interface as fi
        db = str(tmp_path / "pilot.db")
        self._gate_db(db)
        monkeypatch.setattr(fi, "DB_PATH", db)
        monkeypatch.setattr(fi, "MICRO_DB_PATH", str(tmp_path / "ausente.db"))
        monkeypatch.setattr(fi, "_make_tape_agent", lambda: self._fake_agent("UNAVAILABLE", "HOLD"))
        r = client.get("/api/tape-gate/status")
        assert r.status_code == 200
        d = r.get_json()
        assert d["read_only_mode"] is True and d["collector_db_available"] is False
        assert set(d["symbols"]) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
        for s in d["symbols"].values():
            assert s["status"] == "UNAVAILABLE" and s["book_age_s"] is None and s["tick_age_s"] is None
        assert d["gate"]["total"] == 0 and d["gate"]["last"] == [] and d["gate"]["log_available"] is True
        assert d["gate"]["counts"] == {"PASSED": 0, "BLOCKED": 0, "UNAVAILABLE_PASSED": 0}

    def test_table_missing_does_not_error(self, client, tmp_path, monkeypatch):
        import flask_interface as fi
        db = str(tmp_path / "pilot.db")
        self._gate_db(db, create=False)
        monkeypatch.setattr(fi, "DB_PATH", db)
        monkeypatch.setattr(fi, "MICRO_DB_PATH", str(tmp_path / "ausente.db"))
        monkeypatch.setattr(fi, "_make_tape_agent", lambda: self._fake_agent("UNAVAILABLE", "HOLD"))
        d = client.get("/api/tape-gate/status").get_json()
        assert d["gate"]["total"] == 0 and d["gate"]["log_available"] is False

    def test_with_records_counts_last10_and_ages(self, client, tmp_path, monkeypatch):
        import time as _t
        import flask_interface as fi
        db = str(tmp_path / "pilot.db")
        now = int(_t.time() * 1000)
        rows = [(now - i * 1000, "BTCUSDT", "SCALP", "BUY", "PASSED" if i % 2 == 0 else "BLOCKED",
                 "BUY", "OK", "flow_buy_confirmed") for i in range(11)]
        rows.append((now - 99999, "ETHUSDT", "SWING", "SELL", "UNAVAILABLE_PASSED", "HOLD", "UNAVAILABLE", "no_book_data"))
        self._gate_db(db, rows)
        micro = str(tmp_path / "micro.db")
        con = sqlite3.connect(micro)
        con.execute("CREATE TABLE book_features (ts_ms INTEGER, symbol TEXT)")
        con.execute("CREATE TABLE tick_agg (minute_ms INTEGER, symbol TEXT)")
        con.execute("INSERT INTO book_features VALUES (?, 'BTCUSDT')", (now - 5000,))
        con.execute("INSERT INTO tick_agg VALUES (?, 'BTCUSDT')", (now - 30000,))
        con.commit()
        con.close()
        monkeypatch.setattr(fi, "DB_PATH", db)
        monkeypatch.setattr(fi, "MICRO_DB_PATH", micro)
        monkeypatch.setattr(fi, "_make_tape_agent", lambda: self._fake_agent("OK", "BUY"))
        d = client.get("/api/tape-gate/status").get_json()
        g = d["gate"]
        assert g["counts"] == {"PASSED": 6, "BLOCKED": 5, "UNAVAILABLE_PASSED": 1} and g["total"] == 12
        assert len(g["last"]) == 10
        assert set(g["last"][0]) == {"ts", "ts_ms", "symbol", "strategy_type", "direction", "decision", "tape_vote", "reason"}
        assert g["last"][0]["ts_ms"] >= g["last"][-1]["ts_ms"]
        btc = d["symbols"]["BTCUSDT"]
        assert btc["vote"] == "BUY" and btc["status"] == "OK" and btc["confidence"] == 0.62
        assert 4 <= btc["book_age_s"] < 60 and 29 <= btc["tick_age_s"] < 90
        assert d["symbols"]["ETHUSDT"]["status"] == "UNAVAILABLE" and d["symbols"]["ETHUSDT"]["book_age_s"] is None
        assert d["collector_db_available"] is True

    def test_agent_card_mapping_and_not_in_consensus(self, tmp_path, monkeypatch):
        import flask_interface as fi
        monkeypatch.setattr(fi, "DB_PATH", str(tmp_path / "x.db"))
        monkeypatch.setattr(fi, "_make_tape_agent", lambda: self._fake_agent("OK", "BUY"))
        c = fi.get_tape_agent_card("BTCUSDT", "BUY")
        assert c["id"] == "tape" and c["name"] == "Agente Tape Reading" and c["is_gate"] is True
        assert c["bias"] == "BULLISH" and c["confidence"] == 62 and c["data_status"] == "OK"
        assert c["confirms_direction"] is True
        assert c["summary"] == "Fluxo comprador confirmado: taker +0.22, book +0.14, microprice +0.3 bps, spread 0.1 bps"
        assert fi.get_tape_agent_card("BTCUSDT", "SELL")["confirms_direction"] is False
        monkeypatch.setattr(fi, "_make_tape_agent", lambda: self._fake_agent("UNAVAILABLE", "HOLD"))
        u = fi.get_tape_agent_card("BTCUSDT", "BUY")
        assert u["bias"] == "NEUTRAL" and u["status"] == "SEM DADOS" and u["data_status"] == "UNAVAILABLE"

    def test_uncovered_symbol_message_and_status_has_tape_gate(self, client, tmp_path, monkeypatch):
        import flask_interface as fi
        monkeypatch.setattr(fi, "MICRO_DB_PATH", str(tmp_path / "ausente.db"))
        c = fi.get_tape_agent_card("DOGEUSDT", "BUY")
        assert c["summary"] == "Símbolo fora da cobertura do coletor" and c["data_status"] == "UNAVAILABLE"
        d = client.get("/api/status").get_json()
        ids = [a["id"] for a in d["agents"]]
        assert ids[-1] == "tape" and len(ids) == 7
        assert d["agents"][-1]["is_gate"] is True


class TestSignalFunnelApi:
    """GET /api/signal-funnel — leitura somente, banco temporário."""

    @pytest.fixture
    def fdb(self, tmp_path, monkeypatch):
        import flask_interface
        db = str(tmp_path / "funnel.db")
        sqlite3.connect(db).close()
        monkeypatch.setattr(flask_interface, "DB_PATH", db)
        return db

    @staticmethod
    def _create(db, rows):
        import time as _t
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE signal_funnel (hour_ms INTEGER, strategy TEXT, stage TEXT, count INTEGER, "
                    "PRIMARY KEY(hour_ms,strategy,stage))")
        h = int(_t.time() * 1000) // 3_600_000 * 3_600_000
        con.executemany("INSERT INTO signal_funnel VALUES (?,?,?,?)",
                        [(h - off * 3_600_000, s, st, n) for off, s, st, n in rows])
        con.commit()
        con.close()

    def test_table_absent(self, client, fdb):
        r = client.get("/api/signal-funnel?hours=24")
        assert r.status_code == 200
        j = r.get_json()
        assert j["meta"]["tabela_disponivel"] is False
        assert all(c["count"] == 0 for c in j["total"]["conversion"])
        assert j["total"]["rejects"] == [] and j["total"]["bottleneck"] is None

    def test_table_empty(self, client, fdb):
        self._create(fdb, [])
        j = client.get("/api/signal-funnel").get_json()
        assert j["meta"]["tabela_disponivel"] is True
        assert j["hours"] == 24 and j["total"]["counts"]["evaluated"] == 0

    def test_with_data(self, client, fdb):
        self._create(fdb, [
            (0, "SCALP", "evaluated", 1000), (0, "SCALP", "candidate", 100),
            (0, "SCALP", "committee_agree", 50), (0, "SCALP", "confidence_ok", 40),
            (0, "SCALP", "risk_ok", 10), (0, "SCALP", "gate_passed", 5), (0, "SCALP", "executed", 5),
            (0, "SCALP", "reject:no_trigger", 900), (0, "SCALP", "reject:veto", 30),
            (0, "SCALP", "reject:committee_disagree", 50),
            (2, "DAY_TRADE", "evaluated", 200), (2, "DAY_TRADE", "candidate", 20),
            (2, "DAY_TRADE", "reject:veto", 5),
            (30, "SCALP", "evaluated", 777),  # fora da janela de 24h
        ])
        j = client.get("/api/signal-funnel?hours=24").get_json()
        s = j["strategies"]["SCALP"]
        assert s["counts"]["evaluated"] == 1000
        conv = {c["stage"]: c for c in s["conversion"]}
        assert conv["candidate"]["step_pct"] == 10.0 and conv["candidate"]["cum_pct"] == 10.0
        assert conv["committee_agree"]["step_pct"] == 50.0
        assert conv["risk_ok"]["step_pct"] == 25.0 and conv["risk_ok"]["cum_pct"] == 1.0
        assert conv["executed"]["step_pct"] == 100.0
        # rejeições ordenadas desc
        assert [r["reason"] for r in s["rejects"]] == ["no_trigger", "committee_disagree", "veto"]
        # gargalo: confidence_ok->risk_ok perde 75% (evaluated->candidate ignorado)
        assert s["bottleneck"]["stage"] == "risk_ok" and s["bottleneck"]["loss_pct"] == 75.0
        assert s["no_trigger"]["loss_pct"] == 90.0
        # total soma estratégias
        assert j["total"]["counts"]["evaluated"] == 1200
        assert {r["reason"]: r["count"] for r in j["total"]["rejects"]}["veto"] == 35
        assert len(s["series"]) == 1
        # janela maior inclui o ponto de 30h atrás
        j72 = client.get("/api/signal-funnel?hours=72").get_json()
        assert j72["strategies"]["SCALP"]["counts"]["evaluated"] == 1777

    @pytest.mark.parametrize("bad", ["0", "2", "abc", "-1", "9999", ""])
    def test_invalid_hours(self, client, fdb, bad):
        assert client.get(f"/api/signal-funnel?hours={bad}").status_code == 400

    def test_read_only_no_write(self, client, fdb):
        self._create(fdb, [(0, "SCALP", "evaluated", 3)])
        client.get("/api/signal-funnel?hours=1")
        con = sqlite3.connect(fdb)
        assert con.execute("SELECT count FROM signal_funnel").fetchone()[0] == 3
        con.close()
