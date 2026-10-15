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
        assert p["roi_pct"] == 8.0 and p["price_change_pct"] == 2.0
        assert d["summary"]["count"] == 1 and d["summary"]["win_rate_pct"] == 100.0
        assert d["summary"]["total_pnl_pct_initial"] == 0.02

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
