"""Independent security hardening tests for ia_trader/flask_interface.py."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import flask_interface as fi

LEGACY_JWT_SECRET = "super-secret-key-ia-trader-2026"
FLASK_SRC_PATH = ROOT_DIR / "flask_interface.py"


@pytest.fixture
def isolated_client(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setattr(fi, "ENV_FILE_PATH", str(env_file))
    fi.app.config["TESTING"] = True
    with fi.app.test_client() as client:
        yield client, env_file


def test_default_host_is_loopback_when_env_unset(monkeypatch):
    for var in ("IA_TRADER_HOST", "FLASK_HOST", "HOST", "PORT", "FLASK_DEBUG"):
        monkeypatch.delenv(var, raising=False)
    tree = ast.parse(FLASK_SRC_PATH.read_text(encoding="utf-8"), filename=str(FLASK_SRC_PATH))
    main_if = next(
        node for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )
    captured = {}

    class _FakeApp:
        @staticmethod
        def run(**kwargs):
            captured.update(kwargs)

    ns = {"os": os, "sys": type("S", (), {"argv": ["flask_interface.py"]})(), "DB_PATH": "mem", "app": _FakeApp()}
    exec(compile(ast.Module(body=main_if.body, type_ignores=[]), str(FLASK_SRC_PATH), "exec"), ns)
    assert captured["host"] == "127.0.0.1"


def test_jwt_secret_key_is_not_hardcoded_and_ephemeral_when_unset():
    src = FLASK_SRC_PATH.read_text(encoding="utf-8")
    assert LEGACY_JWT_SECRET not in src
    assert fi.app.config["JWT_SECRET_KEY"] != LEGACY_JWT_SECRET

    env = {k: v for k, v in os.environ.items() if k != "JWT_SECRET_KEY"}
    probe = (
        "import sys; sys.path.insert(0, %r); "
        "import dotenv; dotenv.load_dotenv = lambda *a, **kw: False; "
        "import flask_interface as fi; print(fi.app.config['JWT_SECRET_KEY'])" % str(ROOT_DIR)
    )
    s1 = subprocess.check_output([sys.executable, "-c", probe], env=env, text=True).strip()
    s2 = subprocess.check_output([sys.executable, "-c", probe], env=env, text=True).strip()
    assert s1 and s1 != LEGACY_JWT_SECRET and len(s1) >= 32
    assert s1 != s2


def test_login_rejects_legacy_creds_when_env_unset_or_different(isolated_client, monkeypatch):
    client, _ = isolated_client
    monkeypatch.delenv("IA_TRADER_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("IA_TRADER_USERS_JSON", raising=False)

    for user, pw in (("admin", "admin123"), ("trader", "trader2026")):
        resp = client.post("/login", json={"username": user, "password": pw})
        assert resp.status_code == 503
        assert "access_token" not in (resp.get_json() or {})

    monkeypatch.setenv("IA_TRADER_ADMIN_PASSWORD", "StrongAdmin!99")
    monkeypatch.setenv("IA_TRADER_USERS_JSON", '{"trader": "StrongTrader!99"}')

    for user, pw in (("admin", "admin123"), ("trader", "trader2026")):
        resp = client.post("/login", json={"username": user, "password": pw})
        assert resp.status_code == 401
        assert "access_token" not in (resp.get_json() or {})

    for user, pw in (("admin", "StrongAdmin!99"), ("trader", "StrongTrader!99")):
        resp = client.post("/login", json={"username": user, "password": pw})
        assert resp.status_code == 200
        tok = resp.get_json()["access_token"]
        prot = client.get("/protected", headers={"Authorization": f"Bearer {tok}"})
        assert prot.status_code == 200
        assert prot.get_json()["logged_in_as"] == user


def test_api_config_loopback_proxy_and_jwt_enforcement(isolated_client, monkeypatch):
    client, _ = isolated_client
    monkeypatch.setenv("IA_TRADER_ADMIN_PASSWORD", "ConfigAdmin!42")
    monkeypatch.delenv("IA_TRADER_USERS_JSON", raising=False)

    r_local = client.get("/api/config", environ_base={"REMOTE_ADDR": "127.0.0.1"})
    assert r_local.status_code == 200

    r_xff_get = client.get(
        "/api/config",
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
        headers={"X-Forwarded-For": "203.0.113.50"},
    )
    assert r_xff_get.status_code == 403

    r_xff_post = client.post(
        "/api/config",
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
        headers={"X-Forwarded-For": "203.0.113.50"},
        json={"DEFAULT_SYMBOL": "ETHUSDT"},
    )
    assert r_xff_post.status_code == 403

    tok = client.post("/login", json={"username": "admin", "password": "ConfigAdmin!42"}).get_json()["access_token"]
    r_xff_jwt = client.get(
        "/api/config",
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
        headers={"X-Forwarded-For": "203.0.113.50", "Authorization": f"Bearer {tok}"},
    )
    assert r_xff_jwt.status_code == 200


def test_post_api_config_preserves_secret_on_redacted_sentinel(isolated_client, monkeypatch):
    client, env_file = isolated_client
    env_file.write_text("OPENAI_API_KEY=sk-real-secret-9876\nDEFAULT_SYMBOL=BTCUSDT\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-secret-9876")
    monkeypatch.setenv("DEFAULT_SYMBOL", "BTCUSDT")

    resp = client.post(
        "/api/config",
        environ_base={"REMOTE_ADDR": "127.0.0.1"},
        json={"OPENAI_API_KEY": "***REDACTED***", "DEFAULT_SYMBOL": "ETHUSDT"},
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["updated_keys"] == ["DEFAULT_SYMBOL"]

    persisted = fi.read_env_dict()
    assert persisted["OPENAI_API_KEY"] == "sk-real-secret-9876"
    assert persisted["DEFAULT_SYMBOL"] == "ETHUSDT"


def test_cors_rejects_untrusted_origin_and_allows_loopback(isolated_client):
    client, _ = isolated_client
    evil = client.get("/health", headers={"Origin": "https://evil.com"})
    assert evil.status_code == 200
    assert evil.headers.get("Access-Control-Allow-Origin") is None

    local = client.get("/health", headers={"Origin": "http://127.0.0.1:5000"})
    assert local.status_code == 200
    assert local.headers.get("Access-Control-Allow-Origin") == "http://127.0.0.1:5000"
