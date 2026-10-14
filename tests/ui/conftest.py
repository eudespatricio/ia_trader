import math
import os
import random
import sqlite3
import subprocess
import sys
import time
import urllib.request

import pytest

# Chromium precisa de libnspr4/libnss3/libasound; sem sudo, usamos libs extraídas em ~/.cache/pwlibs
# (apt-get download libnspr4 libnss3 libasound2t64 && dpkg-deb -x <deb> ~/.cache/pwlibs)
_LOCAL_LIBS = os.path.expanduser("~/.cache/pwlibs/usr/lib/x86_64-linux-gnu")
if os.path.isdir(_LOCAL_LIBS):
    os.environ["LD_LIBRARY_PATH"] = _LOCAL_LIBS + ":" + os.environ.get("LD_LIBRARY_PATH", "")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SHOT_DIR = os.path.join(HERE, "screenshots")
PORT = 5099
BASE_URL = f"http://127.0.0.1:{PORT}"

VIEWPORTS = {
    "desktop_1920x1080": (1920, 1080),
    "laptop_1366x768": (1366, 768),
    "tablet_768x1024": (768, 1024),
    "mobile_390x844": (390, 844),
}

SEED_ENV = "# comentario\nBITGET_API_KEY=abcd1234SECRETKEYwxyz\nBITGET_API_SECRET=secretvalue9876\nDEFAULT_SYMBOL=BTCUSDT\n"


def _seed(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE candles (symbol TEXT NOT NULL, timeframe TEXT NOT NULL, timestamp INTEGER NOT NULL,
        open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
        quote_volume REAL NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (symbol, timeframe, timestamp))""")
    step = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}
    base = {"BTCUSDT": 65000.0, "ETHUSDT": 3200.0}
    rnd = random.Random(42)
    now = int(time.time() // 3600 * 3600 * 1000)
    rows = []
    for sym, tf in [("BTCUSDT", "1h"), ("BTCUSDT", "15m"), ("BTCUSDT", "4h"), ("ETHUSDT", "1h")]:
        price = base[sym]
        for i in range(200):
            ts = now - (199 - i) * step[tf]
            o = price
            c = o * (1 + rnd.gauss(0, 0.004) + 0.0004 * math.sin(i / 9))
            h = max(o, c) * (1 + abs(rnd.gauss(0, 0.0015)))
            l = min(o, c) * (1 - abs(rnd.gauss(0, 0.0015)))
            v = rnd.uniform(50, 500)
            rows.append((sym, tf, ts, o, h, l, c, v, v * c))
            price = c
    conn.executemany("INSERT INTO candles (symbol,timeframe,timestamp,open,high,low,close,volume,quote_volume) VALUES (?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("uitest")
    db = str(tmp / "market_cache.db")
    envf = str(tmp / ".env")
    _seed(db)
    with open(envf, "w", encoding="utf-8") as f:
        f.write(SEED_ENV)
    env = dict(os.environ, UI_TEST_DB=db, UI_TEST_ENV=envf, UI_TEST_PORT=str(PORT))
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "run_test_server.py")], env=env, cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            urllib.request.urlopen(BASE_URL + "/health", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError("servidor de teste não subiu")
    yield {"url": BASE_URL, "env_path": envf, "db": db}
    proc.terminate()
    try:
        proc.wait(5)
    except Exception:
        proc.kill()


@pytest.fixture(scope="session")
def browser_(playwright):
    b = playwright.chromium.launch(headless=True)
    yield b
    b.close()


class Page:
    """Contexto de página com coleta de console/requests."""

    def __init__(self, page):
        self.page = page
        self.console_errors = []
        self.failed = []
        self.page_errors = []
        page.on("console", lambda m: self.console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: self.page_errors.append(str(e)))
        page.on("response", lambda r: self.failed.append(f"{r.status} {r.url}") if r.status >= 400 else None)
        page.on("requestfailed", lambda r: self.failed.append(f"FAILED {r.url} {r.failure}"))


@pytest.fixture
def make_page(browser_, server):
    ctxs = []

    def _make(viewport=(1366, 768), url_path="/", wait_chart=True, routes=None):
        ctx = browser_.new_context(viewport={"width": viewport[0], "height": viewport[1]})
        ctxs.append(ctx)
        pg = ctx.new_page()
        wrapper = Page(pg)
        for pattern, handler in (routes or []):
            pg.route(pattern, handler)
        pg.goto(server["url"] + url_path, wait_until="load")
        if wait_chart:
            pg.wait_for_selector(".js-plotly-plot", timeout=20000)
            pg.wait_for_function("document.querySelectorAll('#chartContainer .trace').length > 0", timeout=20000)
        return wrapper

    yield _make
    for c in ctxs:
        c.close()
