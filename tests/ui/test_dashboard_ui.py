import json
import os

import pytest

from conftest import SHOT_DIR, VIEWPORTS, SEED_ENV

os.makedirs(SHOT_DIR, exist_ok=True)


def plot_data(p):
    return p.page.evaluate("""() => {
        const el = document.getElementById('chartContainer');
        return (el.data || []).map(t => ({
            type: t.type, name: t.name,
            n: (t.x || []).length,
            bad: [t.y, t.open, t.high, t.low, t.close].filter(Boolean)
                  .reduce((a, arr) => a + arr.filter(v => typeof v === 'number' && !Number.isFinite(v)).length, 0),
            numeric: [t.y, t.close].filter(Boolean).reduce((a, arr) => a + arr.filter(v => typeof v === 'number' && Number.isFinite(v)).length, 0)
        }));
    }""")


def wait_chart_reload(p, expect_fn="true"):
    p.page.wait_for_timeout(1200)
    p.page.wait_for_selector("#chartLoader.hidden", state="attached", timeout=10000)


# ----------------------------------------------------------------- saúde
def test_console_and_network_clean(make_page):
    p = make_page()
    p.page.wait_for_timeout(2500)
    assert p.page_errors == [], p.page_errors
    assert p.console_errors == [], p.console_errors
    assert p.failed == [], p.failed


def test_plotly_traces_valid(make_page):
    p = make_page()
    assert p.page.locator(".js-plotly-plot").count() >= 1
    traces = plot_data(p)
    by_type = {t["type"]: t for t in traces}
    assert "candlestick" in by_type and by_type["candlestick"]["n"] == 120
    assert "bar" in by_type and by_type["bar"]["n"] == 120
    names = [t["name"] for t in traces]
    assert "RSI (14)" in names and "Volume" in names
    for t in traces:
        assert t["bad"] == 0, f"NaN/Inf em {t['name']}"
    rsi = next(t for t in traces if t["name"] == "RSI (14)")
    assert rsi["numeric"] > 50
    vals = p.page.evaluate("document.getElementById('chartContainer').data.find(t=>t.name==='RSI (14)').y.filter(v=>v!==null)")
    assert all(0 <= v <= 100 for v in vals)
    assert p.page.locator("#tickerLastPrice").inner_text() not in ("--", "NaN", "0.00")


def test_no_nan_text_anywhere(make_page):
    p = make_page()
    body = p.page.inner_text("body")
    assert "NaN" not in body and "undefined" not in body and "null" not in body.lower().replace("nullable", "")


def test_ticker_and_cards_populated(make_page):
    p = make_page()
    p.page.wait_for_function("document.getElementById('metricTotalCandles').innerText !== '--'")
    assert p.page.locator("#metricTotalCandles").inner_text() == "800"
    assert p.page.locator("#agentsGrid > div").count() == 6
    assert p.page.locator("#connectorsList > div").count() in (6, 7)
    assert p.page.locator("#cacheTableBody tr").count() == 4


# ----------------------------------------------------------------- interação
def test_symbol_and_timeframe_switch(make_page):
    p = make_page()
    p.page.click("#btnSym_ETHUSDT")
    p.page.wait_for_function("document.getElementById('chartContainer').data && document.getElementById('chartContainer').data[0].name==='ETHUSDT'")
    assert "bg-cyan-600" in p.page.get_attribute("#btnSym_ETHUSDT", "class")
    assert "bg-cyan-600" not in p.page.get_attribute("#btnSym_BTCUSDT", "class")
    price = float(p.page.inner_text("#tickerLastPrice").replace(",", ""))
    assert 2000 < price < 5000
    p.page.click("#btnTf_4h")  # ETH 4h inexistente -> estado vazio
    p.page.wait_for_function("document.getElementById('chartCandleCount').innerText.startsWith('0 ')")
    p.page.click("#btnSym_BTCUSDT")
    p.page.wait_for_function("document.getElementById('chartContainer').data && document.getElementById('chartContainer').data[0] && document.getElementById('chartContainer').data[0].name==='BTCUSDT'")
    assert "bg-cyan-600" in p.page.get_attribute("#btnTf_4h", "class")
    assert float(p.page.inner_text("#tickerLastPrice").replace(",", "")) > 30000
    p.page.click("#btnTf_15m")
    p.page.wait_for_timeout(800)
    assert plot_data(p)[0]["n"] == 120
    assert p.console_errors == [] and p.page_errors == []


@pytest.mark.parametrize("btn,trace_names", [
    ("ema", ["EMA 9", "EMA 21"]),
    ("bb", ["BB Upper", "BB Lower"]),
])
def test_overlay_toggles_on_off(make_page, btn, trace_names):
    p = make_page()

    def names():
        return [t["name"] for t in plot_data(p)]
    assert not set(trace_names) & set(names())
    p.page.click(f"#btnInd_{btn}")
    p.page.wait_for_function(f"document.getElementById('chartContainer').data.some(t=>t.name==='{trace_names[0]}')")
    assert set(trace_names) <= set(names())
    assert all(t["bad"] == 0 for t in plot_data(p))
    p.page.click(f"#btnInd_{btn}")
    p.page.wait_for_function(f"!document.getElementById('chartContainer').data.some(t=>t.name==='{trace_names[0]}')")


def test_sma_and_rsi_toggle(make_page):
    p = make_page()
    p.page.click("#btnInd_rsi")
    p.page.wait_for_function("!document.getElementById('chartContainer').data.some(t=>t.name==='RSI (14)')")
    assert plot_data(p)[0]["type"] == "candlestick"
    p.page.click("#btnInd_sma")
    p.page.wait_for_function("!document.getElementById('chartContainer').data.some(t=>t.name==='SMA 20')")
    p.page.click("#btnInd_rsi")
    p.page.click("#btnInd_sma")
    p.page.wait_for_function("document.getElementById('chartContainer').data.some(t=>t.name==='RSI (14)') && document.getElementById('chartContainer').data.some(t=>t.name==='SMA 20')")
    assert p.console_errors == [] and p.page_errors == []


def test_auto_refresh_toggle(make_page):
    p = make_page()
    p.page.click("#btnAutoRefreshToggle")
    assert p.page.inner_text("#autoRefreshText") == "Pausado"
    p.page.click("#btnAutoRefreshToggle")
    assert "Auto" in p.page.inner_text("#autoRefreshText")


def test_manual_refresh_toast(make_page):
    p = make_page()
    p.page.click("#btnManualRefresh")
    p.page.wait_for_selector("#toastContainer >> text=atualizados", timeout=5000)


def test_auto_refresh_does_not_reset_zoom_or_flicker(make_page):
    """Auto-refresh a cada 5s recria o gráfico com Plotly.newPlot -> zoom do usuário é perdido."""
    p = make_page()
    p.page.evaluate("Plotly.relayout('chartContainer', {'xaxis.range': [10, 40]})")
    p.page.wait_for_timeout(6500)
    rng = p.page.evaluate("document.getElementById('chartContainer').layout.xaxis.range")
    assert rng and rng[0] == 10, f"zoom resetado pelo auto-refresh: {rng}"


# ----------------------------------------------------------------- modal
def open_modal(p):
    # espera a resposta de /api/config: evita corrida em que loadSettings sobrescreve campos preenchidos
    with p.page.expect_response(lambda r: "/api/config" in r.url and r.request.method == "GET"):
        p.page.click("text=Configurações & Chaves")
    p.page.wait_for_selector("#settingsModal.flex")
    p.page.wait_for_function("document.getElementById('hintBitgetKey').innerText.includes('Configurado')")
    p.page.wait_for_timeout(150)


def test_settings_modal_open_close_mask_save(make_page, server):
    p = make_page()
    open_modal(p)
    # mascaramento
    assert p.page.get_attribute("#inputBitgetKey", "type") == "password"
    val = p.page.input_value("#inputBitgetKey")
    assert "SECRETKEY" not in val and val.endswith("wxyz") and val.startswith("•")
    cfg = p.page.evaluate("fetch('/api/config').then(r=>r.text())")
    assert "abcd1234SECRETKEY" not in cfg and "secretvalue9876" not in cfg
    # Cancelar fecha
    p.page.click("text=Cancelar")
    p.page.wait_for_selector("#settingsModal.hidden", state="attached")
    # X fecha
    open_modal(p)
    p.page.click("#settingsModal button:has([data-lucide='x'], svg.lucide-x)")
    p.page.wait_for_selector("#settingsModal.hidden", state="attached")
    # salvar: chave nova + mantém mascarada
    open_modal(p)
    p.page.fill("#inputFredKey", "fredkey-TEST-5555")
    p.page.select_option("#selectDefaultTimeframe", "4h")
    p.page.click("#btnSaveSettings")
    p.page.wait_for_selector("#toastContainer >> text=sucesso", timeout=5000)
    p.page.wait_for_selector("#settingsModal.hidden", state="attached")
    env = open(server["env_path"], encoding="utf-8").read()
    assert "FRED_API_KEY=fredkey-TEST-5555" in env
    assert "DEFAULT_TIMEFRAME=4h" in env
    assert "BITGET_API_KEY=abcd1234SECRETKEYwxyz" in env, "chave mascarada não pode sobrescrever a real"
    assert "# comentario" in env
    assert "•" not in env
    # .env real do usuário intocado
    real = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    assert "fredkey-TEST-5555" not in open(real, encoding="utf-8").read()


def test_settings_modal_closes_with_escape(make_page):
    p = make_page()
    open_modal(p)
    p.page.keyboard.press("Escape")
    p.page.wait_for_timeout(300)
    assert "hidden" in p.page.get_attribute("#settingsModal", "class"), "Modal não fecha com ESC (acessibilidade)"


def test_settings_modal_closes_on_backdrop(make_page):
    p = make_page()
    open_modal(p)
    p.page.mouse.click(5, 5)
    p.page.wait_for_timeout(300)
    assert "hidden" in p.page.get_attribute("#settingsModal", "class"), "Modal não fecha clicando no backdrop"


def test_modal_fits_viewport_mobile(make_page):
    p = make_page(viewport=(390, 844))
    open_modal(p)
    box = p.page.evaluate("""() => { const r = document.querySelector('#settingsModal > div').getBoundingClientRect();
        return {l: r.left, r: r.right, t: r.top, b: r.bottom, w: innerWidth, h: innerHeight}; }""")
    assert box["l"] >= 0 and box["r"] <= box["w"] and box["t"] >= 0 and box["b"] <= box["h"]
    p.page.screenshot(path=os.path.join(SHOT_DIR, "modal_mobile_390x844.png"))


def test_limits_modal_lot_field_validation_and_mobile(make_page):
    p = make_page(viewport=(390, 844))
    p.page.evaluate("openLimitsModal()")
    p.page.wait_for_selector("#inputPaperLotUsd", state="visible")
    assert "máximo 5x" in p.page.inner_text("#leverageReadOnlyInfo")
    assert "A11" in p.page.inner_text("#pilotChangeWarning")
    assert p.page.locator("input[id*=everage]").count() == 0
    p.page.fill("#inputPaperLotUsd", "5")
    p.page.evaluate("document.getElementById('limitsForm').requestSubmit()")
    p.page.wait_for_timeout(300)
    assert "entre US$ 10 e US$ 1000" in p.page.inner_text("#errPaperLotUsd")
    over = p.page.evaluate("document.documentElement.scrollWidth > innerWidth")
    assert not over, "overflow horizontal com modal de limites aberto"
    box = p.page.evaluate("""() => { const r = document.querySelector('#limitsModal > div').getBoundingClientRect();
        return {l: r.left, r: r.right, w: innerWidth}; }""")
    assert box["l"] >= 0 and box["r"] <= box["w"]


def test_fmt_lev_margin_handles_missing(make_page):
    p = make_page()
    out = p.page.evaluate("[fmtLevMargin({}), fmtLevMargin({leverage: 3, margin: 100}), fmtLevMargin(null)]")
    assert out == ["— / —", "3x / $100.00", "— / —"]


# ----------------------------------------------------------------- estados vazios / erro
def test_empty_state_symbol_without_candles(make_page):
    p = make_page()
    p.page.click("#btnSym_SOLUSDT")
    p.page.wait_for_function("document.getElementById('chartCandleCount').innerText.startsWith('0 ')")
    p.page.wait_for_timeout(500)
    title = p.page.text_content("#chartContainer .gtitle")
    assert "Sem velas" in title
    txt = p.page.inner_text("#tickerHigh") + p.page.inner_text("#tickerLow") + p.page.inner_text("#tickerLastPrice")
    assert "NaN" not in txt, f"Ticker mostra NaN no estado vazio: {txt!r}"
    p.page.screenshot(path=os.path.join(SHOT_DIR, "empty_state_sol.png"))
    assert p.page_errors == []


def test_api_500_handled_gracefully(make_page):
    def boom(route):
        route.fulfill(status=500, content_type="application/json", body=json.dumps({"error": "Erro de banco: x"}))
    p = make_page(wait_chart=False, routes=[("**/api/chart/**", boom)])
    p.page.wait_for_timeout(2500)
    body = p.page.inner_text("body")
    assert "undefined" not in body and "NaN" not in body, "UI mostra 'undefined'/'NaN' com API 500"
    # loader não pode ficar eterno
    assert "hidden" in p.page.get_attribute("#chartLoader", "class")
    # usuário deve ser informado do erro (toast/mensagem)
    assert any(w in body.lower() for w in ("erro", "falha", "indispon")), "Nenhuma mensagem de erro exibida ao usuário quando a API falha"


def test_api_offline_handled_gracefully(make_page):
    p = make_page(wait_chart=False, routes=[("**/api/**", lambda r: r.abort())])
    p.page.wait_for_timeout(2500)
    assert p.page_errors == [], p.page_errors
    assert "hidden" in p.page.get_attribute("#chartLoader", "class")
    body = p.page.inner_text("body")
    assert "undefined" not in body and "NaN" not in body
    assert any(w in body.lower() for w in ("erro", "falha", "indispon", "offline")), "Sem feedback visual de API fora do ar"
    p.page.screenshot(path=os.path.join(SHOT_DIR, "api_offline.png"), full_page=True)


def test_chart_api_rejects_bad_limit(server):
    import urllib.error
    import urllib.request
    try:
        urllib.request.urlopen(server["url"] + "/api/chart/BTCUSDT/1h?limit=abc")
        code = 200
    except urllib.error.HTTPError as e:
        code = e.code
    assert code == 400, f"limit inválido deveria retornar 400, retornou {code}"


def test_phase_panel_not_hardcoded_in_progress(make_page):
    """Fase 2 já foi homologada; deve constar como Homologado na tabela de fases."""
    p = make_page()
    if p.page.locator("#tabNavProject").count() > 0:
        p.page.locator("#tabNavProject").click()
    p.page.wait_for_function("document.getElementById('phasesTableBody') && document.getElementById('phasesTableBody').innerText.includes('Feature Engineering')")
    body = p.page.inner_text("body")
    assert "Feature Engineering" in body
    assert "Homologado" in body


def test_footer_port_matches_actual(make_page):
    p = make_page()
    footer = p.page.inner_text("footer")
    assert "5099" in footer or "5000" not in footer, "Rodapé exibe porta 5000 fixa (hardcoded) em vez da porta real"


# ----------------------------------------------------------------- responsividade
def overflow_info(page):
    return page.evaluate("""() => {
        const de = document.documentElement;
        const cut = [];
        document.querySelectorAll('main *, header *, footer *').forEach(e => {
            const r = e.getBoundingClientRect();
            if (r.width === 0 || r.height === 0) return;
            const cs = getComputedStyle(e);
            if (cs.display === 'none' || cs.visibility === 'hidden') return;
            if (r.right > innerWidth + 1 || r.left < -1) cut.push(e.tagName + '#' + e.id + '.' + String(e.className).slice(0, 50) + ' L' + Math.round(r.left) + ' R' + Math.round(r.right));
        });
        const btns = [...document.querySelectorAll('header button, main button')].filter(b => b.offsetWidth > 0);
        const overlaps = [];
        for (let i = 0; i < btns.length; i++) for (let j = i + 1; j < btns.length; j++) {
            const a = btns[i].getBoundingClientRect(), b = btns[j].getBoundingClientRect();
            const w = Math.min(a.right, b.right) - Math.max(a.left, b.left);
            const h = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
            if (w > 2 && h > 2) overlaps.push((btns[i].id || btns[i].innerText.trim()) + ' x ' + (btns[j].id || btns[j].innerText.trim()));
        }
        const textClipped = [...document.querySelectorAll('main span, main div, header span')].filter(e =>
            e.children.length === 0 && e.innerText && e.scrollWidth > e.clientWidth + 2 &&
            !['auto', 'scroll'].includes(getComputedStyle(e).overflowX) && getComputedStyle(e).textOverflow !== 'ellipsis' && getComputedStyle(e).overflowX === 'hidden'
        ).map(e => e.innerText.slice(0, 40));
        return {sw: de.scrollWidth, cw: de.clientWidth, cut: cut.slice(0, 10), overlaps, textClipped};
    }""")


@pytest.mark.parametrize("name", list(VIEWPORTS))
def test_responsive_viewport(make_page, name):
    w, h = VIEWPORTS[name]
    p = make_page(viewport=(w, h))
    p.page.wait_for_function("document.getElementById('agentsGrid').children.length > 0")
    p.page.wait_for_timeout(800)
    p.page.screenshot(path=os.path.join(SHOT_DIR, f"{name}.png"), full_page=True)
    p.page.screenshot(path=os.path.join(SHOT_DIR, f"{name}_fold.png"))
    info = overflow_info(p.page)
    assert info["sw"] <= info["cw"], f"overflow horizontal {info}"
    assert info["cut"] == [], f"elementos cortados fora do viewport: {info['cut']}"
    assert info["overlaps"] == [], f"botões sobrepostos: {info['overlaps']}"
    assert info["textClipped"] == [], f"texto cortado: {info['textClipped']}"
    # gráfico ocupa a largura do container e não vaza
    cw = p.page.evaluate("(() => { const c = document.getElementById('chartContainer').getBoundingClientRect(); const s = document.querySelector('#chartContainer .main-svg').getBoundingClientRect(); return [c.width, s.width]; })()")
    assert cw[1] <= cw[0] + 1 and cw[1] > cw[0] * 0.9, cw
    assert p.console_errors == [] and p.page_errors == []


# ----------------------------------------------------------------- posições abertas / histórico
def _pos_routes(state):
    import time as _t
    now = int(_t.time() * 1000)

    def open_handler(route):
        route.fulfill(status=200, content_type="application/json", body=json.dumps({
            "positions": state["open"], "total": len(state["open"]), "initial_capital": 10000.0,
            "summary": {"count": len(state["open"]), "total_pnl": 12.5, "total_pnl_pct_initial": 0.125,
                        "wins": 1, "losses": 0, "win_rate_pct": 100.0}}))

    def closed_handler(route):
        route.fulfill(status=200, content_type="application/json", body=json.dumps({
            "positions": state["closed"], "total": len(state["closed"]), "initial_capital": 10000.0,
            "summary": {"count": len(state["closed"]), "total_pnl": -3.0, "total_pnl_pct_initial": -0.03,
                        "wins": 0, "losses": 1, "win_rate_pct": 0.0}}))
    return [("**/api/paper-trading/positions/open", open_handler),
            ("**/api/paper-trading/positions/closed", closed_handler)]


def _state():
    import time as _t
    now = int(_t.time() * 1000)
    base = dict(status="OPEN", entry_timestamp=now - 600_000, opened_at_ms=now - 600_000, open_for_s=600)
    return {
        "open": [
            dict(base, position_id="P1", symbol="BTCUSDT", side="LONG", strategy_type="SCALP", entry_price=65000.0,
                 current_price=65100.0, quantity=0.01, notional=650.0, stop_loss=64500.0, take_profit=66000.0,
                 tp1_price=65500.0, tp2_price=66000.0, unrealized_pnl=1.0, leverage=3, margin=216.67,
                 leverage_effective=3, margin_effective=216.67, roi_pct=0.46, price_change_pct=0.15),
            # sem leverage/margin: não pode quebrar
            dict(base, position_id="P2", symbol="ETHUSDT", side="SHORT", strategy_type="SWING", entry_price=3200.0,
                 current_price=3190.0, quantity=0.1, notional=320.0, stop_loss=3300.0, take_profit=3000.0,
                 tp1_price=3100.0, tp2_price=3000.0, unrealized_pnl=1.0),
        ],
        "closed": [dict(status="CLOSED", position_id="C1", symbol="BTCUSDT", side="LONG", strategy_type="SCALP",
                        entry_price=65000.0, exit_price=64900.0, quantity=0.01, notional=650.0, realized_pnl=-3.0,
                        opened_at_ms=now_ms_ago(7200), closed_at_ms=now_ms_ago(3600), duration_s=3600,
                        net_pnl=-3.0, net_pnl_pct_margin=-1.4, price_change_pct=-0.15,
                        leverage_effective=3, margin_effective=216.67)],
    }


def now_ms_ago(sec):
    import time as _t
    return int((_t.time() - sec) * 1000)


def test_open_list_only_open_with_datetime_and_percent(make_page):
    st = _state()
    p = make_page(routes=_pos_routes(st))
    p.page.wait_for_selector("#paperPositionsTableBody tr[data-position-id]")
    rows = p.page.locator("#paperPositionsTableBody tr[data-position-id]")
    assert rows.count() == 2
    txt = p.page.inner_text("#paperPositionsTableBody")
    import re
    assert re.search(r"\d{2}/\d{2} \d{2}:\d{2}:\d{2}", txt), "falta data/hora dd/mm HH:MM:SS"
    assert "ROI" in txt and "%" in txt and "3x" in txt
    assert "undefined" not in txt and "NaN" not in txt
    assert "Abertas" in p.page.inner_text("#positionsSummary") and "Win rate" in p.page.inner_text("#positionsSummary")
    assert p.page_errors == [], p.page_errors


def test_closed_history_has_close_time_duration_and_summary(make_page):
    st = _state()
    p = make_page(routes=_pos_routes(st))
    p.page.evaluate("switchPaperTab('trades')")
    p.page.wait_for_selector("#paperTradesTableBody tr[data-symbol]")
    txt = p.page.inner_text("#paperTradesTableBody")
    assert "FECHAMENTO" in txt.upper() and "DURAÇÃO" in txt.upper() and "1h 00m" in txt and "-1.40%" in txt
    assert "Fechadas" in p.page.inner_text("#tradesSummary")
    assert "Win rate" in p.page.inner_text("#tradesSummary")


def test_click_open_position_projects_live_lines_and_clears_on_close(make_page):
    st = _state()
    p = make_page(routes=_pos_routes(st))
    p.page.wait_for_selector("#paperPositionsTableBody tr[data-position-id='P1']")
    p.page.click("#paperPositionsTableBody tr[data-position-id='P1']")
    p.page.wait_for_function("document.getElementById('chartContainer').layout && (document.getElementById('chartContainer').layout.annotations||[]).length >= 4", timeout=15000)
    info = p.page.evaluate("""() => ({tf: currentTimeframe, sym: currentSymbol,
        ann: document.getElementById('chartContainer').layout.annotations.map(a => a.text),
        shapes: document.getElementById('chartContainer').layout.shapes.length,
        markers: document.getElementById('chartContainer').data.filter(t => t.mode === 'markers').length,
        banner: !document.getElementById('chartOrderOverlayBanner').classList.contains('hidden')})""")
    assert info["tf"] == "15m" and info["sym"] == "BTCUSDT"
    joined = " ".join(info["ann"])
    for k in ("ENTRY LONG", "SL:", "TP1:", "TP2:"):
        assert k in joined, joined
    assert info["shapes"] >= 4 and info["banner"] and info["markers"] == 1
    assert "ring-cyan" in p.page.get_attribute("#paperPositionsTableBody tr[data-position-id='P1']", "class")
    # posição fecha -> linhas removidas + aviso
    st["open"] = [x for x in st["open"] if x["position_id"] != "P1"]
    p.page.evaluate("loadPaperTradingData()")
    p.page.wait_for_function("(document.getElementById('chartContainer').layout.shapes||[]).length === 0", timeout=10000)
    assert "encerrada" in p.page.inner_text("#toastContainer")
    assert "hidden" in p.page.get_attribute("#chartOrderOverlayBanner", "class")
    assert p.page_errors == [], p.page_errors


def test_swing_position_switches_to_1h_and_clear_selection(make_page):
    st = _state()
    p = make_page(routes=_pos_routes(st))
    p.page.wait_for_selector("#paperPositionsTableBody tr[data-position-id='P2']")
    p.page.evaluate("setTimeframe('15m')")
    p.page.wait_for_timeout(800)
    p.page.click("#paperPositionsTableBody tr[data-position-id='P2']")
    p.page.wait_for_function("currentTimeframe === '1h' && currentSymbol === 'ETHUSDT'", timeout=10000)
    p.page.wait_for_function("(document.getElementById('chartContainer').layout.annotations||[]).length >= 3", timeout=15000)
    p.page.click("#chartOrderOverlayBanner button")
    p.page.wait_for_function("(document.getElementById('chartContainer').layout.shapes||[]).length === 0", timeout=10000)
    assert "ring-cyan" not in p.page.get_attribute("#paperPositionsTableBody tr[data-position-id='P2']", "class")


@pytest.mark.parametrize("name", ["mobile_small", "mobile"] if False else list(VIEWPORTS))
def test_position_cards_no_horizontal_overflow(make_page, name):
    w, h = VIEWPORTS[name]
    p = make_page(viewport=(w, h), routes=_pos_routes(_state()))
    p.page.wait_for_selector("#paperPositionsTableBody tr[data-position-id]")
    over = p.page.evaluate("document.documentElement.scrollWidth > innerWidth")
    assert not over, f"overflow horizontal em {name}"
    p.page.evaluate("switchPaperTab('trades')")
    p.page.wait_for_selector("#paperTradesTableBody tr[data-symbol]")
    assert not p.page.evaluate("document.documentElement.scrollWidth > innerWidth")


def _ladder_state():
    st = _state()
    st["open"][0].update(tp3_price=67000.0, tp1=65500.0, tp2=66000.0, tp3=67000.0, tp1_hit=True, tp2_hit=False,
                         tp1_fraction=0.4, tp2_fraction=0.3, runner_fraction=0.3, remaining_fraction=0.6,
                         trailing_active=True, trailing_stop_price=65300.0, exit_ladder=True, stop_loss=65000.0)
    st["closed"][0].update(close_action="TRAILING_STOP", close_action_label="Trailing stop", tp1_fraction=0.4,
                           tp2_fraction=0.3, runner_fraction=0.3,
                           events=[{"action": "TP1_PARTIAL", "label": "TP1 parcial"},
                                   {"action": "TAKE_PROFIT_3", "label": "TP3 (runner encerrado)"}])
    return st


def test_ladder_chart_lines_and_cards(make_page):
    p = make_page(routes=_pos_routes(_ladder_state()))
    p.page.wait_for_selector("#paperPositionsTableBody tr[data-position-id='P1']")
    txt = p.page.inner_text("#paperPositionsTableBody tr[data-position-id='P1']")
    assert "TP1 40% / TP2 30% / runner 30%" in txt and "60%" in txt and "ATIVO" in txt and "(BE)" in txt
    assert "TP1 50%" not in p.page.inner_text("#paperPositionsTableBody")
    # legado (P2): '—' sem erro
    t2 = p.page.inner_text("#paperPositionsTableBody tr[data-position-id='P2']")
    assert "—" in t2 and "undefined" not in t2 and "NaN" not in t2
    p.page.click("#paperPositionsTableBody tr[data-position-id='P1']")
    p.page.wait_for_function("(document.getElementById('chartContainer').layout.annotations||[]).length >= 6", timeout=15000)
    ann = " ".join(p.page.evaluate("document.getElementById('chartContainer').layout.annotations.map(a => a.text)"))
    for k in ("TP3:", "TRAILING STOP (ativo)", "SL→BREAKEVEN", "✔ TP1 atingido", "TP2:"):
        assert k in ann, ann
    assert p.page_errors == [] and p.console_errors == []


def test_ledger_friendly_labels_in_history(make_page):
    p = make_page(routes=_pos_routes(_ladder_state()))
    p.page.evaluate("switchPaperTab('trades')")
    p.page.wait_for_selector("#paperTradesTableBody tr[data-symbol]")
    txt = p.page.inner_text("#paperTradesTableBody")
    assert "Trailing stop" in txt and "TP1 parcial" in txt and "TP3 (runner encerrado)" in txt
    assert p.page.evaluate("ledgerLabel('TP2_PARTIAL')").startswith("TP2 parcial")
    assert p.page.evaluate("ledgerLabel('LIQUIDATION')") == "Liquidação"
    assert p.page_errors == []
