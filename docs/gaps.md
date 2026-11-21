# Gaps, Auditoria & Roadmap — Wave P3 (`v1.0.10`)

Todos os **4 itens P2 diferidos do Patch S5 (`v1.0.8`)** foram resolvidos na **Wave P2 (`v1.0.9`)**, e na **Wave P3 (`v1.0.10`)** foi entregue o motor paralelo **`SWING_4H` (Donchian40 / ExitCh10)** junto com **Option B (Higiene & Rotação de Logs)**.

---

## 0. Entregas Wave P3 (`v1.0.10`, 2026-11-21)

1. **`SWING_4H` Paralelo (`data/swing_4h_strategy.py`, `smart_trade.py`, `scripts/run_paper_simulation.py`, `rollback_guard.py`, `flask_interface.py`, `scripts/pilot_report.py`):**
   - Motor trend-following 4h independente do comitê (`Donchian40/ExitCh10`, `|dist_sma_200| <= 20%`, `SL 2.5xATR`, trailing puro `3.5xATR`, timeout `120` barras, `max_leverage=2.0`).
   - Referência offline (`scripts/eval_swing_4h_reference.py`, `docs/pilot_backtest_reference.json`, `TRIAL_0174..0175`, custos estresse + funding 8h real):
     - **IS (`< 2025-11-20`):** `524` trades, `PF 1.396`, `exp +0.6781%`, `Sharpe 0.9736`, `MaxDD 15.58%`, `+$12,709.02`.
     - **OOS (`>= 2025-11-20`):** `99` trades, `PF 1.178`, `exp +0.1844%`, `Sharpe 0.4688`, `MaxDD 11.76%`, `+$757.03` (âncora em `pilot_backtest_reference.json`).
     - **FULL (`2020-2026`):** `624` trades, `PF 1.363`, `exp +0.6263%`, `Sharpe 0.8993`, `MaxDD 15.58%`, `+$14,907.65`.
   - Pré-registro formal adicionado em `docs/HIPOTESES_PRE_REGISTRADAS.md` (`Experimento E_SWING_4H`).
2. **Option B — Higiene & Rotação de Logs (`scripts/run_paper_simulation.py`, `scripts/system_supervisor.py`):**
   - `stale candle` em `GP` e `SCALP` rebaixado de `INFO` para `DEBUG`; rotação simples (`> 50 MiB -> .1`) adicionada em `ServiceManager.start()`.
3. **Pendência pós-Wave P3 (Meta-Labeling ML):**
   - Pesquisa de **Meta-Labeling binário (`P[win | sinal]`)** sobre gatilhos já filtrados (`SCALP Squeeze Keltner` / Comitê) segue aberta como próxima frente de pesquisa ML.

---

## 1. Itens P2 do Patch S5 — Resolvidos na Wave P2

1. **`scripts/binance_ws_collector.py` [RESOLVED]:**
   - Migrado endpoint default de `wss://fstream.binance.com/stream` (que conectava silenciosamente e dava `TimeoutError` na leitura) para `wss://fstream.binance.com/market/stream` e logger dedicado `collector.ws`.
   - `on_trade` desacoplado de `ws.recv()` via fila `asyncio.Queue(maxsize=10000)` + `_dispatch_worker` (cancelamento e drenagem determinística em `finally`), mantendo compatibilidade síncrona quando `handle_message()` é chamado diretamente sem fila.
   - Verificado ao vivo pós-restart (`15:56 UTC`): `BTCUSDT` saltou de `500–700` trades truncados (`sat=1`, 95% de saturação 24h) para **`4.345` trades com `sat=0`**, `ETHUSDT` `3.844` trades (`sat=0`), `SOLUSDT` `331` trades (`sat=0`), e `liquidations` saiu de `0` para coleta ativa (`7` eventos nos primeiros minutos).
2. **`data/live_context_enricher.py` [RESOLVED]:**
   - `_parse_yf(res, utc_today=...)` agora filtra barras intradiárias parciais (`date_str >= utc_today`).
   - `_fetch_all` diferencia `"no such table"` (`debug`) de erros SQL inesperados (`warning` + incremento de `_enrich_error_count`, exposto em `ctx["enrich_errors"]`).
   - Fallback de `binance_series` (`kind='taker'`) infere o período real entre linhas (`[60_000, 900_000]` ms) em vez de fixar 5 minutos.
3. **`agents/tape_reading_agent.py` [RESOLVED]:**
   - `refresh_covered_symbols_from_db()` registra erros inesperados em `warning` (mantendo `"no such table"` em `debug`).
   - `_fetch_binance_series_taker` infere cadência real (`LIMIT 4`, filtro `row_ts + period_ms <= now_ms`) preservando compatibilidade com registro único.
4. **`scripts/backtest_scalp_15m_calibration.py` [RESOLVED]:**
   - Adicionado `maker_fill_mode="optimistic" | "realistic"` (toque real na barra seguinte `low[i+1] <= close[i]` para `BUY` / `high[i+1] >= close[i]` para `SELL`) e rótulo explícito `[INFORMATIONAL ONLY — NOT FOR PROMOTION DECISION]` no bloco pós-`created_at`.
   - Pós-desduplicação, ambos os modos empataram em `PF = 1.22` (`n=33 < 1.25`) no winner e `PF = 1.32` (`n=12`) na base `35/1.0/0.45`, mantendo os thresholds de produção em `35.0 / 1.0 / 0.45`.

---

## 2. Resultado Honesto do Benchmark Wave P2-B (Antes `v1` × Depois `v2`)

- **Desduplicação Binance + Bitget (`scripts/audit_and_dedup_candles.py`):** removidos **4.512** candles redundantes da Bitget (`192` sobreposições exatas `ETHUSDT:4h` + `4.320` offsets UTC+8 em `1d` BTC/ETH/SOL), `0` duplicatas restantes e `PRAGMA integrity_check = ok`, preservando as 21 costuras não sobrepostas (tratadas via guard por segmento em `data/ml_features_v2.py`).
- **Comparação `v1` vs `v2` sem vazamento (`datasets/reports/p2_ml_v2_benchmark.json`, `TRIAL_0166..TRIAL_0173`):**
  - `1h` Holdout (`2025-11-20..2026-11-20`): `BEFORE_V1_1H` (`ZImp=11/10`, `N=15`, `PF=0.557`, `E[R]=-0.300`) -> `AFTER_V2_1H_H5` (`ZImp=0/0`, `N=9`, `PF=2.435`, `E[R]=+0.354`) / `AFTER_V2_1H_H18` (`ZImp=0/0`, `N=38`, `PF=1.002`, `E[R]=+0.002`), mas `PBO=0.266` e `DSR=0.000` (`N=173`) -> **`FAIL_REFUTED`**.
  - `15m` Holdout (`2026-07-07..2026-11-20`): `BEFORE_V1_15M` (`ZImp=10/10`, `N=714`, `PF=0.302`, `E[R]=-0.621`) -> `AFTER_V2_15M_H30` (`ZImp=1/0`, `N=1776`, `PF=0.528`, `E[R]=-0.424`), `PBO=0.563` -> **`FAIL_REFUTED`**.
  - Ablação informativa 66 dias (`binance_series`): `dPF=-0.0683`, `dE[R]=-0.0928` -> microestrutura curta não justifica inclusão em treino curto.
- **Decisão:** Modelos `v1` `.joblib` preservados em produção (nenhuma promoção cega). Pipeline v2, hot-reload por `mtime_ns/size` e trava `SKIPPED_V2_GUARD` ficaram operacionais e testados para futuras evoluções.

---

## 3. Gaps P2 Residuais & Roadmap Wave P3 (Meta-Labeling)

1. **`_DefaultMLEnsemble` (`agents/technical_agent.py`) & Promoção `.joblib` (`model_trainer.py`, `scripts/eval_and_train_ml_v2.py`) [RESOLVED]:**
   - Escrita atômica via `_atomic_joblib_dump` (`tempfile.mkstemp` + `os.replace`) aplicada em `ModelTrainer.train_full_model()` e `promote_suite_models()`.
   - Sanitização em `eval_and_train_ml_v2.py` unificada em `_sanitize_feature_frame()` usando explicitamente `_NEUTRAL_DEFAULTS` (ex.: `macro_vix_level = 18.0` em vez de `0.0`); manter conjuntos `lgbm+rf` homogêneos na promoção.
2. **Wave P3 — Pivotar ML para Meta-Labeling Condicionado (López de Prado):**
   - Classificador direcional autônomo barra-a-barra em `1h`/`15m` mostrou ser insuficiente após custos (`E[R] <= 0.002` / `DSR=0.0`). A próxima pesquisa ML deve treinar **Meta-Labeling binário (`P[win | sinal]`)** exclusivamente sobre eventos já filtrados por `SCALP Squeeze Keltner` ou pelo Comitê Multi-Agente, em vez de prever todas as barras do mercado.
