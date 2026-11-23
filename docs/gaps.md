# Gaps, Auditoria & Roadmap — Patch S6 (Onda 1 `v1.0.12`) & S5 (`v1.0.11`)

Patch S6 Onda 1 (`v1.0.12`, `2026-11-23`) resolve o vazamento de perps TradFi no `GOLDEN_POT` (`S6-2`) e encerra empiricamente em OOS a hipótese de relaxamento do comitê/`RSI>75` no `SCALP 15m` e `DAY_TRADE 1h` (`S6-1`), preservando produção protegida. Detalhes completos em [PLANO_EVOLUTIVO_S6.md](file:///home/eudespatricio/ia_trader/docs/PLANO_EVOLUTIVO_S6.md).

---

## 0-s6. Auditoria Pós-Pump (`2026-11-23 11:55 BRT`) — Patch S6 Onda 1 (`v1.0.12`)

1. **`S6-2` Filtro Dinâmico TradFi no `GOLDEN_POT` (`agents/golden_pot_agent.py`, `scripts/run_paper_simulation.py`) [RESOLVED]:**
   - `GP_EXCLUDED_BASE_ASSETS` (`42` estáticos) deixava vazar `28+` perps TradFi (`MUUSDT`, `MUUUSDT`, `ZHONGJIUSDT`, `ZHIPUUSDT`, `NATGASUSDT`, `DRAMUSDT`, `NBISUSDT`, `EWYUSDT`, etc.) dos `253` listados em `/fapi/v1/exchangeInfo`.
   - Criada fonte única `NON_CRYPTO_BASE_EXCLUSIONS` (`85` bases) + cache dinâmico via `/fapi/v1/exchangeInfo` (TTL `6h`, backoff `5m` em erro) filtrando Binance e fallback Bitget sem bloquear tokens cripto/RWA `COIN` (`PAXG`, `XAUT`, `MANTRA`, `CFG`).
2. **`S6-1` Conflito de Camadas `SCALP 15m` / `RSI > 75` (`15m` / `1h`) [NO-GO EMPÍRICO]:**
   - `SCALP 15m` (`N=261` Full Squeeze / `98k` barras OOS): Produção (`6-ag conf>=0.70, RSI<=75`) foi superior em IS e OOS (`OOS PF 1.56, +10.30R`) vs remover `Macro+OnChain` (`PF 1.26, +4.52R`), remover `RSI>75` (`PF 1.06, +1.32R`) ou usar `RiskOnly` (`PF 1.21, +6.43R`). Entradas em `RSI > 75 / < 25` no `15m` tiveram `PF 0.59 (-2.86R)` no Full Squeeze OOS e `PF 0.31 (-3.24R)` pós-`ML15m`.
   - `DAY_TRADE 1h`: Trades isolados em `RSI > 75 / < 25` perderam `-238.0R` (IS) e **`-395.6R` (`PF 0.730`, `n=2.143`)** em OOS `2025–2026`. Filtros originais preservados intactos.

---

## 0. Entregas Patch Final S5 (`v1.0.11`, 2026-11-23)

1. **`S5-1` Living Sub-Rule Learner (`agents/technical_agent.py`, `learning_tracker.py`, `signal_generator.py`, `core.py`):**
   - `enable_sweep_trigger=False` por padrão (`Gatilho Estrutural` isolado: `-333.3R` OOS, `PF 0.66`); deprecado `structure_context` (`0.7/0.85x` SL); calibração `Beta(4,4)` com janela `W=20` e auto-silenciamento em `structural_sub_rule_calibration` (`n>=5, post_wr<0.44, pnl<0`; replay real: `+$821.24 -> +$1,317.32`).
   - Preservação de precedência explícita de `context["sub_rule_multipliers"]` em `core.py` para replay causal.
2. **`S5-2` + `S5-3` ML `1h` Multi-Symbol & Calibração (`model_trainer.py`, `learning_tracker.py`, `agents/technical_agent.py`):**
   - Treino `BTC+ETH+SOL` com Triple-Barrier isolado por ativo e embargo por timestamp; `NaN` nativo preservado apenas para `dist_resistance_pct`/`dist_support_pct` no LightGBM; adição de `ExtraTrees`, regularização rasa LGBM (`max_depth=4, num_leaves=15`), pós-calibração `Platt`/`Isotonic` (`ECE` reduzido ~15×–17× OOS), guarda de classes na calibração e gate conjunto `log_loss` + `roc_auc_ovr`.
3. **`S5-4` Reforma `SentimentAgent` + Bug P0 Funding Live (`agents/sentiment_agent.py`, `data/live_context_enricher.py`):**
   - Novo padrão `mode="trend_momentum"` (`F&G >= 55 & fng_7d_chg >= +5 -> BUY`; `<= 45 & <= -5 -> SELL`), superando `S0` em `+241.3R` OOS standalone e melhorando comitê `1h` OOS (`+34.4R`).
   - Corrigido bug P0 em que `funding_rates` (`> 12h` stale) bloqueava fallback para `microstructure.db.derivs` (`funding_rate == 0.0` em 50/50 decisões ao vivo).
4. **`S5-5` `SWING_4H` Shadow Meta-Sizer (`data/swing_4h_meta_sizer.py`, `scripts/run_paper_simulation.py`):**
   - `Swing4HMetaSizer` causal (`LogReg-L2 C=0.1` em 10 features D-1 + `EWMA_VolTarget`, `OOS Sharpe 1.07 -> 1.24`, `Calmar 2.13`) operando em **`mode="shadow"` por padrão** (`applied_mult = 1.0`), gravando `swing_4h_meta_telemetry` sem alterar `data/swing_4h_strategy.py`.
5. **Notas P2/P3 da Revisão Cruzada & Itens Diferidos para S6 / Wave P4b:**
   - **Overlay `atr_risk_parity_mult` e `rs_14d_mult`:** Calculados e logados em `swing_4h_meta_telemetry` apenas como diagnóstico independente; não são multiplicados junto a `vol_target_mult` em `combined_shadow_mult` para evitar dupla contagem de volatilidade (`VT + S5-5` foi a combinação validada em OOS).
   - **`open_air` sem resistência mapeada:** `dist_res_val is None` continua mapeando `open_air` ("céu aberto") conforme baseline `EXP 6C`; separar `"unknown_sr"` só faz sentido caso o extrator passe a falhar sem sinalizar.
   - **Diferidos propositalmente:** **`T2` Carry Treasury** (em `STANDBY` — yield OOS 2025–26 em `+0.12%/ano`), **`T11` Expansão Top 15–20 Perps** (S6), **`DAY_TRADE 1h` geometria `G2/G3`** (`OOS PF 0.90/0.95 < 1.0`, manter E7 congelado), **`T6a 4h` Lead-Lag** (amostra OOS `n=38` insuficiente), **`Wave P4b` liquidations/`tick_agg`** (~30–60d coleta limpa) e Meta-Labeling sobre gatilhos já filtrados.

---

## 0-prev. Entregas Wave P3 (`v1.0.10`, 2026-11-21)

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
