# Gaps & Deferred Backlog — Patch S5 (`v1.0.8`)

Itens **P2** identificados na revisão cross-model (`claude-sonnet-5-5-medium`, Round 1 & Round 2 `APPROVE`) e roadmap da **Wave P2** diferido por governança de cota e escopo. Nenhum item abaixo constitui bloqueador P0/P1 para operação das ondas P0 + P1-A + P1-B.

---

## 1. Itens P2 Deferidos (Patch S5)

1. **`scripts/binance_ws_collector.py` (`on_trade` no loop asyncio):**
   - Callback `on_trade` executa de forma síncrona dentro do loop `asyncio` do WebSocket; caso futuros callbacks passem a fazer I/O ou persistência bloqueante, desacoplar via fila (`asyncio.Queue` / worker dedicado).
   - Higienizar estado interno do loop WS caso ocorram exceções fora de `recv`.
2. **`data/live_context_enricher.py` (barra diária parcial e visibilidade de erros SQL):**
   - No refresh via Yahoo (`3mo/1d` em `_fetch_yf`), filtrar `date_str < utc_today` na tabela diária caso atualizado intradia, garantindo ausência de barra D parcial.
   - `_fetch_all` registra erros inesperados apenas em nível `debug`; considerar incrementar contador explícito (`ctx["enrich_errors"]`) ou warning para erros SQL não relacionados a tabelas opcionais ausentes.
3. **`agents/tape_reading_agent.py` (`active_symbols` / `binance_series` period):**
   - Falhas ao consultar `active_symbols` em `refresh_covered_symbols_from_db()` permanecem registradas apenas em `debug`.
   - Tanto a janela de taker 5m quanto o deslocamento de fechamento de bucket (`ts_ms + 300000 <= ts_ms`) assumem período fixo de 5 minutos para linhas de `binance_series` (`kind='taker'`).
4. **`scripts/backtest_scalp_15m_calibration.py` (premissas de microestrutura e bloco informativo):**
   - Simulação de entrada maker assume preenchimento determinístico em `close[i]`, sem modelo probabilístico de fila/fill; saídas por `max_holding_bars=30` aplicam custo Taker (`SL`) de forma conservadora.
   - O bloco informativo pós-`created_at` do `.joblib` de produção ranqueia candidatos após observar OOS (`n=0` no recorte recente) e não deve ser usado para decisão de promoção.

---

## 2. Roadmap Diferido — Wave P2 (Pós-Reset de Cota Semanal)

- **Limpeza de Features Mortas & Retreino ML v2 (`1h` e `15m`):**
  - Remover ou corrigir as 10 features mortas identificadas no diagnóstico pré-S5.
  - Retreinar modelos `1h` (`lgbm_model_v1`, `rf_model_v1`) e `15m` (`lgbm_15m_v1`, `rf_15m_v1`) incorporando microestrutura enriquecida, cross-asset (`ETH/BTC`, `SOL/BTC`), MTF (`4h`/`1d`) e sazonalidade intradiária sob Purged Walk-Forward CV (`PBO < 20%`, `DSR > 0.95`).
  - Reavaliar promoção dos thresholds `SCALP_SQUEEZE_NATR_BPS_MIN` / `SCALP_SQUEEZE_VOL_Z_MIN` somente após novo modelo `15m` superar o gate OOS (`PF >= 1.25`).
  - Invalidar `TechnicalAgent._cached_default_model` em caso de hot-reload/promoção de artefato em memória.
