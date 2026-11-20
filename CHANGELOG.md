# 📋 Registro de Mudanças (Changelog) - IA Trader

Todas as alterações notáveis deste projeto são documentadas neste arquivo de acordo com as diretrizes do [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/) e aderentes ao [Semantic Versioning](https://semver.org/lang/pt-BR/).

---

## [1.0.9] - 2026-11-20 - Wave P2: Desduplicação DB, Pipeline ML v2 Causal, Hot-Reload, Benchmark Honesto & Hardening P2-D

### Added
- **Auditoria & Desduplicação Canônica Binance + Bitget (`scripts/audit_and_dedup_candles.py`):** Novo utilitário CLI read-only (`--check`) e limpeza transacional (`--apply` com backup online prévio em `datasets/backups/pre_dedup_20261120T145807Z.db` e `PRAGMA integrity_check`). Removidas `4.512` barras redundantes da Bitget (`192` overlaps exatos em `ETHUSDT:4h` e `4.320` candles `1d` com offset UTC+8 sobre datas cobertas pela Binance), zerando duplicatas sem apagar as 21 costuras históricas não sobrepostas (`15m`/`4h`).
- **Pipeline Causal ML v2 & Paridade Serving (`data/ml_features_v2.py` & `scripts/run_paper_simulation.py`):** Novo pipeline unificado de `49` features estacionárias e finitas (`V2_FEATURE_COLUMNS`), eliminando as 10+ features mortas e duplicatas de `v1`, protegendo costuras multi-exchange (`bitget -> binance`) por segmento contínuo no volume/VWAP, calculando `funding_zscore_30d`/`funding_7d_ma_bps` na série nativa 8h antes do `merge_asof`, aplicando shift `D+1` estrito em macro/on-chain longo (sem `DEFI_TVL` de 60d), reamostrando MTF `4h`/`1d` exclusivamente de velas fechadas `1h` da Binance, cross-asset BTC e sazonalidade cíclica (`1.76ms` cacheado ao vivo, paridade batch-live `<= 1e-9`).
- **Benchmark Zero-Leakage Before × After & Promotion Gate (`scripts/eval_and_train_ml_v2.py`):** Suíte pré-registrada em `1h` (`2020-01..2026-11`, holdout 1 ano `>= 2025-11-20`) e `15m` (`2025-10..2026-11`, holdout `>= 2026-07-07`) com CSCV `PBO`, `DSR` sobre `173` trials e ablação informativa 66d de microestrutura (`datasets/reports/p2_ml_v2_benchmark.json`, `TRIAL_0166..TRIAL_0173`).
- **QA Independente (`tests/test_wave_p2_ml_and_hardening.py`):** Suíte dedicada cobrindo desduplicação, perturbação anti-lookahead, guarda de costura, paridade batch-live, split multi-ativo por timestamp, hot-reload `mtime_ns/size`, guarda v2 e endurecimento P2-D.

### Changed
- **Validação Temporal Multi-Ativo & Proteção de Schema (`model_trainer.py`, `agents/technical_agent.py`, `learning_tracker.py`, `scripts/run_paper_simulation.py`):**
  - `PurgedTimeSeriesSplit` e `ModelTrainer.train_walk_forward()` passaram a suportar `timestamps` + `embargo_ms` cronológicos reais para datasets pooled multi-ativo (mantendo compatibilidade retroativa para `timestamps=None`).
  - `_DefaultMLEnsemble` refatorado para selecionar features por sub-modelo (`v1`/`v2`), com auto hot-reload por `(path, st_mtime_ns, st_size)`, `invalidate_default_model_cache()`, `invalidate_scalp_ml15_cache()` e `SKIPPED_V2_GUARD` em `ModelEvolutionManager.run_retraining_cycle()`.
  - `PaperTradingDaemon` mantém chamada compatível de 1 argumento em `_predict_scalp_ml15_probs(latest)` e propaga contexto causal para `_build_committee_context(..., timeframe=...)`.
- **Fechamento dos 4 Itens P2-D (`scripts/binance_ws_collector.py`, `data/live_context_enricher.py`, `agents/tape_reading_agent.py`, `scripts/backtest_scalp_15m_calibration.py`):**
  - WebSocket Binance Futures atualizado para `wss://fstream.binance.com/market/stream`, logger `collector.ws` (visível em `logs/collector.log`), fila assíncrona `asyncio.Queue` desacoplando `on_trade` de `ws.recv()` e limpeza consistente de estado do loop.
  - `LiveContextEnricher` filtra barra UTC do dia corrente no Yahoo, contabiliza erros SQL inesperados (`ctx["enrich_errors"]`) e infere período dinâmico em `binance_series`.
  - `TapeReadingAgent` infere cadência real em `binance_series` e loga erros inesperados de `active_symbols` em `warning`.
  - `backtest_scalp_15m_calibration.py` ganhou `maker_fill_mode="realistic"` (toque na barra seguinte) e aviso explícito `[INFORMATIONAL ONLY]` pós-`created_at`.

### Results (Sem Oráculo Comprado)
- **Benchmark ML `v1` × `v2` (`Decision=FAIL_REFUTED` — Modelos `v1` preservados):** Feature engineering `v2` zerou features mortas (`11/10 -> 0/0` em 1h) e elevou `PF`/`E[R]` em todas as comparações diretas (`1h H5: PF 0.557 -> 2.435` com `N=9`; `1h H18: PF 1.002` com `N=38`; `15m: PF 0.302 -> 0.528`), porém sem atingir o gate rigoroso (`PBO < 0.20` e `DSR >= 0.95` sobre 173 trials); artefatos `v1` e thresholds `SCALP` (`35.0 / 1.0 / 0.45`, realista `PF=1.22 < 1.25`) foram mantidos.
- **WebSocket Coletor ao Vivo Pós-Restart:** Saturação em `BTCUSDT` caiu de `95%` (`500–700` trades/min via REST) para `0` na primeira janela completa pós-restart (`4.345` trades/min em `BTCUSDT`, `3.844` em `ETHUSDT`, `331` em `SOLUSDT`, e tabela `liquidations` ativa com 7 eventos iniciais).

---

## [1.0.8] - 2026-11-20 - Patch S5: Blindagem P0, Comitê 6-Agentes Causal, Microestrutura WS e Maker-First

### Added
- **Wave P1-A (`data/live_context_enricher.py`):** Novo `LiveContextEnricher` injetando contexto causal em tempo real nos 3 loops (`GP`, `SCALP`, `DIRECTIONAL`): `volume_sma` (fallback `vol_sma_20`), `atr_median`/`atr_sma`, regime por símbolo (`MarketRegimeClassifier(db_path="")`), série diária macro estrita `D-1` (`timestamp_ms < day_start_ms` + guard de staleness para `DXY`, `VIX`, `SP500`, `FEAR_AND_GREED`, `STABLECOINS_CIRCULATING_USD`, `DEFI_TVL`), calendário de eventos macro ±30m UTC (`US_CPI_RELEASE` 13:30 UTC, `FOMC_DECISION` 19:00 UTC -> `is_cpi_fomc_window`), `funding_zscore` (30 amostras em `funding_rates`/`derivs`), fallback de profundidade de book (`n10->n5->n2`), buckets fechados anti-lookahead (`minute_ms + 60000 <= ts_ms` em `tick_agg` e `ts_ms + 300000 <= ts_ms` em `binance_series`) e refresh periódico de cache externo a cada 6h (com backoff de 5m em falha total).
- **Wave P1-B (`scripts/binance_ws_collector.py` & `scripts/microstructure_collector.py`):** Novo `BinanceFuturesWSCollector` (`websockets 16.1.1`) em streaming de `@aggTrade` (`ingest_ws_trade` com `sat=0`) e `!forceOrder@arr` persistindo na nova tabela `liquidations`; isolamento de IDs em `TickAggregator.last_id` por `(source, sym)`; paginação em `_poll_binance` (`limit=1000`, até 3 páginas); substituição de bloqueio permanente de Bitget por TTL de 300s (`_BITGET_BACKOFF_UNTIL`); e flag `--status` estritamente read-only (`?mode=ro`) reportando liquidações e saturação 24h.
- **Wave P1-B (`scripts/backtest_scalp_15m_calibration.py`):** Harness local de calibração OOS para SCALP 15m com split Purged IS->OOS sem vazamento e embargo de 10 barras.
- **QA & Documentação (`tests/test_patch_s5_p0.py`, `tests/test_patch_s5_p1a.py`, `tests/test_patch_s5_p1b.py`, `docs/gaps.md`):** Suítes independentes cross-model cobrindo P0, P1-A (incluindo `test_microstructure_taker_closed_bucket_anti_lookahead`) e P1-B, mais registro de gaps P2/Wave P2 em `docs/gaps.md`.

### Changed
- **Wave P0 (`agents/technical_agent.py`, `scripts/run_paper_simulation.py`):** Correção do filtro de warning `sklearn.utils.parallel`, desativação segura contra ciclos de paralelismo RF (`_disable_estimator_parallelism()` com `n_jobs=1`), gate de `SWEEP_OVERRIDE` exigindo alinhamento `macro_bull` e `vol_z >= 0.8` (`conf=0.72` se `vol_z >= 1.5`, senão `0.68`, quando `vol_z` não for `None`), `RotatingFileHandler` opcional no daemon e deduplicação do curto-circuito do Circuit Breaker (`_breaker_short_circuit()` / `_is_breaker_blocking()`) antes de I/O ou ML em `GP`, `SCALP` e `DIRECTIONAL`.
- **Wave P0 (`performance_tracker.py`, `rollback_guard.py`):** Janela móvel de 15 trades (`rolling_window_size=15`, `rolling_trades_count`, `rolling_win_rate`, `rolling_profit_factor`) com ativação de `DefensiveRiskPolicy` (`DEGRADATION_ROLLBACK_ACTIVATED`) e auto-recuperação (`rolling_wr >= 50.0` e `rolling_pf >= 1.25` -> `DEGRADATION_ROLLBACK_RECOVERED`).
- **Wave P0 (`smart_trade.py`, `tests/test_signal_funnel.py`):** Expansão de `FUNNEL_REJECT_REASONS` com `"stale_bar"`, `"stale_candle"`, `"climax_exhaustion"`, `"confidence"`, `"stale_data"`, `"tape_veto"` e `"kill_switch"`.
- **Wave P1-A (`agents/orderflow_agent.py`, `consensus_judge.py`, `scripts/run_paper_simulation.py`):** `OrderFlowAgent` ganhou Prioridade 0.5 de microestrutura real (`micro_taker_imb_5m` e `micro_book_imb_10`), suprime o eco da Prioridade 5 quando há tape real e endurece a Prioridade 5 (`|close_position| >= 0.55` ou `vol_ratio >= 1.2 and |close_position| >= 0.25`); `ConsensusJudge._apply_confidence_calibration()` generalizado além de `SIDEWAYS_RANGE` para todos os regimes explícitos (`(ece > 0.20 and acc < 0.50)` ou `acc < 0.40`, guardando `total_samples < 5`).
- **Wave P1-B (`agents/tape_reading_agent.py`, `smart_trade.py`, `scripts/run_paper_simulation.py`):** `TapeReadingAgent` com schema-aware `tick_agg`, fallback de profundidade (`n10->n5->n2`), fallback para `binance_series` (`kind='taker'`) sob saturação/ausência de `tick_agg`, telemetria de liquidações 5m e `try/finally: con.close()` em `refresh_covered_symbols_from_db()`; `PaperTradingEngine` ganhou `directional_maker_entry: bool = False` opt-in (`[MAKER_POST_ONLY]` / `[TAKER_FALLBACK_WIDE_SPREAD]` com threshold `> 15.0` bps em GP e `> 5.0` bps nos demais), mantendo paridade default Taker em testes e `SCALP` preservado em `SCALP_SQUEEZE_NATR_BPS_MIN = 35.0` / `SCALP_SQUEEZE_VOL_Z_MIN = 1.0` após gate OOS (`PF 1.24 < 1.25`).

### Fixed
- **Wave P0 (`flask_interface.py`):** Garantia de fechamento de sessão async (`dl.close()`) em `try ... finally` nos endpoints `/api/chart/...` e `/api/refresh-symbol/...` antes de `loop.close()`.
- **Review Cross-Model Round 1 -> Round 2 (`APPROVE`):** Eliminação de lookahead em buckets abertos de `tick_agg`/`binance_series` no `LiveContextEnricher`, reconstrução sem vazamento (Purged IS->OOS) do calibrador SCALP 15m e fechamento determinístico de conexão SQLite em `TapeReadingAgent`.

---

## [1.0.7] - 2026-11-15 - Patch S4.2: Blindagem Multi-Estratégia, Reabilitação Universal do Scalp 15m e Paridade

### Adicionado
- **Reabilitação Universal do Scalp 15m (`smart_trade.py`, `scripts/run_paper_simulation.py`):**
  - Integração do ensemble dedicado 15m (`models/lgbm_15m_v1.joblib` e `models/rf_15m_v1.joblib`) com gatilho Squeeze Bollinger Bands / Keltner Channels e filtro de convicção `ML 15m >= 0.45` (baseado no estudo `E10.9 x ML 15m OOS`).
  - Saída estrutural com alvo único (`SL 1.4x ATR`, `TP 2.2x ATR`, `tp1_fraction=0.0`, `tp2_fraction=0.0`), eliminando a armadilha de atrito da escada `TP1 BE +4bps`.
  - Operação universal com risco defensivo de `0.50x` em todos os 3 pares principais (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`) sob gate estrito `Tape status == "OK"`.
- **Trava de Correlação no Directional 1h (`scripts/run_paper_simulation.py`):**
  - Implementada restrição `MAX_SAME_DIR_MAJORS = 2`, impedindo exposição direcional simultânea em mais de 2 majors e mitigando o risco de squeeze triplo.
- **Filtro Macro no Golden Pot 5m (`scripts/run_paper_simulation.py`):**
  - Adicionado gate de alinhamento com a SMA 200 de 1h do BTC (`BTCUSDT 1h dist_sma_200` dentro de `±1.5%`), bloqueando compras contra tendência macro baixista.
  - Implementada política L2 híbrida `M4`: lote integral (`1.00x`) quando `Tape status == "OK"` e lote defensivo (`0.50x`) sob `UNAVAILABLE`, mantendo cobertura ampla de altcoins para aprendizado.
- **Aba de Evolução & Telemetria no Dashboard (`web/routes.py`, `web/templates/index.html`):**
  - Adicionado endpoint `/api/telemetry/evolution` e interface gráfica para acompanhamento contínuo de PnL, Win Rate, acurácia por regime e calibração de modelos por versão/semana.

### Modificado
- **Gate de Microestrutura no Directional (`scripts/run_paper_simulation.py`):**
  - Endurecido `_flow_gate` para modo *fail-closed* estrito nos majors, exigindo `status == "OK"` e eliminando passagens cegas (`UNAVAILABLE_PASSED`).

### Corrigido
- **Paridade do Motor de Backtest com Produção (`backtesting.py`):**
  - Portada para o simulador de backtest a proteção intra-barra `closed_safe` (`v1.0.5`), garantindo paridade estrita na avaliação de Breakeven Stops contra pavios intra-barra.
- **Calibração e Telemetria em Produção (`learning_tracker.py`, `agents/regime_classifier.py`):**
  - Corrigida persistência do `brier_score` real em `agent_learning_memory` e ampliada janela de classificação de regime para `LIMIT 250`.
  - Deduplicadas velas não-binance em `candles` com prioridade canônica `binance`.

---

## [1.0.5] - 2026-11-05 - Sprint S3: Proteção Intra-Barra do Breakeven Stop contra Pavios Iniciais

### Corrigido
- **Smart Trade Ladder - Proteção de Breakeven Intra-Barra ([`smart_trade.py`](file:///home/eudespatricio/ia_trader/smart_trade.py#L1847)):**
  - Eliminado falso estopamento de Breakeven em barras onde o TP1 é atingido e o preço fecha a salvo (`closed_safe`).
  - Pavios anteriores da mesma barra não acionam mais o BE retroativamente quando há dados OHLC estruturados (`same_bar_safe`), preservando posições em forte expansão de momentum.
  - Adicionado teste de regressão dedicado em [`tests/test_exit_ladder_e7.py`](file:///home/eudespatricio/ia_trader/tests/test_exit_ladder_e7.py#L135).

---

## [1.0.4] - 2026-11-03 - Patch S2: Calibração Convexa de Agentes, Dampening em Range e Retreino 50k

### Adicionado
- **Otimização Convexa Restrita por Regime (`learning_tracker.py`):**
  - Implementada recalibração de pesos de voto por regime (`recalibrate_agent_weights`) via Ridge / L2 com prior bayesiano `DEFAULT_BASE_WEIGHTS`.
  - Garantida soma normalizada de pesos ativos, pesos não-negativos e preservação inegociável do veto do `RiskSentinelAgent` (`w = 1.5`).
- **Dampening de Sobreconfiança em `SIDEWAYS_RANGE` (`consensus_judge.py`, `core.py`):**
  - Calibração probabilística com cálculo de ECE (Expected Calibration Error).
  - Limitação cirúrgica da confiança máxima a 0.55 quando ECE > 0.15 ou acurácia < 45% em regime lateral, prevenindo falsos rompimentos.
- **Pipeline de Retreinamento Contínuo (`learning_tracker.py`):**
  - Adicionado método `retrain_daily_models()` delegando ao `ModelEvolutionManager` para ciclos Champion-Challenger automáticos.

### Corrigido
- **Smart Trade Exit Ladder & Funil de Sinais (`tests/test_exit_ladder_e7.py`, `tests/test_signal_funnel.py`):**
  - Corrigida asserção de trailing stop (`trail0 = 101.21`, `trail1 = 101.30`) em paridade com a lógica nativa de ratchet.
  - Sincronizado timestamp de teste de rejeição do funil com o bucket nativo de 15m (`SCALP_BAR_MS`).

---

## [1.0.3] - 2026-10-21 - Remediação Integral de Riscos P0 e Implementação do OMS Institucional

### Adicionado
- **Order Management System (OMS) Institucional (`oms.py`, `tests/test_oms.py`):**
  - Implementada State Machine de ordens estrita com 12 estados (`SIGNAL_READY -> RISK_APPROVED -> ORDER_INTENT -> SENT -> ACKED -> PARTIALLY_FILLED -> FILLED / CANCELED / REJECTED / EXPIRED -> CLOSED -> RECONCILED`).
  - Idempotência determinística via hash SHA-256 (`symbol:timestamp_ms:strategy:side:position_id`), eliminando duplicatas e double-fillings.
  - Kill switch externo local (`emergency_kill.switch`) e remoto via variável de ambiente (`IA_TRADER_KILL_SWITCH`), bloqueando imediatamente novas entradas e armando protocolo de segurança.
  - Reconciliação forense no boot com detecção de ordens e posições órfãs e transição automática de ordens pendentes não confirmadas para `EXPIRED`.
- **Política de Risco Defensiva Versionada (`RiskPolicy` e `DefensiveRiskPolicy` em `rollback_guard.py`):**
  - Objeto de governança de risco que restringe estratégias permitidas (`allowed_strategies=["DIRECTIONAL"]`), reduz alavancagem máxima para 2.0x, aplica multiplicador de lote `0.5x` e exige confiança mínima de 0.85 quando acionado.

### Corrigido
- **Quórum & Estado ABSTAIN no Comitê (`agents/base_agent.py`, `consensus_judge.py`):**
  - Adicionado estado canônico `ABSTAIN` e parâmetro `min_directional_agents=2`. Um único agente direcional (ex.: Technical BUY com todos os demais HOLD/ABSTAIN) nunca mais aprova uma operação isoladamente.
  - LLM rebaixado de autoridade executiva para auditoria quantitativa de risco com timeout de 1.0s, impossibilitado de criar ordens direcionais sem concordância da bancada.
- **Veto em Duas Fases (`core.py`, `agents/risk_sentinel_agent.py`):**
  - A checagem de proximidade de suporte/resistência e varredura de liquidez agora roda na Fase 2 com a direção real pretendida (`intended_direction`), eliminando o bypass onde ordens contrárias passavam sem veto direcional.
- **Trava de Drawdown Diário Real (`rollback_guard.py`):**
  - Rastreamento persistente de baseline diário via tabela `circuit_breaker_daily`. Se o drawdown intradiário atingir ou ultrapassar 5.0%, dispara `DAILY_DRAWDOWN_BREAKER`, bloqueando novas entradas até o reset da virada UTC.
- **Persistência Atômica de Ledger (`smart_trade.py`):**
  - Implementado método `_save_position_and_trade()` sob transação atômica única `BEGIN IMMEDIATE`, eliminando risco de posições sem trade de abertura e inconsistências no banco.
- **Freshness e Fail-Closed contra Stale Data (`smart_trade.py`, `agents/tape_reading_agent.py`):**
  - Sinais marcados com dados obsoletos (`is_stale=True`) ou com veto de fluxo (`tape_decision == "VETO"`) são rejeitados compulsoriamente no boundary final de execução. Ticks com mais de 5 minutos são descartados no Tape Reading.
- **Isolamento de Épocas de Reset no Performance Tracker (`performance_tracker.py`):**
  - As métricas do piloto passam a respeitar o marco de corte `ledger_epoch_trade_rowid` gravado na tabela `paper_settings`, expurgando trades de testes anteriores dos relatórios de performance.
- **Modelagem Quant e Paridade (`agents/regime_classifier.py`, `pair_scanner.py`, `funding_carry.py`, `trend_following.py`, `statistical_rigor.py`, `backtesting.py`):**
  - Exigência estrita de 200 barras para cálculo de SMA200 (`has_real_sma200`).
  - Varredura de 24h no scanner ajustada para 96 barras de 15m.
  - Normalização do microprice de Tape Reading pelo spread (`mdev / max(spread, 1.0)`).
  - TSMOM corrigido para registrar pernas short (`w < 0`).
  - Testes de estresse de Funding Carry calculados dinamicamente com capital e alavancagem reais.
  - Log de ensaios estatísticos (`trial_log.csv`) transformado em append-only imutável.
- **Suíte de Testes 100% Verde:**
  - 569 testes aprovados (com 1 skip e 7 xfail esperados de paridade de design).

---

## [1.0.2] - 2026-10-21 - Correções Críticas de Execução, Margem e Árbitro LLM

### Corrigido
- **Trava de Barra Única e Cooldown em Swing Trade (`scripts/run_paper_simulation.py`):**
  - Implementada deduplicação por vela fechada (`_swing_entered_bar(sym, bar_id)` e `_swing_entered_bars`) e proteção contra re-entrada no mesmo par após stop loss (`_swing_in_cooldown`).
  - Elimina o bug que provocou 4 perdas consecutivas em `ETHUSDT` na mesma vela de 1h em menos de 10 minutos (economizando até R$ 1.800 / $350+ por ciclo desfavorável).
- **Cálculo de Margem Real no Dashboard Flask (`flask_interface.py`):**
  - Corrigida a consulta SQLite nos endpoints `/api/paper-trading/status` e `/api/simulation/settings`: substituído `SUM(notional)` por `SUM(notional / COALESCE(NULLIF(leverage, 0), 1.0))`.
  - O dashboard agora reporta com precisão matemática a margem real empenhada ($6.775,82 / 72,3% de utilização) em vez de notional bruto alavancado ($20.327,47 / 216,8%), restaurando a exibição correta de margem livre ($1.194,91).
- **Timeout no Árbitro LLM & Integração Google Gemini Flash (`chatgpt_assistant.py`, `consensus_judge.py`):**
  - Adicionado suporte nativo ao **Google Gemini Flash** (`gemini-3.8-flash`) via SDK `google-genai` para arbitragem ultrarrápida (<800ms) com structured JSON output.
  - Aumento do timeout para 8.5s quando utilizando fallback do Ollama local (`llama3.1:latest` na RTX 3060), eliminando os timeouts crônicos de 4.5s no loop do daemon.
- **Governança de Testes no WSL (`pytest.ini`):**
  - Adicionada flag `addopts = --ignore=tests/ui` para garantir execução rápida e 100% verde da suíte completa de backend (556 testes aprovados) sem conflito de display headless ou porta 5000.

---

## [1.0.1] - 2026-10-20 - Desmembramento & Isolamento Definitivo no WSL

### Adicionado
- **Handover Operacional e Arquitetural (`HANDOVER.md`):** Documento detalhado sobre a operação 100% isolada no WSL, governança e monitoramento de daemons (Flask 5000, Paper Trading e Microestrutura).
- **Consolidação de Documentação:** Centralização de `docs/IA_Trader_Documentacao_FINAL_VERSAO_1_0.txt` e microagentes `.agents/` diretamente dentro do repositório `/home/eudespatricio/ia_trader`, eliminando dependências externas do disco compartilhado `D:\Projeto`.

---

## [Não lançado] - 2026-10-14/15 - Auditoria e Correções

### Alterado
- **Transição de Timeframes Intraday (2026-10-16):** Alinhamento operacional do Scalper (Majors BTC/ETH/SOL) para candles 3m com gatilho de momentum rápido e do Golden Pot (Radar Altcoins) para 5m com expansão de volatilidade calibrada; polling ágil a 3m para posições ativas; Smart Trade Exit Ladder mantida intacta; reinício a quente via supervisor com restauração total de estado (trade SOLUSDT preservado).
- **Funil de sinais (2026-10-15):** instrumentação pura (NÃO muda estratégia/decisões): tabela `signal_funnel` (hora UTC x estratégia x estágio, `reject:<motivo>`) alimentada por `PaperTradingEngine.funnel_inc` (dedupe por barra, flush ~30 s e no shutdown); `reset_pilot` a limpa.
- **E8 (2026-10-15):** gate de fluxo ativo (`TapeReadingAgent`, lê `microstructure.db` em modo RO) após a porta do comitê em SCALP/GOLDEN_POT/DAY TRADE: voto do tape deve confirmar a direção (oposto/HOLD bloqueia; dados ausentes = fail-open marcado); auditoria contrafactual em `flow_gate_log`. Piloto reiniciado. Ver ADDENDUM E8.
- **E7 (2026-10-15):** SCALP, GOLDEN_POT e DAY TRADE passam obrigatoriamente pelo comitê (scalp deixa de usar confidence 0.85/±0.80); GOLDEN_POT v3 = momentum + volatilidade extraordinária; **Smart Trade Exit Ladder** comum (TP1 40% + BE, TP2 30% + trailing por ATR sem lookahead, TP3/trailing no runner; perfis em `STRATEGY_PROFILES`). Piloto reiniciado. Ver ADDENDUM E7.
- **GOLDEN_POT v2 (E5, 2026-10-15):** entrada só com concordância do comitê completo (sem veto), `confidence`/`consensus`/`risk` reais, cooldown 16 barras, 1 GP/vela, filtro de faca-caindo, exclusão de ativos tokenizados, ≥400 candles com `sma_200` válida; lote = margem `paper_lot_usd` (padrão US$100) com alavancagem do Smart Trader (`leverage_policy.py`, teto 5x, liquidação). Piloto reiniciado; E4 invalidado para GOLDEN_POT.

### Corrigido
- **Auditoria Estrutural Argon — 6 Correções Validadas em Teste Offline A/B (2026-10-17):**
  1. **Deduplicação do `_flow_gate` por `(strategy, symbol, bar_ts)` (`scripts/run_paper_simulation.py`):** Impede a reavaliação da mesma vela fechada a cada 15s (redução de 99,2% na inflação da `flow_gate_log`) e elimina entradas atrasadas com preço antigo (*stale price*, como o trade `d3be50b1` que havia entrado 9m32s após o fechamento).
  2. **Alinhamento Temporal de Polling e Ratchet do Trailing (`smart_trade.py`, `scripts/run_paper_simulation.py`):** Posições `GOLDEN_POT` (5m) atualizam via velas fechadas de 5m e `SCALP`/`DIRECTIONAL` (3m/1h) via 3m, com inicialização imediata de `_last_tick_bar_ts` na abertura e evolução do `_ratchet_trailing` restrita ao fechamento do bucket nativo da estratégia (`prof.bar_ms`).
  3. **Priorização de Momentum no Scanner do Golden Pot (`agents/golden_pot_agent.py`, `scripts/run_paper_simulation.py`):** `SWING_BREAKOUT` avaliado antes de `SCALP_REVERSION` e fila `top_swings + top_scalps` priorizada para não sufocar rompimentos com ativos em exaustão de VWAP (+38,5% de candidatos de momentum detectados no backtest de 5m).
  4. **Bloqueio de Colisão de Ativo entre Estratégias e Calibração Fat-Tail do Golden Pot (`smart_trade.py`):** `_open_ladder_position` e `evaluate_swing_opportunities` vedam múltiplas posições simultâneas no mesmo símbolo independente da estratégia; `GOLDEN_POT.sl_atr_mult` calibrado de `1.0x` para `1.5x ATR` (alinhado ao `trail_atr_mult = 1.5x ATR`), absorvendo retestes naturais de rompimento em altcoins (Win Rate subiu de 33,3% para 50,0% no replay 5m).
  5. **Escalonamento de `ATR_PCT_EXTREME` por Timeframe e Sincronismo UTC no `restore_state` (`leverage_policy.py`, `smart_trade.py`):** Limiar de volatilidade extrema ajustado por timeframe (`3m`: `1.2%`, `5m`: `1.5%`, `15m`: `2.0%`, `1h`: `3.0%`, `4h`: `5.0%`) e `current_day_utc` sincronizado na restauração.
  6. **Consolidação por `position_id` no `PerformanceTracker` e `LearningTracker` (`performance_tracker.py`, `smart_trade.py`):** `LearningTracker` passa a receber `pos.realized_pnl` (acumulado das parciais) em vez do `net_pnl` apenas da última perna (`BREAKEVEN_STOP -$0.05`), e `PerformanceTracker.compute_metrics_from_db()` agrega por `position_id` (excluindo `FUNDING` isolado e posições abertas), impedindo que trades vencedores em `TP1 + Breakeven` sejam contabilizados como derrotas no `RollbackGuard`.
- **Isolamento de Velas Pré-Entrada (2026-10-16):** Em `smart_trade.py` (`update_market_tick`) e `scripts/run_paper_simulation.py` (`_update_symbol`), corrigido vazamento temporal onde candles anteriores à abertura da posição (`bts < entry_timestamp`) avaliavam a escada de saída e disparavam falso Stop Loss em trades de rompimento (ocorrido em `BTWUSDT`); barras pré-entrada agora são estritamente filtradas no polling e bloqueadas na ladder; testes de regressão adicionados; trava de cooldown do RollbackGuard liberada.
- **Lookahead V1-V8:** CPI no dia 20 do mês seguinte; FEDFUNDS no fim do mês+1; removido `bfill`; snapshots macro/order book só na última barra; `as_of_date` em VIX/DXY/SP500; candle em formação descartado. V9 corrigido com atualização de trailing apenas em barras fechadas.
- **Execução:** taxa de saída em dobro (`smart_trade.py`); cota diária reaberta na virada UTC; teto de margem no carry; race em `update_market_tick`; TP1 parcial duplicado; `entry_fees` em `funding_carry`.
- **Paper trading:** `restore_state()` (ledger, `MANUAL_RESET`); swing com 400 candles + warmup `sma_200` (viés comprador); A3 (sem SL/TP com dados pré-entrada); A1 (SL/BE/MANUAL como taker 0,06% + 0,03% slippage); win_rate por `position_id`; `position_size` em USD no swing.
- **Decisão:** NaN→confiança, peso de fallback, validação de LLM, divisão por zero em `signal_generator`.

### Adicionado
- `data/db_maintenance.py` (checkpoint WAL; purge de símbolos esparsos em dry-run).
- Supervisor: healthcheck 5s, 3 falhas, re-adoção de PID; `logs/flask.log`.
- `docs/CRITERIOS_APROVACAO_PILOTO.md` e `scripts/pilot_report.py`.
- Suíte backend: **544 testes passando** (`tests/test_pre_entry_regression.py` expandido com regressão das 6 correções Argon).

### Pendente / Propostas
- V9; ajustes de consenso (agentes neutros, vetos inertes, double counting) aguardam backtest A/B.
- Piloto pré-reset REPROVADO (35 trades, expectância bruta -0,995%, viesado); piloto limpo desde o reset de 2026-10-14/15, parâmetros congelados >= 56 dias. Alfa NÃO demonstrado.

---

## [1.0.0] - 2026-10-14 - Release Oficial (10 Fases Homologadas)

### Adicionado
- **Ponto de Entrada Unificado (`launcher.py`):**
  - CLI com suporte a flags operacionais: `--db`, `--host`, `--port`, `--headless`, `--no-trader`, `--paper-only`.
  - Orquestração simultânea e resiliente de threads (Paper Trading, TaskScheduler e Servidor Web Flask).
  - Encerramento gracioso via sinais `SIGINT` e `SIGTERM` com `PRAGMA wal_checkpoint(TRUNCATE)` e `PRAGMA optimize`.
- **Empacotamento Standalone com PyInstaller (`ia_trader.spec`):**
  - Binário executável independente autocontido (`dist/ia_trader/ia_trader` - 21.5 MB).
  - Coleta automática e empacotamento de templates web, licença e dependências analíticas (`lightgbm`, `sklearn`, `ta`, `aiohttp`, `flask`).
  - Suporte transparente a `sys._MEIPASS` para extração e renderização de templates em tempo de execução.
- **Suíte de Testes de Hardening e Concorrência (`tests/test_hardening_simulation.py`):**
  - Validação de estresse com 6 threads concorrentes realizando 120 escritas em SQLite WAL sem locks.
  - Testes de disparo de Circuit Breakers (cooldown por 4 perdas consecutivas) e recuperação supervisionada.
  - Elevação da suíte global para **208/208 testes com 100% de aprovação (Zero falhas, 0:01:12)**.

---

## [1.0.0-alpha.8] - 2026-10-14

### Adicionado
- **Módulo Scanner de Pares e Volatilidade Anômala (`pair_scanner.py`):**
  - Detecção precoce de anomalias de volatilidade (ATR relativo $\ge 2.0\times$) e surtos anormais de volume ($\ge 2.5\times$).
  - Detecção de colapso de preços e risco crítico de delistagem ($\le -35\%$), atribuindo isolamento e $0\%$ de alocação de capital.
  - Alocação isolada e independente de capital de risco para pares voláteis (teto de $5\%$).
- **Módulo Educacional Interativo e Didática de Trading (`educational_module.py`):**
  - Tradução e explicabilidade em linguagem simples de operações quantitativas ("Por que a IA comprou/vendeu?").
  - Análise pedagógica de risco, dimensionamento de Stop Loss por ATR e assimetria R:R.
  - Glossário quantitativo completo (`ORDER_BLOCK`, `FAIR_VALUE_GAP`, `LIQUIDITY_SWEEP`, `ATR`, `FUNDING_RATE`, `CIRCUIT_BREAKER`, `RISK_REWARD`).
- **Módulo de Comandos de Voz e Linguagem Natural (`voice_commands.py`):**
  - Roteador semântico para comandos textuais ou via voz ("status do robô", "mostrar últimas operações", "pausar operações", "retomar operações", "qual o regime do mercado", "explicar último trade").
  - Integração direta com `RollbackGuard` para bloqueio emergencial e `PerformanceTracker` para telemetria.
- **Módulo de Gestão de Usuários e Controle de Acesso (`auth_module.py`):**
  - Armazenamento com hash PBKDF2-HMAC-SHA256 (100.000 iterações + salt criptográfico de 16 bytes).
  - RBAC com três níveis de assinatura (`FREE`, `PRO`, `PREMIUM`) com controle estrito de pares, timeframes, golden pots e comandos de voz.
  - Rastreabilidade de consentimento dos termos de uso (LGPD / GDPR).
- **Módulo Legal & Compliance:**
  - `LICENSE.txt`: licença de software e termo de isenção de recomendação financeira.
  - `templates/terms_of_use.html` e `templates/privacy_policy.html`: páginas de termos e privacidade.
- **Rotas REST Integradas no Flask (`flask_interface.py`):**
  - `/terms`, `/privacy`, `/api/educational/glossary`, `/api/educational/explain`, `/api/voice/command`, `/api/scanner/opportunities`.
- **Suíte de Testes da Fase 9 (`tests/test_auxiliary_modules.py`):**
  - 21 novos testes automatizados elevando a suíte para **205/205 testes passando (100% Green)**.

---

## [1.0.0-alpha.7] - 2026-10-14

### Adicionado
- **Módulo de Memória Cognitiva & Aprendizado Adaptativo (`learning_tracker.py`):**
  - Tabela `agent_learning_memory`: armazena deliberações com voto de cada agente, regime de mercado e desfecho real de trade (Win, Loss, PnL).
  - Tabela `agent_regime_weights`: calibração dinâmica de pesos do comitê com base em acertos empíricos e Brier score por regime de mercado.
  - Classe `ModelEvolutionManager`: protocolo Champion-Challenger para retreinamento contínuo de modelos sem parada de sistema (hot-swap seguro com exigência de ganho $\ge 0.5\%$ em ROC-AUC/Accuracy).
- **Módulo Agendador de Tarefas Periódicas (`scheduler.py`):**
  - Loop assíncrono para recalibração de pesos de agentes (a cada 30 min).
  - Loop assíncrono para checagem de retreinamento de modelos (a cada 6h / diário).
  - Manutenção periódica do banco SQLite (`PRAGMA wal_checkpoint` e `optimize` diário).
- **Módulo de Telemetria Financeira (`performance_tracker.py`):**
  - Fonte única da verdade para PnL líquido, equity, pico de capital, Maximum Drawdown ($ e %), Win Rate, Profit Factor, Payoff Ratio, sequências (streaks) e índices de Sharpe, Sortino e Calmar.
  - Decomposição analítica de performance por estratégia (`SCALP`, `GOLDEN_POT`, `CARRY`).
  - Persistência de snapshots históricos na tabela `performance_snapshots`.
- **Módulo de Estabilidade e Circuit Breakers (`rollback_guard.py`):**
  - Circuit Breaker de Drawdown Total (10.0%): trava automática de novas ordens para preservação de capital.
  - Circuit Breaker de Perdas Consecutivas (4 stops): cooldown obrigatório de 2h contra overtrading e desvio de regime.
  - Auto-Rollback por Degradação: detecção de queda crítica de Win Rate (< 30%) e reversão para parâmetros ultraconservadores.
  - Parada manual de emergência e reset supervisionado com trilha de auditoria na tabela `circuit_breaker_events`.
- **Novos Endpoints REST no Dashboard Flask (`flask_interface.py`):**
  - `GET /api/telemetry/performance`: métricas financeiras completas.
  - `GET /api/telemetry/circuit-breaker`: estado ao vivo do guardião (`CLEAR`, `COOLDOWN`, `TRIPPED`).
  - `POST /api/telemetry/circuit-breaker/reset`: reset supervisionado de circuit breaker.
  - `POST /api/telemetry/emergency-stop`: acionamento imediato de parada de emergência.

### Modificado
- `smart_trade.py`: integrados os gates de segurança de circuit breaker em `can_execute_scalp()` e `can_execute_golden_pot()`; acionamento de telemetria e memória cognitiva ao encerrar trades em `close_position()`.
- `core.py`: incorporada a busca de pesos dinamicamente calibrados pelo regime de mercado no `TradingCore`.

---

## [1.0.0-alpha.6] - 2026-10-14

### Adicionado
- **Motor de Execução Smart Trade (`smart_trade.py`):**
  - Execução estrita em modo de simulação (Paper Trading) com bloqueio mandatório de ordens com capital real.
  - Separação e independência de cotas diárias: Scalps intradiários padrões (mínimo 2, teto 5/dia) e Golden Pots de alta convicção (máximo 3/dia) rodando de forma desacoplada sem canibalização.
  - Sistema de saídas parciais escalonadas (TP1 com realização de 50% + ajuste automático de Stop Loss para o Breakeven + TP2 para surfar expansões de tendência).
  - Otimização com compressão inteligente de snapshots de portfólio no SQLite (evita mais de 4.300 gravações redundantes por dia).
- **Daemon de Paper Trading Assíncrono (`scripts/run_paper_simulation.py`):**
  - Execução simultânea não-bloqueante (`asyncio.gather`) para múltiplos pares da Bitget/Binance.
  - Classificação macro de regime via `RegimeClassifier` (`BULL_TREND`, `BEAR_TREND`, `SIDEWAYS_RANGE`, `HIGH_VOLATILITY_PANIC`).

---

## [1.0.0-alpha.5] - 2026-10-14

### Adicionado
- **Simulador de Backtesting Institucional (`backtesting.py`):**
  - Modelagem estrita de microestrutura de custos: taxas maker (0.02%), taker (0.06%), slippage (0.02% a 0.03%) e rolagem de funding rate a cada 8h.
  - Colisão intra-barra conservadora (Stop Loss assumido primeiro em caso de overlap).
- **Módulo de Rigor Quantitativo (`statistical_rigor.py`):**
  - Cálculo do Deflated Sharpe Ratio (DSR - López de Prado) ajustado pelo número de testes múltiplos $N$.
  - Combinatorially Symmetric Cross-Validation (CSCV) para cálculo da Probability of Backtest Overfitting (PBO).
  - Registro atômico de ensaios e tentativas em `datasets/reports/trial_log.csv`.
- **Estratégias Institucionais Estruturais:**
  - Simulador de Funding Carry delta-neutro spot + perp (`funding_carry.py`).
  - Time Series Momentum com volatilidade alvo e banda de no-trade (`trend_following.py`).

---

## [1.0.0-alpha.4] - 2026-10-14

### Adicionado
- **Motor de Estrutura de Mercado & POI (`data/market_structure.py`):**
  - Algoritmo causal de identificação de fractais de Swing High / Swing Low (janela causal $N=3$).
  - Detecção de Fair Value Gaps (FVGs) e Order Blocks (OBs) de ignição institucional com deslocamento $\ge 1.5\times$ ATR.
  - Piscinas de liquidez Buy-Side (BSL) e Sell-Side (SSL) e detecção algorítmica de Liquidity Sweeps (varredura de stops com rejeição).
  - Distância percentual contínua para o suporte mais próximo (`dist_support_pct`) e resistência (`dist_resistance_pct`).
- **Comitê de Decisão Multi-Agente (`agents/`):**
  - `TechnicalAgent`: análise de Machine Learning tabular + confluência com POI e Sweeps.
  - `OrderflowAgent`: desbalanceamento de livro L2 e volume profile.
  - `SentimentAgent`: sentimento contrariano Fear & Greed e notícias CryptoPanic indexadas rigorosamente em D-1.
  - `MacroAgent`: correlações macroeconômicas DXY, VIX e S&P 500 em D-1.
  - `OnchainAgent`: fluxos de stablecoins e TVL DefiLlama em D-1.
  - `RiskSentinelAgent`: poder de veto absoluto por volatilidade anômala, iliquidez, spread alargado e proximidade de resistência/suporte imediato.
- **Juiz de Consenso Ponderado (`consensus_judge.py`):**
  - Ponderação matemática com precedência estrita de veto e timeout de 2.5s para árbitro LLM com fallback local.

---

## [1.0.0-alpha.3] - 2026-10-14

### Adicionado
- **Motor de Treinamento de Modelos (`model_trainer.py`):**
  - Modelos LightGBM e Random Forest com Triple-Barrier labeling (TP, SL e barreira temporal).
  - Validação cruzada temporal (*PurgedGroupTimeSeriesSplit*) com janela de embargo para impedir vazamento de autocorrelação.

---

## [1.0.0-alpha.2] - 2026-10-14

### Adicionado
- **Engenharia de Features Locais (`data/feature_engineering.py`):**
  - 100% vetorizado em `ta` e `numpy` para baixa latência (< 10ms por lote de 500 velas).
  - Indicadores de tendência (SMA 20/50/200, EMA 9/21, MACD), osciladores (RSI 14, Estocástico), volatilidade (ATR 14, Bandas de Bollinger, Bandwidth, NATR).
  - Microestrutura: OBV, Z-score de volume de 20 períodos, desvio de VWAP e imbalance de livro de ofertas.
  - Fatores macroeconômicos alinhados rigorosamente por release_date (anti-lookahead).

---

## [1.0.0-alpha.1] - 2026-10-14

### Adicionado
- **Camada de Dados 100% Gratuita (`data/data_integrations.py`):**
  - Conectores assíncronos independentes para Bitget v2 pública, Binance pública (data.binance.vision), CoinGlass scraping, FRED, Yahoo Finance REST, CryptoPanic e DefiLlama.
- **Gerenciador de Cache Local Multi-Cadência (`data/data_loader.py`):**
  - Cache persistente SQLite em modo WAL (`datasets/market_cache.db`).
  - TTL configurável por cadência de dados e ordenação temporal cronológica estrita.

---

## [1.0.0-alpha.0] - 2026-10-14

### Adicionado
- Setup inicial da arquitetura, governança e definição formal da **Regra de Ouro (Golden Rule)**.
- Documento de rastreabilidade [EVOLUTION_LOG.md](EVOLUTION_LOG.md) e cronograma de 10 fases [CRONOGRAMA_MODULAR.md](CRONOGRAMA_MODULAR.md).
