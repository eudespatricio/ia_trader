# PARIDADE BACKTEST × PILOTO (paper) — "laranjas com laranjas"

> Auditoria **somente leitura** (nenhum arquivo de produção alterado). Gerado por inspeção de código +
> harness executável [`scripts/parity_audit.py`](../scripts/parity_audit.py) + testes
> [`tests/parity/`](../tests/parity/test_backtest_vs_paper_parity.py).
> Linhas são de um **snapshot** (outro agente estava editando `smart_trade.py` / `run_paper_simulation.py`
> durante a auditoria — a ladder E7 e o scalp-por-comitê apareceram no meio dela); funções são a âncora estável.
> **Carry/funding-carry: STAND-BY (aceito por design, fora de escopo).** Funding aplicado a posições
> direcionais/scalp **não** é carry e está no escopo.

## 0. Veredito (cínico)

1. **Hoje o piloto NÃO testa o que o backtest validou.** O que o backtest "validou" é (a) um comitê *degradado* (3 de 6
   agentes sempre HOLD, vetos inertes, sem ML) com bracket único SL 1.0/TP 1.5 ATR + timeout 48 barras, e
   (b) mesmo assim concluiu **sem edge** (`AB_CONSENSO_RESULTADOS.md`: hit 39,8% vs 40% de equilíbrio, expectância líquida
   −0,175%/trade, Sharpe ≈ −5). O piloto executa **outra estratégia de saída** (ladder 40/30/30 + BE + trailing, alavancagem ≤5x,
   liquidação, maker/taker por saída) que **nunca passou por nenhum backtest** com a mesma semântica.
2. A única paridade real é **entrada→decisão do comitê em modo PROD** (A/B foi construído como réplica do piloto): mesmos agentes,
   mesmo contexto degradado, mesmo juiz (0.30, pesos default), mesmo gate 0.70. **Mas paridade de um comitê quebrado não é
   funcionalidade**: Sentiment/Macro/OnChain e vetos de volatilidade/iliquidez são inertes **nos dois lados**.
3. Resultados de `run_phase4_*` (ML LGBM walk-forward, F&G/DXY/VIX/stablecoins D-1, pesos 1.0, sem gate 0.70, horizon 6)
   **não** descrevem o piloto: ML, contexto macro e semântica de pesos **não existem** em produção.
4. Harness (forced, BTC/ETH/SOL, 1.200 barras 1h, mesmas velas e mesmos sinais): Σ retorno%-do-notional
   BTC bt −0,86% × pp −6,73%; ETH −3,81% × −11,77%; SOL −0,61% × −1,84%. Divergência sistemática (piloto pior) decorrente de
   ladder+BE (corta vencedores em BREAKEVEN_STOP), slippage perp 0,03% e entrada no close. **Nenhum trade bate em tudo.**

## 1. Matriz de paridade funcionalidade-a-funcionalidade

Status: **IGUAL** · **DIVERGENTE** · **AUSENTE no piloto** · **AUSENTE no backtest** · **AMBOS ausentes (inerte)** ·
**STAND-BY** (aceito). Impacto: 🔴 alto (invalida inferência) · 🟠 médio · 🟢 baixo.
Backtests de referência: **A/B** = `scripts/ab_consensus_experiment.py`; **P4** = `run_phase4_evaluation.py`/`_holdout_and_reports.py`;
**BT** = `backtesting.py::BacktestEngine`; **PP** = `smart_trade.py::PaperTradingEngine` + `scripts/run_paper_simulation.py`.

### 1.1 Geração de sinal

| # | Funcionalidade | Backtest (evidência) | Piloto (evidência) | Status | Impacto |
|---|---|---|---|---|---|
| S1 | Comitê (swing, 1h) | A/B replica `core.evaluate_market_deterministic` (`ab_consensus_experiment.py` L367-L417) com `ConsensusJudge(0.30)`, pesos default, gate 0.70 (L74) | `evaluate_swing_opportunities` → `core.evaluate_multi_strategy` → `evaluate_market_deterministic` (`run_paper_simulation.py` L561-L616; `core.py` L76-L124), gate `SWING_MIN_COMMITTEE_CONFIDENCE`=0.70 | **IGUAL** (A/B) / DIVERGENTE (P4: sem gate, pesos 1.0, `min_consensus` igual) | 🟠 |
| S2 | Comitê (GP / scalp, 15m) | **Nenhum** backtest de comitê em 15m; `eval_surgical_scalp.py` não usa agentes | `_committee_gate` → `core.evaluate_market_async` (`run_paper_simulation.py` L321-L370; `core.py` L126-L192) | **AUSENTE no backtest** | 🔴 |
| S3 | Contexto Technical | P4: `proba_buy/proba_sell` OOS (L184-L196, L298); A/B: só features (CTX_COLS L349) | Só features (`latest.to_dict()`, L589) | A/B **IGUAL**; P4 **DIVERGENTE** | 🔴 |
| S4 | Contexto Sentiment/Macro/OnChain | P4: F&G, DXY, VIX, stablecoins D-1 (L276-L306); A/B mundo PROD: vazio; mundo FULL: enriquecido (L449-L459) | Vazio → sempre HOLD "sem dados" | A/B-PROD **IGUAL (ambos inertes)**; P4/A-B-FULL **AUSENTE no piloto** | 🔴 |
| S5 | `volume_sma` / `atr_median` (vetos VOLATILITY_SPIKE, ILLIQUIDITY) | P4: `volume_sma`=sma20, `atr_median`=atr_14 próprio (ratio=1, **veto nunca dispara**, L295-L297); A/B-P2 mapeia real (L229-L231) mas P2 não adotado | Ausentes do contexto (`risk_sentinel_agent.py` L43,L60) → vetos inertes | **AMBOS ausentes (inerte)** | 🔴 |
| S6 | Veto SCHEDULED_MACRO_EVENT (`is_cpi_fomc_window`) | P4 L305 (calendário CPI/FOMC) | Nunca setado | **AUSENTE no piloto** | 🟠 |
| S7 | Modelo ML (LGBM) | P4: treino walk-forward purgado (`model_trainer.py`, L153-L196), threshold 0.55 | `TradingCore()` → `TechnicalAgent()` sem modelo; `models/*.joblib` nunca carregados (só `ModelTrainer.load_model` em teste) | **AUSENTE no piloto** | 🔴 |
| S8 | Fallback sem modelo (regras RSI/SMA200/POI) | A/B usa como baseline ("réplica fiel") | Idem | **IGUAL** | 🟢 |
| S9 | Pesos do juiz | P4: `step_weights`=1.0 por agente ativo (L259-L260, L315); A/B/piloto: default 1.0/1.0/0.8/0.6/0.6/0.5 (`consensus_judge.py` L53-L60) | default | P4 **DIVERGENTE**; A/B **IGUAL** | 🟠 |
| S10 | Regime / calibração de pesos (LearningTracker, `get_calibrated_weights_for_regime`, `MarketRegimeClassifier`) | Nenhum backtest usa | `record_decision` nunca é chamado e nenhum caminho de decisão lê pesos calibrados; só `update_trade_outcome` em `close_position` | **AMBOS ausentes (inerte)** — mas o piloto grava "Pesos atualizados" (ruído enganoso) | 🟠 |
| S11 | `SignalGenerator` SL/TP assimétrico por POI/sweep (`structure_context`) | Nunca passado | Nunca passado (`core.py` não repassa) | **AMBOS ausentes (inerte)** | 🟢 |
| S12 | Gatilho de oportunidade scalp | `eval_surgical_scalp.py` (volZ≥1.8, VWAP≥1.2%, RSI 32/68, funding z, maker) | `simulation_settings` (VWAP≥2.0%, RSI 35/65, volZ≥0.2) **+ comitê E7** | **DIVERGENTE** (nunca backtestado) | 🔴 |
| S13 | Gatilho GP v2 (VWAP 2.5/RSI 38-62 ou sweep+POI, faca-caindo, cooldown 16, 1/vela) | Nenhum script de validação encontrado | `evaluate_golden_pot_opportunities` | **AUSENTE no backtest** | 🔴 |
| S14 | Anti-lookahead macro | A/B: disponibilidade = data+24h (`load_daily_context` L314-L346); P4: D-1; `data_integrations` com `release_date` | Sem macro no contexto (nada a vazar, nada a testar) | **AUSENTE no piloto** (funcionalidade) | 🟠 |

### 1.2 Dados, janelas, timing

| # | Funcionalidade | Backtest | Piloto | Status | Impacto |
|---|---|---|---|---|---|
| D1 | Janela de histórico p/ features | Histórico completo (A/B L426; P4 `load_candles`) | 400 velas (`SWING_CANDLE_LIMIT`; GP/scalp 400×15m) | **DIVERGENTE** — efeito medido pequeno (teste verde `test_feature_window_effect_400_vs_full_is_small`, ≤5% em atr/rsi/sma200/vwap) | 🟢 |
| D2 | Velas fechadas | A/B descarta a última barra (L291); BT recebe o que o chamador passa | GP/scalp descartam vela em formação (`_fetch_closed_candles_15m` L307-L316); **swing NÃO** (usa `df.iloc[-1]` direto, L582) | **DIVERGENTE (swing)** — decisão em vela aberta | 🔴 |
| D3 | `update_active_positions` | Barra 1h completa com high/low finais | Poll 15m `limit=5`, usa `iloc[-1]` (vela em formação, L138-L156); high/low parciais acumulados | **DIVERGENTE** (mais fiel ao live; pior reprodutibilidade) | 🟠 |
| D4 | Timeframe de decisão | 1h (A/B), 1h/4h (P4) | Swing 1h; scalp/GP 15m | scalp/GP **AUSENTE no backtest** | 🔴 |
| D5 | Entrada | **OPEN de t+1** (+slippage 0.02%) (`backtesting.py` L336-L350; A/B L495-L497) | **CLOSE da vela t** com ts da vela (A3) (`execute_directional_signal(current_price=close, timestamp=ts)`; `_open_ladder_position`) | **DIVERGENTE** (harness: `ENTRY_TIMING`/`ENTRY_PRICE` em 100% dos trades) | 🔴 |
| D6 | Reentrada na barra de saída (A10a) | Proibida (`closed_intrabar`, L329-L336) | Permitida (sinal no close de t pode reabrir) | **DIVERGENTE** (`UNMATCHED_*`) | 🟠 |

### 1.3 Execução e custos

| # | Funcionalidade | Backtest | Piloto | Status | Impacto |
|---|---|---|---|---|---|
| E1 | SL/TP fixo | SL 1.0 / TP 1.5 ATR, âncora = preço de entrada, ATR[t] (`backtesting.py` L343-L348) | SL do comitê (close[t]±1.0 ATR) ou perfil; TP1/TP2/TP3 = 1.5/2.5/4.0 ATR (DIRECTIONAL), SCALP 1.0/1.8/3.0, GP 1.2/2.5/6.0 (`STRATEGY_PROFILES`, `_open_ladder_position`) | **DIVERGENTE** — TP único 1.5 ATR **não existe** no piloto | 🔴 |
| E2 | TP1/TP2/TP3 escalonado (40/30/30) + BE (+0.04%) | **AUSENTE** (`BacktestEngine` só bracket; `eval_staged_tp_trailing_swing.py` usa 25/25/50, 1.5/3.0 ATR, trail 2.5, 4h, regra de entrada própria) | `_process_ladder_bar` (L1365-L1435) | **AUSENTE no backtest** (e o "validado" tem parâmetros diferentes) | 🔴 |
| E3 | Trailing | `trailing_stop=False` em TODOS os experimentos; opção existe (`_update_trailing` L135-L146, mult 1.5) | Ativado no TP2, mult 0.8/1.2/1.5 por perfil, só barras fechadas | **AUSENTE no backtest (validado)** | 🔴 |
| E4 | Ordem intrabarra | SL antes de TP na colisão; saída SL gap-aware `min(open, sl)` (L172, L188) | Stop vigente antes de TP; BE após TP1 na mesma barra; SL fill = `min(sl, high)` (não usa o gap do open) (`_close_ladder_on_stop` L1351-L1363) | **IGUAL (política)**, **DIVERGENTE (gap)** | 🟠 |
| E5 | Timeout de posição | `max_holding_bars` (A/B 48, P4 6) → `HORIZON_TIMEOUT` (L271-L274) | **Inexistente** (posição pode ficar aberta indefinidamente) | **AUSENTE no piloto** | 🔴 |
| E6 | Taxas | Taker 0.06% entrada e saída (todas); cenário maker 0.02% **para tudo**, slip 0 (P4 L230-L242) | Entrada taker (swing) / maker (scalp, GP); TP maker 0.02%; SL/BE/TRAILING/MANUAL/LIQ taker (`close_position` L1196-L1207) | **DIVERGENTE** (fee/notional bt 0.119% × pp 0.104% no harness) | 🟠 |
| E7 | Slippage | 0.02% por fill, simétrico | Long 0.02% (spot), Short 0.03% (perp), maker 0.01%; assimetria arbitrária long×short | **DIVERGENTE** | 🟠 |
| E8 | Funding (direcional/scalp/GP) | Aplica 0.01%/8h fixo (`backtesting.py` L257-L263); A/B usa funding **real** (`funding_rates`, L294-L311) | **Não aplica** (`funding_rate_period` só no carry, L1534-L1540; daemon nunca passa) | **AUSENTE no piloto** (harness: 50/175 trades afetados) | 🟠 |
| E9 | Sizing | 1% de risco, cap `capital×lev` (≈ notional ≈ 100% do capital em 1x) (L353-L363) | Lote fixo = MARGEM $100 × alavancagem; teto por saldo (50%/25%/20%) | **DIVERGENTE — ACEITO POR DESENHO** (comparar % do notional) | 🟢 |
| E10 | Alavancagem | 1x em todos os experimentos | `decide_leverage` até 5x (`leverage_policy.py`), SL antes da liquidação | **AUSENTE no backtest** | 🔴 |
| E11 | Liquidação | **Inexistente** | `LIQUIDATION` a 1/lev−0.5% (`calc_liquidation_price`; tick L1495) | **AUSENTE no backtest** | 🔴 |
| E12 | Margem agregada / multi-ativo | Capital 10k **por símbolo**, MaxDD aditivo (A/B doc L28 admite) | Saldo único, teto 85% de margem, ≤5 posições | **DIVERGENTE** | 🟠 |
| E13 | Cotas diárias (5 scalps / 3 GP) | Nenhuma | `can_execute_scalp/golden_pot` | **AUSENTE no backtest** (swing: nenhuma em ambos) | 🟠 |
| E14 | Cooldown / 1 entrada por vela | Só "1 posição/símbolo" | GP 16 barras, scalp 4 barras; swing nenhum | **AUSENTE no backtest** | 🟠 |
| E15 | RollbackGuard / circuit breakers | **Inexistente** no BT/A/B/P4 | `can_open_new_position` só em scalp/GP; **DIRECTIONAL ignora** (`_open_ladder_position`: `can_exec=True`); `evaluate_telemetry` após cada fechamento | **AUSENTE no backtest**; **DIVERGENTE no piloto (direcional sem trava)** | 🔴 |
| E16 | Reversão de sinal | `allow_reversal=False` (todos) | Não existe | **IGUAL** | 🟢 |
| E17 | Fim de dados | `END_OF_DATA` a mercado | n/a | n/a | 🟢 |
| E18 | Carry / funding_carry | — | `execute_carry_allocation` (standby) | **STAND-BY (aceito por design)** | — |

## 2. Harness executável

```
.venv/bin/python scripts/parity_audit.py --bars 1500 --scenario forced      # sinais determinísticos (amostra garantida)
.venv/bin/python scripts/parity_audit.py --bars 1500 --scenario committee   # sinais do comitê PROD (gate 0.70)
.venv/bin/pytest tests/parity -q -rx
```

* Dados: `datasets/market_cache.db` com `mode=ro` (teste verde prova que escrita falha); 1h binance BTC/ETH/SOL; última barra descartada.
* Sinais: **um único vetor** passa por `BacktestEngine` (config validada: taker 0.06%, slip 0.02%, funding 0.01%, 1%/1x, 1.5/1.0 ATR, timeout 48) e
  por `PaperTradingEngine` (DB temporário, `restore=False`, sem rede) na sequência do daemon (tick → sinal → `execute_directional_signal` ao close, lev 1x).
* Reconciliação por barra do sinal; categorias em `CATEGORIES`; `ACCEPTED_BY_DESIGN = {SIZING}`; exit code ≠ 0 se houver divergência relevante.
* Resultado (forced, 1.200 barras): `ENTRY_TIMING` 175, `PNL_PCT` 106, `EXIT_REASON` 86, `EXIT_FEE_MODEL` 85, `ENTRY_PRICE` 72, `EXIT_TIMING` 59, `FUNDING` 50, `UNMATCHED_BT` 11, `SIZING` 175 (aceita).
* Testes: 7 verdes (inclui "sizing aceito", "BT não tem circuit breaker" documentado, janela 400×full ≤5%, pesos A/B = piloto) e 21 `xfail(strict=False)` com motivo —
  viram XPASS quando corrigidos (remover marcador).
* Limitações do harness (honestas): replay em 1h (o daemon poll em 15m); leverage 1x; não cobre scalp/GP (sem backtest de referência — é a conclusão, não omissão);
  `committee` usa contexto PROD, não FULL.

## 3. O que o backtest validou e NUNCA foi para produção

1. **ML LGBM walk-forward** (`model_trainer.py`, P4) — produção roda só regras; modelos `.joblib` são decorativos; retreino/challenger (`learning_tracker`) não é agendado no daemon.
2. **Contexto macro/sentiment/on-chain real** (F&G, DXY, VIX, stablecoins, CPI/FOMC) e anti-lookahead `release_date`.
3. **Vetos de volatilidade/iliquidez** com baselines reais (P2 do A/B) — e a conclusão do próprio A/B de que **nenhuma** variante P1–P6 tem alfa.
4. **Timeout de horizonte** (6/48 barras) e **funding**.
5. **Trend Diário com vol targeting (S3), momentum cross-sectional (S2)** (`trend_following.py`, `run_phase_r_experiments.py`) — a "virada institucional" do Parecer; piloto roda o comitê intradiário que o A/B reprovou. (Carry S1: stand-by.)
6. Pesos 1.0 por ablação e `min_confidence` 0.55 do TechnicalAgent (P4).

## 4. O que o piloto faz e NUNCA foi backtestado

Ladder 40/30/30 + BE +0.04% + trailing (0.8/1.2/1.5 ATR) · alavancagem ≤5x e liquidação · taxas maker/taker por saída ·
GP v2 (gatilho, faca-caindo, cooldown, exclusões) · scalp por comitê em 15m e seu gatilho (thresholds de `simulation_settings`) ·
cotas diárias, cooldowns, teto 85%/5 posições · RollbackGuard/telemetria · lote fixo em margem · polling 15m com vela parcial ·
modo DEMO (thresholds GP 1.2/42/58). **Qualquer PnL do piloto é, estatisticamente, um experimento novo (zero prior de backtest).**

## 5. Lista priorizada de correções

Legenda esforço: S ≤0,5 d · M 1-2 d · L 3-5 d. Risco = risco de regressão/invalidar pré-registro.
"Fonte de verdade" = lado cuja semântica prevalece.

| P | Item | Corrigir em | Fonte de verdade | Esforço | Risco |
|---|---|---|---|---|---|
| 1 | **Ladder E7 no `BacktestEngine`** (TP1/2/3 40/30/30, BE +0.04%, trailing só barras fechadas, maker/taker por saída, mesmos multiplicadores `STRATEGY_PROFILES`) e **re-rodar A/B/phase4** com ela | Backtest (aproximar do piloto) | **Piloto** (já em produção, pré-registrado E7) | L | Médio: resultados históricos mudam; exige novo `trial_log` |
| 2 | **Entrada**: decidir um único modelo. Recomendo piloto entrar no **OPEN da vela seguinte** (ou backtest entrar no close com slippage extra) — A3 já força `high=low=close` na vela da entrada | Piloto → open t+1 (preferível: backtest é o conservador/validado) | **Backtest** | M | Médio: muda timestamps/ledger, reinicia relógio do piloto |
| 3 | **Swing só em vela fechada** (descartar `iloc[-1]` em formação como GP/scalp) | Piloto | Backtest | S | Baixo |
| 4 | **Timeout de posição** no `PaperTradingEngine` (48 barras 1h DIRECTIONAL; definir para scalp/GP) | Piloto | Backtest | S-M | Baixo |
| 5 | **Funding** em direcional/scalp/GP: daemon passa funding real (tabela `funding_rates` / Bitget) a `update_market_tick`; `_update_market_tick_locked` aplica a posições LONG/SHORT a cada 8h | Piloto | Backtest (funding real, não 0.01% fixo) | M | Baixo |
| 6 | **Circuit breaker no DIRECTIONAL** (`can_execute_*` equivalente) **e** modelar RollbackGuard/cotas/cooldown no backtest (ou declarar explicitamente que não se valida risco) | Ambos | Piloto | M-L | Médio |
| 7 | **Alavancagem + liquidação** no `BacktestEngine` (ou fixar piloto em 1x até haver backtest) | Backtest (ou Piloto 1x) | Piloto | M | Médio |
| 8 | **Taxas/slippage unificados**: slippage long=short (0.02%/0.03% perp?) em ambos; maker 0.02% só nas saídas limite | Ambos | Piloto (custos reais Bitget) | S | Baixo |
| 9 | **Contexto do comitê**: injetar `volume_sma`, `atr_median` (A/B-P2), F&G/DXY/VIX/stablecoins (D-1, `release_date`) no daemon, OU desligar explicitamente Sentiment/Macro/OnChain do juiz (P1a) — hoje 3 agentes só diluem consenso | Piloto | Backtest (A/B-FULL) | M | **Alto**: muda o sinal; novo pré-registro |
| 10 | **ML**: decidir — conectar `TechnicalAgent(model=...)` com features idênticas e inferência pós-treino OOS, ou declarar P4 inválido p/ o piloto | Piloto (ou doc) | Backtest P4 | L | Alto |
| 11 | Unificar **pesos/gate**: phase4 com pesos default + gate 0.70, ou documentar P4 como "ablação teórica" | Backtest | Piloto | S-M | Baixo |
| 12 | Backtest de **scalp/GP em 15m com comitê** + gatilhos reais (o piloto roda sem prior) | Backtest (novo script) | Piloto | L | Baixo (só mede) |
| 13 | Margem agregada / saldo único / 5 posições no backtest multi-ativo (MaxDD composto) | Backtest | Piloto | M | Baixo |
| 14 | SL gap-aware no piloto (`fill = min(sl, open)` quando `open` gapeou); poll 15m: processar só barras fechadas | Piloto | Backtest | S | Baixo |
| 15 | Remover/atualizar log "Pesos atualizados" (LearningTracker inerte) e `structure_context` morto | Piloto | — | S | Baixo |

**Recomendação estratégica (cínica):** antes de investir nos itens 1-14, aceitar que o baseline que se quer "paridar" **não tem edge**.
Paridade só vale como pré-condição para *medir* qualquer estratégia candidata (S3 trend, S2 momentum) — prioridade real é
(1)+(2)+(5)+(6) para que o piloto passe a ser um *replicador fiel* do `BacktestEngine`, e só então trocar a estratégia.
