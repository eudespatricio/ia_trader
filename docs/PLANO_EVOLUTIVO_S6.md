# Plano Evolutivo S6 — Pacote Evolutivo Contínuo

**Data de abertura:** `2026-11-23`  
**Status:** **Onda S6.3 (`v1.0.14`, `2026-11-25`) — Frente B ATIVA em produção (`microstructure_collector` + `paper_trading_daemon` reiniciados) & Frente A `T11` Backfill/OOS concluída (`NO-GO` Prudent Hold)**  
**Regra Mandatória:** *Toda alteração de lógica ou filtro exige prova empírica zero-leakage em OOS (custos estresse + embargo) antes de ir para produção, testes independentes em janela limpa e revisão cross-model.*

---

## 1. Onda S6.1 (`v1.0.12`) — Entregas Aprovadas (`GO`)

### Item `S6-2` — `GOLDEN_POT`: Filtro Dinâmico de Perps TradFi / Não-Cripto (`exchangeInfo` + Fallback Estático) **[IMPLEMENTADO]**

- **Diagnóstico ao Vivo (`2026-11-23 12:17 BRT`):**
  - A lista estática `GP_EXCLUDED_BASE_ASSETS` (`42` tickers) não acompanhou novas listagens TradFi/Stock Perps na Binance Futures e Bitget: o endpoint `/fapi/v1/exchangeInfo` marca **`253` contratos não-cripto** (`EQUITY`, `KR_EQUITY`, `CN_EQUITY`, `HK_EQUITY`, `COMMODITY`, `PREMARKET`, `INDEX`, `underlyingSubType=['TradFi']`), dos quais `28+` vazavam para a tabela `golden_pots` e para avaliação 5m do daemon (`MUUSDT`, `MUUUSDT`, `ANTUSDT`, `DRAMUSDT`, `EWYUSDT`, `NBISUSDT`, `ZHIPUUSDT`, `ZHONGJIUSDT`, `BEUSDT`, `MRVLUSDT`, `NVDLUSDT`, `CRCLUSDT`, `IRENUSDT`, `NATGASUSDT`, `CBRSUSDT`, `SPCXUSDT`, `MINIMAXUSDT`, etc.).
- **Implementação Cirúrgica:**
  1. [golden_pot_agent.py](file:///home/eudespatricio/ia_trader/agents/golden_pot_agent.py#L47-L182): Fonte única `NON_CRYPTO_BASE_EXCLUSIONS` (`85` ativos-base estáticos, preservando tokens `COIN`/`RWA` cripto como `PAXG`, `XAUT`, `MANTRA`, `CFG`, `AIGENSYN`), classificadores `extract_base_asset()` e `classify_non_crypto_from_exchange_info()`, cache dinâmico via `/fapi/v1/exchangeInfo` com TTL `6h` e backoff de falha `5m` em `refresh_non_crypto_symbols()`, e filtro `is_non_crypto_symbol()` aplicado tanto no loop Binance quanto no fallback Bitget em `fetch_market_universe_tickers()`.
  2. [run_paper_simulation.py](file:///home/eudespatricio/ia_trader/scripts/run_paper_simulation.py#L159-L162): `GP_EXCLUDED_BASE_ASSETS = NON_CRYPTO_BASE_EXCLUSIONS` (eliminando duplicação) e `_is_gp_symbol_excluded()` na entrada de `evaluate_golden_pot_opportunities()`.

---

## 1-B. Onda S6.2 (`v1.0.13`, `2026-11-24`) — Destravamento Pós-S5 (`GO`)

### Item `S6-3` — Anti-Deadlock de Calibração por Regime: Isolamento por Época (`pilot_start_utc`) + TTL `72h` ([`learning_tracker.py`](file:///home/eudespatricio/ia_trader/learning_tracker.py)) **[IMPLEMENTADO]**

- **Diagnóstico (`2026-11-24 13:43 BRT`):** `agent_regime_weights` e `agent_regime_calibration` ainda carregavam `n=16` trades pré-S5 em `BEAR_TREND` (último em `19/11 19:00 BRT`, `TechnicalAgent` e `OrderFlowAgent` com `acc=31.25%`). Com `acc < 35%`, `ConsensusJudge._apply_confidence_calibration()` travava ambos em `0.55`, gerando teto matemático de `0.65–0.698 < 0.70` mesmo em unanimidade (`SOLUSDT 12:00 BRT`: `4/5 SELL` bloqueado em `0.6537`, `32` `reject:low_confidence` em `DAY_TRADE` nas últimas 25h) — um **deadlock auto-travante** sem recuperação possível sem abrir novos trades em `BEAR_TREND`.
- **Prova Empírica Zero-Leakage (`TRIAL_S6_04`):**
  - **Replay Live 24h (`23/11 13:00 → 24/11 12:00 BRT`, `16 bps` RT, `SL 1.0x / TP 1.5x ATR`):** `V0_PROD` (`1` sinal `>=0.70`, `30` `low_conf`, `+1.38R`) -> **`V1_CLEAN_EPOCH`** (`pilot_start_utc` preservando `S5-1`: **`27` sinais `>=0.70`, `0` `low_conf`, `6W/1L/2open`, `+8.10R`**). No candle `SOLUSDT 12:00 BRT`, consenso saltou de `0.6537` para **`0.7936`** (`-0.53 <= -0.30 | Unânime`).
  - **Simulação Causal em Loop Fechado (`2022–2026`, `64.260` barras `2h`, `IS 2022–24 | 7d | OOS 2025–26`, `16 bps` RT):**
    - `V0_PROD`: deadlock permanente de **`1.673,9` dias** (`OOS n=0`, `5.360` bloqueios em `BEAR_TREND`).
    - **`V1_TTL_72H` (Vencedor):** Limita seca máxima em `BEAR_TREND` a **`3.0d`** e entrega **melhor `PF` OOS em `BEAR_TREND` (`PF 0.77`, `WR 41.3%`, `-62.3R`)** vs `48h` (`PF 0.65, -105.5R`), `120h` (`PF 0.49, -78.1R`), `168h` (`PF 0.69, -44.0R`), **`V2_TTL72_UNAN3` (`NO-GO`: `PF 0.69, -112.0R`)** e **`V3_NO_DAMPEN_TREND` (`NO-GO`: `PF 0.75, -297.1R`)**.
- **Implementação Cirúrgica:** Apenas em [`learning_tracker.py`](file:///home/eudespatricio/ia_trader/learning_tracker.py) (mantendo `consensus_judge.py` e `recalibrate_structural_sub_rules()` 100% intactos): constante `CALIBRATION_STALE_TTL_MS = 72 * 3_600_000`, isolamento estrito `max(s3_epoch_ms, pilot_start_utc)` sem fallback pré-piloto, quebra de cluster por janela de silêncio `>= 72h`, expiração read-time em `get_calibrated_weights_for_regime()`/`get_agent_calibration_for_regime()` (`total_samples=0, stale_expired=True`) e fechamento explícito `with closing(...)` (`0` vazamento de FD).

### Item `S6-4` — Promoção do Baseline ML `1h` Multi-Symbol + `Platt` (`LGBM` & `RF`) **[IMPLEMENTADO]**

- **Diagnóstico:** Os Champions legados (`rf` `08/11`, `lgbm` `17/11`, BTC-only sem calibração) não possuíam `log_loss`/`roc_auc_ovr` em `metrics`, fazendo `evaluate_challenger_joint()` cair no fallback de `emitted_signals_accuracy` e rejeitar os Challengers calibrados multi-symbol no cron de 24h (`RETAINED -7.18%`).
- **Prova Empírica Head-to-Head OOS (`TRIAL_S6_05`):**
  - **OOS 2 Anos Purgado (`IS <= 2024 [n=99.498] | 7d | OOS >= 2025-01-08 [n=49.353]`):** `LGBM` `LogLoss 1.0760 -> 1.0733` (`AUC 0.5847 -> 0.5870`, `emAcc 56.8% -> 57.8%`); `RF` `LogLoss 1.0764 -> 1.0753` (`AUC 0.5856 -> 0.5852`).
  - **Cauda Inédita Pós-Champions (`>= 2026-11-18, n=465`):** `LGBM` `LogLoss 1.0837 -> 1.0752` (`AUC 0.5691 -> 0.5745`); `RF` `LogLoss 1.0790 -> 1.0720` (`AUC 0.5817 -> 0.5809`).
- **Execução Ao Vivo (`force_promote=True` pós-aprovação OOS):** `LGBM` (`LL=1.0746, AUC=0.5871, ECE=0.0097`, backup `lgbm_model_v1_20261124T174515Z.joblib`) e `RF` (`LL=1.0751, AUC=0.5869, ECE=0.0093`, backup `rf_model_v1_20261124T174559Z.joblib`) promovidos via `CalibratedMulticlassModel(method="platt")` sobre `89.772` amostras (`BTC+ETH+SOL`), destravando o gate conjunto `evaluate_challenger_joint()`.

---

## 2. Itens Investigados & Refutados em OOS (**NO-GO** — Manter Produção Intacta)

### Item `S6-1a` — Remoção/Relaxamento do `_committee_gate` e Teto `RSI > 75` no `SCALP 15m` **[NO-GO]**

- **Motivação:** Durante o pump de `2026-11-23 13:45Z` (`BTC +4.18%`, `ETH +4.97%`, `SOL +5.54%`), o gatilho Squeeze `15m` disparou `BUY`, mas `ETH`/`SOL` foram barrados pelo ML `15m < 45%` e `BTC` foi barrado pelo comitê (`RSI=84.2 > 75` neutralizou `TechnicalAgent`, enquanto `Macro+OnChain` votaram `SELL`). *(Nota live: entrar em `BTC` às `13:45Z` em `63,833.30` teria acionado `SL 63,276.44` na barra seguinte `14:00Z`.)*
- **Ablação Purgada Zero-Leakage (`98.116` barras `15m` BTC+ETH+SOL, custo estresse `16 bps` RT, `SL 1.4x / TP 2.2x`, split `2026-07-29`, embargo `10` barras, LGBM+RF purgado em `65.683` linhas IS):**

| Variante Testada | Full Squeeze IS (`N=174`) | Full Squeeze OOS (`N=87`) | Squeeze + `ML15m >= 0.45` OOS (`N=11`) | Veredicto |
|---|---|---|---|---|
| **Produção (`6-ag conf>=0.70, RSI<=75`)** | `n=66, PF 0.64, -17.67R` | **`n=37, WR 59.5%, PF 1.56, +10.30R (+0.278R)`** | **`n=4, WR 50.0%, PF 1.18, +0.41R`** | **MANTER** |
| `V0b` Produção `conf >= 0.55` | — | — | `n=5, WR 40.0%, PF 0.78, -0.75R` | **NO-GO** |
| `V1` Sem teto `RSI > 75` (`6-ag`) | — | — | `n=5, WR 40.0%, PF 0.77, -0.80R` | **NO-GO** |
| `V2b` Intraday `Tech+OF` com `RSI<=75` (sem Macro/OC) | `n=74, PF 0.61, -20.70R` | `n=31, WR 54.8%, PF 1.26, +4.52R` | `n=2, WR 100%, +2.60R` | **NO-GO** (corta `>50%` do OOS) |
| `V2` Intraday `Tech+OF` sem `RSI` (sem Macro/OC) | `n=87, PF 0.59, -26.05R` | `n=36, WR 50.0%, PF 1.06, +1.32R` | `n=6, WR 50.0%, PF 1.15, +0.52R` | **NO-GO** |
| `V3b` `RiskOnly + RSI<=75/25` | `n=98, PF 0.56, -32.06R` | `n=48, WR 56.2%, PF 1.38, +9.64R` | `n=4, WR 75.0%, PF 3.02, +2.58R` | Inferior no Full Squeeze (`PF 1.38 < 1.56`) |
| `V3` `RiskOnly` (sem comitê, sem RSI) | `n=111, PF 0.53, -39.94R` | `n=53, WR 52.8%, PF 1.21, +6.43R` | `n=9, WR 44.4%, PF 0.89, -0.66R` | **NO-GO** |
| Isolado: `RSI > 75 (BUY) / < 25 (SELL)` | `n=20, PF 0.74, -3.82R` | `n=9, WR 33.3%, PF 0.59, -2.86R` | `n=5, WR 20.0%, PF 0.31, -3.24R` | **Confirmado nocivo** |
| Isolado: `RSI 25..75` normal | `n=154, PF 0.60, -47.40R` | `n=78, WR 52.6%, PF 1.18, +8.04R` | `n=5, WR 80.0%, PF 4.02, +3.86R` | Cap protetor validado |

### Item `S6-1b` — Relaxamento do Teto `RSI > 75 / < 25` no `TechnicalAgent` (`DAY_TRADE 1h`) **[NO-GO]**

- **Backtest 6 Anos (`2020–2026`, `BTC+ETH+SOL`, IS `<=2024` | embargo `7d` | OOS `>=2025`, `SL 1.0x / TP 1.5x`, `16 bps` RT):**
  - Trades isolados em extremo (`RSI > 75` acima da SMA200 / `RSI < 25` abaixo da SMA200): IS `n=5.337, PF 0.928, -238.0R` | **OOS `n=2.143, WR 39.6%, PF 0.730, -395.6R` (`-0.1846R`/trade)**.
  - Relaxar o teto `25..75` para `20..80`, `15..85` ou `0..100` degradou o OOS de `-3.894,3R` para `-4.072,7R` / `-4.103,0R` / `-4.076,2R`. Trava mantida intacta.

### Item `S6-5a` — Ablação OOS `SCALP 15m` (`S0..S7`, `TRIAL_0176..0183`) **[NO-GO]**

- **Escopo ([eval_scalp_s6_oos.py](file:///home/eudespatricio/ia_trader/scripts/eval_scalp_s6_oos.py), [s6_scalp_oos_ablation.json](file:///home/eudespatricio/ia_trader/datasets/reports/s6_scalp_oos_ablation.json)):** `119,1d` OOS (`2026-07-29..11-25`, `BTC/ETH/SOL`, entrada **taker**, modelos `15m`/`1h` purgados pré-split, `learning_tracker=None`) vs `DIRECTIONAL 1h` (`4,30` trades/dia, `n=512`, WR `39,1%`, PF `0,80`, `-74,96R`).
- **Resultados:** Todas as 8 variantes fecharam **`NO-GO`** (`DSR ≈ 0`): `S0_PROD` (`n=11, 0,09/d, PF 3,39` -> `n<15`), `S1_NO_ML15` (`n=37, 0,31/d, PF 1,47, +9,17R`, `IS PF 0,74`), `S2_ML15_DIR_ONLY`/`S3_ML15_THR38` (`n=28, 0,24/d, PF 1,62`, `IS PF 0,99`), `S4_SQZ1B_NO_ML15` (`n=43, 0,36/d, PF 1,57`, `IS PF 0,68`), `S5_SQZ2B_RELAX_VOL_NATR` (`n=51, 0,43/d, PF 1,32`, `IS PF 0,78`), `S6_MOM20_H1_NO_ML15` (`n=191, 1,60/d, PF 0,57, -68,51R`), `S7_META_LABEL_SQZ` (`n=13, 0,11/d, PF 1,96, IS PF 1,24` -> `n<15`). A trava live de mesmo par com `DIRECTIONAL` corta `~53%–73%` dos scalps (`S1: 37→13`).

### Item `S6-5b` — Ablação OOS `GOLDEN_POT 5m` (`G0..G7`, `TRIAL_0184..0191`) **[NO-GO]**

- **Escopo ([eval_golden_pot_s6_oos.py](file:///home/eudespatricio/ia_trader/scripts/eval_golden_pot_s6_oos.py), [s6_golden_pot_oos_ablation.json](file:///home/eudespatricio/ia_trader/datasets/reports/s6_golden_pot_oos_ablation.json)):** `220` altcoins cripto puras (excluídas `54` TradFi: `32` canônicas + `22` extras de `/exchangeInfo`), `597` segmentos contíguos (`359.465` barras), `20,1d` OOS (`2026-11-05..11-25`, sendo `19,19d` em `OOS_BEAR`) vs `DIRECTIONAL 1h` (`4,77` trades/dia, `n=96`, WR `50,0%`, PF `1,49`, `+26,20R`).
- **Resultados:** Todas as 8 variantes fecharam **`NO-GO`**: `G0_PROD` (`n=54, 2,69/d [4,62 uncap], PF 1,01, IS PF 0,78`), `G1_RELAX_CLIMAX` (`n=59, 2,93/d [9,40 uncap], PF 1,01`), `G2_NO_CLIMAX` (`PF 0,85, -3,60R`), `G3_MACRO_BTC_35`/`G4_NO_MACRO_BTC`/`G5`/`G6_INTRADAY` (`PF 0,91`), `G7_EARLY_TRIGGER` (`n=60, 2,98/d [12,88 uncap], WR 66,7%, PF 1,09, +1,98R`, mas `Bear PF 0,97 < 1,25`). **Caveat:** chunks `5m` de ~400 barras pós-seleção do scanner no DB + ausência do cap Top-6 e `_flow_gate`.

---

## 3. Backlog Evolutivo S6 (Próximas Ondas — Aguardando Coleta / Execução Dedicada)

### Onda S6.3 Executada (`2026-11-25`) — Frente B `[IMPLEMENTADA / GO]` & Frente A `T11` `[BACKFILL OK / PROMOÇÃO NO-GO HOLD]`

- **Frente B (`TapeReadingAgent` Fallback REST + Cobertura L2 GP `top_n=6`):**
  - [agents/tape_reading_agent.py](file:///home/eudespatricio/ia_trader/agents/tape_reading_agent.py): adicionados `LIVE_FALLBACK_TIMEOUT_S = 2.5`, `LIVE_FALLBACK_CACHE_TTL_MS = 10_000` e `LIVE_FALLBACK_ERR_COOLDOWN_MS = 15_000`; fechamento explícito da conexão SQLite RO antes da chamada REST; acionamento de `_evaluate_live_binance_fallback()` apenas quando `is_live_call=True` em `book_stale`, `tick_stale`, `tick_insufficient` (`n_min`/`n_trades`/`sat_ratio > 0.50`) e `tick_zero_volume` (mantendo zero rede em backtest determinístico).
  - [scripts/microstructure_collector.py](file:///home/eudespatricio/ia_trader/scripts/microstructure_collector.py): `DEFAULT_GP_TOP_N = 6` e `MAX_ACTIVE_SYMBOLS = 16`, deduplicação case-insensitive e união com `NON_CRYPTO_BASE_EXCLUSIONS` (`log.error` explícito se falhar import).
  - **Validação Frente B:** `96 passed` ([test_tape_reading_gate.py](file:///home/eudespatricio/ia_trader/tests/test_tape_reading_gate.py) + [test_patch_s6_onda3_tape_fallback.py](file:///home/eudespatricio/ia_trader/tests/test_patch_s6_onda3_tape_fallback.py)).
- **Frente A (`T11` Expansão Majors `3 -> 8`: Backfill + Validação OOS Zero-Leakage, `TRIAL_0196..TRIAL_0202`):**
  - **Backfill A1 ([backfill_candidate_majors.py](file:///home/eudespatricio/ia_trader/scripts/backfill_candidate_majors.py)):** `52.828` velas `1h` (`2020-11-15..2026-11-25`), `40.338` velas `15m` (`2025-10-01..2026-11-25`) e `6.604` `funding_rates` para `XRPUSDT, NEARUSDT, DOGEUSDT, LINKUSDT, AAVEUSDT`, após backup online + `--dedup` (`30` duplicatas `NEARUSDT:15m` Bitget removidas, `0` gaps, `PRAGMA quick_check = ok`).
  - **OOS Zero-Leakage ([majors_s6_sim.py](file:///home/eudespatricio/ia_trader/scripts/majors_s6_sim.py), [eval_majors_expansion_s6_oos.py](file:///home/eudespatricio/ia_trader/scripts/eval_majors_expansion_s6_oos.py), [s6_majors_expansion_oos.json](file:///home/eudespatricio/ia_trader/datasets/reports/s6_majors_expansion_oos.json)):** zero `.joblib`, paridade causal `enricher=0.0` / `ensemble=0.0` (`n=25`), `corr_oos` vs BTC `0.639..0.836`, `beta_30d` `1.00..1.81`:

| Estratégia / Janela | `M0_BASE_3` (`BTC/ETH/SOL`) | `M1_TIER1_8` Melhor Config | Veredicto Gate A3 |
|---|---|---|---|
| `DIRECTIONAL 1h` (`>= 2025-01-08`) | `n=2844, PF 0.798, -406.34R, DD 99.02%` | `base3_cc2_nr2_beta`: `n=2583, PF 0.855, -255.51R, DD 94.93%, DSR 0.000` | **`NO-GO`** (`TRIAL_0197`) |
| `SWING_4H` Primária (`>= 2025-01-08`) | `n=190, PF 1.478, +35.98R, DD 12.06%` | `cc4_nrnone_unit`: `n=381, PF 1.340, +54.26R, DD 21.75%, DSR 0.650` | **`NO-GO`** (`TRIAL_0201`, `cc3_nr3_beta +35.51R`) |
| `SWING_4H` Paridade (`>= 2025-11-20`) | `n=99, PF 1.220, +9.67R, DD 11.76%` | `cc4_nrnone_unit`: `n=207, PF 1.078, +7.42R, DD 21.75%, DSR 0.310` | **`NO-GO`** (`XRP/DOGE/LINK/AAVE < 1.0`) |
| `SCALP 15m` Standalone (`>= 2026-07-29`) | `n=11, PF 3.393, +8.09R, DSR 0.954` | `base3_cc2_nrnone`: `n=33, PF 1.891, +13.61R, DD 3.68%, DSR 0.917` | **`NO-GO`** (`TRIAL_0202`, `XRP 0.58 / AAVE 0.76`) |
| `SCALP 15m` com `dirlock` (`>= 2026-07-29`) | `n=6, PF 6.414, +5.98R, DSR 0.516` | `base3_cc2_nrnone`: `n=18, PF 6.159, +18.08R, DD 1.24%, DSR 0.962` | **Hold `NO-GO`** (`TRIAL_0199` contrafactual sem `DIR` nos 5 pares) |

  - **Notas Complementares:** `ML pooled8` degradou `M0` em `SCALP` (`PF 3.393 -> 2.491`), `M3 (NEAR/DOGE/LINK)` (`n=20, PF 3.770, DSR 0.995`) fica em observação exploratória, `G8_ATR_EXP_CAP_3X` fechou neutro (`n=55, sumR=-2.0126` na Run 1), e código de produção das estratégias permanece **inalterado** em `BTC/ETH/SOL` (revisão cross-model `Claude Opus 5.5` — `APPROVE`).

### Itens Diferidos
1. **`S6-3` / `S5-5` Ativação Controlada (`shadow -> active`):** Avaliar promoção de `Swing4HMetaSizer` (`LogReg-L2` + `EWMA_VolTarget`) após amostra limpa de trades `SWING_4H` em produção.
2. **`T2` Carry Treasury:** Reavaliar ativação apenas quando funding anualizado OOS superar custo + hurdle institucional (hoje `STANDBY`).
3. **Expansão Complementar (`T11` Sleeve Beta-Neutro `XSMOM/Carry`):** Após consolidar a expansão `3 -> 8` majors.
4. **Wave P4b (`~dez/2026–jan/2027`):** Estudo OOS zero-leakage com `30–60` dias limpos de `liquidations` WS + `tick_agg` (`sat=0`).
5. **Meta-Labeling ML sobre Sinais Filtrados:** Pesquisar meta-rotulagem binária `P[win | sinal]` especificamente sobre candidatos já filtrados (ex.: `SCALP` e `DAY_TRADE`) quando houver série estendida.
