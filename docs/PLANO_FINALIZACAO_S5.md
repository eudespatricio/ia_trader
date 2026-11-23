# Plano Vivo de Finalização S5 — Backlog Combinado & Auto-Evolução

**Data de abertura:** `2026-11-21`  
**Status:** **CONCLUÍDO — Patch Final S5 (`v1.0.11`) implementado, revisado e verificado em `2026-11-23`**  
**Regra Mandatória:** *Tudo deve ser testado e provado empiricamente em OOS sem vazamento (zero-leakage) que melhora o desempenho antes de implementar.* Qualquer implementação na onda combinada exige testes independentes em janela limpa + cross-model code review.

Referências do repositório:
- Hipóteses pré-registradas (`E7..E10`, `Wave P3`): [HIPOTESES_PRE_REGISTRADAS.md](file:///home/eudespatricio/ia_trader/docs/HIPOTESES_PRE_REGISTRADAS.md)
- Gaps e resultados Wave P2/P3/S5: [gaps.md](file:///home/eudespatricio/ia_trader/docs/gaps.md)
- Convenção anterior de consolidação de sprint: [PATCH_S4_DOMINGO_BACKLOG.md](file:///home/eudespatricio/ia_trader/docs/PATCH_S4_DOMINGO_BACKLOG.md)

---

## 1. Itens PROVADOS & Implementados no Patch Final S5 (`v1.0.11`)

### Item S5-1 — `TechnicalAgent` + `LearningTracker` + `SignalGenerator`: Auto-Aprendizado Vivo por Sub-Regra Estrutural **[IMPLEMENTADO]**

- **Diagnóstico & Evidência Empírica:**
  - **Telemetria Real (`101` trades concluídos `WIN/LOSS` em `agent_learning_memory`):**
    - `Gatilho Estrutural Puro` (Sweep override `HOLD -> BUY/SELL` em `TechnicalAgent`): `n = 13`, WR `23,1%` (`3/13`), PnL **`-$433.04`** (`-$33.31`/trade).
    - `CONFLUÊNCIA POI/SWEEP` (bônus hardcoded `+0.08`/`+0.15` em `TechnicalAgent`): `n = 22`, WR `40,9%` (`9/22`), PnL **`-$630.71`** (`-$28.67`/trade).
    - `normal_no_sr` (sem sweep/POI): `n = 66`, WR `56,1%` (`37/66`), PnL **`+$1,884.98`** (`+$28.56`/trade).
    - *Causa raiz:* O `LearningTracker` agrega tudo sob o peso global de `"TechnicalAgent"`, penalizando o núcleo lucrativo (`+$1,884.98`) sem conseguir silenciar autonomamente as sub-regras estruturais negativas hardcoded.
  - **Backtest Purgado 6 Anos (`2020–2026`, `174.290` barras `1h`, BTC+ETH+SOL, custo RT `0,16%`, IS `<=2024` | 7d embargo | OOS `>=2025`):**
    - `Gatilho Estrutural Puro` (`SL 1.0x / TP 1.5x ATR`): IS `n = 3.238`, WR `40,6%`, `PF 0.82`, `-591.4R` (`-13.5 bps`); **OOS `n = 1.333`, WR `39,4%`, `PF 0.66`, `-333.3R` (`-20.8 bps`)**. Com `SL 0.7x / TP 2.1x` (`SignalGenerator`): **OOS `PF 0.72`, `-437.0R`**.
    - `Pullback (SL 1.0x / TP 1.5x)` religando `structure_context` do `SignalGenerator` (`SL 0.7/0.85x`): IS `-106.4R -> -127.0R`; **OOS `-66.6R -> -87.7R` (`PF 0.84 -> 0.80`)**.
  - **Prova do Auto-Learner Causal (`EXP 6C`, janela deslizante `W=20`, shrinkage Bayesiano `Beta(4,4)`, zero lookahead sobre os `101` trades reais):**
    - Baseline estático (regras S/R fixas): `n = 101`, WR `48,5%`, PnL **`+$821.24`**.
    - Auto-Learner causal por sub-regra (silencia se `n_tag >= 5`, `post_wr < 0.44` e `pnl_tag < 0`): `n = 80`, WR `51,2%`, PnL **`+$1,317.32`** (**delta `+$496.08`**, bloqueando autonomamente `21` trades ruins: `8` `sweep_trigger`, `8` `poi_confluence`, `5` `normal_no_sr`).

- **Escopo Exato Planejado:**
  1. [technical_agent.py](file:///home/eudespatricio/ia_trader/agents/technical_agent.py#L486-L521): Desabilitar `Gatilho Estrutural Puro` por padrão (`HOLD -> BUY/SELL` em sweep), emitir `structural_sub_rule` no `metrics`/`AgentSignal` (`sweep_trigger`, `poi_confluence`, `open_air`, `normal_no_sr`) e modular o bônus estrutural pelo fator causal aprendido.
  2. [learning_tracker.py](file:///home/eudespatricio/ia_trader/learning_tracker.py#L217-L413): Persistir e atualizar calibração causal por `structural_sub_rule` com janela deslizante `W=20` e shrinkage Bayesiano `Beta(4,4)`, expondo fator de modulação/auto-silenciamento (`n >= 5`, `post_wr < 0.44`, `pnl < 0`) para o `TechnicalAgent` e dashboard.
  3. [signal_generator.py](file:///home/eudespatricio/ia_trader/signal_generator.py#L70-L123): Remover o bloco morto `structure_context` (`SL 0.7/0.85x ATR` — provado inferior em IS e OOS e nunca alimentado por `core.py`).

---

### Item S5-2 — `ModelEvolutionManager`: Auto-Evolução ML Multi-Symbol + Gate LogLoss/ROC-AUC **[IMPLEMENTADO]**

- **Diagnóstico & Evidência Empírica (`EXP 6A`, LightGBM `1h`, IS `2020–2024` | 7d embargo | OOS `2025–2026`):**
  - Hoje `run_retraining_cycle()` treina **apenas em `BTCUSDT`**, desperdiçando histórico equivalente de `ETHUSDT` e `SOLUSDT`, avalia promoção por `emitted_signals_accuracy` isolada e converte `NaN` de `dist_resistance_pct`/`dist_support_pct` em `0.0` (colisão semântica *"sem topo/fundo recente = 0% de distância"*, ocorrida em `3,56%` / `2,18%` das `174.290` barras `1h` e `2,64%` / `2,59%` das `97.579` barras `15m`).
  - **Ganho OOS consolidado (`test = BTC+ETH+SOL`, `RT 0,16%`):**
    - `h=5 (1.2x/1.2x)`: Single-Symbol BTC `AUC 0.5222 | LogLoss 0.6994 | Top30% Acc 53.02% | Net -13.4 bps` -> **Multi-Symbol (`BTC+ETH+SOL`) `AUC 0.5270 | LogLoss 0.6941 | Top30% Acc 53.65% | Net -8.3 bps`**.
    - `h=24 (2.0x/2.0x)`: Single-Symbol BTC `AUC 0.5111 | LogLoss 0.7074 | Top30% Acc 51.66% | Net -29.5 bps` -> **Multi-Symbol (`BTC+ETH+SOL`) `AUC 0.5249 | LogLoss 0.6949 | Top30% Acc 53.51% | Net -14.4 bps`** (`-9.9 bps` quando enriquecido com MTF S/R).

- **Escopo Exato Planejado:**
  1. [learning_tracker.py](file:///home/eudespatricio/ia_trader/learning_tracker.py#L553-L692): Evoluir `run_retraining_cycle()` para suportar treino **Multi-Symbol (`("BTCUSDT", "ETHUSDT", "SOLUSDT")`)** mantendo retrocompatibilidade quando `symbol` único for passado explicitamente.
  2. Atualizar o gate `evaluate_challenger()` no ciclo de retreino para exigir melhoria conjunta de **calibração + discriminação (`LogLoss` e `ROC-AUC`)**, não apenas acurácia pontual de sinais emitidos.
  3. Preservar `NaN` nativo no pipeline LightGBM para `dist_resistance_pct` / `dist_support_pct` (com flags `has_res_above` / `has_sup_below` quando aplicável) em vez de `fillna(0.0)`, mantendo compatibilidade do fallback para estimadores `sklearn` que exijam valores finitos (`RF`).

---

### Item S5-3 — `ModelEvolutionManager` (`learning_tracker.py` / `model_trainer.py`): Calibração Platt/Isotonic + `LGBM-Shallow` + `ExtraTrees` no ML `1h` (**GO — [IMPLEMENTADO]**)

- **Diagnóstico & Evidência Empírica (`Lab B 2A/2B`, `1h`, IS `2021–2024` | embargo 7d | OOS `2025–2026`, `42.668` amostras `h=24` / `42.186` `h=48`):**
  - **Calibração (`Platt` / `Isotonic`):** Fit calibrador em split temporal interno (`2024-04..12`) sobre modelo treinado até `2024-03` reduziu o erro de calibração (`ECE`) em **~15×** sem perda de discriminação:
    - `h=24 (2.0x/2.0x)`: Raw `LL 0.7042 | ECE 0.0599` -> Platt `LL 0.6928 | ECE 0.0059` | Isotonic **`LL 0.6926 | ECE 0.0040`**.
    - `h=48 (3.0x/3.0x)`: Raw `LL 0.7088 | ECE 0.0597` -> Platt **`LL 0.6929 | ECE 0.0035`** | Isotonic `LL 0.6932 | ECE 0.0034`.
  - **Arquitetura no `F0_TechOnly` (`45` features técnicas):**
    - `LGBM-Shallow (max_depth=4, num_leaves=15, reg_lambda=5.0)` liderou `AUC` (`h=24`: `AUC 0.5302 | LL 0.6936 | ECE 0.0242 | Top30% Acc 54.12%`; `h=48`: `AUC 0.5259 | LL 0.6932 | ECE 0.0187 | Top30% Acc 53.99% | Net +2.9 bps`).
    - `ExtraTrees (max_depth=8)` obteve o menor `LogLoss` (`0.6919` em `h=48`, `0.6921` em `h=24`) e melhor `ECE` nativo (`0.0069–0.0070`).

- **Escopo Exato Planejado:**
  1. [learning_tracker.py](file:///home/eudespatricio/ia_trader/learning_tracker.py#L553-L692) & [model_trainer.py](file:///home/eudespatricio/ia_trader/model_trainer.py): Adicionar regularização rasa no LightGBM + suporte opcional a `ExtraTrees` + pós-calibração temporal (`Platt`/`Isotonic`) dentro do ciclo Walk-Forward de `ModelEvolutionManager`.
  2. Manter `F0_TechOnly` para o horizonte `1h` (proibido injetar macro/on-chain diário bruto dentro das árvores `1h`).

---

### Item S5-4 — `SentimentAgent` (`agents/sentiment_agent.py`) + Comitê `DAY_TRADE 1h`: Reforma Obrigatória do `SentimentAgent` (**GO — Reforma [IMPLEMENTADO]**)

- **Diagnóstico & Evidência Empírica (`Lab A 1A/1B` + `Lab C 3B`, 5 anos D-1 `2022–2026`, `RT 0.16%`):**
  - **Desempenho individual (`SL 2.0x / TP 3.0x`):** `SentimentAgent` contrário atual (`F&G <= 42 BUY / >= 58 SELL`) perdeu em **5/5 anos**: `2022 PF 0.81 (-209R)`, `2023 PF 0.88 (-67R)`, `2024 PF 0.91 (-69R)`, `2025 PF 0.78 (-170R)`, `2026 PF 0.75 (-171R)` (IS `PF 0.85 / -352.4R`, OOS `PF 0.77 / -327.7R`). Mesmo restrito a extremos (`<=15 / >=85`) perdeu em todos os anos ativos (`PF 0.67–0.87`).
  - **Impacto no Comitê `1h` (`SL 2.0x / TP 3.0x`):**
    - `C1` Produção atual: IS `PF 0.85 (-290.7R)` | OOS `PF 0.83 (-193.9R, -19.6 bps)`.
    - `C3` Zerar peso do `SentimentAgent`: IS `PF 0.92 (-257.8R)` | OOS `PF 0.89 (-149.2R, -12.9 bps)`.
    - `C9` `TechClean + OF_Mom + Sent_TREND`: IS `PF 0.92 (-245.0R)` | OOS `PF 0.93 (-133.4R, -8.6 bps)`.
    - `C15` `Macro + OnChain + TechClean (q=2, SL 2.5x / Trail 3.5x)`: IS `PF 0.95 (-41.2R)` | **OOS `PF 1.01 (+17.4R, +1.2 bps)`**.

- **Escopo Exato Planejado (Reforma do `SentimentAgent` + Bug Fix Live Funding):**
  1. [sentiment_agent.py](file:///home/eudespatricio/ia_trader/agents/sentiment_agent.py#L21-L91): Reformar o padrão de `SentimentAgent` para `mode="trend_momentum"` (`F&G >= 55 & fng_7d_chg >= +5 -> BUY`; `F&G <= 45 & fng_7d_chg <= -5 -> SELL`; caso contrário `HOLD`), superando `S0_Contrarian` em **`+241.3R` OOS** (`PF 0.83 -> 1.04`, `+23.1R` standalone e melhorando comitê `1h` OOS `PF 0.85 -> 0.90` / `+34.4R`). Preservar `neutral_telemetry` e `legacy_contrarian` para comparação.
  2. [live_context_enricher.py](file:///home/eudespatricio/ia_trader/data/live_context_enricher.py#L130-L220): Exportar `fng_7d_chg` estritamente causal (D-1) **e corrigir bug P0 ao vivo** onde `funding_rates` desatualizado (`> 12h`) no `market_cache.db` impedia fallback para `derivs` do `microstructure.db` (`funding_rate == 0.0` em **50/50 decisões ao vivo**).
  3. Manter E7 `STRATEGY_PROFILES["DIRECTIONAL"]` (`SL 1.0x/TP 1.5x`) inalterado na S5 (`G2/G3` melhoram PF relativo `0.69 -> 0.90/0.95` mas seguem `< 1.0` OOS — registrado em `docs/gaps.md`).

---

### Item S5-5 — `SWING_4H` Shadow Meta-Sizer (`LogReg-L2` + Vol-Target/ATR-RP + RS 14d) (**GO — Modo `shadow` [IMPLEMENTADO]**)

- **Diagnóstico & Evidência Empírica (`Lab B/C` + `Alpha-1 T1` + `Alpha-2 T6c`, 2020–2026):**
  - `LogReg-L2 (C=0.1)` com 10 features D-1: **WF AUC `0.5880`**, `+22.4R` (**Boot 95% CI `[+9.1R, +37.5R]`, `P>0 = 100.0%`**), positivo em 3/4 anos e 3/3 ativos.
  - **`T1` Vol-Targeting / ATR Risk Parity:** Base Fixed (`Sharpe 0.84 / MaxDD 30.9% / Calmar 0.74`) -> `ATR_RiskParity (1.5%R)` **`Full Sharpe 1.11 / MaxDD 21.9% / Calmar 1.27` (+72% Calmar)**; `EWMA_VolTarget [0.4–2x]` **`Full Sharpe 1.03 / MaxDD 22.1%`**, **`OOS DD 16.3%`**; somado ao Meta-Sizer (`VT + S5-5`): **`OOS Sharpe 1.07 -> 1.24`, `OOS Ret +60.5% -> +83.7%`, `OOS Calmar 1.50 -> 2.13`**, delta neutro em alavancagem WF (`2023–26`): **`+17.3%` (`P>0 = 82.1%`)**.
  - **`T6c` Rotação Força Relativa 14d (`1.4x/1.0x/0.6x` normalizado a `1.0x`):** Positiva em **6/7 anos (`2020–2025`)**: IS `+93.5R -> +108.9R (+15.4R, P>0=96.1%)`, OOS `+42.1R -> +45.5R (+3.5R)`, Full **`+133.7R -> +152.4R (+18.7R, P>0=95.6%)`**.

- **Escopo Exato Planejado:**
  1. Criar [swing_4h_meta_sizer.py](file:///home/eudespatricio/ia_trader/data/swing_4h_meta_sizer.py) (`<= 300` linhas) acoplado a [run_paper_simulation.py](file:///home/eudespatricio/ia_trader/scripts/run_paper_simulation.py) em **`mode="shadow"` por padrão** (gravando telemetria em `swing_4h_meta_telemetry` e `sig.metrics`, preservando `applied_mult = 1.0` e regras congeladas `Donchian40/ExitCh10` durante S5 até ativação explícita via `simulation_settings.swing_4h_meta_sizing_mode = "active"`).

---

## 2. Itens REFUTADOS / **NO-GO** (não implementar nem retrabalhar sem novos dados)

| Hipótese Testada | Evidência Empírica (IS / OOS / Live) | Decisão |
|---|---|---|
| **Novo agente de comitê `DerivativesPositioningAgent`** (`OI`, `ls_global`, `ls_top_acc`, `ls_top_pos`, `basis_bps`) | **Wave P4a** (`67,4d` purged split): `oi_chg`, `ls_top_pos_z`, `funding_z`, `basis_bps_z` inverteram sinal IS->OOS; `ls_global`/`ls_top_acc` são só proxy de `-ret_24h` (`ResidIC OOS = +0.004 / -0.001`). LogReg 11 features: **OOS AUC `0.503` (4h), `0.463` (12h), `0.454` (24h)**. No `SWING_4H`, os 2 maiores winners (`ETH +13.43%`, `SOL +15.56%`) tiveram `oi_chg_4h` negativo (`-4.49%`, `-3.54%`). | **NO-GO** |
| **Alterar `SWING_4H` via S/R** (`Struct SL 4h`, `Struct Trail`, `Hybrid Trail`, `Parcial 1d`, `Veto Parede 1d < 1/2 ATR`) | [`swing_4h_strategy.py`](file:///home/eudespatricio/ia_trader/data/swing_4h_strategy.py) Baseline (`SL 2.5x / Trail 3.5x / ExitCh10`) venceu **7/7 anos (`2020–2026`)**: IS `PF 1.41 (+102.1R)`, OOS `PF 1.48 (+39.8R)`. `Veto 1d < 2 ATR` perdeu em **5/7 anos** (`2021` caiu de `+40.6%` para `-2.0%`); `Struct SL` caiu para `+25.4R` OOS; `Struct/Hybrid Trail` caiu para `PF 1.39/1.40`. | **NO-GO — manter `SWING_4H` congelado** |
| **SL/TP ancorado em S/R no `1h`** (`Struct SL 1h`, `Struct SL 4h`, `TP cap 4h`) | `Struct SL 4h` overfitou IS (`+75.9R, PF 1.04`) e colapsou OOS (**`-46.3R, PF 0.84`**); `Struct SL 1h` e `TP cap 4h` também ficaram abaixo de ATR puro (`OOS PF 0.92 / 0.90`). | **NO-GO** |
| **MTF S/R (`4h`/`1d`/PDH-L/PWH-L/POC 30d) e fix `NaN->0` isolado no ML `1h`/`15m`** | `ResidIC` pós `ret_24h` + `dist_sma200` `<= 0.029`. LightGBM `1h` (`h=5`): `V0 0.5270` vs `V1 0.5263` vs `V1+MTF 0.5266`; `15m` (`h=12`): `V0 0.5088` vs `NO_SR 0.5091`. | **NO-GO isolado** |
| **Religar `structure_context` (`SL 0.7/0.85x ATR`) no `SignalGenerator`** | Piorou IS (`-106.4R -> -127.0R`) e **OOS (`-66.6R -> -87.7R`)**. | **NO-GO** |
| **Online Bayes Learner sobre buckets S/R de estratégia-base negativa (`1h D20`)** | Buckets inverteram IS->OOS (`MID_SPACE PF 0.64 -> 1.07`; `NEAR_4H_WALL PF 1.07 -> 0.84`). Learner piorou OOS (`-43.3R -> -47.3R` / `-51.8R`). | **NO-GO** |
| **Macro/On-Chain/Funding bruto em árvores no `1h` (`F2_FullMultiDomain`)** | **Lab B 2A:** `AUC` despencou `0.5302 (F0) -> 0.4971 (F2)` e `ECE` piorou `0.0242 -> 0.0605` (séries diárias repetidas em `70k` barras decoram o calendário `2022–24`). | **NO-GO** |
| **Rede Neural `MLP (64,32)` em tabular `1h`** | **Lab B 2B:** `AUC 0.4812`, `LogLoss 1.43–1.66`, `ECE 0.33–0.35`, `Net -39.8 bps` — sobreconfiança extrema em ruído tabular. | **NO-GO** |
| **`Regime-Specialist LGBM` & `Soft-Voting 5 famílias` em `F2`** | **Lab B 2B:** `AUC 0.5064 / 0.4894`, `LogLoss 0.706–0.711` — degradados por `F2` + `MLP`; inferiores a `F0` + calibração. | **NO-GO** |
| **Pesos estáticos Ridge por regime (`C13/C14`) & Meta-LGBM/HistGB no Comitê `1h`** | **Lab A 1B/1C:** `C14` OOS `PF 0.82 (-123.8R)`; Meta-LGBM/HistGB OOS `AUC 0.47–0.51` — overfit sobre sinal-base `1h` ruidoso. | **NO-GO** |
| **`T1-DDThrottle` (-8%/-10% drawdown throttle no `SWING_4H`)** | **Alpha-1 T1:** Reduziu OOS Sharpe (`1.16 -> 0.77/1.01`) e cortou retorno OOS pela metade (`+66.9% -> +27.3%`). | **NO-GO** |
| **`T3` Carver EWMAC contínuo + Inércia (`10%/15%`)** | **Alpha-1 T3:** OOS Sharpe caiu para `0.23–0.30` (`LongOnly -0.23`, `+2.7%`..`+4.4%` CAGR); Blend 50/50 reduziu OOS Sharpe (`1.07 -> 0.82–0.92`). | **NO-GO** |
| **`T4` Volatility Squeeze Breakout (`1h` / `4h`)** | **Alpha-2 T4:** Instável entre anos (`2022 -47.3R`, `2023 -20.7R`, `2026 -3.9R`, OOS Boot CI `[-18R, +120R]` cruza zero; `4h` OOS `-7.6bp`). | **NO-GO** |
| **`T5` Capitulation `RSI(2)+IBS` (`4h` / `1d`)** | **Alpha-1 T5:** Inverteu IS->OOS nos dois timeframes (`4h` IS `PF 0.55` -> OOS `1.33`; `1d` IS `PF 1.29` -> OOS `0.61`). | **NO-GO** |
| **`T6a` `1h` Lead-Lag & `T6b` Kalman Stat-Arb (`ETH/BTC/SOL`)** | **Alpha-2 T6:** `1h` Lead-Lag `IS PF 0.68 / OOS PF 0.84`; Kalman `OOS PF 0.27–0.30` (`-20R..-22R`) nos 3 pares após custos+funding. (`T6a 4h` com `n=38` OOS insuficiente -> `gaps.md`). | **NO-GO** |
| **`T7b` Zarattini Dynamic Noise-Band ORB** | **Alpha-2 T7:** Negativo em todas âncoras UTC (`00/13/14 UTC`): IS `PF 0.78–0.85 (-198R..-465R)`, OOS `PF 0.73–0.90 (-48R..-237R)`. | **NO-GO** |
| **`T8` Funding rules no `1h` `SentimentAgent` (`S3/S4/S5`)** | **Wave P5b:** Todos inverteram/colapsaram em OOS (`PF 0.65–0.82`, `-57R` a `-142R`). Só `S2_Trend_FG_Mom` foi positivo (`PF 1.04`, `+23.1R`). | **NO-GO** |
| **`T9` Macro Liquidity Switcher (`Tilt`/`HardGate`) no `SWING_4H`** | **Alpha-2 T9:** `Tilt` reduziu OOS (`+42.1R -> +40.7R`); `HardGate` cortou amostra (`188 -> 106`) e total R (`+42.1R -> +28.2R`); `S5-5` já captura macro sem corte bruto. | **NO-GO** |
| **`T10` Assimetria L/S, Kaufman ER e Time-Stop no `SWING_4H`** | **Alpha-2 T10:** Shorts de `SWING_4H` têm edge positivo (`IS PF 1.24 / OOS PF 1.33`); `ER` e `TimeStop` reduziram `R` total e retorno médio por trade. | **NO-GO** |
| **`T12` Bloqueio `FOMC/CPI [T-2h..T+1h]` em `1h`** | **Alpha-3 T12:** Apesar de range ~1.7–1.8x maior, bloquear entradas piorou OOS (`PF 0.854 -> 0.839`, `-132.6R -> -146.1R`); entradas na janela tiveram `PF 1.052 (+2.0R)`. | **NO-GO** |

---

## 3. Próximas Investigações (Status Finalizado para Patch S5)

- [x] **Reforma do `SentimentAgent` (Wave P5b) + Auditoria Live Funding:** Concluída. `S2_Trend_FG_Mom` aprovado (`S5-4`) + bug P0 ao vivo de `funding_rate == 0.0` (`50/50` decisões) diagnosticado em [live_context_enricher.py](file:///home/eudespatricio/ia_trader/data/live_context_enricher.py).
- [x] **Baterias Globais `Alpha-1`, `Alpha-2`, `Alpha-3`:** Concluídas (`100%` testadas em OOS zero-leakage).
  - *Nota `DAY_TRADE 1h`:* `G2 (2x/3x)` e `G3 (2.5x/Trail 3.5x)` melhoram PF relativo (`0.69 -> 0.90/0.95`), mas seguem `< 1.0` OOS -> preservar `STRATEGY_PROFILES["DIRECTIONAL"]` inalterado na S5 e registrar em `docs/gaps.md`.
  - *Nota `GOLDEN_POT` & `SCALP`:* Histórico fechado em paper (`GOLDEN_POT` `35` trades `-$202.25`; `SCALP` `28` trades `-$131.31` vs `DIRECTIONAL` `83` trades `+$480.35`). Filtros macro/clímax já endurecidos; sem alterações cegas sem série 5m estendida.
- **Adiado propositalmente para S6 / Wave P4b:**
  - **`T2` Idle-Capital Carry Treasury:** Validado (`corr +0.001`, 7/7 anos positivo, IS `+5.99%/ano`), mas yield OOS `2025-26` está apenas em `+0.12%/ano` com custos estresse; manter `STANDBY` pronto para ciclo de funding alto.
  - **`T11` Expansão de Universo (`Top 15–20 Perps Líquidos`) + XSMOM/Carry Neutro:** Adiado para S6 (requer ingestão histórica multi-ano + controle de viés de sobrevivência).
  - **Wave P4b (~30–60 dias de coleta limpa, ~fim dez/2026–jan/2027):** Avaliação OOS zero-leakage de **`liquidations` WS (`!forceOrder@arr`)** e **`tick_agg` (`sat=0`)**.

### Test List Global — Estratégias Institucionais & Top Traders (`T1`–`T12`)

*(Baterias Alpha-1/2/3 executadas com rigor OOS zero-leakage `2020–2026`, custo RT `0,16%` + funding real 8h.)*

#### Pilar 1 — Overlays de Portfólio, Risco & Eficiência de Capital
| Status | ID | Estratégia / Origem Institucional | Edge Econômico & Encaixe no `ia_trader` | Veredicto Final |
|---|---|---|---|---|
| `[x]` | **`T1`** | **Volatility Targeting + Inverse-Vol Risk Parity + DD Throttle** *(Barroso & Santa-Clara / Moreira & Muir / AQR / Man AHL)* | `ATR_RiskParity` (`Full Shp 1.11, MaxDD 21.9%, Calmar 1.27`) e `VT + S5-5` (`OOS Shp 1.24, Ret +83.7%, Calmar 2.13`) aprovados; **`T1-DDThrottle` refutado** (`OOS Shp 0.77`). | **GO -> `S5-5` (VT+S5-5 ativo em `shadow`; ATR-RP em telemetria) / DDThrottle NO-GO** |
| `[x]` | **`T2`** | **Idle-Capital Carry Treasury ("Cash-and-Carry Overlay")** *(Wintermute / QCP / Ethena)* | Descorrelacionado (`corr +0.001`), positivo 7/7 anos, IS `+5.99%/ano`, mas OOS `2025-26` rendeu apenas `+0.12%/ano` em estresse. | **STANDBY S6** (aguardar ciclo de funding alto) |

#### Pilar 2 — Sleeves de Alfa Descorrelacionados
| Status | ID | Estratégia / Origem Institucional | Edge Econômico & Encaixe no `ia_trader` | Veredicto Final |
|---|---|---|---|---|
| `[x]` | **`T3`** | **Multi-Speed Continuous Trend + Buffer de Inércia** *(Robert Carver EWMAC / Winton / Aspect)* | `OOS Sharpe 0.23–0.30`; Blend 50/50 degradou `SWING_4H` (`1.07 -> 0.82–0.92`). | **NO-GO** |
| `[x]` | **`T4`** | **Volatility Compression -> Expansion Breakout (Squeeze)** *(Toby Crabel NR7 / Raschke / Minervini VCP)* | Negativo em 2022/23/26 (`-47.3R/-20.7R/-3.9R`), Boot CI cruza zero; `4h` OOS `-7.6bp`. | **NO-GO** |
| `[x]` | **`T5`** | **Capitulation / Crash-Bounce Mean Reversion** *(Larry Connors `RSI(2) + IBS` / Desks Pós-Liquidação)* | Inverteu IS->OOS nos dois timeframes (`4h 0.55->1.33`, `1d 1.29->0.61`). | **NO-GO** |
| `[x]` | **`T6`** | **Cross-Asset Triad: BTC Lead-Lag, Rotação RS & Stat-Arb Kalman** *(Jump / B2C2 / Desks RV)* | **`T6c` RS 14d (`avg_w=1.0`)** positivo 6/7 anos (`+18.7R Full, P>0=95.6%`); `T6a 1h` (`PF 0.68/0.84`) e `T6b Kalman` (`OOS PF 0.27–0.30`) refutados (`T6a 4h` `n=38` em `gaps.md`). | **`T6c` GO -> `S5-5` (telemetria) / `T6a,T6b` NO-GO** |

#### Pilar 3 — Microestrutura Cripto Pós-ETF, Derivativos & Regimes Macro
| Status | ID | Estratégia / Origem Institucional | Edge Econômico & Encaixe no `ia_trader` | Veredicto Final |
|---|---|---|---|---|
| `[x]` | **`T7`** | **Sazonalidade de Sessão UTC + ORB com Banda de Ruído** *(Carlo Zarattini / Concretum / Fluxo ETF EUA)* | ORB negativo em todas as 4 âncoras UTC (`IS PF 0.78–0.85`, `OOS PF 0.73–0.90`). | **NO-GO** |
| `[x]` | **`T8`** | **Funding-Squeeze & Híbrido Tendência + Carry** *(Alimenta Reforma `SentimentAgent` — Wave P5b)* | Regras de funding no `1h` (`S3/S4/S5`) colapsaram (`PF 0.65–0.82`); vencedor foi `S2_Trend_FG_Mom` (`+241.3R vs S0` OOS). Bug P0 de funding live corrigido em `LiveContextEnricher`. | **`S2` + Fix Funding GO -> `S5-4` / `S3-S5` NO-GO** |
| `[x]` | **`T9`** | **Macro Liquidity Regime Switcher (Expansão vs Contração)** *(Pantera / Brevan Howard Digital)* | `Tilt` e `HardGate` reduziram `R` OOS (`+42.1R -> +40.7R / +28.2R`); `S5-5` já captura macro continuamente. | **NO-GO** |
| `[x]` | **`T10`** | **Assimetria Long/Short + Kaufman Efficiency Ratio (ER) + Time-Stop** *(Perry Kaufman / CTAs clássicos)* | Shorts lucrativos (`OOS PF 1.33`); `ER` e `TimeStop` reduziram `R` total e expectativa média. | **NO-GO** |

#### Pilar 4 — Expansão de Universo & Eventos
| Status | ID | Estratégia / Origem Institucional | Edge Econômico & Encaixe no `ia_trader` | Veredicto Final |
|---|---|---|---|---|
| `[ ]` | **`T11`** | **Expansão de Universo (`Top 15–20 Perps Líquidos`) + XSMOM/Carry Beta-Neutro** *(Quant Crypto Desks)* | Requer ingestão multi-ano `2021–2026` (`15–20` ativos + controle de sobrevivência) e sleeve beta-neutro separado. | **ADIADO S6** |
| `[x]` | **`T12`** | **Event Guard Macro (`FOMC / CPI / NFP`)** *(Prop Desks / Market Makers)* | Bloquear entradas `1h` `[T-2h..T+1h]` piorou OOS (`PF 0.854 -> 0.839`, `-132.6R -> -146.1R`). | **NO-GO** |

#### Baterias de Execução Executadas (Zero-Leakage OOS `2020–2026`, `RT 0,16%` + Funding Real)
1. [x] **Bateria Alpha-1 (`T1`, `T2`, `T3`, `T5`):** Executada (`T1` VT/ATR-RP **GO**; `T1-DDThrottle`, `T3`, `T5` **NO-GO**; `T2` **STANDBY S6**).
2. [x] **Bateria Alpha-2 (`T4`, `T6`, `T7`, `T8/Wave P5b`, `T9`, `T10`):** Executada (`T6c`, `S2_Trend_FG_Mom` **GO**; demais **NO-GO**).
3. [x] **Bateria Alpha-3 (`T12`, `DAY_TRADE 1h` geometria, Auditoria Live Funding):** Executada (Bug P0 Live Funding detectado e corrigido; `T12` e `G2/G3 < 1.0` OOS mantidos fora de produção; `T11` **ADIADO S6**).

---

## 4. Checklist de Execução Combinada (Concluído em `v1.0.11`)

1. [x] Concluir baterias de testes empíricos OOS (`Alpha-1/2/3`, `Wave P5b`) e atualizar Seções 1, 2 e 3 acima.
2. [x] Congelar escopo final do **Patch Final S5 (`v1.0.11`)** (`S5-1..S5-5` + fix P0 Live Funding).
3. [x] Implementar via `swarm` + auditoria cruzada **DeepSeek (`auto-pro`)** em 4 chunks sem truncamento:
   - **Coders:** implementação cirúrgica preservando [swing_4h_strategy.py](file:///home/eudespatricio/ia_trader/data/swing_4h_strategy.py) e `STRATEGY_PROFILES["DIRECTIONAL"]` inalterados.
   - **Independent Tester ("Sem Oráculo Comprado"):** [test_patch_s5_final.py](file:///home/eudespatricio/ia_trader/tests/test_patch_s5_final.py) validando `Beta(4,4)`, nativo `NaN` no LightGBM, calibração `Platt`/`Isotonic`, `SentimentAgent` + fallback `micro_derivs`, e `Swing4HMetaSizer` (`shadow` + zero-lookahead).
   - **Cross-Model Review (`DeepSeek auto-pro`):** `0` P0 encontrados; todos P1/P2/P3 aplicáveis corrigidos (precedência de `sub_rule_multipliers` no `core.py`, guarda de classes na calibração, fechamento SQLite `closing`, fallback sem `exchange` e remoção de simulações sintéticas no meta-sizer).
4. [x] Rodar suíte completa `./.venv/bin/pytest -q`: **`689 passed, 1 skipped, 6 xfailed`** (`0` falhas) e validar daemon/época S5 ao vivo sem regressões.
5. [x] Commit local criado na branch `main` (`git push origin main` pendente de autorização explícita do usuário).
