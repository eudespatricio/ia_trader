# Backlog Oficial — Patch S4.2 de Domingo (Pré-Aprovado para Revisão/Aplicação)

Data de consolidação: `2026-11-15` (atualizado com fechamento final isolado S4 N=33) | Status: **Aplicado em produção (`v1.0.7`, 2026-11-15)**  
Referências auditadas:
- Auditoria empírica S4: [S4_AUDITORIA_EMPIRICA_E_CALIBRACAO.md](file:///home/eudespatricio/ia_trader/docs/S4_AUDITORIA_EMPIRICA_E_CALIBRACAO.md)
- Hipóteses pré-registradas (`E10.7..E10.9`): [HIPOTESES_PRE_REGISTRADAS.md](file:///home/eudespatricio/ia_trader/docs/HIPOTESES_PRE_REGISTRADAS.md)
- Estudo de viabilidade do `SCALP` (`TRIAL_0106..TRIAL_0130`): [ESTUDO_SCALP_S4_ALTERNATIVAS.md](file:///home/eudespatricio/ia_trader/docs/ESTUDO_SCALP_S4_ALTERNATIVAS.md)

---

## Item 1 — `GOLDEN_POT`: Alinhamento Macro via BTC `±1.5% SMA200` (**GO**)
- **Regra proposta:** Bloquear entradas `LONG` no `GOLDEN_POT` quando `dist_sma_200` do `BTCUSDT` 1h estiver abaixo de `-1.5%` (`< -0.015`) — e, simetricamente, bloquear `SHORT` contra macro altista forte (`BTC dist_sma_200 > +1.5%`, salvo se houver confirmação explícita na spec de domingo).
- **Evidência empírica S4 (janela `N=11` trades fechados ao vivo desde `2026-11-09 00:00 UTC`):**
  - **Realizado sem filtro:** `11` trades (`5 LONG` com `3 wins` totalizando `-$46.31`; `6 SHORT` com `4 wins` totalizando `+$23.05`) -> resultado líquido **`-$23.27`**.
  - **Contrafactual com filtro macro BTC `±1.5% SMA200`:** bloqueia os `LONG` contra tendência macro de baixa, mantendo `6` trades `SHORT` com **`+$23.05`** (melhora líquida de **`+$46.32`** na janela S4).
- **Plano de validação pós-patch no domingo:** Testar unitariamente o gate macro no `GOLDEN_POT` e confirmar ausência de regressão nos 3 majors antes de reiniciar o daemon.

---

## Item 2 — `DIRECTIONAL`: `max_same_dir_majors=2` + `_flow_gate` Estrito `status == "OK"` nos Majors (**GO**)
- **Regras propostas:**
  1. **Trava de correlação (`DIR_CAL2_CORR_CAP`):** Limitar a no máximo `2` posições simultâneas na mesma direção nos majors (`max_same_dir_majors=2`).
  2. **Gate estrito de microestrutura (`TapeReadingAgent`):** Em `BTCUSDT`, `ETHUSDT` e `SOLUSDT`, exigir `status == "OK"` no `_flow_gate` — eliminando o fallback permissivo quando os dados de book/ticks L2 estiverem indisponíveis (`unavailable_passed=63` no replay 29d em `microstructure.db`).
- **Evidência empírica consolidada:**
  - **Holdout OOS 1h (`2024-02-24 .. 2026-11-13`, `TRIAL_0098..TRIAL_0102`):** `DIR_BASE` (`6.883 tr`, `-$5.329,03`, DD `53,68%`) -> `DIR_CAL2_CORR_CAP` (`5.924 tr`, `-$4.492,47`, corte de **`-8,57 pp` no DD máximo para `45,11%`**) -> combinado S4.2 (`5.144 tr`, WR `39,6%`, PF `0,649`, `-$3.722,59`, DD **`37,60%`**).
  - **Replay Real 29d em `microstructure.db` (`2026-10-15 .. 2026-11-13`):** `DIR_S42_NO_TAPE` (`139 tr`, WR `39,6%`, PF `0,542`, `-$100,14`) -> `DIR_S42_WITH_TAPE` (`87 tr`, WR `46,0%`, PF `0,711`, **`-$35,09`**, redução de **`65%` nas perdas**; `tape_gate_stats`: `evaluated=314, passed=77, unavailable_passed=63, blocked=174`). Tornar o gate estrito (`status == "OK"`) evita exposição cega nesses `63` sinais sem leitura L2.
  - **Contrafactual Ao Vivo S4 (`N=11` fechados):** `11 SHORT` (`8 wins`, WR `72,7%`, `+$1.064,93`) -> com `corr_cap=2`: **`9` trades (`7 wins`, WR `77,8%`, `+$1.095,00`)**.
- **Nota de verificação L2 no domingo:** Garantir monitoramento de saúde de `book_features`/`tick_agg` para que `status == "OK"` não silencie os majors por falta de ingestão do collector.

---

## Item 3 — `SCALP` 15m: Veredito da Pesquisa `E10.9` e Decisão de Domingo (**NO-GO para Ativação Lucrativa**)
- **Diagnóstico estrutural confirmado (`smart_trade.py` vs `backtesting.py`):**
  1. **Atrito vs `1R`:** No 15m (`1R = 0.9 * ATR_14`), a mediana de `1R` é `~29,5 bps` (`24,0 bps` no BTC); um round-trip Taker real (`8+8 = 16 bps`) consome **`54%–67%` de `1R`**.
  2. **Armadilha da escada `40/30/30` + `BE +4 bps`:** Ao bater `TP1 (+1.2 ATR)`, `79,8%` retornam antes de `TP2` e encerram em `BREAKEVEN_STOP` pagando `8 bps` taker nos `60%` restantes (`+0,27R` líquido vs `-1,34R a -1,42R` nos `STOP_LOSS`).
  3. **Discrepância de paridade:** `backtesting.py:781` não possuía a trava `closed_safe` `v1.0.5` de `smart_trade.py:1851`.
- **Resultados empíricos da suíte `E10.9_SCALP_RESEARCH` (`25` trials `TRIAL_0106..TRIAL_0130` em Horizontes A [9-13m], B [60d Binance derivs] e C [29d L2 real]):**
  - **Redução estrutural de sangria comprovada (6x menor perda por trade):** Trocar a entrada taker + escada `TP1 BE` (`A0_LIVE_REAL_TAKER`: `1027 tr`, PF `0,271`, `E[R] = -0,6707`, `-$733,46`) por `Limit Pullback` maker (`2bps` in, slippage `0`, sem `TP1 BE`) reduziu o prejuízo esperado em mais de 5x–6x: `A4_BO20_DELAYED_BE_PULLBACK` (`535 tr`, PF `0,838`, `E[R] = -0,1120`, `-$149,91`) e `A8_RANGE_VWAP_REV_PULLBACK` (`69 tr`, WR `47,8%`, PF `0,877`, `E[R] = -0,1155`, `-$11,35`).
  - **Insuficiência estatística para operação lucrativa em 15m:** **Todas as `25` variantes** apresentaram `E[R] < 0` e `DSR <= 0,0004`. O único trial com PF acima de `1,0` foi `B4_FUNDING_WINDOW_FADE` (janela `±30m` de funding `00/08/16 UTC`, limit pullback: `N=31 tr`, WR `48,4%`, PF `1,218`, PnL `+$7,14`, DD `0,13%`), porém com **`E[R] = -0,0573` (IC 95% `[-0,5112, +0,3415]`), `boot_ci [-6,758, +6,924]` e `DSR = 0,0000`** (`H1 E[R] = -0,0084`, `H2 E[R] = -0,1095`), sem robustez estatística para promoção. Em 29d L2 real, `C2b_L2_TOXIC_OBI_PULLBACK` ficou em PF `0,951`, `-$0,86`, `E[R] = -0,0995` em apenas `N=15 tr` (colapsando em H2 para PF `0,279`, `-$11,10`).
- **Recomendação Objetiva para Domingo:**
  1. **NO-GO para ativar `SCALP` 15m com capital paper/real** -> manter **pausado (`enabled=False`)** ou em **modo `shadow` (custo/risco zero)** para estancar sangria imediata (`SCALP_BASE -$598,88 / PF 0,349` em OOS; ao vivo S4 `-$50,58` em 2 trades).
  2. **Se mantido em `shadow` para telemetria futura:** remover a escada `40/30/30 + BE +4bps` + entrada taker (quebrada matematicamente) em favor do esqueleto `Limit Pullback` + alvo único/BE tardio (`A4`/`A8`) e coletar `60–90` dias adicionais de `book_features` + `derivs` — ou migrar pesquisa futura para `30m/1h` onde `1R >= 60–80 bps` dilua o atrito de execução.

---

## Item 4 — Anotações de Debate: Impacto e Ponderação do `TapeReadingAgent` nas 3 Estratégias (`E10.10`)
Referência completa: [NOTAS_IMPACTO_TAPE_READING_3_ESTRATEGIAS.md](file:///home/eudespatricio/ia_trader/docs/NOTAS_IMPACTO_TAPE_READING_3_ESTRATEGIAS.md) | Script: [eval_tape_reading_3strategies_s4.py](file:///home/eudespatricio/ia_trader/scripts/eval_tape_reading_3strategies_s4.py) (`TRIAL_0131..TRIAL_0165`).

### 4.1 Como Garantir a Validação Estrita no Código no Domingo
- **O que existe hoje:** As 3 estratégias (`DIRECTIONAL` L1057, `SCALP` L951, `GOLDEN_POT` L786 de [run_paper_simulation.py](file:///home/eudespatricio/ia_trader/scripts/run_paper_simulation.py)) já passam obrigatoriamente por `_flow_gate(...)` após `_committee_gate`, **mas:**
  1. `TapeReadingAgent` **não participa** do `ConsensusJudge` (`core.py` L62–69 e `consensus_judge.py` L57–64 têm apenas `OrderFlowAgent` com peso `0.8`, baseado em OHLCV/derivativos).
  2. `_flow_gate` opera em modo **fail-open** (`UNAVAILABLE_PASSED`, L553–572), deixando sinais entrarem mesmo sem book/tick válidos.
- **Como blindar por estratégia no patch S4.2:**
  - **`DIRECTIONAL` & eventual `SCALP` (Majors `BTC/ETH/SOL`):** Exigir política *fail-closed* por configuração (`require_tape_ok=True`: passar só se `status == "OK"` e `vote == direction`), com teste unitário garantindo rejeição se `status != "OK"`.
  - **`GOLDEN_POT` (Altcoins 5m):** Não usar `M2_GATE_STRICT_OK` puro sem cobertura L2 prévia — derrubou para `2` trades em 29d (`0/31` no log live). Em altcoins fora de `covered_symbols` (`symbol_not_covered`), usar política híbrida (`M4_STRICT_OK_PLUS_WEIGHT` / veto estrito quando `status == "OK"` oposto, e se `UNAVAILABLE` + macro BTC alinhado, operar apenas com redução defensiva de lote para `0.50x`).

### 4.2 Evidência Empírica: Contrafactual Ao Vivo (`flow_gate_log`, `N=804`) & Replay 29d L2 Real (`microstructure.db`)
- **Log real `flow_gate_log` (`801` avaliados com candle):**
  - `DIRECTIONAL`: dos `639` candidatos sem gate (`PF 0,509`, `-$505,35`), os **`562` bloqueados pelo gate tinham `PF 0,474` e `-$479,47`** — o gate evitou quase toda a sangria. Os `59` `StrictOK` ficaram em `WR 37,3% / PF 0,691` (vs `41,6% / PF 0,780` nos `77` fail-open que incluíram `18` `UNAVAILABLE_PASSED` por sorte amostral `+$4,92`).
  - `SCALP`: `131` sem gate (`PF 0,099`, `-$175,52`) vs `103` bloqueados (`PF 0,080`, `-$158,82`) vs `24` `StrictOK` (`PF 0,220`, `-$14,50`).
  - `GOLDEN_POT`: **`0/31` tiveram `status == "OK"`**, **`27/31` entraram via `UNAVAILABLE_PASSED`** (`63,0% WR`, `PF 0,850`, `-$16,87`), e `4` foram bloqueados (`+$4,09`).
- **Replay 29d L2 Real — Comparativo Sem Tape (`M0`), Fail-Open (`M1`), Estrito (`M2`), Peso (`M3`: `0.65*base + 0.35*tape`) e Estrito+Peso (`M4`):**
  - **`DIRECTIONAL` 1h (`corr_cap=2`):** `M0` (`138 tr`, WR `38,4%`, PF `0,530`, `-$103,38`) -> `M1` (`84 tr`, WR **`44,0%` (`+5,64 pp`)**, PF **`0,661`**, `-$41,17`) -> `M2` (`55 tr`, WR `38,2%`, PF `0,541`, `-$43,32`) -> `M3` (`68 tr`, WR `38,2%`, PF `0,504`, `-$65,93`). **Conclusão:** em `DIRECTIONAL`, dar peso permitindo `HOLD` neutro (`M3`) reinjetou falsos positivos; o gate pós-comitê binário (`M1`/`M2`) é superior ao peso contínuo.
  - **`GOLDEN_POT` 5m (`224` ativos 5m -> `243` aprovações comitê -> `185` após macro BTC `±1.5%`):** com macro BTC, `M0` (`83 tr`, `-$155,12`) -> `M1` (`83 tr`, `-$148,56`) -> `M2` (`2 tr`, `-$0,10`) -> **`M4` (`39 tr`, `41,0% WR`, PF `0,750`, `-$7,62`, `H2 PF 1,01 / +$0,10`)** — redução de perda de `-$155` para `-$7,62`.
  - **`SCALP` 15m (Produção & Melhores Win Rates do Backtest):**
    - `SCALP_PROD_BASE`: WR sobe `+7,03 pp` em `M2` (`31,4% -> 38,5%`) e `+8,57 pp` em `M3` (`40,0%`), corte de perda de `-$56,85` para `-$8,38` (`M2`), ainda negativo (`E[R] -0,81`).
    - `B4_FUNDING_WINDOW_FADE` (campeão WR 60d `48,4%`): `M0` (`15 tr`, WR `46,7%`, PF `1,234`, `+$3,82`) -> **`M2`/`M4`** (`6 tr`, WR **`50,0%` (`+3,33 pp`)**, PF `1,160 / 1,182`, `E[R] +0,0060`, `+$1,17 / +$1,60`) vs **`M3` (`12 tr`, WR `50,0%`, PF `1,462`, `+$4,43`, DD `0,06%`)**, mostrando que em setups *fade/contrarian* o peso contínuo (`M3`) evita o corte excessivo de frequência do gate binário (`32->8` vs `32->23`), mas a amostra (`N=6..12`, `H1 -$4,70`) ainda é insuficiente para promoção com dinheiro real.
    - `A8_RANGE_VWAP_REV_PULLBACK` (`47,8%` WR 9-13m): em 29d só teve `8` candidatos (`7 tr M0`, `42,9% WR`, `-$1,43`), caindo para `2 tr` em `M2` (`0% WR`, `-$4,23`).
    - `A9b_SWEEP_FAKEOUT_PULLBACK` (`104 tr M0`): `M2` piorou WR (`33,6% -> 28,0%`, `-5,65 pp`) e `E[R]` (`-0,299 -> -0,450`), provando que **estratégias de reversão de varredura não combinam com gate direcional padrão** (exigem detecção específica de *absorção* ou apenas veto a cascata extrema contrária).

### 4.3 Pontos de Decisão para Discussão no Domingo
1. **Regra no patch S4.2:** Bloquear fallback `UNAVAILABLE_PASSED` em `BTC/ETH/SOL` (`DIRECTIONAL` / `SCALP`) com flag por estratégia; em `GOLDEN_POT`, manter proteção contra `symbol_not_covered` reduzindo mão (`0.50x`) em vez de bloquear 99% dos altcoins.
2. **Peso no Comitê vs Gate Binário:** Não injetar `TapeReadingAgent` com `HOLD` permissivo no `DIRECTIONAL` (`M3` piorou `M1`/`M2`); estudar uso de `M3` **apenas** em pesquisa futura de *fades* (`B4`) em `shadow`.
3. **Pesquisa futura `SCALP`:** Se quiser reabilitar `SCALP` além de `shadow`, focar em `B4_FUNDING_WINDOW_FADE + M3/M4` ou em timeframes `30m/1h`, após coletar mais base L2 para aumentar `N` além de `6–15` trades/mês.

---

## Item 5 — Auditoria Isolada da S4 (09/11–14/11, N=31) & Hipóteses Extras para Revisão de Domingo
Referência de código: [technical_agent.py](file:///home/eudespatricio/ia_trader/agents/technical_agent.py#L224-L230) | Fonte: [market_cache.db](file:///home/eudespatricio/ia_trader/datasets/market_cache.db) (`mode=ro`).

### 5.1 Fechamento Isolado S4 (`2026-11-09 00:00 UTC` .. `2026-11-14 17:20 UTC`, `0` posições abertas)
- **Equity & Risco:** `$9.892,11 -> $11.796,83` (`+$1.904,72` / **`+19,25%`**), MaxDD `-4,49%`, `5` de `6` dias positivos.
- **Resultado líquido por posição (`true_net`, descontando taxas de entrada, saída e funding):**
  - **Total S4 (`31` tr):** WR nominal `67,7%` (`21/31`), WR real `TP2+` `48,4%` (`15/31`, `6` em `TP1+BE`), PF `4,89`, Ganho/Perda médios `+$114,19` / `-$49,08`, Líquido **`+$1.907,15`**.
  - **`DIRECTIONAL` 1h (`16` tr, `16 SHORT / 0 LONG`):** WR `81,2%` (`13/16`), `TP2+` `75,0%` (`12/16`), PF `7,60`, Líquido **`+$2.008,30`**.
  - **`GOLDEN_POT` 5m (`13` tr, `8 SHORT / 5 LONG`):** WR `61,5%` (`8/13`), `TP2+` `23,1%` (`3/13`, `5` em `TP1+BE`), PF `0,64`, Líquido **`-$48,99`** (`LONG` `-$46,87`, `SHORT` `-$2,12`).
  - **`SCALP` 15m (`2` tr, `1 SHORT / 1 LONG`):** WR `0,0%` (`0/2`), Líquido **`-$52,17`**.

### 5.2 Méritos Comprovados dos Mecanismos da S4 (Sem Misturar `S1..S3`)
- **Funil + `_flow_gate` S4 no `DIRECTIONAL` (`321` avaliadas -> `49` aprovadas no comitê -> `16` executadas):** barrou **`4/4` tentativas de `BUY` em repiques** (`0,0%` WR contrafactual, `-1,19R` médio) e separou os `12` `SELL PASSED (OK)` (**`83,3%` WR, `+1,47R` médio**) dos `27` `SELL BLOCKED` (`48,1%` WR, `+0,31R` médio); nos 2 dias de alta do BTC (`09/11` `+0,67%` e `11/11` `+0,29%`), manteve disciplina (`0` trades no dia `09` e `1/1` win `+$50,25` no dia `11`).
- **Achado S4 — Convergência do Ensemble `ML 50k` ([technical_agent.py](file:///home/eudespatricio/ia_trader/agents/technical_agent.py#L224-L230)):**
  - `DIRECTIONAL`: com `ML 50k` convergente fez **`5/5` (`100,0%` WR, `+$804,47`)** vs `8/11` (`72,7%`, `+$1.203,83`) sem convergência — **todos os `3` únicos stops da semana (`-$304,19` simultâneos em `BTC+ETH+SOL` em `10/11`) ocorreram sem convergência da `ML 50k`**.
  - `GOLDEN_POT`: com `ML 50k` convergente fez **`7/8` (`87,5%` WR, `+$41,40`)** vs **`1/5` (`20,0%` WR, `-$90,39`)** sem convergência.
  - Consolidado S4: `12/14` (**`85,7%` WR**, `+$835,33`) com `ML 50k` convergente vs `9/17` (`52,9%` WR) sem convergência (`+32,8 pp`).

### 5.3 Fragilidades Exclusivas da S4 e Decisões Pendentes para Domingo
1. **Squeeze triplo correlacionado:** `100%` do DD da S4 (`-4,49%`, `-$304,19`) veio de `BTC+ETH+SOL` estopados juntos em `10/11` (depois `11/11` wins seguidos) -> resolvido pelo Item 2 (`max_same_dir_majors=2`).
2. **Agentes contextuais mudos na S4:** `MacroAgent`, `SentimentAgent`, `OnChainAgent` e `RiskSentinelAgent` votaram **`31/31 HOLD`** em `agent_learning_memory`.
3. **Falta de L2 efetivo no `GOLDEN_POT`:** `13/13` executados entraram via `UNAVAILABLE_PASSED` e `5/8` gains pararam em `TP1+BE`.
4. **Novo teste candidato para domingo (`GOLDEN_POT + ML 50k Gate`):** Validar em backtest exigir convergência da `ML 50k` (`>= 35%` na direção do gatilho) no `GOLDEN_POT` junto ao filtro Macro BTC (Item 1) e lote `0.50x` sem L2.
5. **Diretriz de Meta (`~70%+` Sniper vs Payoff):** Perseguir `~70%+` por hiper-seletividade (`Comitê + ML 50k + Tape OK` + abstenção em mercado lateral na S5), guiado por `PF >= 1.50`, `E[R] >= +0.15R` e WR real (`TP2+`), sem inflar WR nominal com `BREAKEVEN_STOP`.

---

## Resolução Final & Aplicação em Produção — Patch S4.2 (`v1.0.7`, 2026-11-15)

O backlog foi homologado e aplicado em produção em `2026-11-15` estruturado em 5 frentes de ação:

1. **`GOLDEN_POT` 5m (Item 1 & 4 — Macro BTC + Política L2 Híbrida `M4`):**
   - Ativado filtro macro `BTCUSDT 1h dist_sma_200` (`±1,5%`), vetando entradas contra a tendência estrutural do Bitcoin.
   - Adotada a **política L2 híbrida `M4`**: dimensionamento integral (`1.00x`) quando `Tape status == "OK"`; sob `UNAVAILABLE` (`symbol_not_covered`), opera com dimensionamento defensivo `0.50x`. Nenhuma altcoin é excluída ou silenciada, preservando telemetria integral de aprendizado. `ML 50k` mantida para modulação/pesquisa sem bloqueio rígido.
2. **`DIRECTIONAL` 1h (Item 2 — Trava de Correlação + Gate Estrito):**
   - Ativada restrição `MAX_SAME_DIR_MAJORS = 2`, eliminando o risco de squeeze triplo simultâneo em `BTC`, `ETH` e `SOL`.
   - `_flow_gate` endurecido para modo *fail-closed* estrito (`status == "OK"` obrigatório nos 3 pares principais).
   - Teste pós-sábado confirmou refutação de `ML 50k` como trava rígida (`5/7` wins com vs `8/11` sem; ambos os stops de domingo continham `ML 50k`), mantendo-se a seletividade via microestrutura e correlação.
3. **`SCALP` 15m (Item 3 — Reabilitação Universal Opção 3B):**
   - O estudo cruzado `E10.9 x ML 15m OOS` (`PurgedTimeSeriesSplit` Maio–Nov/2026) comprovou viabilidade matemática positiva com gatilho `A6_SQUEEZE_KELTNER_PULLBACK + ML 15m >= 0.45` (`N=37, WR 54,0%, PF 1,447, E[R] = +0,293R`, positivo em `H1` e `H2`).
   - Implementada a **Opção 3B Universal**: modelos dedicados de 15m (`lgbm_15m_v1` + `rf_15m_v1`), trigger Squeeze BB/Keltner, filtro `ML 15m >= 0.45`, `Tape status == "OK"`, saída limpa com alvo único (`SL 1.4x ATR`, `TP 2.2x ATR`, removendo o atrito e armadilha da escada `TP1 BE +4bps`) e risco defensivo `0.50x` em **todos os 3 majors (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`)**, sem quarentena ou isolamento de moedas, maximizando telemetria até dezembro.
4. **Infraestrutura, Paridade e Governança (Frentes 0 & 4):**
   - Ativada persistência de Brier score no `LearningTracker`, janela `LIMIT 250` no `RegimeClassifier`, descarte de velas sobrepostas não-binance em `candles` e nova aba de evolução no dashboard.
   - Portada para `backtesting.py` a proteção intra-barra `closed_safe` (`v1.0.5`), garantindo paridade total entre backtest e produção.
   - Suíte completa de testes aprovada, governança registrada sob `E10.11` e hot-restart do supervisor executado com sucesso preservando as 2 posições abertas da S4.

