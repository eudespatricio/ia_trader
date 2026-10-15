# A/B de Ajustes de Consenso (P1–P6) — Resultados e Recomendação

**Autor:** quant_developer · **Data:** 2026-10-15 · **Escopo:** 100% offline; piloto congelado e produção intocada.
**Reprodução:** `.venv/bin/python scripts/ab_consensus_experiment.py` (≈70 s) · testes: `tests/test_ab_consensus_experiment.py` · dados brutos: `docs/ab_consenso_resultados.json`.

## 0. RECOMENDAÇÃO (leia primeiro)

| Proposta | Efeito medido | Decisão |
|---|---|---|
| **P1a** (excluir agentes sem dado do denominador) | ×2,1 trades; Δ retorno holdout −779 pp do capital; 0/6 blocos | **NÃO APLICAR** (piora) |
| **P1b** (limiar 0.30→0.22) | ×1,6 trades; Δ −316 pp; 0/6 blocos | **NÃO APLICAR** (piora) |
| **P2** (mapear `volume_sma`/`atr_median`) | Δ ≈ 0 (−14 pp, IC inclui 0); 1.614 barras vetadas em 6 anos | **Aplicar só após o piloto** — é correção de *bug* (vetos documentados que nunca disparam), não de alfa |
| **P3** (TVL 1×, DXY ±1.0) | **Inerte** no piloto (agentes On-Chain/Macro sem contexto); TVL sem histórico | **Aplicar só após o piloto** (higiene), sem evidência de efeito |
| **P4** (OrderFlow prio. 5 mais estrita) | Menos trades (−9%), Δ +77 pp (p=0,001; IC Bonferroni inf. > 0), 5/6 blocos — **mas 78% do ganho é "operar menos"**; Δ exp./trade +0,006% com IC [−0,053; +0,063]% | **Aplicar só após o piloto**, se o piloto reprovar por excesso de custo; não é alfa |
| **P5** (RSI 30–70 no momentum) | Δ ≈ 0; expectância/trade **piora** (−0,203% vs −0,175%) | **NÃO APLICAR** (dado contradiz a hipótese) |
| **P6** (F&G monotônico) | **Inerte** em PROD e em FULL (decisões idênticas) | **Aplicar só após o piloto** (higiene, efeito nulo) |

**Nenhuma variante é "melhora defensável" pelo critério completo** (DSR ≥ 0,95 etc.), e **nenhuma — nem o baseline — tem alfa líquido**. Nada deve ser aplicado durante o piloto (alterar parâmetros reinicia o relógio e viola A11).

## 0.1 Achados cínicos (mais importantes que o A/B)

1. **O baseline não tem edge bruto.** Com TP 1,5×ATR / SL 1,0×ATR o hit rate de equilíbrio bruto é 1/(1+1,5) = **40,0%**; o baseline acerta **39,8%** (bruto), expectância bruta −0,054%/trade e líquida −0,175% (custo 0,16% + funding). Sharpe líquido ≈ −5 em todas as janelas. Todas as variantes, em ambos os mundos, ficam entre −4,4 e −8,2. As diferenças entre variantes são ruído em torno de "coin-flip menos custos".
2. **Frequência absurda:** ~4.400 trades em 2,7 anos de holdout × 3 ativos (~1 trade a cada ~5 h por ativo). O gate `confidence ≥ 0.70` é frouxo porque Technical (fallback RSI/SMA200) quase sempre vota. Custo domina: o único "ganho" real de P4 é reduzir trades.
3. **"Melhorar" = perder menos.** A decomposição mostra que quase todo Δ positivo vem de menos trades × expectância negativa (P4: +60 de +77 pp; P5: +93 pp "menos trades" anulados por −88 pp de pior qualidade). Isso não é alfa; um filtro que bloqueasse *tudo* "venceria" todos os testes de Δ pareado. Por isso a coluna "Alfa líquido" (IC95 inferior da expectância > 0) é o critério que importa: **todas = não**.
4. **P1a/P1b pioram porque funcionam como desenhados**: o denominador inflado (4,5 vs ~1,8) é o que hoje *freia* um Technical sem edge. Corrigi-lo libera mais trades ruins. A "falha" apontada pela auditoria é, na prática, um freio acidental.
5. **P1a_F ≡ P1b_F** (mundo FULL): decisões idênticas — o score é bimodal e quase não há barras entre os dois cortes (0,99 vs 1,05 soma ponderada). Não são duas evidências independentes.
6. **P3/P6 são inertes no piloto:** o `evaluate_swing_opportunities` nunca fornece F&G/DXY/VIX/stablecoins/TVL; os 3 agentes votam HOLD "sem dados" sempre. Mesmo com os dados (mundo FULL) P6 não altera nenhuma decisão e P3 só muda DXY ±1,5→±1,0 sem cruzar limiares. Dois dos seis "achados" não têm efeito algum mensurável.
7. **Overfitting/data-mining:** 17 configurações, 3 símbolos, 6 blocos e ~15 comparações no mesmo holdout; correção de Bonferroni aplicada (α=0,0033) e DSR com N=17 (e N+83 do `trial_log`). Nenhum parâmetro foi ajustado (janela ATR=100, limiar volume 1,2 e RSI 30/70 fixados a priori, sem sweep) — mas o próprio P4 seria o "vencedor" escolhido pós-hoc entre 15; sem correção de seleção seu p=0,001 viraria p≈0,015 só por Bonferroni e continua não sendo alfa. DSR=0 é esperado: Sharpe observado é negativo.
8. **Limitações:** (a) execução simplificada (bracket SL/TP fixo, sem TP1/TP2 escalonado do engine, sem limite de margem entre símbolos — MaxDD "cap." é aditivo, não composto, e supera 100% porque até 3 posições de 100% de notional se sobrepõem; usar para *comparar*, não como risco real); (b) features calculadas sobre histórico completo (piloto usa janela de 400 barras: EMA/ATR/RSI diferem ligeiramente no warmup); (c) TVL não existe no cache → parte TVL de P3 só tem teste unitário; (d) P5/P6 implementados por neutralização de entrada / pós-ajuste de confiança (equivalentes funcionais, não reescrita do agente); P6 aplicado simetricamente em 75 para não criar viés direcional (desvio do texto, que cita só 25); (e) **sem ML**: TechnicalAgent do piloto roda no fallback de regras, replicado fielmente.
9. **Possível bug de produção (não corrigido, só reportado):** o piloto pode usar a última vela *em formação* do DataLoader; aqui todas as velas são fechadas, então o baseline do backtest é mais favorável que o live.

## 1. Metodologia e tabelas (geradas automaticamente)


- **Dados:** Binance 1h do cache (BTC/ETH/SOL), candles fechados (última barra descartada), features `FeatureEngineer.compute_all_features(drop_na=False)` como no piloto.
- **Baseline (BASE)** = réplica do `evaluate_swing_opportunities`: contexto só com features 1h (sem F&G/macro/on-chain, **sem `volume_sma` nem `atr_median`**), TechnicalAgent sem modelo, `ConsensusJudge(0.30)`, gate `confidence ≥ 0.70`.
- **Mundo PROD** = contexto idêntico ao piloto. **Mundo FULL** = + F&G, DXY, VIX, S&P, stablecoins diários defasados +24h (somente ≥ 2021-12-01). TVL não existe no cache → metade TVL de P3 só é coberta por teste unitário.
- **Execução:** decisão no fechamento de *t*, entrada na abertura de *t+1*; SL 1.0×ATR / TP 1.5×ATR (`SignalGenerator`, sizing 1% risco, 1× alavancagem, capital $10k); SL vence TP na mesma barra; timeout 48 barras; 1 posição por símbolo; funding real do cache.
- **Custos:** taker 0,06% + slippage 0,02% por perna em *todos* os fills (0,16% round-trip) — SL tratado como taker.
- **Split PROD:** 2020-01-09 → TREINO até 2024-01-30; embargo 10d; HOLDOUT 2024-02-09 → 2026-10-15 (trades que cruzam a fronteira são descartados).
- **Split FULL:** 2021-12-01 → TREINO até 2024-11-02; embargo 10d; HOLDOUT 2024-11-12 → 2026-10-15 (trades que cruzam a fronteira são descartados).
- **Estatística:** DSR com N = 17 variantes (e N+83 do trial_log como cenário mais severo); variância dos Sharpes = entre variantes do mundo no holdout; bootstrap por blocos móveis (bloco 7d, 5000 reamostras) da diferença diária pareada vs baseline; IC com correção de Bonferroni (K = 15; α = 0.0033); consistência em 6 blocos temporais.
- **Critério de 'melhora defensável' (TODOS):** diferença pareada média > 0 no holdout, limite inferior Bonferroni > 0, DSR ≥ 0,95, melhora também no treino, ≥ 4/6 blocos positivos, ≥ 100 trades no holdout, e variante não-inerte.

## 2.1 Mundo PROD

### PROD — HOLDOUT

| Variante | Sinais exec. | Decisões brutas | Vetos (barras) | Hit rate | Hit bruto | Exp. bruta/trade | Exp. líquida/trade | IC95 exp. líq. | Sharpe | MaxDD (cap.) | Retorno total (cap.) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| BASE | 4389 | — | — | 39.8% | 39.8% | -0.054% | -0.175% | [-0.217%, -0.135%] | -4.97 | -675.05% | -659.85% |
| P1a | 9377 | — | — | 39.5% | 39.6% | -0.054% | -0.176% | [-0.204%, -0.148%] | -6.99 | -1441.67% | -1438.61% |
| P1b | 6903 | — | — | 40.2% | 40.3% | -0.046% | -0.167% | [-0.199%, -0.135%] | -5.36 | -983.17% | -975.70% |
| P2 | 4470 | — | — | 39.9% | 39.9% | -0.055% | -0.176% | [-0.216%, -0.138%] | -5.00 | -687.15% | -673.77% |
| P3 | 4389 | — | — | 39.8% | 39.8% | -0.054% | -0.175% | [-0.217%, -0.135%] | -4.97 | -675.05% | -659.85% |
| P4 | 3988 | — | — | 39.9% | 39.9% | -0.048% | -0.170% | [-0.212%, -0.129%] | -4.62 | -601.56% | -582.52% |
| P5 | 3770 | — | — | 39.0% | 39.0% | -0.082% | -0.203% | [-0.245%, -0.160%] | -5.51 | -675.89% | -654.67% |
| P6 | 4389 | — | — | 39.8% | 39.8% | -0.054% | -0.175% | [-0.217%, -0.135%] | -4.97 | -675.05% | -659.85% |
| P2+P4 | 4234 | — | — | 39.9% | 39.9% | -0.051% | -0.173% | [-0.213%, -0.132%] | -4.81 | -638.91% | -626.20% |
| ALL_a | 9764 | — | — | 39.2% | 39.2% | -0.066% | -0.187% | [-0.213%, -0.160%] | -8.18 | -1596.37% | -1595.01% |
| ALL_b | 5835 | — | — | 39.4% | 39.4% | -0.072% | -0.193% | [-0.229%, -0.159%] | -5.97 | -969.73% | -957.71% |

### PROD — ALL

| Variante | Sinais exec. | Decisões brutas | Vetos (barras) | Hit rate | Hit bruto | Exp. bruta/trade | Exp. líquida/trade | IC95 exp. líq. | Sharpe | MaxDD (cap.) | Retorno total (cap.) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| BASE | 10902 | 18270 | 52 | 39.8% | 39.9% | -0.037% | -0.161% | [-0.196%, -0.126%] | -4.36 | -1475.82% | -1460.17% |
| P1a | 22635 | 109413 | 52 | 40.0% | 40.0% | -0.052% | -0.175% | [-0.198%, -0.151%] | -6.20 | -3071.93% | -3060.51% |
| P1b | 17131 | 53788 | 52 | 40.0% | 40.0% | -0.036% | -0.160% | [-0.188%, -0.133%] | -5.06 | -2300.14% | -2279.76% |
| P2 | 11044 | 18517 | 1614 | 39.9% | 40.0% | -0.037% | -0.161% | [-0.195%, -0.127%] | -4.39 | -1488.01% | -1472.56% |
| P3 | 10902 | 18270 | 52 | 39.8% | 39.9% | -0.037% | -0.161% | [-0.196%, -0.126%] | -4.36 | -1475.82% | -1460.17% |
| P4 | 9968 | 15387 | 52 | 39.6% | 39.7% | -0.042% | -0.166% | [-0.202%, -0.130%] | -4.24 | -1380.18% | -1362.87% |
| P5 | 9401 | 14953 | 52 | 39.1% | 39.2% | -0.061% | -0.185% | [-0.223%, -0.148%] | -4.61 | -1430.30% | -1415.82% |
| P6 | 10902 | 18270 | 52 | 39.8% | 39.9% | -0.037% | -0.161% | [-0.196%, -0.126%] | -4.36 | -1475.82% | -1460.17% |
| P2+P4 | 10471 | 16494 | 1614 | 39.8% | 39.8% | -0.042% | -0.166% | [-0.200%, -0.132%] | -4.38 | -1447.80% | -1431.92% |
| ALL_a | 23599 | 116591 | 1614 | 39.3% | 39.4% | -0.074% | -0.196% | [-0.219%, -0.173%] | -7.30 | -3555.93% | -3548.54% |
| ALL_b | 14467 | 33108 | 1614 | 39.4% | 39.4% | -0.059% | -0.183% | [-0.213%, -0.152%] | -5.36 | -2179.79% | -2162.23% |

### PROD — Defensabilidade estatística vs baseline (HOLDOUT)

| Variante | Δ média diária (cap.) | IC95 bloco | IC Bonferroni inf. | p(Δ≤0) | DSR (N) | DSR (N+log) | Δ treino | Blocos + | Veredito |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| BASE | — | — | — | — | 0.00 | 0.00 | — | — | referência |
| P1a | -0.7955% | [-0.9982%, -0.5850%] | -1.0998% | 1.000 | 0.00 | 0.00 | -0.5480% | 0/6 | ❌ holdout não melhora; IC Bonferroni inclui 0; DSR 0.00<0.95; treino não melhora; só 0/6 blocos melhoram |
| P1b | -0.3226% | [-0.4634%, -0.1812%] | -0.5324% | 1.000 | 0.00 | 0.00 | -0.3383% | 0/6 | ❌ holdout não melhora; IC Bonferroni inclui 0; DSR 0.00<0.95; treino não melhora; só 0/6 blocos melhoram |
| P2 | -0.0142% | [-0.0448%, +0.0164%] | -0.0611% | 0.831 | 0.00 | 0.00 | +0.0018% | 3/6 | ❌ holdout não melhora; IC Bonferroni inclui 0; DSR 0.00<0.95; só 3/6 blocos melhoram |
| P3 | +0.0000% | [+0.0000%, +0.0000%] | +0.0000% | 1.000 | 0.00 | 0.00 | +0.0000% | 0/6 | ❌ inerte (decisões idênticas ao baseline) |
| P4 | +0.0790% | [+0.0271%, +0.1293%] | +0.0076% | 0.001 | 0.00 | 0.00 | +0.0156% | 5/6 | ❌ DSR 0.00<0.95 |
| P5 | +0.0053% | [-0.0642%, +0.0694%] | -0.1034% | 0.434 | 0.00 | 0.00 | +0.0265% | 4/6 | ❌ IC Bonferroni inclui 0; DSR 0.00<0.95 |
| P6 | +0.0000% | [+0.0000%, +0.0000%] | +0.0000% | 1.000 | 0.00 | 0.00 | +0.0000% | 0/6 | ❌ inerte (decisões idênticas ao baseline) |
| P2+P4 | +0.0344% | [-0.0093%, +0.0818%] | -0.0331% | 0.076 | 0.00 | 0.00 | -0.0008% | 4/6 | ❌ IC Bonferroni inclui 0; DSR 0.00<0.95; treino não melhora |
| ALL_a | -0.9552% | [-1.1672%, -0.7417%] | -1.2630% | 1.000 | 0.00 | 0.00 | -0.7660% | 0/6 | ❌ holdout não melhora; IC Bonferroni inclui 0; DSR 0.00<0.95; treino não melhora; só 0/6 blocos melhoram |
| ALL_b | -0.3043% | [-0.4425%, -0.1707%] | -0.5040% | 1.000 | 0.00 | 0.00 | -0.2664% | 0/6 | ❌ holdout não melhora; IC Bonferroni inclui 0; DSR 0.00<0.95; treino não melhora; só 0/6 blocos melhoram |

### PROD — O que explica o Δ? (HOLDOUT; retorno total em % do capital, não composto)

| Variante | Δ retorno total | do qual: menos trades | do qual: trade melhor | Δ exp./trade | IC95 | IC Bonferroni inf. | Alfa líquido (IC95 inf.>0)? |
|---|---:|---:|---:|---:|---|---:|---|
| BASE | — | — | — | — | — | — | não |
| P1a | -778.8% | -749.9% | -28.9% | -0.000% | [-0.050%, +0.047%] | -0.074% | não |
| P1b | -315.9% | -378.0% | +62.1% | +0.008% | [-0.044%, +0.059%] | -0.068% | não |
| P2 | -13.9% | -12.2% | -1.8% | -0.001% | [-0.057%, +0.057%] | -0.091% | não |
| P3 | +0.0% | -0.0% | +0.0% | +0.000% | [-0.055%, +0.056%] | -0.084% | não |
| P4 | +77.3% | +60.3% | +17.0% | +0.006% | [-0.053%, +0.063%] | -0.080% | não |
| P5 | +5.2% | +93.1% | -87.9% | -0.028% | [-0.086%, +0.031%] | -0.110% | não |
| P6 | +0.0% | -0.0% | +0.0% | +0.000% | [-0.055%, +0.056%] | -0.084% | não |
| P2+P4 | +33.6% | +23.3% | +10.3% | +0.002% | [-0.055%, +0.058%] | -0.080% | não |
| ALL_a | -935.2% | -808.1% | -127.1% | -0.011% | [-0.059%, +0.038%] | -0.083% | não |
| ALL_b | -297.9% | -217.4% | -80.5% | -0.018% | [-0.073%, +0.034%] | -0.095% | não |

## 2.2 Mundo FULL

### FULL — HOLDOUT

| Variante | Sinais exec. | Decisões brutas | Vetos (barras) | Hit rate | Hit bruto | Exp. bruta/trade | Exp. líquida/trade | IC95 exp. líq. | Sharpe | MaxDD (cap.) | Retorno total (cap.) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| BASE_F | 2638 | — | — | 40.3% | 40.3% | -0.035% | -0.156% | [-0.208%, -0.105%] | -4.37 | -387.90% | -375.04% |
| P1a_F | 3028 | — | — | 40.1% | 40.1% | -0.043% | -0.165% | [-0.211%, -0.121%] | -4.82 | -467.84% | -455.17% |
| P1b_F | 3028 | — | — | 40.1% | 40.1% | -0.043% | -0.165% | [-0.211%, -0.121%] | -4.82 | -467.84% | -455.17% |
| P3_F | 2638 | — | — | 40.3% | 40.3% | -0.035% | -0.156% | [-0.208%, -0.105%] | -4.37 | -387.90% | -375.04% |
| P6_F | 2638 | — | — | 40.3% | 40.3% | -0.035% | -0.156% | [-0.208%, -0.105%] | -4.37 | -387.90% | -375.04% |
| ALL_a_F | 2508 | — | — | 39.7% | 39.7% | -0.060% | -0.182% | [-0.231%, -0.132%] | -5.13 | -422.17% | -411.85% |

### FULL — ALL

| Variante | Sinais exec. | Decisões brutas | Vetos (barras) | Hit rate | Hit bruto | Exp. bruta/trade | Exp. líquida/trade | IC95 exp. líq. | Sharpe | MaxDD (cap.) | Retorno total (cap.) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| BASE_F | 6683 | 13326 | 52 | 40.0% | 40.1% | -0.030% | -0.153% | [-0.190%, -0.117%] | -4.13 | -938.97% | -913.49% |
| P1a_F | 7660 | 38995 | 52 | 39.9% | 39.9% | -0.034% | -0.158% | [-0.192%, -0.124%] | -4.50 | -1112.00% | -1087.49% |
| P1b_F | 7660 | 38995 | 52 | 39.9% | 39.9% | -0.034% | -0.158% | [-0.192%, -0.124%] | -4.50 | -1112.00% | -1087.49% |
| P3_F | 6683 | 13326 | 52 | 40.0% | 40.1% | -0.030% | -0.153% | [-0.190%, -0.117%] | -4.13 | -938.97% | -913.49% |
| P6_F | 6683 | 13326 | 52 | 40.0% | 40.1% | -0.030% | -0.153% | [-0.190%, -0.117%] | -4.13 | -938.97% | -913.49% |
| ALL_a_F | 6429 | 23961 | 1614 | 39.3% | 39.3% | -0.065% | -0.188% | [-0.223%, -0.152%] | -4.80 | -1037.95% | -1021.00% |

### FULL — Defensabilidade estatística vs baseline (HOLDOUT)

| Variante | Δ média diária (cap.) | IC95 bloco | IC Bonferroni inf. | p(Δ≤0) | DSR (N) | DSR (N+log) | Δ treino | Blocos + | Veredito |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| BASE_F | — | — | — | — | 0.00 | 0.00 | — | — | referência |
| P1a_F | -0.1142% | [-0.1843%, -0.0408%] | -0.2193% | 0.999 | 0.00 | 0.00 | -0.0928% | 0/6 | ❌ holdout não melhora; IC Bonferroni inclui 0; DSR 0.00<0.95; treino não melhora; só 0/6 blocos melhoram |
| P1b_F | -0.1142% | [-0.1843%, -0.0408%] | -0.2193% | 0.999 | 0.00 | 0.00 | -0.0928% | 0/6 | ❌ holdout não melhora; IC Bonferroni inclui 0; DSR 0.00<0.95; treino não melhora; só 0/6 blocos melhoram |
| P3_F | +0.0000% | [+0.0000%, +0.0000%] | +0.0000% | 1.000 | 0.00 | 0.00 | +0.0000% | 0/6 | ❌ inerte (decisões idênticas ao baseline) |
| P6_F | +0.0000% | [+0.0000%, +0.0000%] | +0.0000% | 1.000 | 0.00 | 0.00 | +0.0000% | 0/6 | ❌ inerte (decisões idênticas ao baseline) |
| ALL_a_F | -0.0524% | [-0.1579%, +0.0620%] | -0.2149% | 0.841 | 0.00 | 0.00 | -0.0632% | 2/6 | ❌ holdout não melhora; IC Bonferroni inclui 0; DSR 0.00<0.95; treino não melhora; só 2/6 blocos melhoram |

### FULL — O que explica o Δ? (HOLDOUT; retorno total em % do capital, não composto)

| Variante | Δ retorno total | do qual: menos trades | do qual: trade melhor | Δ exp./trade | IC95 | IC Bonferroni inf. | Alfa líquido (IC95 inf.>0)? |
|---|---:|---:|---:|---:|---|---:|---|
| BASE_F | — | — | — | — | — | — | não |
| P1a_F | -80.1% | -55.4% | -24.7% | -0.009% | [-0.076%, +0.060%] | -0.117% | não |
| P1b_F | -80.1% | -55.4% | -24.7% | -0.009% | [-0.076%, +0.060%] | -0.117% | não |
| P3_F | +0.0% | -0.0% | +0.0% | +0.000% | [-0.070%, +0.070%] | -0.100% | não |
| P6_F | +0.0% | -0.0% | +0.0% | +0.000% | [-0.070%, +0.070%] | -0.100% | não |
| ALL_a_F | -36.8% | +18.5% | -55.3% | -0.026% | [-0.095%, +0.043%] | -0.131% | não |

## 3. Legenda das variantes

- **BASE** (PROD): Réplica do piloto congelado
- **P1a** (PROD): Juiz exclui agentes sem dados do denominador
- **P1b** (PROD): min_consensus_threshold 0.30→0.22
- **P2** (PROD): volume_sma/atr_median mapeados (vetos + vol_ratio reais)
- **P3** (PROD): OnChain TVL 1x + DXY ±1.0 (inerte em PROD)
- **P4** (PROD): OrderFlow prio 5: |pos|>=0.5 ou volume
- **P5** (PROD): Technical momentum RSI 30..70
- **P6** (PROD): Sentiment monotônico F&G=25 (inerte em PROD)
- **P2+P4** (PROD): P4 com vol_ratio real (requer P2)
- **ALL_a** (PROD): Tudo, com P1a
- **ALL_b** (PROD): Tudo, com P1b
- **BASE_F** (FULL): Baseline com contexto enriquecido (F&G/macro/stable)
- **P1a_F** (FULL): P1a com contexto enriquecido
- **P1b_F** (FULL): P1b com contexto enriquecido
- **P3_F** (FULL): DXY ±1.0 (TVL inobservável)
- **P6_F** (FULL): Sentiment monotônico
- **ALL_a_F** (FULL): Tudo, com P1a

_Tempo de execução: 70s._
