# A/B de Ajustes de Consenso (P1–P6) — Resultados OFFLINE

> Gerado por `scripts/ab_consensus_experiment.py` (produção intocada; cache SQLite em modo somente-leitura). Piloto congelado desde 2026-10-15 00:47 UTC — **nada aqui altera parâmetros do piloto**.

## 1. Metodologia

- **Dados:** Binance 1h do cache (BTC/ETH/SOL), candles fechados (última barra descartada), features `FeatureEngineer.compute_all_features(drop_na=False)` como no piloto.
- **Baseline (BASE)** = réplica do `evaluate_swing_opportunities`: contexto só com features 1h (sem F&G/macro/on-chain, **sem `volume_sma` nem `atr_median`**), TechnicalAgent sem modelo, `ConsensusJudge(0.30)`, gate `confidence ≥ 0.70`.
- **Mundo PROD** = contexto idêntico ao piloto. **Mundo FULL** = + F&G, DXY, VIX, S&P, stablecoins diários defasados +24h (somente ≥ 2021-12-01). TVL não existe no cache → metade TVL de P3 só é coberta por teste unitário.
- **Execução (`BacktestEngine(pilot_parity=True)`, entrada `signal_close`):** MESMO engine do piloto — decisão no fechamento de *t*, entrada a mercado em close(t) (== open(t+1)); ladder TP1 40% (1,5×ATR) → SL=BE(+0,04%), TP2 30% (2,5×ATR) → trailing 1,2×ATR(entrada) só com barras fechadas, runner 30% em TP3 (4,0×ATR) ou trailing; SL 1,0×ATR; lote=margem $100, alavancagem `decide_leverage(confiança do sinal)` ≤5×, SL antes da liquidação; saldo único ($10k) com margem agregada ≤85% e ≤5 posições; timeout 48 barras; funding UTC 8h com taxa real do cache.
- **Custos (iguais ao piloto):** taker 0,06% (entrada, SL/BE/trailing/timeout/liquidação) + slippage de mercado 0,02%; TP1/TP2/TP3 maker 0,02% (TP1/TP2 sem slippage, TP3 0,01%). Retorno por trade = % do *notional* (invariante à alavancagem).
- **Split PROD:** 2020-01-09 → TREINO até 2024-01-30; embargo 10d; HOLDOUT 2024-02-09 → 2026-10-15 (trades que cruzam a fronteira são descartados).
- **Split FULL:** 2021-12-01 → TREINO até 2024-11-02; embargo 10d; HOLDOUT 2024-11-12 → 2026-10-15 (trades que cruzam a fronteira são descartados).
- **Estatística:** DSR com N = 17 variantes (e N+83 do trial_log como cenário mais severo); variância dos Sharpes = entre variantes do mundo no holdout; bootstrap por blocos móveis (bloco 7d, 5000 reamostras) da diferença diária pareada vs baseline; IC com correção de Bonferroni (K = 15; α = 0.0033); consistência em 6 blocos temporais.
- **Critério de 'melhora defensável' (TODOS):** diferença pareada média > 0 no holdout, limite inferior Bonferroni > 0, DSR ≥ 0,95, melhora também no treino, ≥ 4/6 blocos positivos, ≥ 100 trades no holdout, e variante não-inerte.

## 2.1 Mundo PROD

### PROD — HOLDOUT

| Variante | Sinais exec. | Decisões brutas | Vetos (barras) | Hit rate | Hit bruto | Exp. bruta/trade | Exp. líquida/trade | IC95 exp. líq. | Sharpe | MaxDD (cap.) | Retorno total (cap.) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| BASE | 4313 | — | — | 39.6% | 39.6% | -0.163% | -0.276% | [-0.316%, -0.235%] | -6.86 | -38.11% | -37.77% |
| P1a | 305 | — | — | 42.3% | 42.3% | -0.114% | -0.235% | [-0.426%, -0.047%] | -1.17 | -0.24% | -0.24% |
| P1b | 2570 | — | — | 39.3% | 39.3% | -0.160% | -0.273% | [-0.333%, -0.215%] | -4.15 | -18.36% | -18.32% |
| P2 | 4386 | — | — | 39.6% | 39.6% | -0.163% | -0.275% | [-0.315%, -0.235%] | -6.75 | -37.94% | -37.71% |
| P3 | 4313 | — | — | 39.6% | 39.6% | -0.163% | -0.276% | [-0.316%, -0.235%] | -6.86 | -38.11% | -37.77% |
| P4 | 3970 | — | — | 39.5% | 39.6% | -0.169% | -0.282% | [-0.324%, -0.239%] | -6.87 | -37.09% | -36.55% |
| P5 | 3720 | — | — | 39.0% | 39.0% | -0.166% | -0.279% | [-0.323%, -0.236%] | -6.45 | -33.47% | -32.95% |
| P6 | 4313 | — | — | 39.6% | 39.6% | -0.163% | -0.276% | [-0.316%, -0.235%] | -6.86 | -38.11% | -37.77% |
| P2+P4 | 4215 | — | — | 39.5% | 39.5% | -0.167% | -0.280% | [-0.319%, -0.240%] | -6.90 | -38.19% | -37.97% |
| ALL_a | 0 | — | — | 0.0% | 0.0% | 0.000% | 0.000% | [nan, nan] | +0.00 | 0.00% | 0.00% |
| ALL_b | 3455 | — | — | 39.0% | 39.0% | -0.171% | -0.284% | [-0.332%, -0.235%] | -5.25 | -27.26% | -27.16% |

### PROD — ALL

| Variante | Sinais exec. | Decisões brutas | Vetos (barras) | Hit rate | Hit bruto | Exp. bruta/trade | Exp. líquida/trade | IC95 exp. líq. | Sharpe | MaxDD (cap.) | Retorno total (cap.) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| BASE | 10800 | 18274 | 52 | 39.5% | 39.6% | -0.174% | -0.288% | [-0.321%, -0.255%] | -5.68 | -99.02% | -98.99% |
| P1a | 12902 | 109428 | 52 | 40.5% | 40.6% | -0.169% | -0.284% | [-0.320%, -0.248%] | -4.89 | -99.80% | -99.80% |
| P1b | 12474 | 53796 | 52 | 39.8% | 39.8% | -0.158% | -0.273% | [-0.309%, -0.239%] | -5.06 | -99.85% | -99.80% |
| P2 | 10960 | 18521 | 1614 | 39.6% | 39.6% | -0.172% | -0.286% | [-0.320%, -0.252%] | -5.78 | -99.51% | -99.49% |
| P3 | 10800 | 18274 | 52 | 39.5% | 39.6% | -0.174% | -0.288% | [-0.321%, -0.255%] | -5.68 | -99.02% | -98.99% |
| P4 | 9970 | 15391 | 52 | 39.2% | 39.2% | -0.179% | -0.294% | [-0.330%, -0.258%] | -5.57 | -95.31% | -94.77% |
| P5 | 9318 | 14955 | 52 | 38.8% | 38.9% | -0.176% | -0.290% | [-0.327%, -0.252%] | -5.18 | -84.74% | -84.57% |
| P6 | 10800 | 18274 | 52 | 39.5% | 39.6% | -0.174% | -0.288% | [-0.321%, -0.255%] | -5.68 | -99.02% | -98.99% |
| P2+P4 | 10500 | 16498 | 1614 | 39.3% | 39.3% | -0.178% | -0.293% | [-0.326%, -0.259%] | -5.81 | -99.10% | -99.06% |
| ALL_a | 11854 | 116608 | 1614 | 39.9% | 39.9% | -0.196% | -0.310% | [-0.348%, -0.274%] | -5.10 | -99.80% | -99.80% |
| ALL_b | 11839 | 33111 | 1614 | 38.7% | 38.8% | -0.180% | -0.295% | [-0.328%, -0.261%] | -5.27 | -99.80% | -99.80% |

### PROD — Defensabilidade estatística vs baseline (HOLDOUT)

| Variante | Δ média diária (cap.) | IC95 bloco | IC Bonferroni inf. | p(Δ≤0) | DSR (N) | DSR (N+log) | Δ treino | Blocos + | Veredito |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| BASE | — | — | — | — | 0.00 | 0.00 | — | — | referência |
| P1a | +0.0383% | [+0.0316%, +0.0459%] | +0.0281% | 0.000 | 0.00 | 0.00 | -0.0260% | 3/6 | ❌ DSR 0.00<0.95; treino não melhora; só 3/6 blocos melhoram |
| P1b | +0.0199% | [+0.0125%, +0.0278%] | +0.0092% | 0.000 | 0.00 | 0.00 | -0.0135% | 2/6 | ❌ DSR 0.00<0.95; treino não melhora; só 2/6 blocos melhoram |
| P2 | +0.0001% | [-0.0015%, +0.0017%] | -0.0022% | 0.461 | 0.00 | 0.00 | -0.0004% | 2/6 | ❌ IC Bonferroni inclui 0; DSR 0.00<0.95; treino não melhora; só 2/6 blocos melhoram |
| P3 | +0.0000% | [+0.0000%, +0.0000%] | +0.0000% | 1.000 | 0.00 | 0.00 | +0.0000% | 0/6 | ❌ inerte (decisões idênticas ao baseline) |
| P4 | +0.0013% | [-0.0011%, +0.0032%] | -0.0020% | 0.098 | 0.00 | 0.00 | +0.0021% | 6/6 | ❌ IC Bonferroni inclui 0; DSR 0.00<0.95 |
| P5 | +0.0049% | [+0.0019%, +0.0079%] | +0.0005% | 0.001 | 0.00 | 0.00 | +0.0064% | 6/6 | ❌ DSR 0.00<0.95 |
| P6 | +0.0000% | [+0.0000%, +0.0000%] | +0.0000% | 1.000 | 0.00 | 0.00 | +0.0000% | 0/6 | ❌ inerte (decisões idênticas ao baseline) |
| P2+P4 | -0.0002% | [-0.0020%, +0.0018%] | -0.0029% | 0.578 | 0.00 | 0.00 | +0.0002% | 3/6 | ❌ holdout não melhora; IC Bonferroni inclui 0; DSR 0.00<0.95; só 3/6 blocos melhoram |
| ALL_a | +0.0386% | [+0.0318%, +0.0461%] | +0.0284% | 0.000 | 0.00 | 0.00 | -0.0262% | 3/6 | ❌ DSR 0.00<0.95; treino não melhora; só 3/6 blocos melhoram; n_holdout=0<100 |
| ALL_b | +0.0108% | [+0.0040%, +0.0183%] | +0.0011% | 0.002 | 0.00 | 0.00 | -0.0075% | 1/6 | ❌ DSR 0.00<0.95; treino não melhora; só 1/6 blocos melhoram |

### PROD — O que explica o Δ? (HOLDOUT; retorno total em % do capital, não composto)

| Variante | Δ retorno total | do qual: menos trades | do qual: trade melhor | Δ exp./trade | IC95 | IC Bonferroni inf. | Alfa líquido (IC95 inf.>0)? |
|---|---:|---:|---:|---:|---|---:|---|
| BASE | — | — | — | — | — | — | não |
| P1a | +37.5% | +35.1% | +2.4% | +0.041% | [-0.158%, +0.240%] | -0.242% | não |
| P1b | +19.5% | +15.3% | +4.2% | +0.002% | [-0.067%, +0.074%] | -0.101% | não |
| P2 | +0.1% | -0.6% | +0.7% | +0.000% | [-0.055%, +0.058%] | -0.083% | não |
| P3 | +0.0% | -0.0% | +0.0% | +0.000% | [-0.057%, +0.058%] | -0.088% | não |
| P4 | +1.2% | +3.0% | -1.8% | -0.006% | [-0.064%, +0.051%] | -0.099% | não |
| P5 | +4.8% | +5.2% | -0.4% | -0.003% | [-0.062%, +0.057%] | -0.091% | não |
| P6 | +0.0% | -0.0% | +0.0% | +0.000% | [-0.057%, +0.058%] | -0.088% | não |
| P2+P4 | -0.2% | +0.9% | -1.1% | -0.004% | [-0.061%, +0.053%] | -0.086% | não |
| ALL_a | +37.8% | +37.8% | +0.0% | — | — | — | não |
| ALL_b | +10.6% | +7.5% | +3.1% | -0.008% | [-0.071%, +0.058%] | -0.101% | não |

## 2.2 Mundo FULL

### FULL — HOLDOUT

| Variante | Sinais exec. | Decisões brutas | Vetos (barras) | Hit rate | Hit bruto | Exp. bruta/trade | Exp. líquida/trade | IC95 exp. líq. | Sharpe | MaxDD (cap.) | Retorno total (cap.) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| BASE_F | 2627 | — | — | 40.0% | 40.1% | -0.150% | -0.262% | [-0.309%, -0.213%] | -6.27 | -20.74% | -20.45% |
| P1a_F | 2978 | — | — | 39.9% | 40.0% | -0.153% | -0.266% | [-0.312%, -0.221%] | -6.58 | -23.63% | -23.34% |
| P1b_F | 2978 | — | — | 39.9% | 40.0% | -0.153% | -0.266% | [-0.312%, -0.221%] | -6.58 | -23.63% | -23.34% |
| P3_F | 2627 | — | — | 40.0% | 40.1% | -0.150% | -0.262% | [-0.309%, -0.213%] | -6.27 | -20.74% | -20.45% |
| P6_F | 2627 | — | — | 40.0% | 40.1% | -0.150% | -0.262% | [-0.309%, -0.213%] | -6.27 | -20.74% | -20.45% |
| ALL_a_F | 2465 | — | — | 39.5% | 39.5% | -0.157% | -0.270% | [-0.318%, -0.218%] | -6.34 | -19.55% | -19.43% |

### FULL — ALL

| Variante | Sinais exec. | Decisões brutas | Vetos (barras) | Hit rate | Hit bruto | Exp. bruta/trade | Exp. líquida/trade | IC95 exp. líq. | Sharpe | MaxDD (cap.) | Retorno total (cap.) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| BASE_F | 6721 | 13331 | 52 | 39.8% | 39.8% | -0.153% | -0.266% | [-0.300%, -0.231%] | -5.43 | -53.96% | -53.21% |
| P1a_F | 7636 | 39003 | 52 | 39.8% | 39.9% | -0.151% | -0.265% | [-0.298%, -0.231%] | -5.81 | -60.85% | -60.09% |
| P1b_F | 7636 | 39003 | 52 | 39.8% | 39.9% | -0.151% | -0.265% | [-0.298%, -0.231%] | -5.81 | -60.85% | -60.09% |
| P3_F | 6721 | 13331 | 52 | 39.8% | 39.8% | -0.153% | -0.266% | [-0.300%, -0.231%] | -5.43 | -53.96% | -53.21% |
| P6_F | 6721 | 13331 | 52 | 39.8% | 39.8% | -0.153% | -0.266% | [-0.300%, -0.231%] | -5.43 | -53.96% | -53.21% |
| ALL_a_F | 6437 | 23964 | 1614 | 38.9% | 39.0% | -0.175% | -0.289% | [-0.324%, -0.253%] | -6.07 | -54.53% | -54.20% |

### FULL — Defensabilidade estatística vs baseline (HOLDOUT)

| Variante | Δ média diária (cap.) | IC95 bloco | IC Bonferroni inf. | p(Δ≤0) | DSR (N) | DSR (N+log) | Δ treino | Blocos + | Veredito |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| BASE_F | — | — | — | — | 0.00 | 0.00 | — | — | referência |
| P1a_F | -0.0041% | [-0.0064%, -0.0016%] | -0.0075% | 1.000 | 0.00 | 0.00 | -0.0039% | 0/6 | ❌ holdout não melhora; IC Bonferroni inclui 0; DSR 0.00<0.95; treino não melhora; só 0/6 blocos melhoram |
| P1b_F | -0.0041% | [-0.0064%, -0.0016%] | -0.0075% | 1.000 | 0.00 | 0.00 | -0.0039% | 0/6 | ❌ holdout não melhora; IC Bonferroni inclui 0; DSR 0.00<0.95; treino não melhora; só 0/6 blocos melhoram |
| P3_F | +0.0000% | [+0.0000%, +0.0000%] | +0.0000% | 1.000 | 0.00 | 0.00 | +0.0000% | 0/6 | ❌ inerte (decisões idênticas ao baseline) |
| P6_F | +0.0000% | [+0.0000%, +0.0000%] | +0.0000% | 1.000 | 0.00 | 0.00 | +0.0000% | 0/6 | ❌ inerte (decisões idênticas ao baseline) |
| ALL_a_F | +0.0015% | [-0.0020%, +0.0056%] | -0.0041% | 0.275 | 0.00 | 0.00 | -0.0017% | 3/6 | ❌ IC Bonferroni inclui 0; DSR 0.00<0.95; treino não melhora; só 3/6 blocos melhoram |

### FULL — O que explica o Δ? (HOLDOUT; retorno total em % do capital, não composto)

| Variante | Δ retorno total | do qual: menos trades | do qual: trade melhor | Δ exp./trade | IC95 | IC Bonferroni inf. | Alfa líquido (IC95 inf.>0)? |
|---|---:|---:|---:|---:|---|---:|---|
| BASE_F | — | — | — | — | — | — | não |
| P1a_F | -2.9% | -2.7% | -0.2% | -0.004% | [-0.068%, +0.063%] | -0.100% | não |
| P1b_F | -2.9% | -2.7% | -0.2% | -0.004% | [-0.068%, +0.063%] | -0.100% | não |
| P3_F | +0.0% | -0.0% | +0.0% | +0.000% | [-0.069%, +0.068%] | -0.101% | não |
| P6_F | +0.0% | -0.0% | +0.0% | +0.000% | [-0.069%, +0.068%] | -0.101% | não |
| ALL_a_F | +1.0% | +1.3% | -0.2% | -0.008% | [-0.078%, +0.061%] | -0.116% | não |

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

_Tempo de execução: 57s._
