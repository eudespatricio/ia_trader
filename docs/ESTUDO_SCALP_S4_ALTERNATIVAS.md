# Estudo Estrutural de Viabilidade do `SCALP` 15m — Pesquisa S4 (2026-11-13)

Gerado em `2026-11-13T22:46:43.798153+00:00` em `BTCUSDT`, `ETHUSDT`, `SOLUSDT` usando `_ScalpResearchSimulator` (paridade Smart Trade + funding real 8h + simulação causal de Limit Pullback com verificação de Stop-Loss/seleção adversa na barra do preenchimento) e conexões `mode=ro`.

## 1. Diagnóstico Matemático do Problema Atual (`SCALP` Produção)
- **Atrito intradiário vs 1R:** No 15m (`1R = 0.9 * ATR_14`), a mediana de `1R` é `~29.5 bps` (`24.0 bps` em BTC). Um round-trip Taker real (`8 bps` in + `8 bps` out = `16 bps`) consome **`54%–67%` de 1R**.
- **Inversão de Payoff na Escada `40/30/30` + `BE +4 bps`:** Ao atingir `TP1 (+1.2 ATR = +1.33R)`, `79.8%` das operações retornam antes do `TP2 (+2.0 ATR)` e são estopadas em `BREAKEVEN_STOP` pagando `8 bps` de saída taker sobre `60%` da mão -> ganho líquido médio **`+0.27R`** por acerto de TP1 contra **`-1.34R a -1.42R`** nos `STOP_LOSS`.

## 2. Horizonte A — Longo Prazo 15m (`9-13 meses`: Execução, Saídas & Estrutura)
Janela: `2025-10-01 08:00 UTC` .. `2026-11-13 22:15 UTC` (Split H1/H2: `2026-04-23 15:07 UTC`)

| Estratégia | Execução | Trades (Fill%) | WR% | PF | E[R] (IC 95%) | PnL (`$10k`) | MaxDD | Sharpe Diário | DSR | Taxas/Fund |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `A0_LIVE_SIM_PARITY` | `sim_parity` | 1027 (93.1%) | 39.0% | 0.349 | -0.5396 [-0.6014, -0.4797] | $-598.88 | 5.99% | -10.383 | 0.0000 | $220.94 |
| `A0_LIVE_REAL_TAKER` | `real_taker` | 1027 (93.1%) | 38.0% | 0.271 | -0.6707 [-0.7302, -0.6114] | $-733.46 | 7.33% | -12.149 | 0.0000 | $342.82 |
| `A1_V105_BE_GUARD_TAKER` | `real_taker` | 1022 (97.1%) | 38.2% | 0.519 | -0.4642 [-0.5380, -0.3891] | $-480.33 | 4.80% | -7.409 | 0.0000 | $334.17 |
| `A2_BO20_SINGLE_TP_TAKER` | `real_taker` | 657 (89.8%) | 38.4% | 0.784 | -0.1612 [-0.2572, -0.0677] | $-248.19 | 2.62% | -2.385 | 0.0000 | $207.70 |
| `A3_BO20_SINGLE_TP_PULLBACK` | `limit_pullback` | 572 (79.9%) | 36.4% | 0.793 | -0.1537 [-0.2545, -0.0485] | $-206.28 | 2.20% | -2.074 | 0.0000 | $113.57 |
| `A4_BO20_DELAYED_BE_PULLBACK` | `limit_pullback` | 535 (83.1%) | 32.5% | 0.838 | -0.1120 [-0.2231, +0.0068] | $-149.91 | 1.64% | -1.523 | 0.0000 | $113.57 |
| `A5_SQUEEZE_KELTNER_TAKER` | `real_taker` | 212 (95.1%) | 37.7% | 0.680 | -0.2192 [-0.3874, -0.0489] | $-105.28 | 1.16% | -2.267 | 0.0000 | $67.31 |
| `A6_SQUEEZE_KELTNER_PULLBACK` | `limit_pullback` | 188 (87.4%) | 36.7% | 0.731 | -0.1844 [-0.3595, -0.0080] | $-74.07 | 0.86% | -1.586 | 0.0000 | $37.57 |
| `A7_SQUEEZE_US_EU_PULLBACK` | `limit_pullback` | 77 (88.5%) | 37.7% | 0.839 | -0.1411 [-0.4258, +0.1501] | $-18.06 | 0.34% | -0.652 | 0.0000 | $15.25 |
| `A8_RANGE_VWAP_REV_PULLBACK` | `limit_pullback` | 69 (78.4%) | 47.8% | 0.877 | -0.1155 [-0.3663, +0.1386] | $-11.35 | 0.46% | -0.371 | 0.0000 | $12.98 |
| `A9_SWEEP_FAKEOUT_TAKER` | `real_taker` | 1625 (91.3%) | 38.0% | 0.673 | -0.2707 [-0.3280, -0.2082] | $-733.26 | 7.34% | -5.336 | 0.0000 | $513.70 |
| `A9b_SWEEP_FAKEOUT_PULLBACK` | `limit_pullback` | 1335 (75.9%) | 36.9% | 0.719 | -0.2214 [-0.2881, -0.1539] | $-497.12 | 4.98% | -3.818 | 0.0000 | $264.22 |

## 3. Horizonte B — 60 Dias Derivativos Binance 5m (`oi`, `taker`, `ls_global`, `ls_top_pos`)
Janela: `2026-09-15 12:00 UTC` .. `2026-11-13 22:15 UTC` (Split H1/H2: `2026-10-15 05:07 UTC`)

| Estratégia | Execução | Trades (Fill%) | WR% | PF | E[R] (IC 95%) | PnL (`$10k`) | MaxDD | Sharpe Diário | DSR | Taxas/Fund |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `B0_BASE_60D_TAKER` | `real_taker` | 152 (95.6%) | 36.2% | 0.180 | -0.7794 [-0.9237, -0.6368] | $-111.79 | 1.12% | -17.672 | 0.0000 | $50.79 |
| `B1_OI_BUILDUP_TAKER` | `real_taker` | 74 (91.4%) | 33.8% | 0.602 | -0.3360 [-0.6193, -0.0435] | $-42.52 | 0.60% | -3.744 | 0.0000 | $23.83 |
| `B1b_OI_BUILDUP_PULLBACK` | `limit_pullback` | 67 (87.0%) | 32.8% | 0.632 | -0.2936 [-0.5825, -0.0024] | $-33.19 | 0.42% | -3.834 | 0.0000 | $13.80 |
| `B2_OI_FLUSH_FADE_TAKER` | `real_taker` | 279 (83.0%) | 38.7% | 0.640 | -0.3078 [-0.4488, -0.1668] | $-157.68 | 1.78% | -5.865 | 0.0000 | $88.92 |
| `B2b_OI_FLUSH_FADE_PULLBACK` | `limit_pullback` | 239 (73.3%) | 38.9% | 0.735 | -0.2259 [-0.3665, -0.0825] | $-91.34 | 1.10% | -4.289 | 0.0000 | $47.74 |
| `B3_SMART_RETAIL_DIV_PULLBACK` | `limit_pullback` | 27 (84.4%) | 37.0% | 0.648 | -0.2868 [-0.7206, +0.1592] | $-13.18 | 0.19% | -2.906 | 0.0000 | $5.26 |
| `B4_FUNDING_WINDOW_FADE` | `limit_pullback` | 31 (73.8%) | 48.4% | 1.218 | -0.0573 [-0.5112, +0.3415] | $+7.14 | 0.13% | +1.271 | 0.0000 | $5.53 |

## 4. Horizonte C — 29 Dias Microestrutura Real (`book_features` + `tick_agg` L2)
Janela: `2026-10-15 12:30 UTC` .. `2026-11-13 22:15 UTC` (Split H1/H2: `2026-10-30 05:22 UTC`)

| Estratégia | Execução | Trades (Fill%) | WR% | PF | E[R] (IC 95%) | PnL (`$10k`) | MaxDD | Sharpe Diário | DSR | Taxas/Fund |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `C0_BASE_29D_WITH_TAPE_TAKER` | `real_taker` | 28 (100.0%) | 28.6% | 0.106 | -0.9843 [-1.2589, -0.6795] | $-23.12 | 0.24% | -13.708 | 0.0000 | $9.20 |
| `C1_L2_ABSORPTION_PULLBACK` | `limit_pullback` | 33 (86.8%) | 33.3% | 0.514 | -0.3944 [-0.7546, -0.0286] | $-24.23 | 0.26% | -5.750 | 0.0000 | $6.74 |
| `C2_L2_TOXIC_OBI_TAKER` | `real_taker` | 18 (100.0%) | 22.2% | 0.274 | -0.6685 [-1.0308, -0.2303] | $-23.00 | 0.24% | -8.784 | 0.0000 | $6.20 |
| `C2b_L2_TOXIC_OBI_PULLBACK` | `limit_pullback` | 15 (83.3%) | 40.0% | 0.951 | -0.0995 [-0.5888, +0.4481] | $-0.86 | 0.13% | -0.235 | 0.0004 | $3.10 |
| `C3_RAW_A4_BO20_DELAYED_BE_PULLBACK` | `limit_pullback` | 43 (84.3%) | 20.9% | 0.314 | -0.5354 [-0.8630, -0.1707] | $-58.65 | 0.67% | -9.657 | 0.0000 | $9.50 |
| `C3b_L2FILTER_A4_BO20_DELAYED_BE_PULLBACK` | `limit_pullback` | 41 (85.4%) | 21.9% | 0.343 | -0.4891 [-0.8325, -0.0957] | $-51.48 | 0.59% | -8.829 | 0.0000 | $8.98 |
