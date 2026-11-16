# Anotações Empíricas — Impacto e Ponderação do `TapeReadingAgent` nas 3 Estratégias (`E10.10`)

Gerado em `2026-11-13T23:15:47.811888+00:00` | Janela 29d real (`microstructure.db`): `2026-10-15 12:30 UTC` .. `2026-11-13` | Conexão SQLite `mode=ro`.

## 1. Como o Código Funciona Hoje vs O Que Testamos
- **No `ConsensusJudge` ([core.py](file:///home/eudespatricio/ia_trader/core.py#L62-L69), [consensus_judge.py](file:///home/eudespatricio/ia_trader/consensus_judge.py#L57-L64)):** Apenas `OrderFlowAgent` (features OHLCV/derivativos) participa da votação ponderada com peso `0.8`. O `TapeReadingAgent` ([tape_reading_agent.py](file:///home/eudespatricio/ia_trader/agents/tape_reading_agent.py#L279-L323)) **não está dentro do consenso**.
- **No daemon live ([run_paper_simulation.py](file:///home/eudespatricio/ia_trader/scripts/run_paper_simulation.py#L531-L581)):** Ele roda obrigatoriamente **depois do comitê** como barreira binária (`_flow_gate`) nas 3 estratégias (`DIRECTIONAL`, `GOLDEN_POT`, `SCALP`), **porém com política fail-open (`UNAVAILABLE_PASSED`)** quando falta dado de book/ticks.
- **Simulação `M3_TAPE_WEIGHTED_SCORE` vs `M2_GATE_STRICT_OK`:** Testamos transformar o `TapeReadingAgent` em componente ponderado (`0.65 * convicção_comitê + 0.35 * tape_unit`, usando `flow_score` + `confidence` + veto em `spread_wide`/fluxo oposto e sizing `0.50x..1.25x`) versus o gate estrito binário (`status == 'OK' and vote == direction`).

## 2. Auditoria Contrafactual Ao Vivo de `flow_gate_log` (`N=804`)
| Estratégia | Corte | Sinais | WR% | ΔWR vs Sem Gate | PF | E[R] | PnL Sim (`$100`/lote) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `DIRECTIONAL` | Sem Gate (Todos Candidatos) | 639 | 48.4% | - | 0.509 | -0.2401 | $-505.35 |
| `DIRECTIONAL` | Produção Atual (Fail-Open) | 77 | 41.6% | -6.80 pp | 0.780 | -0.1777 | $-25.88 |
| `DIRECTIONAL` | Gate Estrito (OK + Direção) | 59 | 37.3% | -11.07 pp | 0.691 | -0.2321 | $-30.80 |
| `DIRECTIONAL` | Passou Sem Dado (UNAVAILABLE) | 18 | 55.6% | - | 1.273 | +0.0005 | $+4.92 |
| `DIRECTIONAL` | Bloqueados pelo Gate | 562 | 49.3% | - | 0.474 | -0.2486 | $-479.47 |
| `SCALP` | Sem Gate (Todos Candidatos) | 131 | 25.9% | - | 0.099 | -1.1661 | $-175.52 |
| `SCALP` | Produção Atual (Fail-Open) | 28 | 25.0% | -0.95 pp | 0.250 | -1.4225 | $-16.70 |
| `SCALP` | Gate Estrito (OK + Direção) | 24 | 25.0% | -0.95 pp | 0.220 | -1.5341 | $-14.50 |
| `SCALP` | Passou Sem Dado (UNAVAILABLE) | 4 | 25.0% | - | 0.402 | -0.7532 | $-2.20 |
| `SCALP` | Bloqueados pelo Gate | 103 | 26.2% | - | 0.080 | -1.0964 | $-158.82 |
| `GOLDEN_POT` | Sem Gate (Todos Candidatos) | 31 | 61.3% | - | 0.906 | +0.0266 | $-12.78 |
| `GOLDEN_POT` | Produção Atual (Fail-Open) | 27 | 63.0% | +1.67 pp | 0.850 | +0.0160 | $-16.87 |
| `GOLDEN_POT` | Gate Estrito (OK + Direção) | 0 | 0.0% | - | 0.000 | +0.0000 | $+0.00 |
| `GOLDEN_POT` | Passou Sem Dado (UNAVAILABLE) | 27 | 63.0% | - | 0.850 | +0.0160 | $-16.87 |
| `GOLDEN_POT` | Bloqueados pelo Gate | 4 | 50.0% | - | 1.171 | +0.0979 | $+4.09 |

## 3. Replay 29d L2 Real — Comparativo dos 5 Modos nas 3 Estratégias
### 3.1 `DIRECTIONAL` 1h (`BTC/ETH/SOL`, S4.2 + `corr_cap=2`)
| Modo | Trades (Cand->Keep) | WR% | ΔWR | PF | E[R] | PnL (`$10k`) | MaxDD | H1 -> H2 (PF / PnL) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `M0_NO_TAPE` | 138 (307->307) | 38.4% | +0.00 pp | 0.530 | -0.3642 | $-103.38 | 1.15% | `0.39` ($-72.0) / `0.69` ($-31.4) |
| `M1_GATE_FAIL_OPEN` | 84 (307->136) | 44.0% | +5.64 pp | 0.661 | -0.2597 | $-41.17 | 0.65% | `0.69` ($-13.1) / `0.65` ($-28.1) |
| `M2_GATE_STRICT_OK` | 55 (307->73) | 38.2% | -0.23 pp | 0.541 | -0.3294 | $-43.32 | 0.55% | `0.83` ($-5.1) / `0.40` ($-38.3) |
| `M3_TAPE_WEIGHTED_SCORE` | 68 (307->156) | 38.2% | -0.17 pp | 0.504 | -0.3722 | $-65.93 | 0.79% | `0.88` ($-5.1) / `0.32` ($-60.8) |
| `M4_STRICT_OK_PLUS_WEIGHT` | 55 (307->73) | 38.2% | -0.23 pp | 0.542 | -0.3294 | $-54.03 | 0.69% | `0.83` ($-6.5) / `0.41` ($-47.5) |

### 3.2a `GOLDEN_POT` 5m — Sem Filtro Macro BTC (Universo Completo 5m)
| Modo | Trades (Cand->Keep) | WR% | ΔWR | PF | E[R] | PnL (`$10k`) | MaxDD | H1 -> H2 (PF / PnL) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `M0_NO_TAPE` | 87 (243->243) | 49.4% | +0.00 pp | 0.460 | -0.2476 | $-125.49 | 1.48% | `0.36` ($-103.7) / `0.69` ($-21.8) |
| `M1_GATE_FAIL_OPEN` | 87 (243->231) | 49.4% | +0.00 pp | 0.459 | -0.2615 | $-121.12 | 1.44% | `0.37` ($-95.0) / `0.64` ($-26.2) |
| `M2_GATE_STRICT_OK` | 2 (243->2) | 50.0% | +0.57 pp | 0.894 | -0.1461 | $-0.10 | 0.01% | `99.00` ($+0.9) / `0.00` ($-1.0) |
| `M3_TAPE_WEIGHTED_SCORE` | 86 (243->213) | 48.8% | -0.59 pp | 0.482 | -0.2660 | $-60.09 | 0.72% | `0.36` ($-49.5) / `0.72` ($-10.6) |
| `M4_STRICT_OK_PLUS_WEIGHT` | 50 (243->58) | 36.0% | -13.43 pp | 0.529 | -0.5116 | $-21.44 | 0.31% | `0.58` ($-10.5) / `0.47` ($-11.0) |

### 3.2b `GOLDEN_POT` 5m — Com Item 1 Macro BTC `±1.5% SMA200` (Universo Completo 5m)
| Modo | Trades (Cand->Keep) | WR% | ΔWR | PF | E[R] | PnL (`$10k`) | MaxDD | H1 -> H2 (PF / PnL) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `M0_NO_TAPE` | 83 (185->185) | 44.6% | +0.00 pp | 0.384 | -0.3275 | $-155.12 | 1.78% | `0.42` ($-81.8) / `0.34` ($-73.3) |
| `M1_GATE_FAIL_OPEN` | 83 (185->176) | 45.8% | +1.20 pp | 0.385 | -0.3236 | $-148.56 | 1.71% | `0.44` ($-73.0) / `0.32` ($-75.5) |
| `M2_GATE_STRICT_OK` | 2 (185->2) | 50.0% | +5.42 pp | 0.894 | -0.1461 | $-0.10 | 0.01% | `99.00` ($+0.9) / `0.00` ($-1.0) |
| `M3_TAPE_WEIGHTED_SCORE` | 82 (185->167) | 45.1% | +0.54 pp | 0.376 | -0.3353 | $-76.93 | 0.88% | `0.42` ($-39.2) / `0.32` ($-37.7) |
| `M4_STRICT_OK_PLUS_WEIGHT` | 39 (185->43) | 41.0% | -3.55 pp | 0.750 | -0.4440 | $-7.62 | 0.18% | `0.66` ($-7.7) / `1.01` ($+0.1) |

### 3.2c `GOLDEN_POT` 5m — Com Macro BTC Apenas nos 117 Símbolos Cobertos L2
| Modo | Trades (Cand->Keep) | WR% | ΔWR | PF | E[R] | PnL (`$10k`) | MaxDD | H1 -> H2 (PF / PnL) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `M0_NO_TAPE` | 78 (144->144) | 44.9% | +0.00 pp | 0.383 | -0.3058 | $-154.68 | 1.85% | `0.49` ($-69.2) / `0.26` ($-85.5) |
| `M1_GATE_FAIL_OPEN` | 76 (144->135) | 44.7% | -0.13 pp | 0.349 | -0.3183 | $-164.54 | 1.95% | `0.45` ($-76.9) / `0.23` ($-87.7) |
| `M2_GATE_STRICT_OK` | 2 (144->2) | 50.0% | +5.13 pp | 0.894 | -0.1461 | $-0.10 | 0.01% | `99.00` ($+0.9) / `0.00` ($-1.0) |
| `M3_TAPE_WEIGHTED_SCORE` | 75 (144->127) | 44.0% | -0.87 pp | 0.344 | -0.3202 | $-83.92 | 0.99% | `0.43` ($-41.1) / `0.24` ($-42.8) |
| `M4_STRICT_OK_PLUS_WEIGHT` | 2 (144->2) | 50.0% | +5.13 pp | 1.032 | -0.1461 | $+0.04 | 0.01% | `99.00` ($+1.1) / `0.00` ($-1.1) |

### 3.3 `SCALP` 15m (`BTC/ETH/SOL`) — Produção vs Melhores Win Rates/Estruturas do Backtest
### `SCALP` — `SCALP_PROD_BASE`
| Modo | Trades (Cand->Keep) | WR% | ΔWR | PF | E[R] | PnL (`$10k`) | MaxDD | H1 -> H2 (PF / PnL) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `M0_NO_TAPE` | 70 (76->76) | 31.4% | +0.00 pp | 0.146 | -0.8759 | $-56.85 | 0.57% | `0.13` ($-21.7) / `0.16` ($-35.1) |
| `M1_GATE_FAIL_OPEN` | 26 (76->27) | 26.9% | -4.51 pp | 0.104 | -1.0151 | $-21.84 | 0.23% | `0.07` ($-9.2) / `0.12` ($-12.7) |
| `M2_GATE_STRICT_OK` | 13 (76->13) | 38.5% | +7.03 pp | 0.183 | -0.8053 | $-8.38 | 0.08% | `0.12` ($-5.4) / `0.27` ($-3.0) |
| `M3_TAPE_WEIGHTED_SCORE` | 25 (76->44) | 40.0% | +8.57 pp | 0.150 | -0.7422 | $-18.92 | 0.20% | `0.10` ($-9.2) / `0.20` ($-9.7) |
| `M4_STRICT_OK_PLUS_WEIGHT` | 13 (76->13) | 38.5% | +7.03 pp | 0.182 | -0.8053 | $-10.29 | 0.10% | `0.13` ($-6.5) / `0.26` ($-3.9) |

### `SCALP` — `B4_FUNDING_WINDOW_FADE`
| Modo | Trades (Cand->Keep) | WR% | ΔWR | PF | E[R] | PnL (`$10k`) | MaxDD | H1 -> H2 (PF / PnL) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `M0_NO_TAPE` | 15 (32->32) | 46.7% | +0.00 pp | 1.234 | -0.1095 | $+3.82 | 0.10% | `0.00` ($-9.5) / `2.95` ($+13.3) |
| `M1_GATE_FAIL_OPEN` | 8 (32->10) | 37.5% | -9.17 pp | 0.852 | -0.3187 | $-1.48 | 0.08% | `0.00` ($-7.0) / `2.86` ($+5.5) |
| `M2_GATE_STRICT_OK` | 6 (32->8) | 50.0% | +3.33 pp | 1.160 | +0.0060 | $+1.17 | 0.06% | `0.00` ($-4.3) / `2.86` ($+5.5) |
| `M3_TAPE_WEIGHTED_SCORE` | 12 (32->23) | 50.0% | +3.33 pp | 1.462 | -0.0188 | $+4.43 | 0.06% | `0.00` ($-4.7) / `2.87` ($+9.1) |
| `M4_STRICT_OK_PLUS_WEIGHT` | 6 (32->8) | 50.0% | +3.33 pp | 1.182 | +0.0060 | $+1.60 | 0.07% | `0.00` ($-5.3) / `3.00` ($+6.9) |

### `SCALP` — `A8_RANGE_VWAP_REV_PULLBACK`
| Modo | Trades (Cand->Keep) | WR% | ΔWR | PF | E[R] | PnL (`$10k`) | MaxDD | H1 -> H2 (PF / PnL) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `M0_NO_TAPE` | 7 (8->8) | 42.9% | +0.00 pp | 0.866 | -0.1768 | $-1.43 | 0.07% | `0.63` ($-1.6) / `1.03` ($+0.2) |
| `M1_GATE_FAIL_OPEN` | 3 (8->3) | 33.3% | -9.53 pp | 0.599 | -0.4094 | $-1.70 | 0.05% | `0.00` ($-2.0) / `1.12` ($+0.3) |
| `M2_GATE_STRICT_OK` | 2 (8->2) | 0.0% | -42.86 pp | 0.000 | -1.1661 | $-4.23 | 0.05% | `0.00` ($-2.0) / `0.00` ($-2.3) |
| `M3_TAPE_WEIGHTED_SCORE` | 4 (8->5) | 25.0% | -17.86 pp | 0.094 | -0.5955 | $-5.72 | 0.07% | `0.17` ($-2.9) / `0.00` ($-2.8) |
| `M4_STRICT_OK_PLUS_WEIGHT` | 2 (8->2) | 0.0% | -42.86 pp | 0.000 | -1.1661 | $-5.27 | 0.07% | `0.00` ($-2.6) / `0.00` ($-2.7) |

### `SCALP` — `A9b_SWEEP_FAKEOUT_PULLBACK`
| Modo | Trades (Cand->Keep) | WR% | ΔWR | PF | E[R] | PnL (`$10k`) | MaxDD | H1 -> H2 (PF / PnL) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `M0_NO_TAPE` | 104 (144->144) | 33.6% | +0.00 pp | 0.632 | -0.2990 | $-51.29 | 0.72% | `1.67` ($+19.1) / `0.36` ($-70.4) |
| `M1_GATE_FAIL_OPEN` | 47 (144->62) | 31.9% | -1.74 pp | 0.584 | -0.3486 | $-26.96 | 0.37% | `1.51` ($+6.1) / `0.38` ($-33.0) |
| `M2_GATE_STRICT_OK` | 25 (144->35) | 28.0% | -5.65 pp | 0.492 | -0.4501 | $-18.53 | 0.25% | `2.06` ($+5.9) / `0.21` ($-24.4) |
| `M3_TAPE_WEIGHTED_SCORE` | 63 (144->89) | 33.3% | -0.32 pp | 0.593 | -0.3049 | $-24.19 | 0.38% | `2.46` ($+12.1) / `0.29` ($-36.3) |
| `M4_STRICT_OK_PLUS_WEIGHT` | 25 (144->35) | 28.0% | -5.65 pp | 0.502 | -0.4501 | $-21.96 | 0.30% | `1.93` ($+6.5) / `0.23` ($-28.5) |

### `SCALP` — `A4_BO20_DELAYED_BE_PULLBACK`
| Modo | Trades (Cand->Keep) | WR% | ΔWR | PF | E[R] | PnL (`$10k`) | MaxDD | H1 -> H2 (PF / PnL) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `M0_NO_TAPE` | 43 (66->66) | 20.9% | +0.00 pp | 0.315 | -0.5308 | $-58.24 | 0.67% | `0.17` ($-25.5) / `0.40` ($-32.7) |
| `M1_GATE_FAIL_OPEN` | 11 (66->18) | 9.1% | -11.84 pp | 0.135 | -0.8888 | $-22.89 | 0.26% | `1.65` ($+1.4) / `0.00` ($-24.3) |
| `M2_GATE_STRICT_OK` | 4 (66->8) | 0.0% | -20.93 pp | 0.000 | -1.1393 | $-10.06 | 0.11% | `0.00` ($-2.2) / `0.00` ($-7.9) |
| `M3_TAPE_WEIGHTED_SCORE` | 33 (66->47) | 21.2% | +0.28 pp | 0.308 | -0.4812 | $-27.01 | 0.31% | `0.08` ($-12.6) / `0.43` ($-14.4) |
| `M4_STRICT_OK_PLUS_WEIGHT` | 4 (66->8) | 0.0% | -20.93 pp | 0.000 | -1.1393 | $-11.99 | 0.13% | `0.00` ($-2.4) / `0.00` ($-9.6) |

