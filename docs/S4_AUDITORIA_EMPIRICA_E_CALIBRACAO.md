# Auditoria Empírica S4, Saneamento do Banco e Calibração S4.2 (2026-11-13)

Gerado em `2026-11-13T21:57:13.005401+00:00` em `BTCUSDT`, `ETHUSDT`, `SOLUSDT` com **`BacktestEngine(..., pilot_parity=True)`** (Smart Trade 40/30/30, breakeven +0,04%, trailing pós-TP2, funding real 8h e slippage real) e ML estritamente out-of-sample (`PurgedTimeSeriesSplit` com embargo).

## 1. Resumo Executivo das Correções já Aplicadas no Código & Banco (Steps 1–3)
- **Poluição de testes isolada:** `47` linhas de `'Reset de teste'` removidas de `circuit_breaker_events` (backup em `datasets/backups/cb_test_pollution_backup_20261113.json`); `tests/test_performance_and_rollback.py` isolado com `monkeypatch.setattr("flask_interface.DB_PATH", temp_db)`.
- **Deduplicação canônica de velas:** `63.008` linhas sobrepostas não-`binance` removidas de `candles` (backup em `datasets/backups/candles_overlap_backup_20261113.sqlite`, `0` duplicatas remanescentes); loaders endurecidos com prioridade `binance` + `drop_duplicates`.
- **Regime, Brier e TapeReading ao vivo corrigidos:** `core.py` propaga `regime` para `TradeSignal.metrics`; `MarketRegimeClassifier` usa janela `LIMIT 250` e detecta consolidação em tendência; `learning_tracker.py` persiste `brier_score` real; `run_paper_simulation.py` passa `allow_live_fallback=True` em `_flow_gate`.

### Antes vs Depois da Matriz de Calibração no Banco (`agent_regime_weights` & `agent_regime_calibration`)
| Snapshot | Regime | Agente | Peso | Acurácia | Amostras | ECE | Brier |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Antes (pré-backfill) | `BEAR_TREND` | `OrderFlowAgent` | 0.800 | 100.0% | 1 | 0.0000 | 0.0000 |
| Antes (pré-backfill) | `BEAR_TREND` | `TechnicalAgent` | 1.000 | 100.0% | 1 | 0.0000 | 0.0000 |
| Antes (pré-backfill) | `SIDEWAYS_RANGE` | `OrderFlowAgent` | 0.577 | 62.1% | 29 | 0.1881 | 0.0000 |
| Antes (pré-backfill) | `SIDEWAYS_RANGE` | `TechnicalAgent` | 1.223 | 62.1% | 29 | 0.1991 | 0.0000 |
| Depois (reparado) | `BEAR_TREND` | `OrderFlowAgent` | 0.486 | 68.2% | 22 | 0.2546 | 0.2408 |
| Depois (reparado) | `BEAR_TREND` | `TechnicalAgent` | 1.314 | 68.2% | 22 | 0.2101 | 0.2171 |
| Depois (reparado) | `BULL_TREND` | `OrderFlowAgent` | 0.880 | 66.7% | 6 | 0.0529 | 0.2122 |
| Depois (reparado) | `BULL_TREND` | `TechnicalAgent` | 0.920 | 66.7% | 6 | 0.1403 | 0.2525 |
| Depois (reparado) | `SIDEWAYS_RANGE` | `OrderFlowAgent` | 0.800 | 0.0% | 2 | 0.0000 | 0.4880 |
| Depois (reparado) | `SIDEWAYS_RANGE` | `TechnicalAgent` | 1.000 | 0.0% | 2 | 0.0000 | 0.7260 |

## 2. Bloco 1 — ML Walk-Forward Antes vs Depois da Deduplicação (`25.000` linhas 1h)
| Ativo | Cenário | Linhas Brutas | Únicas | Duplicatas | Modelo | Acc Global | F1 Macro | Acc Sinais (>=0.50) | Cobertura |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: |
| `BTCUSDT` | Contaminado | 25000 | 12720 | 12280 | LGBM | 50.05% | 0.4533 | 61.80% (1419) | 9.48% |
| `BTCUSDT` | Limpo Dedup | 25000 | 25000 | 0 | LGBM | 40.44% | 0.4039 | 40.06% (689) | 4.60% |
| `BTCUSDT` | Contaminado | 25000 | 12720 | 12280 | RF | 50.45% | 0.4504 | 68.99% (1177) | 7.86% |
| `BTCUSDT` | Limpo Dedup | 25000 | 25000 | 0 | RF | 40.65% | 0.4037 | 43.01% (93) | 0.62% |
| `ETHUSDT` | Contaminado | 25000 | 12893 | 12107 | LGBM | 46.33% | 0.4292 | 58.29% (1225) | 8.18% |
| `ETHUSDT` | Limpo Dedup | 25000 | 25000 | 0 | LGBM | 40.38% | 0.3999 | 38.24% (672) | 4.49% |
| `ETHUSDT` | Contaminado | 25000 | 12893 | 12107 | RF | 44.89% | 0.4206 | 70.94% (843) | 5.63% |
| `ETHUSDT` | Limpo Dedup | 25000 | 25000 | 0 | RF | 41.01% | 0.4050 | 41.64% (269) | 1.80% |
| `SOLUSDT` | Contaminado | 25000 | 12893 | 12107 | LGBM | 45.37% | 0.4187 | 65.14% (1090) | 7.28% |
| `SOLUSDT` | Limpo Dedup | 25000 | 25000 | 0 | LGBM | 39.09% | 0.3915 | 42.07% (397) | 2.65% |
| `SOLUSDT` | Contaminado | 25000 | 12893 | 12107 | RF | 44.15% | 0.4098 | 74.11% (811) | 5.42% |
| `SOLUSDT` | Limpo Dedup | 25000 | 25000 | 0 | RF | 38.74% | 0.3877 | 36.27% (102) | 0.68% |

### Auditoria dos Modelos `.joblib` em Produção
| Artefato | Criado em | TP/SL | Amostras OOS | Features | Acc Global | F1 Macro | Acc Emissão (>=0.50) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `models/archive/lgbm_model_v1_20261111T231642Z.joblib` | 2026-11-08T20:44:03.598985+00:00 | 1.2/1.2 | 29848 | 45 | 41.44% | 0.4114 | 41.94% (496) |
| `models/lgbm_model_v1.joblib` | 2026-11-11T23:16:44.931570+00:00 | 1.2/1.2 | 29848 | 45 | 41.73% | 0.4146 | 42.71% (487) |
| `models/rf_model_v1.joblib` | 2026-11-08T20:45:04.425523+00:00 | 1.2/1.2 | 29848 | 45 | 41.82% | 0.4104 | 47.37% (57) |

## 3. Bloco 2 — Perfil 21:00 BRT (`174,341` velas 1h limpas) & Estratégias em `pilot_parity=True`
| Hora (BRT) | Velas | Range/ATR | Vol Ratio | Reversão 4h (Mov >=0.3%) |
| :--- | ---: | ---: | ---: | ---: |
| `11h` | 7265 | 1.352 | 1.810 | 51.55% |
| `18h` | 7262 | 0.953 | 0.783 | 53.39% |
| `20h` | 7262 | 0.902 | 0.817 | 54.29% |
| `21h` | 7264 | 1.093 | 1.013 | 53.09% |
| `22h` | 7264 | 1.008 | 0.900 | 50.17% |
| `05h` | 7265 | 0.953 | 0.966 | 51.73% |

### Reavaliação Consolidada & Por Ativo (Somente Folds Estritamente OOS, Smart Trade Ladder + Funding Real)
| Estratégia | TF | Port. Trades | Port. WR | Port. PF | Port. E[R] | Port. PnL ($10k) | Max DD | BTC WR / PF / PnL | ETH WR / PF / PnL | SOL WR / PF / PnL |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- | :--- |
| Baseline Momentum (1h) | 1h | 2995 | 40.5% | 0.56 | -0.342R | $-838.15 | 8.48% | 39.5% / 0.44 / $-282 | 42.6% / 0.59 / $-263 | 39.3% / 0.61 / $-294 |
| Opção A: Flow Reversal (Pura) | 1h | 1979 | 39.6% | 0.65 | -0.278R | $-483.38 | 4.87% | 39.7% / 0.55 / $-134 | 40.1% / 0.70 / $-148 | 39.0% / 0.66 / $-201 |
| Opção A: Flow Reversal (ML OOS) | 1h | 711 | 39.2% | 0.69 | -0.302R | $-160.11 | 1.79% | 36.0% / 0.46 / $-64 | 37.3% / 0.69 / $-58 | 43.2% / 0.82 / $-39 |
| Opção B: Session Sweep 21/05/11h (Pura) | 1h | 553 | 38.9% | 0.61 | -0.329R | $-147.29 | 1.53% | 34.6% / 0.50 / $-48 | 42.5% / 0.74 / $-31 | 39.4% / 0.56 / $-68 |
| Opção B: Session Sweep 21/05/11h (ML OOS) | 1h | 194 | 36.1% | 0.49 | -0.417R | $-68.31 | 0.68% | 34.4% / 0.38 / $-23 | 40.3% / 0.63 / $-16 | 33.8% / 0.47 / $-30 |
| Opção B-21h: Daily Open 21h BRT (Pura) | 1h | 143 | 38.5% | 0.74 | -0.248R | $-26.07 | 0.34% | 31.9% / 0.64 / $-10 | 47.3% / 0.95 / $-2 | 34.1% / 0.63 / $-14 |
| Opção B-21h: Daily Open 21h BRT (ML OOS) | 1h | 33 | 39.4% | 0.51 | -0.321R | $-10.50 | 0.11% | 46.1% / 0.72 / $-2 | 41.7% / 0.57 / $-3 | 25.0% / 0.28 / $-6 |
| Opção C: Scalp VWAP (Baseline 2.0%) | 15m | 129 | 44.2% | 0.67 | -0.245R | $-17.38 | 0.21% | 40.0% / 0.53 / $-3 | 40.5% / 0.71 / $-5 | 46.8% / 0.68 / $-10 |
| Opção C: Scalp VWAP (Live DB 0.8%) | 15m | 781 | 40.3% | 0.55 | -0.371R | $-112.48 | 1.26% | 38.1% / 0.43 / $-32 | 43.4% / 0.61 / $-35 | 38.9% / 0.56 / $-52 |
| Opção C: Scalp VWAP (Calibrado 0.6%) | 15m | 1080 | 40.7% | 0.50 | -0.403R | $-157.64 | 1.64% | 40.2% / 0.47 / $-37 | 41.9% / 0.50 / $-55 | 40.6% / 0.52 / $-67 |
| Opção C: Scalp VWAP (Calibrado 0.6% + ML OOS 15m) | 15m | 392 | 42.6% | 0.55 | -0.365R | $-49.65 | 0.52% | 44.4% / 0.52 / $-13 | 39.3% / 0.47 / $-21 | 44.4% / 0.63 / $-16 |

## 4. Bloco 3 — A/B de Calibrações S4.2 em Holdout OOS (`DIRECTIONAL 1h` e `SCALP 15m`)
- Janela Holdout 1h (`60% treino + 10d embargo`): `2024-02-24 07:12 UTC` a `2026-11-13 20:00 UTC` (ensemble ML treinado só antes de `2024-02-14 07:12 UTC`).
- Janela 15m OOS disponível: `2026-02-20 14:45 UTC` a `2026-11-13 21:15 UTC`.

| Variante | Trades | WR% | PF | E[R] (IC 95%) | PnL ($10k) | Ret% | Max DD | Sharpe Diário | DSR |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| `DIR_BASE` | 6883 | 39.4% | 0.629 | -0.2904 [-0.3190, -0.2606] | $-5329.03 | -53.29% | 53.68% | -7.118 | 0.0000 |
| `DIR_CAL1_P2` | 6869 | 39.3% | 0.626 | -0.2930 [-0.3228, -0.2630] | $-5335.18 | -53.35% | 53.72% | -7.256 | 0.0000 |
| `DIR_CAL2_CORR_CAP` | 5924 | 39.4% | 0.628 | -0.2929 [-0.3230, -0.2612] | $-4492.47 | -44.92% | 45.11% | -7.315 | 0.0000 |
| `DIR_CAL3_REGIME_WEIGHTS` | 5956 | 39.7% | 0.647 | -0.2775 [-0.3098, -0.2472] | $-4467.98 | -44.68% | 45.17% | -6.473 | 0.0000 |
| `DIR_S4_2_COMBINED` | 5144 | 39.6% | 0.649 | -0.2815 [-0.3163, -0.2497] | $-3722.59 | -37.23% | 37.60% | -6.493 | 0.0000 |
| `SCALP_BASE_DB` | 775 | 39.4% | 0.365 | -0.5296 [-0.5980, -0.4580] | $-427.02 | -4.27% | 4.29% | -10.842 | 0.0000 |
| `SCALP_CAL_TRIGGER` | 721 | 39.8% | 0.349 | -0.5383 [-0.6101, -0.4646] | $-405.30 | -4.05% | 4.07% | -10.387 | 0.0000 |
| `SCALP_CAL_ALIGNED` | 721 | 39.8% | 0.349 | -0.5383 [-0.6101, -0.4646] | $-405.30 | -4.05% | 4.07% | -10.387 | 0.0000 |

## 5. Bloco 4 — Replay 29d em `microstructure.db` (`122` símbolos) & Contrafactual S4 (`N=24`)
### 5.1 Replay 29 dias (`2026-10-15 .. 2026-11-13`)
| Cenário Replay 29d | Trades | WR% | PF | E[R] | PnL ($10k) | Max DD | Gate Tape (Blk/Pass/Unav) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `DIR_BASE_NO_TAPE` | 194 | 41.2% | 0.650 | -0.2703R | $-103.93 | 1.21% | Sem Gate |
| `DIR_BASE_WITH_TAPE` | 105 | 45.7% | 0.664 | -0.2377R | $-50.11 | 0.75% | 216/99/63 |
| `DIR_S42_NO_TAPE` | 139 | 39.6% | 0.542 | -0.3629R | $-100.14 | 1.12% | Sem Gate |
| `DIR_S42_WITH_TAPE` | 87 | 46.0% | 0.711 | -0.2287R | $-35.09 | 0.65% | 174/77/63 |
| `SCALP_BASE_NO_TAPE` | 74 | 33.8% | 0.218 | -0.7191R | $-48.47 | 0.49% | Sem Gate |
| `SCALP_BASE_WITH_TAPE` | 28 | 32.1% | 0.191 | -0.7944R | $-17.70 | 0.18% | 51/15/14 |
| `SCALP_ALIGNED_NO_TAPE` | 64 | 37.5% | 0.259 | -0.6638R | $-36.83 | 0.37% | Sem Gate |
| `SCALP_ALIGNED_WITH_TAPE` | 22 | 31.8% | 0.167 | -0.8252R | $-13.98 | 0.14% | 45/11/12 |

### 5.2 Contrafactual S4 Ao Vivo (`N=24` trades fechados desde `2026-11-09 00:00 UTC`; `2` posições abertas intactas)
| Estratégia | Antes (N / WR / PnL) | Filtro S4.2 Candidato (N / WR / PnL) |
| :--- | :--- | :--- |
| `DIRECTIONAL` | 11 / 72.7% / $+1064.93 | 9 / 77.8% / $+1095.00 (after_corr_cap2) |
| `GOLDEN_POT` | 11 / 63.6% / $-23.27 | 6 / 66.7% / $+23.05 (after_btc_macro_15pct) |
| `SCALP` | 2 / 0.0% / $-50.58 | 1 / 0.0% / $-10.04 (after_counter_regime_block) |
| `TOTAL` | 24 / 62.5% / $+991.08 | 16 / 68.8% / $+1108.01 (after) |
