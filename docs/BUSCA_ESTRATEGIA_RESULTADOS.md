# Busca de estratégia (2026-10-15) — resultados no período de desenvolvimento

Holdout (2025-10-14 .. 2026-10-14) **NÃO utilizado**. Custos: taker 0,06% + slippage 0,02% por perna (0,16% ida e volta) + funding real quando aplicável.
Tentativas desta rodada: 11 (7 em `scripts/strategy_search.py`, 4 em `scripts/online_ml_search.py`). Contagem total considerada no DSR: 95–97 (84 históricas + as desta rodada). **Estas 11 ainda não estão no `trial_log.csv`** — qualquer nova avaliação deve somá-las ao N.

## 1. Hipóteses estruturais/diárias (`strategy_search.py`, 2020-09-14 .. 2025-10-13)
| Hipótese | Sharpe líq. | Observação |
|---|---|---|
| Beta com vol-management (A) | +1,20 | = Buy&Hold (1,18); MDD −40% vs −85%; correlação 0,97 → é beta, não alfa |
| Funding contrário z>2.0 / 1.5 | −0,34 / −0,22 | sem edge |
| Momentum cross-sectional 28d / 14d | +0,37 / +0,76 | IC95 inclui 0; DSR≈0; negativo em 2024–25 |
| Reversão cross-sectional 3d | −0,96 | custos 22%/ano |
| Fear & Greed contrário | −0,01 | sem edge |

## 2. Aprendizado contínuo sobre indicadores (`online_ml_search.py`)
LightGBM binário por lado, retreino **mensal** (janela expandida, embargo H+10), predição OOS pura 2022-01..2025-09, pool BTC/ETH/SOL 1h, regra de entrada fixa (EV_R>0,10), uma posição por ativo.

| Config | AUC OOS (L/S) | Expectância líq./trade | IC95 | p(permutação) | Sharpe | DSR |
|---|---|---|---|---|---|---|
| H=24, TP1.5/SL1.0 | 0,517/0,518 | −0,053% | [−0,175; +0,062] | 0,003 | −0,43 | 0,000 |
| H=48, TP2.5/SL1.5 | 0,511/0,516 | +0,026% | [−0,115; +0,186] | 0,001 | +0,17 | 0,001 |
| H=96, TP4.0/SL2.5 | 0,497/0,505 | −0,049% | [−0,287; +0,188] | 0,117 | −0,21 | 0,000 |
| H=48 + features estendidas (funding, F&G, fator BTC, hora) | 0,516/0,513 | −0,075% | [−0,202; +0,052] | 0,054 | −0,58 | 0,000 |

## Conclusão (honesta)
- Os indicadores contêm **informação preditiva real, porém pequena** (AUC ≈ 0,515; modelo bate entradas aleatórias, p≈0,001–0,003). O edge **bruto** estimado é ≈ 0,1–0,2% por trade.
- O custo de 0,16% por trade **consome** esse edge: expectância líquida estatisticamente indistinguível de zero em todas as configurações, e instável por ano (ex.: H=48: 2022 −0,24%, 2024 +0,24%).
- Enriquecer features **não melhorou** o resultado. A calibração do topo é ruim (previsto 0,5 → realizado 0,4): limiares de confiança mais altos não ajudam.
- Nenhuma configuração passa o critério de promoção (IC95 inferior > 0 e DSR ≥ 0,95). **Nada é promovido ao piloto; o holdout permanece trancado.**
- Hipótese ainda não testada (e que NÃO está validada): se o custo por trade cair de forma comprovada (execução maker com modelo de fill-rate e seleção adversa, ou menor taxa por volume), o mesmo sinal passaria a ter margem. Isso exige modelar execução limit antes de qualquer afirmação.
- O piloto atual NÃO usa este modelo de aprendizado contínuo (o `TechnicalAgent` roda sem modelo; `LearningTracker` não é chamado).

## 3. Teste com execução MAKER realista (scripts/maker_exec_test.py; entrada limite com trade-through de 0,02%, TP maker, SL/timeout taker)
| Config | Fill rate | Expectância líq./trade | IC95 | p(permutação) | Sharpe |
|---|---|---|---|---|---|
| H=48, TP2.5/SL1.5 | 96,9% | -0,126% | [-0,243; -0,004] | 0,464 | -1,04 |
| H=24, TP1.5/SL1.0 | 96,6% | +0,007% | [-0,084; +0,101] | 0,000 | +0,08 |

Leitura: a execução maker NÃO resgata o sinal. O custo médio cai pouco (média nula de -0,168% para -0,132%) porque ~60% das saídas são stops a mercado (taker + slippage). O H=48 deixa de ser distinguível de aleatório (p=0,46) por seleção adversa; o H=24 fica em +0,007%/trade com IC95 [-0,084; +0,101]. Nenhum candidato passa o critério (IC95 inferior > 0, DSR >= 0,95). Tentativas totais desta rodada: 13 (N do DSR = 97). Ressalva: só OHLC 1h; fill-rate é estimativa, não substitui ordens reais.

## 4. Rodada de padrões de price action + trailing no TP1 (3 tentativas; N do DSR = 97 + 2 = 99, +1 = 100)
Holdout (2025-10-14..2026-10-14) **continua trancado** (nada rodou com `--holdout`; o experimento de trailing foi truncado em 2025-10-13 23:00).

### 4.1 `--patterns` (`data/pattern_features.py`, 13 features causais; mesmo walk-forward/regra de entrada/custo taker 0,16%)
Pin bar de alta/baixa, engolfo, inside bar, divergência regular de RSI(14) (pivots confirmados com atraso n=3, janela ≤30 barras), distância % ao último swing high/low confirmado (`market_structure`) e flag "padrão a ≤0,5 ATR da zona". Testes de causalidade (truncar/alterar barras futuras não muda o valor em t) em `tests/test_pattern_features.py`.

| Config | AUC OOS (L/S) | Trades | Expectância líq./trade | IC95 | p(perm.) | Sharpe | DSR (N=99) |
|---|---|---|---|---|---|---|---|
| H=24, TP1.5/SL1.0 + patterns | 0,5185/0,5208 | 1300 | −0,064% | [−0,184; +0,061] | 0,009 | −0,51 | 0,000 |
| H=48, TP2.5/SL1.5 + patterns | 0,5131/0,5151 | 1566 | −0,099% | [−0,240; +0,055] | 0,126 | −0,73 | 0,000 |

Comparação honesta com a seção 2 (sem patterns): H=24 AUC 0,517/0,518 → 0,5185/0,5208 (+0,002, dentro do ruído) e expectância −0,053% → −0,064%; H=48 AUC 0,511/0,516 → 0,5131/0,5151 e expectância +0,026% → −0,099% (p da permutação 0,001 → 0,126). **Padrões não agregam informação utilizável**: AUC praticamente igual, expectância líquida não melhora e permanece indistinguível de zero (IC95 inclui 0 em ambos; DSR 0). Instável por ano (H=48: 2024 +0,084%, 2025 −0,278%). Observação: as duas rodadas diferem em dezenas de bps apenas por ruído de seleção de trades; não há evidência de melhora.

### 4.2 Trailing liga já no TP1 (`BacktestEngine(pilot_parity=True, trailing_activation='tp1')`, `scripts/trailing_tp1_experiment.py`)
Baseline PROD (variante BASE do `ab_consensus_experiment`), 2020-09 .. 2025-10-13, portfólio $10k, execução idêntica ao piloto; trailing = 1,2×ATR de entrada, só barras fechadas, nunca afrouxa; restante (60%) sob trailing desde o TP1.

| Modo | Trades | Hit | Expectância líq. (notional) | IC95 | Bruta | Sharpe | MDD | Motivos de saída |
|---|---|---|---|---|---|---|---|---|
| `tp2` (piloto) | 9260 | 39,5% | −0,291% | [−0,329; −0,251] | −0,176% | −5,54 | −86% | SL 5591, BE 2195, TRAIL 971, TP3 469 |
| `tp1` | 9732 | 39,5% | −0,320% | [−0,356; −0,284] | −0,205% | −6,69 | −99% | SL 5881, TRAIL 3535, TP3 289, BE 1 |

Diferença diária pareada (tp1 − tp2): −0,62 bps/dia do capital, IC95 por blocos [−0,80; −0,45], p(≤0)=1,000. **Conclusão: piorou** (−0,03 p.p./trade; negativo em todos os anos 2020–2025). O trailing de 1,2 ATR logo após o TP1 (1,5 ATR) é apertado demais: transforma o breakeven em saídas a mercado (taker) e reduz os TP3 (469 → 289) sem salvar ganho; o baseline PROD já é fortemente negativo (o gargalo é o sinal, não a saída). Proposta **não recomendada**.

### N do DSR
Anterior: 97 (84 históricas + 13 desta busca). Esta rodada: +2 (patterns H=24, H=48) = **99** (`PRIOR_TRIALS=97`, `NEW_TRIALS=2` em `online_ml_search.py`), +1 (trailing tp1) = **100** acumulado. O DSR do trailing não foi calculado (expectância já significativamente negativa). Nenhuma das 3 tentativas passa o critério de promoção; nada vai ao piloto. Ainda não estão no `trial_log.csv`.
