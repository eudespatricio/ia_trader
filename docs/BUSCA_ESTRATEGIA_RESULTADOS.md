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
