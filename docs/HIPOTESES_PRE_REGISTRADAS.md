# Hipóteses Pré-Registradas – Fase R (Pesquisa Empírica de Estratégias)

> **Documento Vinculante:** Criado e congelado em 14/10/2026, ANTES da execução de qualquer backtest da Fase R.
> Em conformidade com o `docs/PROTOCOLO_EXPERIMENTAL.md` e `docs/PESQUISA_ESTRATEGIAS.md`, todos os parâmetros, regras operacionais e critérios objetivos de refutação estão fixados a priori. Nenhuma reinterpretação a posteriori é permitida.

---

## 0. Regras Globais e Protocolo de Higiene

1. **Tentativas e Registro Obrigatório:** Toda e qualquer execução de backtest deve ser registrada atômica e compulsoriamente em `datasets/reports/trial_log.csv`. O total de tentativas acumuladas (incluindo as 56 registradas das Fases 3 e 4) é insumo do Deflated Sharpe Ratio (DSR) e correção de Bonferroni/Holm.
2. **Holdout Final Trancado:** O período dos últimos 12 meses (2025-10-14 a 2026-10-14) permanece estritamente bloqueado. Ninguém abre o holdout final nesta etapa de pesquisa. O desenvolvimento e validação ocorrem exclusivamente em dados anteriores.
3. **Custos Realistas:** 
   - Taxa Taker: 0,06% por perna.
   - Slippage: 0,02% por perna em spot e 0,03% em perp.
   - Funding Rate: débito/crédito real a cada 8h indexado às taxas históricas.
   - Cenário Maker: estritamente informativo se aplicável, sem compor conclusões de viabilidade sem modelo de fill rate.
4. **Amostra Mínima:** Nenhuma conclusão afirmativa sobre estratégias com menos de 100 trades OOS ou menos de 2 anos de histórico avaliado.

---

## Experimento E1: Trend Following Diário com Volatility Targeting (TSMOM)

### 1. Razão Econômica
O Time Series Momentum (TSMOM) fundamenta-se em sub-reação inicial a fluxos informacionais institucionais, past-return chasing e desalavancagem forçada em tendências de baixa (Moskowitz, Ooi, Pedersen, 2012; Liu & Tsyvinski, 2021). O dimensionamento de risco por volatilidade alvo (vol targeting) estabiliza o risco da carteira, reduzindo a exposição em regimes de alta volatilidade e pânico (Barroso & Santa-Clara, 2015; Moreira & Muir, 2017).

### 2. Universo e Dados
- Ativos: BTCUSDT, ETHUSDT (+ SOLUSDT como teste de robustez cruzada).
- Timeframe: Diário (1d).
- Período de Treino/Validação: Início do histórico disponível até 2025-10-14 (Holdout trancado de 12 meses excluído).

### 3. Especificação do Modelo e Parâmetros Pré-Registrados
- **Sinal de Tendência (Ensemble TSMOM):**
  - Média aritmética dos sinais de retorno passado em 4 horizontes: 20 dias, 50 dias, 100 dias e 200 dias.
  - Cada lookback $L \in \{20, 50, 100, 200\}$ emite $s_L = \text{sign}(P_t - P_{t-L}) \in \{-1, +1\}$ (ou $\{0, +1\}$ na versão Long/Flat).
  - Sinal agregado do ensemble: $S_t = \frac{1}{4} \sum_{L} s_L$.
  - Modo primário: **Long / Flat** (opera comprado quando $S_t > 0$; neutro quando $S_t \le 0$).
  - Modo secundário (sensibilidade): Long / Short simétrico.
- **Volatility Targeting:**
  - Volatilidade anualizada estimada via EWMA ou janela móvel de 30 dias de retornos logarítmicos: $\sigma_{\text{ann}, t} = \sigma_{30d} \cdot \sqrt{365}$.
  - Alvo de volatilidade: $\sigma_{\text{target}} = 20\%$ a.a.
  - Peso teórico da posição: $w_t = S_t \cdot \min\left(1.5, \frac{\sigma_{\text{target}}}{\max(\sigma_{\text{ann}, t}, 0.05)}\right)$.
  - Banda de no-trade (filtro de turnover): Só rebalanceia se $|w_t - w_{t-1}| > 0.10$.
- **Custos:** Taxa taker 0,06% + slippage 0,03% por rotação, funding real diário.
- **Validação:** Walk-forward com janelas de treino expansivas/deslizantes e embargo de 10 dias.

### 4. Critérios Mecânicos de Refutação de E1
A estratégia E1 será considerada **REFUTADA** se qualquer um dos seguintes critérios for verdadeiro:
1. **Sharpe OOS < 0.50:** Sharpe ratio líquido anualizado fora da amostra for inferior a 0,50.
2. **Derrota para Buy & Hold:** Sharpe ratio líquido OOS for menor que o Sharpe do Buy & Hold **OU** Maximum Drawdown OOS for pior que o do Buy & Hold.
3. **DSR Insignificante:** Deflated Sharpe Ratio ($DSR$) < 0,95, considerando a contagem total de testes em `trial_log.csv` e a variância dos testes.
4. **Fragilidade Temporal (Melhor Ano):** Remoção do melhor ano do histórico reduz o Sharpe líquido OOS para menos de 0,30.
5. **Hipersensibilidade de Hiperparâmetros:** Variação de $\pm 30\%$ nos lookbacks ($L \times 0.7$ e $L \times 1.3$) derruba o Sharpe OOS em mais de 50%.
6. **Arrasto Excessivo de Fricção:** Custos totais (taxas + slippage + funding) representam mais de 30% do retorno bruto gerado.

---

## Experimento E2: Funding Carry Delta-Neutro (Spot Long + Perp Short)

### 1. Razão Econômica
O prêmio de liquidez e alavancagem em criptoativos decorre da preferência estrutural de participantes de varejo por posições compradas perpétuas alavancadas, pagando taxa de financiamento (funding rate positivo) para vendedores de perp (Schmeling, Schrimpf, Todorov, BIS WP 1087, 2023). Montar Long Spot e Short Perp no mesmo valor notional neutraliza a variação direcional do preço ($\Delta \approx 0$) e colhe o fluxo de funding.

### 2. Universo e Dados
- Ativos: BTCUSDT, ETHUSDT.
- Frequência: 8 horas (janela natural de funding) e diário.
- Período: Início do histórico com dados confiáveis de funding e spot até 2025-10-14.

### 3. Especificação do Modelo e Parâmetros Pré-Registrados
- **Alavancagem:** $\le 2.0x$ na perna perpétua (margem colateral de pelo menos 50% do notional em USDT/USD).
- **Regra de Entrada e Saída (Filtro de Funding Médio):**
  - Entra se o funding anualizado médio móvel de 7 dias ($MA7$) for superior a $8.0\%$ a.a. ($> 0.0073\%$ por 8h).
  - Desmonta a estrutura se o funding médio de 7 dias cair abaixo de $2.0\%$ a.a.
- **Custos de Montagem/Desmontagem (4 Pernas):**
  - Entrada: Taker spot (0.06% + 0.02% slip) + Taker perp (0.06% + 0.03% slip) = 0.17%.
  - Saída: Taker spot (0.06% + 0.02% slip) + Taker perp (0.06% + 0.03% slip) = 0.17%.
  - Custo total round-trip de 4 pernas: ~0.34%.
- **Stress Test de Choque Instantâneo:** Simulação de salto de preço de $\pm 30\%$ em 24h para avaliar risco de liquidação da perna short perp e margem de manutenção remanescente.

### 4. Critérios Mecânicos de Refutação de E2
A estratégia E2 será considerada **REFUTADA** se qualquer um dos seguintes critérios for verdadeiro:
1. **Retorno Líquido Insuficiente:** Retorno líquido anualizado no período pós-2022 for inferior à taxa livre de risco (SOFR/T-Bill ~4.5%) + 3.0% a.a. (ou seja, retorno líquido < 7.5% a.a.).
2. **Drawdown Excessivo:** Maximum Drawdown da curva patrimonial delta-neutra for $\ge 5.0\%$.
3. **Risco de Liquidação em Stress:** No choque simulado de $+30\%$ em 1 dia, a margem livre cair abaixo de 30% do colateral ou violar margem de manutenção da exchange.
4. **Persistência de Funding Negativo:** O funding rate for negativo ou nulo por mais de 40% dos períodos de 8h avaliados.
5. **Custo de Rotação Destrutivo:** Custos de entrada, saída e rebalanceamento consumirem mais de 30% do rendimento bruto de funding colhido.
6. **Concentração Unilateral em Ciclos Antigos:** Todo o retorno positivo depender exclusivamente do bull market de 2021, apresentando carry próximo a zero ou negativo entre 2022 e 2025.

---

## Experimento E3: Meta-Labeling sobre E1 com CPCV, DSR e PBO

### 1. Razão Econômica
Conforme demonstrado por Marcos López de Prado (2018) e Joubert (2022), o meta-labeling não gera sinais a partir do nada; ele atua como uma camada secundária de Machine Learning para calibrar o dimensionamento ou filtrar falsos positivos de uma regra primária determinística fundamentada.

### 2. Condição de Ativação Pré-Registrada
- **Condição Mandatória:** E3 SÓ SERÁ EXECUTADO se o Experimento E1 demonstrar edge estatístico fraco e positivo (Sharpe OOS > 0.40, retorno líquido positivo).
- **Regra de Pulo/Abandono:** Se E1 for integralmente refutado ($H_0$ não rejeitada, Sharpe < 0.40 ou custos inviabilizando a regra), o Experimento E3 **NÃO TERÁ BASE OPERACIONAL E SERÁ DECLARADO PULADO/NULO**, relatando formalmente que ML não resgata sinal primário nulo.

### 3. Especificação do Modelo (Se Ativado)
- Modelo Secundário: LightGBM Classifier com restrição de profundidade (`max_depth=3`, `num_leaves=7`).
- Features de Contexto Primárias (calculadas estritamente em D-1): Volatilidade realizada de 14 e 30 dias, distância percentual à SMA 200, momentum relativo de 5 e 10 dias, funding rate médio recente.
- Rotulagem: Triple-barrier proporcional ao ATR da barra diária.
- Validação: Combinatorially Symmetric Cross-Validation (CPCV) e Purged K-Fold com embargo.

### 4. Critérios Mecânicos de Refutação de E3
1. **Percentil 5 OOS $\le$ Regra Pura:** O percentil 5 ($p_5$) da distribuição de Sharpe OOS no CPCV for menor ou igual ao da regra E1 pura.
2. **PBO Elevado:** Probability of Backtest Overfitting ($PBO$) > 0.30.
3. **DSR Insignificante:** Deflated Sharpe Ratio ($DSR$) < 0.95.
4. **Ganho de Sharpe Marginal:** Incremento de Sharpe ratio OOS frente à regra pura for inferior a 0,20 ($\Delta SR < 0.20$).
5. **AUC Fraca:** Área sob a curva ROC OOS média $< 0.53$.
6. **Desaparecimento sob Embargo:** Ganho evapora ao expandir o embargo temporal para 15 barras.
