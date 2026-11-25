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


---

# ADDENDUM E4 — Pré-Registro das Estratégias em Paper Trading (GOLDEN_POT, SCALP, DIRECTIONAL/SWING)

> **Data do registro:** 2026-10-15 (UTC). **Natureza:** registro retroativo-ao-código, prospectivo-ao-dado. As regras abaixo descrevem o código **como ele já estava** (commit `c600ea052aa87146b85bbd7f7784c8fde5b81f75` + working tree de 2026-10-15; SHA-256 de `smart_trade.py` = `dca527e60028eaf0acc8162ff9d4b4b92c70dbef582d8839b8bfc99363cbe3d1`, `scripts/run_paper_simulation.py` = `8bea852a08a69337d06b1010810406522bfc14226c8a2dff9f12a4a7af87d60b`, `core.py` = `5a55fb7c61aabe9369ad7d0c4f8fb00270e81c22a5955e23904e9035afc6da7e`, `signal_generator.py` = `88ff42d1842cbea8cf2fe242fdd823502a7cfe34f0ad77efa7f1a3d8dd46d68f`, `agents/golden_pot_agent.py` = `df37976ecb8a463d572c6c946a60e13fab57d7087d194fe0c93fc4e5239df8ce`). Nenhuma regra foi inventada ou "melhorada" ao documentá-la.
>
> **Divulgação de integridade (obrigatória):** (i) este registro foi escrito **depois** do reset que abriu a janela limpa atual (`reset_epoch_utc = 2026-10-15T00:47:53Z`) — mas com **0 trades fechados** nessa janela (`pilot_report.json`: `closed_trades = 0`); nenhum resultado limpo foi observado antes de congelar. (ii) Existe histórico **pré-reset** (35 trades fechados, 11 `MANUAL_RESET`, ≈ 8 h de dados, parâmetros alterados) que **já foi visto** — ele motivou a escolha de limiares de refutação mais duros abaixo e **não conta como evidência a favor**. (iii) Este registro **não** foi feito antes de a estratégia existir; ele apenas impede reinterpretação a posteriori a partir de agora.

## E4.0 Identificadores reconhecidos pelo `pilot_report.py` (não renomear)

O critério A3 compara `strategy_type` (coluna de `paper_positions`) com este documento. Os valores reais gravados pelo motor são:

| `strategy_type` no DB | Estratégia | Origem no código |
|---|---|---|
| `GOLDEN_POT` | Golden Pot (reversão de VWAP / sweep+POI sobre scanner de universo) | `evaluate_golden_pot_opportunities` → `execute_golden_pot_signal` |
| `SCALP` | Scalp Padrão Core (reversão de VWAP+RSI+volume em BTC/ETH/SOL 15m) | `evaluate_scalp_opportunities` → `execute_scalp_signal` |
| `DIRECTIONAL` | **"Swing"** (comitê de agentes 1h; o nome `SWING` nunca é gravado no DB) | `evaluate_swing_opportunities` → `execute_directional_signal` |

Para que fique inequívoco: **GOLDEN_POT**, **SCALP**, **DIRECTIONAL** (a.k.a. SWING). (`DELTA_NEUTRAL_CARRY` permanece em STANDBY e fora deste registro.)

## E4.1 Razão econômica (honesta) e declaração de ausência de alfa

- **Tese nominal:** desvios intradiários do preço em relação ao VWAP, acompanhados de RSI extremo/volume, revertem (microestrutura: provisão de liquidez, exaustão de fluxo); varreduras de liquidez em zonas de POI sinalizam reversão (heurística "Smart Money Concepts"); o comitê direcional capta tendência/regime de 1h.
- **Declaração honesta:** **não existe evidência prévia de alfa para nenhuma das três.** O pré-piloto teve **expectância bruta negativa** (antes de custos) e é **viesado** (resets, parâmetros alterados, janela de ~8 h, universo de tickers de baixa capitalização). Reversão de VWAP em cripto de 15m tem literatura fraca e é a classe de estratégia que o `PARECER_CHIEF_TRADING_SCIENTIST.md` §2.2 condenou. SMC/POI/"liquidity sweep" **não possui validação acadêmica** conhecida; nenhum backtest pré-registrado dessas regras existe em `datasets/reports/` (A9 inatingível). **A hipótese prior é que todas as três são nulas após custos.** A probabilidade a priori de alfa líquido real é baixa; o piloto existe para **refutar**, não para confirmar.
- **Multiplicidade:** três hipóteses testadas simultaneamente, sobre um `trial_log.csv` com 83 tentativas anteriores (84 linhas incl. cabeçalho) → o DSR e a correção de Holm abaixo são obrigatórios.

## E4.2 Especificação congelada (regras como no código)

### Parâmetros globais do motor (`PaperTradingEngine`, defaults + `simulation_settings`)
- Capital inicial $10.000; taker 0,06%; maker 0,02%; slippage spot 0,02%; slippage perp 0,03%; modo simulação obrigatório (live bloqueado por `PermissionError`).
- `max_daily_golden_pots = 3`; `max_daily_scalps = 5` (cotas **independentes**, reset por dia UTC); `max_concurrent_positions = 5` (settings default); `max_margin_utilization_pct = 0,85`; mínimo de notional $10; `available_margin < $200` bloqueia novas entradas de GP e SCALP.
- Um único `OPEN` por (símbolo, `strategy_type`). Ciclo do daemon: `poll_interval` padrão 20 s (`main`) / 15 s (`__init__`); ordem por ciclo: atualizar posições → GP → SCALP → DIRECTIONAL.
- Circuit breaker `RollbackGuard` (`can_open_new_position`, `CONSECUTIVE_LOSS_COOLDOWN`, `DEGRADATION_ROLLBACK_ACTIVATED`) bloqueia aberturas para SCALP/GP; seus limiares (arquivo `rollback_guard.py`) fazem parte do congelamento **por referência ao hash do repositório** e não são alterados durante o piloto.
- `operation_mode = INSTITUTIONAL` (limiares de GP abaixo). O modo `DEMO` (VWAP 1,2% / RSI 42/58) **não é** a especificação registrada; usá-lo na janela é violação (A11).
- Entradas ocorrem no `close` da última vela retornada por `DataLoader.get_candles` (Binance; fallback Bitget). **Não verificado neste registro** se essa última vela já está fechada — risco conhecido de repainting/lookahead intra-barra, declarado como limitação.

### E4.2.a GOLDEN_POT
- **Universo:** `GoldenPotAgent(min_volume_usd=20_000_000)`; `scan_market` → `top_scalps[:4]` (ordem por `score`). Candidato exige `|dist_vwap_24h| ≥ 2,5%` **e** amplitude 24h ≥ 4,0% (setup `SCALP_REVERSION`; direção BUY se abaixo do VWAP, SELL se acima). Universo dinâmico (pode incluir micro-caps listados, ex. SOXSUSDT/WALUSDT/BTWUSDT) — **risco de seleção adversa e de slippage subestimado**.
- **Entrada (15m, 100 velas, mín. 50, `compute_all_features(drop_na=True)`):** BUY se `(dist_vwap_pct ≤ −2,5 e rsi_14 ≤ 38)` **OU** `(liquidity_sweep_bull = 1 e at_bullish_poi = 1)`; SELL se `(dist_vwap_pct ≥ +2,5 e rsi_14 ≥ 62)` **OU** `(liquidity_sweep_bear = 1 e at_bearish_poi = 1)`. O ramo sweep+POI **não exige** desvio de VWAP nem RSI.
- **Execução simulada:** fill **100% garantido** a `close·(1 ± 0,0001)` com taxa **maker 0,02%** (sem modelo de fill rate nem seleção adversa).
- **Saídas:** SL = 1,0×ATR14(15m); TP1 = 1,2×ATR → vende 50% (maker); após TP1, SL → entrada ±0,04% (breakeven); TP2 = 2,5×ATR (maker) fecha os 50% restantes. SL e `BREAKEVEN_STOP` são **a mercado**: taker 0,06% + slippage perp 0,03%. Sem stop de tempo (posição pode ficar aberta indefinidamente).
- **Tamanho:** `position_size = min(1200, available_margin·0,95, balance·0,20)`; no motor, `notional = min(position_size, free_margin, balance·0,20)`. `risk_dollars = 50` é **rótulo cosmético** (o risco real = notional × distância do SL ≈ 20%·1,0·ATR/preço do saldo).
- **Alavancagem/risco:** sem alavancagem explícita além do teto de margem 85%; funding **não** é debitado/creditado (`funding_rate_period = 0`).

### E4.2.b SCALP
- **Universo:** `ETHUSDT`, `SOLUSDT`, `BTCUSDT` (nesta ordem de prioridade), 15m, 100 velas.
- **Entrada:** BUY se `dist_vwap_pct ≤ −2,0` **e** `rsi_14 ≤ 35` **e** `volume_zscore_20 ≥ 0,2`; SELL se `dist_vwap_pct ≥ +2,0` **e** `rsi_14 ≥ 65` **e** `volume_zscore_20 ≥ 0,2` (valores de `simulation_settings`: `scalp_vwap_dev_pct`, `scalp_rsi_oversold/overbought`, `scalp_volume_z`). Máx. 5/dia (global). `dist_vwap_pct` aqui vem de `feature_engineering` (VWAP de sessão/janela das 100 velas, **não** o VWAP 24h do scanner).
- **Execução simulada:** maker 0,02%, fill 100% a `close·(1 ± 0,0001)`.
- **Saídas:** SL = 0,9×ATR14; TP1 = 1,0×ATR (50%, maker); breakeven ±0,04% após TP1; TP2 = 1,8×ATR (maker). SL/BE a mercado (taker + 0,03%).
- **Tamanho:** `position_size = min(1500, available_margin·0,95, balance·0,20)`; no motor `notional = min(.., free_margin, balance·0,25)`. Sem funding.

### E4.2.c DIRECTIONAL ("Swing", 1h)
- **Universo:** BTCUSDT, ETHUSDT, SOLUSDT; 1h; 400 velas buscadas; mínimo bruto 250; exige `sma_200` finita > 0 (warmup, `_compute_swing_features`).
- **Entrada:** `TradingCore.evaluate_multi_strategy` → `evaluate_market_deterministic`: `RiskSentinel` com poder de veto; demais agentes `is_backtestable` avaliados em `ConsensusJudge.judge_deterministic` (`min_consensus_threshold = 0,30`); abre se `direction ∈ {BUY, SELL}`, não vetado e `confidence ≥ 0,70`. Pesos/regras internas dos agentes: as de `agents/*.py` e `core.py` no hash congelado (não reproduzidas aqui para evitar divergência).
- **Execução simulada:** **taker** 0,06% + slippage (spot 0,02% long / perp 0,03% short) na entrada e na saída.
- **Saídas:** SL = 1,0×ATR14(1h); TP = 1,5×ATR (`SignalGenerator` defaults; `structure_context` não é passado → sem assimetria SMC); sem parciais, sem trailing, sem stop de tempo.
- **Tamanho:** risco fixo 1% do capital do `SignalGenerator` (capital **fixo de construção = $10.000, não o saldo corrente**) → `size = 1%·capital / distância_do_SL`, teto por alavancagem 1,0×; convertido para notional (`size·close`); motor limita a `min(notional, free_margin, balance·0,5)`. Um único DIRECTIONAL por símbolo. Funding não aplicado.

### Premissas de simulação que **favorecem** o resultado (declaradas ex ante, não corrigidas)
1. Maker com fill 100% sem seleção adversa (GP/SCALP) — o fill real de ordens limite correlaciona-se negativamente com o resultado.
2. Dentro da mesma vela, TP1 é avaliado **antes** do SL (otimista quando a vela toca ambos).
3. Stops executam no preço do stop (sem gap-through); funding ignorado.
4. Slippage de 0,01–0,03% é irrealista para micro-caps do universo GP.
O relatório decisório (`pilot_report.py`) recalcula tudo com **estresse de 0,08% por perna (taker 0,06% + slippage 0,02%)**; é essa a métrica que vale, não o P&L contábil do motor.

## E4.3 Hipóteses (uma por estratégia; testadas separadamente)

Para cada `S ∈ {GOLDEN_POT, SCALP, DIRECTIONAL}`, sendo $\mu_S$ a expectância líquida por posição fechada (soma de todos os fills da `position_id`, incl. `TP1_PARTIAL`), em % do notional, **sob custo estresse**:

- **$H_0$:** $\mu_S \le 0$.
- **$H_1$:** $\mu_S > 0$ (e, para chamar de implementável, $\mu_S \ge 0{,}10\%$ do notional por posição — abaixo disso o resultado não sobrevive a erro de modelagem de custo).

## E4.4 Métrica primária e regra de decisão

- **Primária:** expectância líquida estresse por posição fechada (% do notional), por `strategy_type`. Secundárias (já em `pilot_report.py`): Sharpe diário anualizado ×√365, DSR (N = tentativas do `trial_log.csv` + 1), MaxDD estresse, fração de semanas positivas, hit-rate vs breakeven (Wilson).
- **Aprovação de uma estratégia exige** o conjunto A1–A14 de `docs/CRITERIOS_APROVACAO_PILOTO.md` **e**: limite inferior bootstrap (5º pct, 5000 reamostras) de $\mu_S$ > 0 **com correção de Holm** para as 3 hipóteses (α = 0,05 família; limiar mais rígido 0,0167 para a menor p-valor). Pool de estratégias **não** pode ser usado para aprovar uma delas isoladamente.
- A3 só valida o *registro*; **não** constitui evidência de alfa.

## E4.5 Tamanho mínimo de amostra

- **Mínimo operacional (necessário, não suficiente):** ≥ 100 posições fechadas **por estratégia** na janela limpa e ≥ 56 dias (≥ 84 alvo) sem `MANUAL_RESET` nem mudança de `simulation_settings` (A1/A2/A11); abaixo disso nenhuma conclusão afirmativa.
- **Potência (aprox., one-sided, Holm: $z_\alpha \approx 2{,}13$, potência 80%: $z_\beta \approx 0{,}84$):** $n \ge \big((z_\alpha+z_\beta)\,\sigma/\mu\big)^2 \approx 8{,}8\,(\sigma/\mu)^2$. Com σ por posição **assumido** (não medido) ≈ 0,5% (SCALP/GP) e μ = 0,10% → **n ≈ 220**; com σ ≈ 1,0% (DIRECTIONAL 1h) e μ = 0,15% → **n ≈ 390**. Logo, 100 trades só detectam efeitos ≥ ~0,15% (σ=0,5%) ou ≥ ~0,30% (σ=1,0%).
- **Consequência esperada:** DIRECTIONAL (≤ 3 símbolos, uma posição por vez, sinal conservador) provavelmente **não atingirá N=100 em 56 dias** → resultado **INCONCLUSIVO por construção**, não aprovação. SCALP/GP (cotas 5 e 3/dia) podem atingir N em semanas, mas com risco de trades correlacionados (mesma barra, mesmos ativos, mesmo regime) → N efetivo < N nominal.

## E4.6 Critérios mecânicos de refutação (qualquer um ⇒ a estratégia é REFUTADA)

Herdam R1–R7 de `docs/CRITERIOS_APROVACAO_PILOTO.md` e adicionam, por estratégia:

1. **Futilidade (R2):** N ≥ 30 posições fechadas e limite **superior** do IC95 unilateral de $\mu_S$ (estresse) < 0.
2. **Sem alfa bruto (R6):** N ≥ 30 e expectância **bruta** (antes dos custos) ≤ 0. Dado o pré-piloto com bruta negativa, este é o desfecho **mais provável**; se ocorrer, nenhuma otimização de execução (maker, filtros, horário) é admissível.
3. **Custos dominam:** custos totais (taxa + slippage) > 30% do retorno bruto positivo agregado, ou breakeven WR (A8) > WR observado no limite Wilson.
4. **Drawdown:** MaxDD estresse ≥ 20% (R1) do portfólio, ou ≥ 10% atribuível a uma única estratégia.
5. **Cauda/ruína:** uma única posição com perda > 3× a perda planejada (risco de gap/stop não executado), ou perda diária > 5% do capital.
6. **Concentração:** > 50% do P&L líquido positivo vindo de um único símbolo ou de ≤ 3 trades (indica sorte/outlier, não processo).
7. **Prazo (R7):** 56 dias com Sharpe < 1,0 ou DSR < 0,95.
8. **Integridade:** qualquer `MANUAL_RESET` ou alteração em `simulation_settings` **reinicia o relógio de aprovação**, mas **nunca** o de refutação (o ledger inteiro conta).

**Efeito da refutação:** a estratégia é **abandonada**; é proibido "ajustar e reiniciar" sem nova hipótese pré-registrada, +1 no `trial_log.csv` e novo holdout (Protocolo §1.1–1.2, 1.7). Refutar as três é um resultado válido e esperado (Protocolo §3).

## E4.7 Lista de proibições durante a janela

Não alterar: limiares de entrada/saída, múltiplos de ATR, cotas diárias, tamanho, `operation_mode`, universo, RollbackGuard, nem o código de `smart_trade.py`, `core.py`, `signal_generator.py`, `agents/*`, `scripts/run_paper_simulation.py`. Correções de bug exigem registro de nova versão e **reinício do relógio de aprovação** (A1) com nota neste documento. Qualquer *novo* `strategy_type` negociado sem registro aqui faz A3 falhar.

# ADDENDUM E5 — GOLDEN_POT v2, lote = margem e alavancagem do Smart Trader (2026-10-15)

**Status:** substitui a regra de GOLDEN_POT do E4. **O piloto é REINICIADO** após esta mudança (relógio de aprovação zerado, A1/A11) e **o E4 fica INVALIDADO para GOLDEN_POT** (regra de entrada, tamanho e universo mudaram). SCALP, DIRECTIONAL/SWING mantêm a regra de entrada do E4, mas o **tamanho de lote mudou para todos** (E5.6), o que também reinicia o piloto. Tokens de estratégia inalterados: `GOLDEN_POT`, `SCALP`, `DIRECTIONAL`, `SWING`.

## E5.1 Motivo (falhas confirmadas da v1)
(a) `confidence=0.90`, `consensus_score=±0.85`, `risk_dollars=50` eram constantes; o comitê e o RiskSentinel (veto) nunca eram consultados; (b) sem cooldown nem limite por vela (5 GPs na mesma vela 15m, 4 stopados em ~20 min); (c) compra de faca caindo sem filtro de regime; (d) perps de ações/ETFs tokenizados e listagens novas (~100–140 candles).

## E5.2 Regra exata do GOLDEN_POT v2 (`scripts/run_paper_simulation.py::evaluate_golden_pot_opportunities`)
Por ciclo, só se `can_execute_golden_pot` (cota **3/dia**, inalterada; posições simultâneas; margem; circuit breaker):
1. **Descoberta:** `GoldenPotAgent.scan_market()` → `top_scalps`, removendo símbolos da lista de exclusão (E5.4) e limitando a `GP_MAX_CANDIDATES=4`. Pula símbolo com GP aberto.
2. **Dados:** `GP_CANDLE_LIMIT=400` velas 15m (binance, fallback bitget); mínimo bruto `GP_MIN_WARMUP_CANDLES=250`; **descarta a vela em formação** (somente velas fechadas); features só valem com `sma_200` finita e > 0 (exclui listagens novas).
3. **Limite por vela:** no máximo `GP_MAX_ENTRIES_PER_BAR=1` entrada GP por vela de 15m (chave = `timestamp//15min` da vela; lido de `trade_history`, restaurado do ledger ⇒ sobrevive a reinício).
4. **Cooldown:** sem reentrada no símbolo por `GP_COOLDOWN_BARS=16` barras de 15m após qualquer saída total (`TP1_PARTIAL` não conta), também via ledger.
5. **Gatilho de descoberta (inalterado do E4):** (VWAP ≤ −2,5% e RSI ≤ 38) ou (sweep SSL + POI bullish) ⇒ BUY; simétrico ⇒ SELL (limiares DEMO 1,2%/42/58). É só gatilho: não decide a entrada.
6. **Filtro de regime / faca-caindo:** "tendência forte contra" = (close < sma_200 e [RSI < `GP_KNIFE_RSI_LOW=20` ou queda > `GP_KNIFE_ATR_MOVE=3.0` ATR em `GP_KNIFE_LOOKBACK_BARS=4` barras]) para BUY; simétrico (close > sma_200, RSI > `GP_KNIFE_RSI_HIGH=80`, alta > 3 ATR) para SELL. Em tendência forte contra, **só entra com confirmação de reversão** (última vela fechada a favor e RSI virando); sem confirmação ⇒ bloqueia.
7. **Comitê completo:** `core.evaluate_market_async(context=última linha de features, ...)` (RiskSentinel + Technical/OrderFlow/Sentiment/Macro/OnChain + ConsensusJudge). Entra **somente** se: sem veto do RiskSentinel, `direction` do comitê == direção do gatilho e `confidence ≥ GP_MIN_COMMITTEE_CONFIDENCE=0.70`.
8. **Sinal:** `confidence`, `consensus_score` e `rationale` vêm do `TradeSignal` real do comitê (nada de constantes); `risk_dollars = (notional/preço) × distância do SL`. Saídas inalteradas: SL 1,0 ATR, TP1 1,2 ATR (50% + breakeven), TP2 2,5 ATR (`GP_*_ATR_MULT`).

## E5.4 Exclusão de ativos tokenizados
`GP_EXCLUDED_BASE_ASSETS` (topo do script, casamento por ativo-base exato após remover USDT/USDC/USD/PERP): SKHYNIX, SKHY, SAMSUNG, KORU, SOXL, SOXS, TQQQ/SQQQ/SPXL/SPXS/UVXY, TSLA, NVDA, AAPL, MSFT, GOOGL/GOOG, AMZN, META, NFLX, AMD, INTC, COIN, MSTR, HOOD, PLTR, BABA, TSM, SPY, QQQ, IWM, GLD, SLV, USO. Motivo: perps TradFi têm sessão/gaps e liquidez fora do pregão, fora da hipótese de dinâmica cripto 24/7. Editar a lista = mudança de universo (reinicia o relógio).

## E5.6 Lote = MARGEM; alavancagem decidida pelo Smart Trader
- **Lote** = margem (capital alocado) por trade = `simulation_settings.paper_lot_usd` (padrão **US$ 100**, válido 10–1000; lido via `sync_settings_from_db` em tempo real). Vale para GOLDEN_POT, SCALP e DIRECTIONAL/SWING (substitui 1200/1500/risco-1%-de-10k). `notional = margem × alavancagem`; margem = `min(lote, margem livre×0,95, cap do saldo, position_size/alavancagem)`. Cotas diárias e `max_concurrent_positions` **não** foram alterados.
- **Alavancagem não é configurável pelo usuário** (`leverage_policy.decide_leverage`, teto duro `MAX_LEVERAGE=5`, também reforçado no engine: clamp 1..5 em passos de 0,5). Usa a confiança **real** do comitê:

| Confiança | Alavancagem base |
|---|---|
| < 0,60 | 1x |
| 0,60–0,70 | 2x |
| 0,70–0,80 | 3x |
| 0,80–0,90 | 4x |
| ≥ 0,90 | 5x |

  −1 degrau se ATR% ≥ 3% (volatilidade extrema); −1 degrau se a entrada é contra a tendência (`sma_200`); 1x forçado com veto ou símbolo sem histórico/baixa liquidez (volume 24h < US$ 50M, `GP_LOW_LIQUIDITY_USD`). SCALP padrão (sem comitê) usa 1x.
- **Liquidação:** preço aproximado LONG `entry×(1−1/lev+0,005)`, SHORT `entry×(1+1/lev−0,005)` (manutenção 0,5%); fechamento `LIQUIDATION` se tocado. O engine **garante SL antes da liquidação** (senão reduz a alavancagem em 0,5). PnL e taxas sobre o notional; `margin_used = Σ notional/lev`; teto de margem 85% mantido; funding ignorado como antes. `PaperPosition.leverage` persistido (migração `ADD COLUMN`, default 1.0) e restaurado.
- **Expectância:** taxas e slippage são proporcionais ao notional, logo a expectância em % do notional **não muda** com lote ou alavancagem. **Não há evidência de que alavancar melhore o resultado**: amplifica ganhos e perdas igualmente (e adiciona risco de liquidação); métricas de aprovação seguem em % do notional.
- **Amostra:** com cotas inalteradas (GP 3/dia, SCALP 5/dia) e 1 posição por par, o lote/alavancagem não aumenta o nº de trades; o teto efetivo continua ~8/dia (GP+SCALP) + swing. A v2 deve **reduzir** o nº de GPs (comitê+cooldown+1/vela), portanto N=100 para GOLDEN_POT tende a levar mais que 56 dias.

## E5.7 Reinício do piloto
Mudanças de regra (GP v2), de universo (E5.4) e de tamanho/alavancagem (E5.6) reiniciam o relógio de aprovação; o ledger anterior (incluindo GP v1) permanece contado para **refutação** (E4.6 item 8). `trial_log.csv` +1 (nova versão de GOLDEN_POT).



---

# ADDENDUM E6 — Parâmetros efetivos do SCALP no piloto limpo (2026-10-15, reset_20261015T024946Z)

**Contexto.** O piloto limpo iniciou em 2026-10-15 02:49:47 UTC (reset total). Os parâmetros do SCALP gravados em `simulation_settings` (modo `CUSTOM`, salvo pela UI em 2026-10-15 02:15 UTC) divergem dos valores do E4 (que eram os defaults do código no modo INSTITUTIONAL). Em vez de sobrescrever as configurações, este addendum registra os valores EFETIVOS, que passam a ser os valores congelados do piloto. Este addendum prevalece sobre o E4.2 para o SCALP.

| Parâmetro | E4 (default INSTITUTIONAL) | Efetivo no piloto (E6) |
|---|---|---|
| `scalp_vwap_dev_pct` | 2.0 | **0.8** |
| `scalp_rsi_oversold` / `scalp_rsi_overbought` | 35 / 65 | **30 / 70** |
| `scalp_volume_z` | 0.2 | **-0.5** (filtro de volume praticamente inativo) |
| `max_daily_scalps` | 5 | **15** (teto; o limitador real é o gatilho: 1 posição por símbolo, 3 símbolos core, máx. 6 posições simultâneas) |
| `max_daily_golden_pots` | 3 | 3 (inalterado) |
| `max_concurrent_positions` | 6 | 6 |
| `max_margin_utilization_pct` | 0.85 | 0.85 |
| `paper_lot_usd` | — | 100 (margem; ver E5.6) |

**Por que o valor efetivo foi preferido a reverter ao E4.** (1) Evidência observacional: em todo o piloto anterior ao reset (46 posições), houve **zero** scalps; só GOLDEN_POT operou. Com VWAP >= 2,0% em 15m de BTC/ETH/SOL (aprox. 4–6 ATR), o gatilho do E4 é raríssimo, e o SCALP seria, na prática, uma estratégia sem amostra. (2) Os valores do E6 foram ajustados pelo usuário em testes anteriores para evitar overtrading; a cota 15/dia não é o limitador real.

**Ressalvas honestas (obrigatórias).**
- Esses valores NÃO são pré-registrados no sentido forte: foram escolhidos com observação de dados anteriores (in-sample), e o registro foi escrito após o início da janela, com 0 trades fechados. Qualquer evidência do SCALP deve ser lida com esse desconto (ver o N de tentativas no `trial_log.csv` para o DSR).
- O SCALP **não passa pelo comitê**: `confidence=0.85` e `consensus_score=±0.80` são constantes no código (`evaluate_scalp_opportunities`). É uma limitação conhecida, igual à da v1 do GOLDEN_POT, e NÃO foi alterada neste piloto.
- Com `scalp_volume_z = -0.5` o filtro de volume só barra volume muito abaixo da média.
- Amostra esperada: baixa. O A/B de consenso e o prior (alfa nulo após custos) permanecem; a hipótese nula do E4.3 e os critérios de refutação do E4.6 valem sem alteração.

**Congelamento.** Commit de referência do código: `b65c0a1`. Proibido alterar esses parâmetros durante a janela (E4.7); qualquer alteração reinicia a contagem de dias limpos (critério A11 do `pilot_report.py`).


---

# ADDENDUM E7 — Comitê obrigatório + Smart Trade Exit Ladder comum (2026-10-15)

**Contexto.** Diretriz do usuário: SCALP, GOLDEN POT e DAY TRADE (fluxo 1h `DIRECTIONAL`) passam **obrigatoriamente** pelo comitê (`core.evaluate_market_async`/`evaluate_market_deterministic`: agentes + consenso + veto do RiskSentinel + SignalGenerator). Mudam apenas o **tipo de oportunidade** (gatilho/candidato) e o perfil de alvos; a gestão de saída/risco é **comum**. Este addendum prevalece sobre E4/E5/E6 onde conflitar (E6: "SCALP não passa pelo comitê" fica **revogado**; E5: gatilho VWAP/RSI/sweep+POI do GOLDEN_POT fica **substituído**; E4/E5: "TP1 50% + TP2" e alvo único do direcional ficam **substituídos** pela escada abaixo). Os valores de `simulation_settings` do SCALP (E6) **não foram alterados**. **O piloto é reiniciado** (relógio de dias limpos zerado; ledger anterior segue contando para refutação, E4.6). `trial_log.csv` +1 (novas versões de SCALP, GOLDEN_POT e DIRECTIONAL).

## E7.1 Regra de entrada por estratégia (porta do comitê comum)
Condição comum a todas: comitê na **mesma direção** do gatilho, **sem veto**, `confidence >= 0.70` (constante por perfil, `STRATEGY_PROFILES[*].min_committee_confidence`); `confidence/consensus/risk_dollars/rationale` são os do sinal real; alavancagem = `decide_leverage(confiança real, ATR%, regime)` (máx 5x); lote = margem `paper_lot_usd`; SL sempre antes da liquidação (reduz alavancagem se preciso); cotas/cooldown/limite por vela.

| Estratégia | Candidato / gatilho | Dados | Cota / cooldown |
|---|---|---|---|
| **SCALP** (BTC/ETH/SOL, 15m) | Setup atual: `dist_vwap_pct <= -scalp_vwap_dev_pct` & `rsi <= scalp_rsi_oversold` & `volume_z >= scalp_volume_z` (BUY); simétrico (SELL). Valores de E6, inalterados. | 400 velas 15m **fechadas**, `sma_200` válida | `max_daily_scalps`; cooldown 4 barras (`SCALP_COOLDOWN_BARS`); 1 entrada/vela |
| **GOLDEN_POT v3** (candidatos do scanner volume/volatilidade, sem tokenizados) | **Momentum com volatilidade extraordinária** na última vela fechada (todas): ATR atual / mediana(ATR das 96 barras anteriores) `>= 1.8`; range da vela `>= 1.5×ATR`; `volume_zscore_20 >= 2.0`; fecha `>= 75%` (BUY) / `<= 25%` (SELL) da vela, vela na direção; `close` rompe a máxima (mín.) das 20 barras anteriores. Direção = a do momentum. Sweep/POI = apenas confluência no rationale. Gatilho de fade (VWAP+RSI) e sweep+POI independente **removidos**. Filtro de faca-caindo **não** é mais aplicado (era do fade). | 400 velas 15m fechadas, `sma_200` válida; liquidez 24h `< 50M` ⇒ 1x | 3/dia; cooldown 16 barras; 1/vela |
| **DAY TRADE / DIRECTIONAL** (1h) | Sinal do comitê (SignalGenerator, regra E3/E4 inalterada); SL = o do sinal | 400 velas 1h, `sma_200` válida | — |

Limiares nomeados em `scripts/run_paper_simulation.py` (`GP_*`, `SCALP_*`, `SWING_*`). Janelas de mediana/rompimento excluem a barra atual (sem lookahead).

## E7.2 Smart Trade Exit Ladder (comum a SCALP, GOLDEN_POT, DIRECTIONAL)
- **TP1** (40% da qtd. inicial, maker) ⇒ SL vai a **breakeven** (entrada ±0,04%).
- **TP2** (30%, maker) ⇒ **ativa o trailing stop** no runner.
- **Runner** (30%) sai em **TP3** (maker) ou no **trailing** (taker), o que vier antes.
- **Trailing:** distância = `trail_atr_mult × ATR_de_entrada` (congelado); o stop é atualizado **somente com extremos de barras já fechadas** (a barra corrente só alimenta o trailing quando chega a seguinte; ticks repetidos da mesma barra só acumulam), **nunca afrouxa**. Ordem intra-barra conservadora: stop vigente antes de qualquer TP; BE tocado na mesma barra do TP1 sai em BE. Stops não preenchem fora do range da barra.
- **Taxas:** TP1/TP2/TP3 maker (0,02%); SL/BREAKEVEN_STOP/TRAILING_STOP/MANUAL_CLOSE/LIQUIDATION taker (0,06% + slippage). Entrada: SCALP/GP maker; DIRECTIONAL taker (como antes). PnL e taxa por parcial no ledger (`TP1_PARTIAL`, `TP2_PARTIAL`, depois `TAKE_PROFIT_3`/`TRAILING_STOP`/…).
- **Perfis ATR** (`smart_trade.STRATEGY_PROFILES`, entrada = preço de execução):

| Perfil | SL | TP1 | TP2 | TP3 | Trailing |
|---|---|---|---|---|---|
| SCALP (alvos curtos) | 0,9 | 1,0 | 1,8 | 3,0 | 0,8 |
| DIRECTIONAL / day trade (médios) | 1,0 (usa o SL do comitê) | 1,5 | 2,5 | 4,0 | 1,2 |
| GOLDEN_POT (maiores) | 1,0 | 1,2 | 2,5 | 6,0 | 1,5 |

Posições abertas antes do E7 (sem `tp3_price`) mantêm o comportamento legado (migração `ADD COLUMN` com defaults legados).

## E7.3 Ressalvas honestas
- **Sem evidência de alfa.** Nada disto demonstra vantagem estatística; o prior (alfa nulo após custos), a hipótese nula do E4.3 e os critérios de refutação do E4.6 valem sem alteração. Os multiplicadores, frações e limiares do gatilho GP foram escolhidos por julgamento, **sem otimização em dados** — mas também sem validação fora da amostra; contam como tentativas no DSR.
- **Premissas do simulador:** execução a limite sempre preenchida ao tocar o nível (sem fila/rejeição), slippage fixo, ordem intra-barra assumida (stop primeiro), ATR de entrada congelado no trailing, gaps não modelados além do clamp ao range da barra, funding ignorado em direcionais. Escada com parciais aumenta a taxa de acerto aparente sem aumentar, por si só, a expectância.
- A porta do comitê reduz a amostra (SCALP e GP serão mais raros que em E5/E6); N mínimo de E4 tende a levar mais dias.

---

# ADDENDUM E8 — Gate de fluxo (Tape Reading) na entrada (2026-10-15)

## E8.1 Regra exata (pré-registrada, fixa)
`agents/tape_reading_agent.py::TapeReadingAgent` lê **somente leitura** (`sqlite mode=ro`) `datasets/microstructure.db` (coletor: `book_features`, `tick_agg`), usando apenas linhas com `ts <= now_ms` (sem lookahead):

| Feature | Definição |
|---|---|
| `taker_imb_5m` | `(Σbuy_notional − Σsell_notional)/(Σbuy_notional + Σsell_notional)` sobre os minutos de `tick_agg` com `now−5min < minute_ms <= now`; exige ≥ 3 minutos e ≥ 30 trades |
| `book_imb_10` | média dos 4 últimos snapshots de `(bid_n10 − ask_n10)/(bid_n10 + ask_n10)` |
| `microprice_dev` | média de `microprice_dev_bps` dos 4 últimos snapshots |
| `spread_bps` | do último snapshot |

Voto: **BUY** se `taker_imb_5m >= +0.15` **E** `book_imb_10 >= +0.10` **E** `microprice_dev >= 0`; **SELL** espelhado (`<= −0.15`, `<= −0.10`, `<= 0`); senão **HOLD**. `spread_bps > 5.0` ⇒ HOLD (`spread_wide`). **UNAVAILABLE** (status; fail-open): símbolo fora de BTC/ETH/SOL, DB ausente/ilegível, sem dados, dados insuficientes, book > 120 s ou tick (`minute_ms` mais recente) > 180 s. Constantes nomeadas no topo do módulo; não serão ajustadas.

**Gate** (`PaperTradingDaemon._flow_gate`, `scripts/run_paper_simulation.py`), aplicado a SCALP, GOLDEN_POT e DAY TRADE **depois** da porta do comitê (E7.1: mesma direção, sem veto, confiança ≥ 0,70): voto do tape **igual** à direção ⇒ `PASSED`; voto oposto ou HOLD ⇒ `BLOCKED` (nenhum trade); UNAVAILABLE ⇒ `UNAVAILABLE_PASSED` (entra, marcado). Símbolos do Golden Pot fora de BTC/ETH/SOL serão quase sempre UNAVAILABLE_PASSED (sem cobertura de fluxo).

**Auditoria:** tabela `flow_gate_log` (market_cache.db) registra TODO sinal que chega ao gate (decisão, voto/status/motivo do tape, features, entrada, SL, TP1–TP3 teóricos do perfil, ATR, confiança, consenso, alavancagem), permitindo reproduzir offline os bloqueados. Limpa por `scripts/reset_pilot.py`. Telemetria (último voto e contagens P/B/U) na linha `[STATUS DEMO]` do daemon.

## E8.2 Hipótese, métrica e refutação
- **H0 (nula):** o gate **não** melhora a expectância líquida (após taxas/slippage/funding) das entradas do comitê.
- **Métrica:** expectância líquida por trade, grupo **PASSED** (+ UNAVAILABLE_PASSED reportado à parte) vs **BLOCKED contrafactual**, este reproduzido no backtest `pilot_parity` a partir de `flow_gate_log` (entrada/SL/TP1–3/ATR registrados, mesma escada E7 e mesmos custos).
- **Amostra mínima:** ≥ 100 sinais por grupo (PASSED e BLOCKED) antes de qualquer conclusão; antes disso, nenhuma inferência.
- **Refutação de utilidade (H0 não rejeitada):** se, com N ≥ 100 por grupo, `E[PASSED] − E[BLOCKED]` tiver IC 95% (bootstrap por bloco/dia) incluindo 0 ou negativo, o gate é considerado sem valor e removido/desativado. Rejeição de H0 exige diferença positiva com IC95% > 0, contada como mais uma tentativa no DSR (E4).

## E8.3 Declarações honestas
- **Não há evidência prévia** de que o gate melhore resultados; limiares escolhidos por julgamento, sem otimização em dados.
- O gate **não é backtestável** de forma ordinária: não existe histórico de book/tape; a avaliação só é possível prospectivamente, via `flow_gate_log`.
- O gate reduz a amostra (E7 já a reduziu) e pode ser dominado por UNAVAILABLE_PASSED se o coletor estiver parado; a taxa de indisponibilidade deve ser reportada.
- **Piloto reiniciado com esta mudança.** E4–E7 permanecem inalterados, exceto que as entradas de SCALP/GP/DAY TRADE agora exigem também confirmação de fluxo (E8); a amostra anterior ao gate não é comparável.

---

# ADDENDUM E9 — Calibração Intraday (3m/5m) e Saneamento Estrutural Argon (2026-10-17)

## E9.1 Especificação técnica consolidada
1. **Timeframes e Perfis (`smart_trade.STRATEGY_PROFILES`):**
   - `SCALP` (BTC/ETH/SOL em `3m`): `sl_atr_mult = 0.9`, `tp1_atr_mult = 1.0`, `tp2_atr_mult = 1.8`, `tp3_atr_mult = 3.0`, `trail_atr_mult = 0.8`, `max_holding_bars = 30` (`90 min`).
   - `GOLDEN_POT` (Altcoins em `5m`): `sl_atr_mult = 1.5` (alinhado ao `trail_atr_mult = 1.5` para absorver o reteste da barra de expansão em 5m), `tp1_atr_mult = 1.2`, `tp2_atr_mult = 2.5`, `tp3_atr_mult = 6.0`, `max_holding_bars = 36` (`3h`).
   - `DIRECTIONAL` (BTC/ETH/SOL em `1h`): `sl_atr_mult = 1.0`, `tp1_atr_mult = 1.5`, `tp2_atr_mult = 2.5`, `tp3_atr_mult = 4.0`, `trail_atr_mult = 1.2`, `max_holding_bars = 48` (`48h`).
2. **Saneamento de Execução e Telemetria (6 Correções Argon):**
   - **Deduplicação do Gate de Fluxo (`_flow_gate`):** cada sinal de uma barra fechada `(strategy_type, symbol, bar_ts)` é avaliado e registrado em `flow_gate_log` uma única vez no fechamento da vela, impedindo entradas com preço defasado (*stale price*) minutos após um bloqueio inicial.
   - **Isolamento Pós-Entrada e Bucket Nativo do Trailing:** velas anteriores à abertura são estritamente ignoradas; posições `GOLDEN_POT` são monitoradas no grid de `5m` e `SCALP`/`DIRECTIONAL` no grid de `3m`, e o `_ratchet_trailing` só avança quando fecha uma janela completa de `prof.bar_ms`.
   - **Priorização de Momentum no Scanner GP:** `SWING_BREAKOUT` precede `SCALP_REVERSION` no `GoldenPotAgent`, e o daemon consome `top_swings + top_scalps`.
   - **Veto de Colisão por Símbolo:** proibida a abertura de mais de uma posição simultânea no mesmo ativo, mesmo entre estratégias distintas.
   - **Escalonamento de `ATR_PCT_EXTREME` por Timeframe (`leverage_policy.py`):** reducción de `-1x` na alavancagem aplicada em `>= 1.2%` (`3m`), `>= 1.5%` (`5m`), `>= 2.0%` (`15m`), `>= 3.0%` (`1h`) e `>= 5.0%` (`4h`).
   - **Consolidação por `position_id` (`PerformanceTracker` e `LearningTracker`):** métricas de assertividade, `consecutive_losses` (que acionam o `RollbackGuard`) e feedback ao `LearningTracker` utilizam o PnL líquido consolidado da posição (`pos.realized_pnl`), não tratando saídas parciais no `BREAKEVEN_STOP` (`-$0.05`) ou liquidações de `FUNDING` como derrotas independentes.

---

# ADDENDUM E10 — Consolidação S2/S3 → S4 e Alinhamento de Invariantes (`v1.0.6`, 2026-11-08)

## E10.1 Parâmetros Vivos de `STRATEGY_PROFILES` (Resolução C1, C2, C3, C9/I9, C12)
Este addendum reconcilia a documentação com os parâmetros vivos em `smart_trade.py` validados a partir da calibração da S2 e prevalece sobre E7/E9 onde houver divergência:

| Perfil | Timeframe (`bar_ms`) | `sl_atr_mult` | `tp1_atr_mult` (40%, BE) | `tp2_atr_mult` (30%, ativa Trail) | `tp3_atr_mult` (30%, Runner) | `trail_atr_mult` | `max_holding_bars` |
|---|---|---|---|---|---|---|---|
| **SCALP** (BTC/ETH/SOL) | `15m` (`900_000 ms`) | **0.9** | **1.2** (`1.33R`) | **2.0** (`2.22R`) | **3.6** (`4.00R`) | **0.8** | `30` (`7h30`) |
| **DIRECTIONAL** (Majors) | `1h` (`3_600_000 ms`) | **1.0** | **1.5** (`1.50R`) | **2.5** (`2.50R`) | **4.0** (`4.00R`) | **1.2** | `48` (`48h`) |
| **GOLDEN_POT** (Altcoins) | `5m` (`300_000 ms`) | **1.5** | **1.5** (`1.00R`) | **2.5** (`1.67R`) | **6.0** (`4.00R`) | **1.5** | `36` (`3h`) |

- **Justificativa da migração SCALP `3m -> 15m` (C1/C2):** Redução do atrito de taxas/slippage de `0.79R` em `3m` para `0.16R` em `15m`, com alvos escalonados `1.2 / 2.0 / 3.6 ATR`.
- **Justificativa do `tp1_atr_mult = 1.5` no GOLDEN_POT (C3/C12):** Com `sl_atr_mult = 1.5`, o TP1 em `1.5x ATR` garante realização parcial de `1.0R` exato na ida ao breakeven.
- **Invariante I9:** Coberto por teste de regressão automatizado (`tests/test_strategy_profiles_invariants.py`).

## E10.2 Patches de Execução S3 (`v1.0.5` e `v1.0.6`) (Resolução I1)
1. **Proteção de Pavio Pré-Rompimento no Breakeven na Mesma Barra (`406c1fd`, `v1.0.5`):** Na barra em que `TP1` é atingido e arma o `BREAKEVEN_STOP`, o pavio adverso da própria barra de expansão (ocorrido antes do rompimento a favor) só aciona o breakeven na mesma barra se o fechamento (`close`) reverter contra o `breakeven_price`, eliminando falsos stops retroativos de intra-barra.
2. **Ajuste de Tolerância de Barra Recente (`d75fff8`):** Alinhamento de janela de frescor para barras fechadas no loop de avaliação.
3. **Correção do Falso Veto Estrutural em Rompimentos (`v1.0.6`):** Quando o preço opera em rompimento de topo/fundo sem resistência acima (`nearest_bsl` ausente -> `dist_resistance_pct = NaN`) ou sem suporte abaixo (`nearest_ssl` ausente -> `dist_support_pct = NaN`), a limpeza `clean_features(drop_na=False)` converte `NaN` para `0.0`. O `RiskSentinelAgent` e o `TechnicalAgent` passam a exigir distância estritamente positiva (`0.0 < dist_pct < threshold` e nível `nearest_bsl/ssl > 0`) para acionar `RESISTANCE_PROXIMITY` / `SUPPORT_PROXIMITY`, destravando entradas legítimas de rompimento na S4.

## E10.3 Derivação Dinâmica de `PRIOR_TRIALS` e Registro no `trial_log.csv` (Resolução C4, C5, I2)
- `statistical_rigor.get_prior_trials_count()` lê diretamente o número de tentativas acumuladas em `datasets/reports/trial_log.csv`.
- `scripts/strategy_search.py` e `scripts/online_ml_search.py` derivam `PRIOR_TRIALS` dinamicamente de `get_prior_trials_count()`, eliminando constantes literais divergentes.
- As calibrações de S1 (`TRIAL_0084`), S2 (`TRIAL_0085`) e S3→S4 (`TRIAL_0086`) estão registradas em `datasets/reports/trial_log.csv`.

## E10.4 Critério Operacional de Parada / Alerta por Trajetória de Versão (Resolução C6 / D28 / I11)
O critério oficial de avaliação até dezembro/2026 é medido **por versão carimbada** (`strategy_version`):
- **Estagnação:** `E[R]` líquido por trade com IC95 incluindo `0` **e** delta entre versões consecutivas também incluindo `0`.
- **Retração (Vermelho):** `E[R]` líquido negativo com IC95 excluindo `0`, **ou** drawdown acumulado total `>= 10.0%` (`RollbackGuard.max_total_drawdown_pct = 10.0`, com trava diária operacional `max_daily_drawdown_pct = 5.0`).

## E10.5 Patch S4 Abertura (`v1.0.6-s4`, 2026-11-09) — Destravamento de Calibração e Regime Dinâmico
1. **Resolução Dinâmica de Regime no Orquestrador (`core._resolve_regime`):**
   - Evita o fallback estático cego para `SIDEWAYS_RANGE` quando o contexto não traz a chave `regime`.
   - Avalia a distância estrutural do preço em relação à SMA 200 (`|dist_sma_200| >= 0.8%`), classificando ativos em expansão tendencial como `BULL_TREND` ou `BEAR_TREND`.
   - Consulta o registro mais recente em `market_regime` antes de aplicar qualquer fallback.
2. **Janela Deslizante no `LearningTracker` Pós-S2/S3:**
   - O cálculo de acurácia, pesos e ECE em `recalibrate_agent_weights` foca na janela pós-virada da S3 (`timestamp >= 1793577600000`), expurgando ruídos legados da S1/S2 e calculando o gap de calibração adaptado para amostragens em formação.
3. **Dampening Proporcional no Juiz de Consenso (`consensus_judge`):**
   - A trava defensiva de confiança máxima em `0.55` no regime lateral passa a agir estritamente sobre agentes com sobreconfiança improdutiva (`ECE > 0.15` **e** `Acurácia < 55%`) ou baixa assertividade (`Acurácia < 45%`).
   - Agentes com assertividade empírica validada na S3 (`Acurácia >= 55%`, como os `64.3%` alcançados na S3) mantêm suas convicções nominais, permitindo que sinais legítimos superem os limiares operacionais (`0.65` Scalp, `0.70` Golden Pot, `0.75` Day Trade).

## E10.6 Patch S4 Contínuo (`v1.0.6-s4.1`, 2026-11-09) — Blindagem Contra Compras de Exaustão no Topo
1. **Teto de Exaustão por RSI no `TechnicalAgent`:**
   - Adicionado teto de sobrecompra (`RSI <= 75.0`) para emissão de `BUY` em tendência de alta. Leituras de `RSI > 75.0` são forçadas a `HOLD (0.55)`, impedindo que o momentum tabular confunda topo esticado com aceleração compradora.
   - Adicionado piso de sobrevenda simétrico (`RSI >= 25.0`) para emissão de `SELL` em tendência de baixa, evitando vendas no fundo esticado (`RSI < 25.0`).
2. **Aperto e Janela Ampliada no Filtro de Clímax do Golden Pot (`run_paper_simulation.py`):**
   - Redução de `GP_CLIMAX_RSI_OVERBOUGHT` de `85.0` para **`75.0`** e elevação de `GP_CLIMAX_RSI_OVERSOLD` de `15.0` para **`25.0`**.
   - Redução de `GP_CLIMAX_MAX_ATR_MOVE` de `2.5x` para **`2.0x ATR`**, avaliando simultaneamente a variação acumulada de 3 e 4 barras (capturando pumps e dumps parabólicos que se estendem por 15-20 minutos).

## E10.7 Validação de Estratégias de Abertura & Reversão de Fluxo — Reauditado em Pilot-Parity & OOS Estrito (2026-11-13, S4)
Reauditoria completa do estudo de 10/11/2026 (baseado no dump das 21:00 BRT em 09/11/2026 e no perfil horário completo de `174.341` velas 1h limpas pós-`sma_200` em `BTCUSDT`/`ETHUSDT`/`SOLUSDT`):

1. **Invalidação Empírica dos Números Anteriores (10/11/2026):**
   - **`68,1%` WR (bug inicial):** Causado por duplicação intra-barra de timestamps (`binance` + `bitget` coexistindo em ~`12.107–12.280` das `25.000` velas no SQLite), bug de escala `atr_pct` e vazamento in-sample de modelos `.joblib`.
   - **`51,3%` (Opção A) e `47,5%` (Opção B) / `+13,0%` em BTC:** Invalidados por **vazamento in-sample** (uso dos `.joblib` finais de produção sobre todo o histórico de BTC), simulador *toy* simplificado (sem escada Smart Trade 40/30/30, sem BE `+0,04%`, sem trailing pós-TP2 e sem funding real 8h) e mistura de modelo `1h` no timeframe `15m`.
   - **Perfil horário `174.341` velas 1h limpas (`21:00 BRT`):** Range/ATR médio `1,093`, Vol Ratio `1,013` e taxa de reversão 4h em movimentos >=0,3% de **`53,09%`** (inferior às `20:00 BRT` com `54,29%` e menor que a expansão das `11:00 BRT` com Range/ATR `1,352` e Vol `1,810`).

2. **Resultados Reais Estritamente OOS em `BacktestEngine(..., pilot_parity=True)` (`TRIAL_0087..TRIAL_0097`):**
   - **1h OOS (`2025-02-26 23:00 UTC .. 2026-11-13 19:00 UTC`, ML `PurgedTimeSeriesSplit` 4-folds em 1h):**
     - **Baseline Momentum (1h):** Portfólio `2.995 trades`, WR `40,5%`, PF `0,56`, E[R] `-0,342R`, PnL `-$838,15`, MaxDD `8,48%` (`BTC 39,5%/0,44/-$282` | `ETH 42,6%/0,59/-$263` | `SOL 39,3%/0,61/-$294`).
     - **Opção A: Flow Reversal (Pura / ML OOS):** Pura `1.979 tr`, WR `39,6%`, PF `0,65`, E[R] `-0,278R`, PnL `-$483,38`; com filtro ML OOS cai para `711 tr`, WR `39,2%`, PF `0,69`, E[R] `-0,302R`, PnL `-$160,11` (`BTC 36,0%/0,46/-$64` | `ETH 37,3%/0,69/-$58` | `SOL 43,2%/0,82/-$39`).
     - **Opção B: Session Sweep 21/05/11h (Pura / ML OOS):** Pura `553 tr`, WR `38,9%`, PF `0,61`, PnL `-$147,29`; com ML OOS `194 tr`, WR `36,1%`, PF `0,49`, E[R] `-0,417R`, PnL `-$68,31`.
     - **Opção B-21h: Daily Open 21h BRT (Pura / ML OOS):** Pura `143 tr`, WR `38,5%`, PF `0,74`, E[R] `-0,248R`, PnL `-$26,07` (`ETH 47,3%/0,95/-$2`); com ML OOS `33 tr`, WR `39,4%`, PF `0,51`, E[R] `-0,321R`, PnL `-$10,50`.
   - **15m OOS (`2026-06-05 08:00 UTC .. 2026-11-13 21:15 UTC`, ML `PurgedTimeSeriesSplit` treinado nativamente em 15m):**
     - **Opção C Scalp VWAP (Baseline 2.0%):** `129 tr`, WR `44,2%`, PF `0,67`, E[R] `-0,245R`, PnL `-$17,38`.
     - **Opção C Scalp VWAP (Live DB 0.8%):** `781 tr`, WR `40,3%`, PF `0,55`, E[R] `-0,371R`, PnL `-$112,48`.
     - **Opção C Scalp VWAP (Calibrado 0.6% / + ML OOS 15m):** Pura `1.080 tr`, WR `40,7%`, PF `0,50`, PnL `-$157,64`; com ML OOS 15m `392 tr`, WR `42,6%`, PF `0,55`, E[R] `-0,365R`, PnL `-$49,65`.
   - **Conclusão E10.7:** Nenhuma variante autônoma de abertura/reversão em 1h ou 15m sem microestrutura de book/CVD obteve `E[R] > 0` ou `PF > 1,0` líquido de custos reais (`Status: REJECTED` para ativação autônoma).

## E10.8 Saneamento Estrutural do Banco, Reparo de Telemetria/Calibração e Benchmark S4.2 (2026-11-13, S4)
Auditoria cirúrgica executada em `2026-11-13` (preservando daemons live e `2` posições abertas) com artefatos completos em [`docs/S4_AUDITORIA_EMPIRICA_E_CALIBRACAO.md`](S4_AUDITORIA_EMPIRICA_E_CALIBRACAO.md) e [`datasets/reports/s4_empirical_audit_20261113.json`](../datasets/reports/s4_empirical_audit_20261113.json):

1. **Saneamento de Dados & Isolamento de Testes:**
   - Removidas `47` linhas `'Reset de teste'` de `circuit_breaker_events` (backup em `datasets/backups/cb_test_pollution_backup_20261113.json`) e isolado `tests/test_performance_and_rollback.py`.
   - Deduplicadas **`63.008`** velas sobrepostas não-`binance` em `candles` (backup `datasets/backups/candles_overlap_backup_20261113.sqlite`, `0` duplicatas restantes; `174.344` velas 1h limpas em majors) + loaders endurecidos com prioridade `binance`.
   - **Impacto ML Walk-Forward Antes vs Depois (`25.000` linhas 1h, `14.976` amostras OOS/ativo):** A contaminação inflava artificialmente a acurácia de sinais emitidos (`BTC RF 68,99% -> 43,01%`, `LGBM 61,80% -> 40,06%`; `ETH RF 70,94% -> 41,64%`, `LGBM 58,29% -> 38,24%`; `SOL RF 74,11% -> 36,27%`, `LGBM 65,14% -> 42,07%`) e a acurácia global (`~45–50% -> ~39–41%`), alinhando-se aos `.joblib` auditados (`lgbm_model_v1.joblib`: `41,73%` global / `42,71%` em `487` sinais; `rf_model_v1.joblib`: `41,82%` global / `47,37%` em `57` sinais).

2. **Correções de Telemetria, Regime e Fluxo ao Vivo (Aplicadas e Verificadas):**
   - `core.py` propaga `regime` para `TradeSignal.metrics`; `agents/regime_classifier.py` usa janela `LIMIT 250` e detecta tendência consolidada (`BEAR_TREND` confirmado no vivo); `learning_tracker.py` persiste `brier_score` real; `TapeReadingAgent` diferencia replay (`allow_live_fallback=False`) de `_flow_gate` live (`allow_live_fallback=True` em `run_paper_simulation.py`).
   - Backfill de `40` registros pós-S3 em `agent_learning_memory` e recálculo via `LearningTracker.recalibrate_agent_weights()` corrigiu pesos degenerados (`BEAR_TREND`: `Technical 1,314`, `OrderFlow 0,486` com `68,2%` acc em `22` amostras e `Brier 0,217/0,241`; `BULL_TREND`: `0,920 / 0,880` em `6` amostras).

3. **Benchmark S4.2 Antes vs Depois & Gate GO / NO-GO (Thresholds Candidatos Pendentes de Aprovação — Não Aplicados em Produção):**
   - **Holdout OOS `DIRECTIONAL` 1h (`2024-02-24 .. 2026-11-13`, `TRIAL_0098..TRIAL_0102`):** `DIR_BASE` (`6.883 tr`, `39,4%`, PF `0,629`, `-$5.329,03`, DD `53,68%`, SR `-7,118`) vs `DIR_CAL1_P2` (`6.869 tr`, `-0,2930R`, neutro/descartado como filtro de alfa) vs `DIR_CAL2_CORR_CAP` (`max_same_dir_majors=2`: `5.924 tr`, `-$4.492,47`, DD `-8,57pp` para `45,11%`) vs `DIR_CAL3_REGIME_WEIGHTS` (`5.956 tr`, WR `39,7%`, PF `0,647`, E[R] `-0,2775R`, SR `-6,473`; mantém ganho na subjanela pré-S3 `2024-02-24..2026-10-30` com PF `0,628 -> 0,647` e DD `53,36% -> 44,74%`) vs `DIR_S4_2_COMBINED` (`5.144 tr`, WR `39,6%`, PF `0,649`, E[R] `-0,2815R`, `-$3.722,59`, DD `37,60%`, DSR `0,0000`).
   - **Holdout OOS `SCALP` 15m (`2026-02-20 .. 2026-11-13`, `TRIAL_0103..TRIAL_0105`):** `SCALP_BASE_DB` (`775 tr`, WR `39,4%`, PF `0,365`, `-$427,02`) vs `SCALP_CAL_TRIGGER` e `SCALP_CAL_ALIGNED` (`721 tr`, WR `39,8%`, PF `0,349`, `-$405,30`; aperto de VWAP/VolZ cortou fallback de `2.688` para `146`, e guarda contra-regime bloqueou `352` candidatos).
   - **Replay 29d Real em `microstructure.db` (`2026-10-15 .. 2026-11-13`, `122` símbolos):** `TapeReadingAgent` no `DIRECTIONAL` reduziu perdas em `52%` no Base (`194 -> 105 tr`, WR `41,2% -> 45,7%`, `-$103,93 -> -$50,11`) e `65%` no S4.2 combinado (`139 -> 87 tr`, WR `39,6% -> 46,0%`, PF `0,542 -> 0,711`, `-$100,14 -> -$35,09`); no `SCALP` 15m continuou negativo em todas variantes (`PF 0,167..0,259`).
   - **Contrafactual S4 Ao Vivo (`N=24` fechados desde `2026-11-09 00:00 UTC`):** Realizado antes `+$991,08` (`15/24`, WR `62,5%`) -> com filtros candidatos **`+$1.108,01`** (`11/16`, WR `68,8%`): `DIRECTIONAL` Cap 2 (`+$1.064,93 -> +$1.095,00`), `GOLDEN_POT` macro BTC `dist_sma_200 >= -1,5%` (`-$23,27` em `11 tr` -> **`+$23,05`** em `6 tr`), `SCALP` bloqueio contra-regime (`-$50,58 -> -$10,04`, removendo `ETHUSDT LONG -$40,54` a `-11,75%` da SMA200).
   - **Decisão Recomendada S4.2:** **GO** como mitigação de risco/drawdown para `DIR_CAL2_CORR_CAP` (`max_same_dir_majors=2`), **GO** para filtro macro BTC em `GOLDEN_POT` e **GO** (como trava/rebaixamento defensivo ou pausa) para `SCALP_CAL_ALIGNED`; **NO-GO** para promover `DIR_CAL1_P2` isolado ou tratar qualquer modelo autônomo sem gate de fluxo como gerador líquido positivo no OOS longo (`E[R] < 0`).

---

## E10.9 Backlog do Patch de Domingo (Itens 1 & 2) e Pesquisa Estrutural de Viabilidade do `SCALP` (2026-11-13, S4)

1. **Backlog Congelado para o Patch S4.2 de Domingo ([PATCH_S4_DOMINGO_BACKLOG.md](file:///home/eudespatricio/ia_trader/docs/PATCH_S4_DOMINGO_BACKLOG.md) — sem alterações em produção nesta sessão):**
   - **Item 1 (`GOLDEN_POT`, GO):** Alinhamento macro via `BTCUSDT 1h dist_sma_200` (`±1,5%`), bloqueando `LONG` quando BTC `< -1,5%` da SMA200 (`N=11` S4: `-$23,27` -> `+$23,05` em `6` trades `SHORT`).
   - **Item 2 (`DIRECTIONAL`, GO):** `max_same_dir_majors=2` (DD OOS `53,68% -> 45,11%`, combinado `37,60%`; ao vivo S4 `+$1.064,93 -> +$1.095,00`) + `_flow_gate` estrito exigindo `status == "OK"` em `BTCUSDT`, `ETHUSDT` e `SOLUSDT` (eliminando `unavailable_passed=63` no replay 29d em `microstructure.db`, onde o gate já havia cortado perdas em `65%`: `-$100,14 -> -$35,09`).

2. **Pesquisa Estrutural & Microestrutura do `SCALP` 15m (`E10.9_SCALP_RESEARCH`, `TRIAL_0106..TRIAL_0130`, `BTCUSDT/ETHUSDT/SOLUSDT`):**
   - **Hipótese pré-registrada:** Testar se a substituição de entrada Taker + escada `40/30/30` com `BE +4bps` (que paga `54%–67%` de `1R` em RT taker e limita ganhos de `TP1` em `+0,27R` líquido contra `-1,34R` no stop em `79,8%` de retornos) por `Limit Pullback` maker (`2 bps`, `0 bp` slip, guarda anti-lookahead na barra de fill), alvos únicos/BE tardio, gatilhos de volatilidade/sessão (`Horizonte A`, 9-13m), séries Binance 5m `oi`/`taker`/`ls_*`/`funding` (`Horizonte B`, 60d) e leitura L2 `book_features`/`tick_agg` (`Horizonte C`, 29d) viabilizaria `E[R] > 0` e `DSR >= 0,95`.
   - **Resultados verificados ([ESTUDO_SCALP_S4_ALTERNATIVAS.md](file:///home/eudespatricio/ia_trader/docs/ESTUDO_SCALP_S4_ALTERNATIVAS.md), [scalp_research_s4_20261113.json](file:///home/eudespatricio/ia_trader/datasets/reports/scalp_research_s4_20261113.json)):**
     - **Horizonte A (`TRIAL_0106..TRIAL_0117`):** `A0_LIVE_SIM_PARITY` (`1.027 tr`, PF `0,349`, `E[R] -0,5396`, `-$598,88`) e `A0_LIVE_REAL_TAKER` (`1.027 tr`, PF `0,271`, `E[R] -0,6707`, `-$733,46`). Melhorias estruturais reduziram a perda esperada por trade em até **6x**: `A4_BO20_DELAYED_BE_PULLBACK` (`535 tr`, PF `0,838`, `E[R] -0,1120`, `-$149,91`, DD `1,64%`) e `A8_RANGE_VWAP_REV_PULLBACK` (`69 tr`, WR `47,8%`, PF `0,877`, `E[R] -0,1155`, `-$11,35`), porém todas permaneceram com `E[R] < 0` (`DSR 0,0000`).
     - **Horizonte B (`TRIAL_0118..TRIAL_0124`, 60d Binance 5m):** `B0_BASE_60D_TAKER` (`152 tr`, PF `0,180`, `E[R] -0,7794`, `-$111,79`). `B4_FUNDING_WINDOW_FADE` atingiu PF `1,218` e `+$7,14` (`N=31 tr`, WR `48,4%`), mas com `E[R] = -0,0573` (IC 95% `[-0,5112, +0,3415]`), `DSR = 0,0000` e `E[R]` negativo em `H1 (-0,0084)` e `H2 (-0,1095)`.
     - **Horizonte C (`TRIAL_0125..TRIAL_0130`, 29d L2 real):** `C0_BASE_29D_WITH_TAPE_TAKER` (`28 tr`, PF `0,106`, `E[R] -0,9843`, `-$23,12`). Melhor variante foi `C2b_L2_TOXIC_OBI_PULLBACK` (`15 tr`, WR `40,0%`, PF `0,951`, `E[R] -0,0995`, `-$0,86`, `DSR 0,0004`, instável entre H1 PF `5,713` e H2 PF `0,279`).
   - **Veredito GO / NO-GO (`E10.9`):** **NO-GO** para promoção lucrativa de `SCALP` 15m (`0/25` variantes com `E[R] > 0` e `DSR >= 0,95`). **GO** para **pausar (`enabled=False`) ou rebaixar para modo `shadow` (risco zero)** no Patch de Domingo, mantendo o esqueleto `Limit Pullback` + saída sem armadilha `TP1 BE` apenas para observação passiva enquanto se coleta base L2 mais longa ou se pesquisa `30m/1h`.

---

## E10.10 Impacto do Gate vs Ponderação do `TapeReadingAgent` nas 3 Estratégias (2026-11-13, S4)

1. **Hipótese pré-registrada:** Medir nos dados reais de `flow_gate_log` (`N=804`) e nos 29 dias L2 reais (`microstructure.db`, `2026-10-15 12:30 UTC .. 2026-11-13`) o ganho incremental (`ΔWR`, `PF`, `E[R]`, `PnL`) do `TapeReadingAgent` nas 3 estratégias (`DIRECTIONAL 1h`, `GOLDEN_POT 5m`, `SCALP 15m` — incluindo setups de maior WR do backtest `B4` e `A8`), comparando Sem Tape (`M0`), Gate Fail-Open Live (`M1`), Gate Estrito `OK` (`M2`), Ponderação Contínua `0.65*comitê + 0.35*tape` + sizing (`M3`) e Estrito+Peso (`M4`) ([NOTAS_IMPACTO_TAPE_READING_3_ESTRATEGIAS.md](file:///home/eudespatricio/ia_trader/docs/NOTAS_IMPACTO_TAPE_READING_3_ESTRATEGIAS.md), `TRIAL_0131..TRIAL_0165`).
2. **Principais achados verificados (`E10.10_TAPE_WEIGHT_3STRAT`):**
   - **Arquitetura atual:** [core.py](file:///home/eudespatricio/ia_trader/core.py#L62-L69) e [consensus_judge.py](file:///home/eudespatricio/ia_trader/consensus_judge.py#L57-L64) ponderam apenas `OrderFlowAgent` (`0.8`); o `TapeReadingAgent` atua pós-comitê via `_flow_gate` em [run_paper_simulation.py](file:///home/eudespatricio/ia_trader/scripts/run_paper_simulation.py#L531-L581) com brecha `fail-open` (`UNAVAILABLE_PASSED`). Em `flow_gate_log`, `562` sinais bloqueados de `DIRECTIONAL` acumulavam `PF 0,474 / -$479,47` e `103` de `SCALP` acumulavam `PF 0,080 / -$158,82`, enquanto `27/31` entradas do `GOLDEN_POT` passaram com `status == "UNAVAILABLE"`.
   - **`DIRECTIONAL` (`TRIAL_0131..TRIAL_0135`):** `M1` elevou WR em **`+5,64 pp`** (`38,4% -> 44,0%`, PF `0,530 -> 0,661`, `-$103,38 -> -$41,17`); `M2` reduziu para `55 tr` (`-$43,32`); `M3` ponderado piorou para `-$65,93` (`PF 0,504`) por reintroduzir sinais `HOLD`.
   - **`GOLDEN_POT` com macro BTC (`TRIAL_0136..TRIAL_0140`):** `M2` estrito estrangula o universo 5m para `2 tr` em 29d (`-$0,10`); `M4` (estrito quando `OK`, `0.50x` quando `symbol_not_covered`) cortou o prejuízo de `-$155,12` (`M0`) para `-$7,62` (`PF 0,384 -> 0,750`, `H2 PF 1,01`).
   - **`SCALP` campeões de WR (`TRIAL_0141..TRIAL_0165`):** Em `B4_FUNDING_WINDOW_FADE`, `M2`/`M4` atingiram `CANDIDATE` exploratório com `N=6 tr` (`WR 50,0%`, `+3,33 pp`, PF `1,160 / 1,182`, `E[R] +0,0060`) e `M3` reteve `12 tr` (`WR 50,0%`, PF `1,462`, `+$4,43`, `E[R] -0,0188`), ainda sem significância estatística para promoção live. Em `A8` (`8->2 tr`, `-$4,23`) e `A9b` (`104->25 tr`, WR `-5,65 pp`), gate direcional padrão penalizou setups contrarian/sweep.
3. **Veredito:** Manter `_flow_gate` binário fail-closed nos majors (`DIRECTIONAL`), política híbrida com sizing defensivo em altcoins do `GOLDEN_POT`, e `SCALP` pausado/`shadow` no domingo.

---

## E10.11 Homologação e Aplicação do Patch S4.2 (v1.0.7, 2026-11-15)

1. **Contexto & Fechamento Consolidado da S4 (2026-11-09 a 2026-11-15):**
   - **Janela isolada S4:** `33` posições fechadas (`+2` posições abertas `SHORT` em `ETHUSDT` e `BTCUSDT`).
   - **Equity & Risco:** `$9.892,04 -> $11.604,55`, resultado líquido `true_net` **`+$1.649,79`** (`WR 63,6%`, `TP2+ 45,5%`, `PF 3,21`, `Payoff 1,83`, `Sharpe 5,28`, `MaxDD 5,11%`).
   - **Performance por Estratégia:**
     - `DIRECTIONAL` 1h: `18` trades (`18 SHORT`), `WR 72,2%` (`TP2+ 66,7%`), `PF 4,12`, líquido **`+$1.750,94`**.
     - `GOLDEN_POT` 5m: `13` trades (`8 SHORT / 5 LONG`), `WR 61,5%` (`TP2+ 23,1%`), `PF 0,64`, líquido **`-$48,99`** (`LONG -$46,87`, `SHORT -$2,12`).
     - `SCALP` 15m: `2` trades, `WR 0,0%`, líquido **`-$52,17`**.
   - **Reavaliação Pós-Sábado da Convergência `ML 50k` no `DIRECTIONAL`:** Teste refutou a hipótese de filtro rígido (*hard filter*): fez `5/7` wins (`71,4%`, `+$549,68`) com ML50k vs `8/11` (`72,7%`, `+$1.201,26`) sem; ambos os stops de domingo continham `[ML 50k]`. No `GOLDEN_POT`, manteve `7/8` (`87,5%`, `+$41,40`) com `ML 50k`, sendo mantida como modulação/telemetria sem bloqueio rígido para preservar amostragem estatística.

2. **Descoberta do Estudo Cruzado `E10.9 x ML 15m OOS` (`PurgedTimeSeriesSplit`, Maio a Nov/2026):**
   - Treinamento dedicado de ensemble 15m (`models/lgbm_15m_v1.joblib` + `models/rf_15m_v1.joblib`) cruzado com os gatilhos estruturais da suíte `E10.9`:
   - `A6_SQUEEZE_KELTNER_PULLBACK + ML15m >= 0.45`: `N=37`, `WR 54,0%`, `PF 1,447`, $\mathbb{E}[R] = +0,293R$, positivo em ambas metades temporais (`H1 50,0% / PF 1,39` e `H2 56,0% / PF 1,48`). Por ativo: `SOL 60,0%` (`PF 1,75`), `ETH 52,9%` (`PF 1,38`), `BTC 40,0%` (`PF 0,72`).
   - `A5_SQUEEZE_KELTNER_TAKER + ML15m >= 0.45`: `N=38`, `WR 52,6%`, `PF 1,217`, $\mathbb{E}[R] = +0,189R$.
   - **Decisão Arquitetural:** Em vez de manter o Scalp desativado ou isolar pares em quarentena, adotou-se a **Opção 3B Universal** em todos os 3 majors (`BTCUSDT`, `ETHUSDT`, `SOLUSDT`) com risco defensivo de `0.50x`, gerando aprendizado estatístico contínuo até dezembro.

3. **As 5 Frentes Implementadas e Certificadas no Patch S4.2 (`v1.0.7`):**
   - **Frente 0 (Base S4.1 + Paridade Histórica):** Ativação definitiva via hot-restart da persistência de `brier_score` no `LearningTracker`, janela `LIMIT 250` no `RegimeClassifier`, deduplicação canônica `binance` em `candles`, aba de telemetria e evolução no dashboard web, e porte da proteção intra-barra `closed_safe` (`v1.0.5`) para o motor `backtesting.py`.
   - **Frente 1 (`DIRECTIONAL` 1h):** Trava de correlação `MAX_SAME_DIR_MAJORS = 2` contra squeezes simultâneos + `_flow_gate` estrito *fail-closed* exigindo `status == "OK"` nos majors.
   - **Frente 2 (`SCALP` 15m Universal — Opção 3B):** Ensemble nativo 15m (`lgbm_15m_v1` + `rf_15m_v1`) + gatilho `Squeeze BB/Keltner` + filtro de convicção `ML 15m >= 0.45` + `Tape status == "OK"` + saída limpa com alvo único (`SL 1.4x ATR`, `TP 2.2x ATR`, `tp1_fraction=0.0`, `tp2_fraction=0.0`, eliminando a armadilha do `TP1 BE +4bps`) com dimensionamento defensivo a `0.50x` em todos os 3 majors (`BTC`, `ETH`, `SOL`).
   - **Frente 3 (`GOLDEN_POT` 5m):** Gate macro via `BTCUSDT 1h dist_sma_200` (`±1,5%`) bloqueando contra-tendência + política L2 híbrida `M4` (`1.00x` quando `status == "OK"` e `0.50x` defensivo em `UNAVAILABLE`), mantendo todas as altcoins ativas para telemetria.
   - **Frente 4 (Governança, Continuidade & Rollback):** Registro formal da hipótese `E10.11`, certificação da suíte de testes unitários/regressão, commit pontual de rollback e hot-restart do supervisor preservando posições abertas.

---

# Experimento E_SWING_4H — Motor Paralelo `SWING_4H` Donchian40 / ExitCh10 (Wave P3, `v1.0.10`, 2026-11-21)

**Status:** Pré-registrado em 2026-11-21 antes do início da coleta limpa em papel (`strategy_type="SWING_4H"`). Implementado como motor paralelo independente de `SCALP`, `GOLDEN_POT` e `DIRECTIONAL`, sem uso do comitê ML/LLM.

## 1. Hipótese & Arquitetura Congelada (`data/swing_4h_strategy.py`, `smart_trade.py`, `scripts/run_paper_simulation.py`)
- **Universo & Dados:** `BTCUSDT`, `ETHUSDT`, `SOLUSDT`; barras `4h` (`14.400.000 ms`) reamostradas estritamente a partir de velas `1h` fechadas (`cnt == 4` e `timestamp + 14_400_000 <= now_ms`), mantendo prioridade `binance` (fallback `bitget` apenas se vazio ou menor). Warmup mínimo: `205` barras `4h` fechadas.
- **Indicadores:** Wilder `ATR(14)` (`ewm(alpha=1/14, adjust=False)`), `SMA(200)`, canais Donchian causais com `shift(1)`: entrada `h40`/`l40` (40 barras), saída `h10`/`l10` (10 barras), `dist_sma_200 = (close - sma_200) / sma_200`.
- **Gatilho de Entrada:**
  - `BUY`: `close > h40` **e** `0.0 <= dist_sma_200 <= 0.20` (`|dist_sma_200| <= 20%`, filtro anti-exaustão).
  - `SELL`: `close < l40` **e** `-0.20 <= dist_sma_200 <= 0.0`.
- **Execução, Sizing & Alavancagem:** Entrada taker a mercado (`0.06%` taxa + `0.02%` slippage) no `open(t+1)` (`entry_timestamp = ts(t) + 14_400_000`). Lote = `1%` de risco da banca (`risk_per_trade_pct`, cap `0.25` do saldo), alavancagem decidida por `decide_leverage(..., timeframe="4h", max_leverage=2.0)` (teto duro `2.0x`), 1 posição aberta por símbolo, cooldown `2` barras `4h` após fechamento, 1 entrada por barra `(symbol, bar_id)`.
- **Saídas (Trailing Puro + Canal 10 + Timeout):**
  - Sem parciais (`tp1_fraction = 0.0`, `tp2_fraction = 0.0`, `tp1/tp2/tp3 = 0.0`).
  - Stop inicial `2.5 × ATR(14)`; `trailing_active = True` desde a abertura com `trailing_atr_mult = 3.5` (atualizado apenas em barras `4h` fechadas; nunca afrouxa).
  - Saída por canal oposto (`CHANNEL_EXIT_10`, taker): `close < l10` para `LONG` ou `close > h10` para `SHORT` avaliada em barras `4h` fechadas após `entry_timestamp` (nunca bloqueada por circuit breaker/cota/margem).
  - Timeout máximo: `120` barras `4h` (`20` dias, `HORIZON_TIMEOUT`).
  - Funding real 8h (`funding_rates`, fallback `0.01%/8h`).

## 2. Referência Offline & Trials Registrados (`scripts/eval_swing_4h_reference.py`, `docs/pilot_backtest_reference.json`)
- **Custos de referência:** modelo estresse (`0.06%` taker + `0.02%` slippage por lado) + funding 8h histórico real (`market_cache.db` read-only).
- **IS (`2020-01-01 .. < 2025-11-20`):** `524` trades (`0.2437/dia`), `WR 40.65%`, `PF 1.396`, expectância líquida `+0.6781%/trade`, `Sharpe 0.9736`, `MaxDD 15.58%`, `PnL +$12,709.02`.
- **OOS (`>= 2025-11-20 .. 2026-11-21`, `TRIAL_0175`):** `99` trades (`0.2705/dia`), `WR 37.37%`, `PF 1.178`, expectância líquida `+0.1844%/trade`, `Sharpe 0.4688`, `MaxDD 11.76%`, `PnL +$757.03` (âncora conservadora em `docs/pilot_backtest_reference.json`).
- **FULL (`2020-01-01 .. 2026-11-21`, `TRIAL_0174`):** `624` trades (`0.2480/dia`), `WR 40.22%`, `PF 1.363`, expectância líquida `+0.6263%/trade`, `Sharpe 0.8993`, `MaxDD 15.58%`, `PnL +$14,907.65`.

## 3. Portas de Refutação & Aceite (`pilot_report.py`)
- Herda todos os critérios A1–A14 e R1–R7 de `docs/CRITERIOS_APROVACAO_PILOTO.md` (N >= 100 em >= 56 dias limpos, `MaxDD estresse >= 20%` refuta, `IC95 sup < 0` com N >= 30 refuta, `expectância bruta <= 0` com N >= 30 refuta).
- **Aditivo Pós-S5 (2026-11-23 — Patch Final S5 `v1.0.11`, Telemetria `shadow`):**
  - Mantidos **100% intocados** todos os parâmetros congelados do `SWING_4H` (`Donchian40/ExitCh10`, `SL 2.5x`, `Trail 3.5x`, `|dist_sma_200|<=0.20`, `max_holding=120`, `cooldown=2`, `max_leverage=2.0`).
  - Adicionado módulo causal `data/swing_4h_meta_sizer.py` (`mode="shadow"` por padrão, `applied_mult == 1.0`) gravando telemetria (`meta_prob`, `meta_size_mult`, `vol_target_mult`, `atr_risk_parity_mult`, `rs_14d_mult`) em `swing_4h_meta_telemetry`, além de `S5-1` (auto-learner por sub-regra estrutural `Beta(4,4)` `W=20`), `S5-2/S5-3` (`Platt`/`Isotonic` + `ExtraTrees` + Multi-Symbol `F0_TechOnly` + gate conjunto `LogLoss`/`ROC-AUC`), `S5-4` (`SentimentAgent trend_momentum` + fix P0 `micro_derivs` funding) e registro formal de `Alpha-1/2/3` (**NO-GOs:** `T1-DDThrottle`, `T3 EWMAC`, `T4 Squeeze`, `T5 Capitulation`, `T6a/T6b`, `T7b ORB`, `T8`, `T9`, `T10`, `T12`).

---

# Experimento E_PATCH_S6_ONDA1 — Auditoria SCALP 15m / Cap RSI 1h (`S6-1`) & Filtro TradFi `GOLDEN_POT` (`S6-2`, `v1.0.12`, 2026-11-23)

- **`TRIAL_S6_01` (`SCALP 15m` Squeeze Ablação OOS — `98.116` barras BTC+ETH+SOL, split `2026-07-29`, embargo `10` barras, `SL 1.4x / TP 2.2x`, `16 bps` RT):**
  - Full Squeeze (`N=261`, `IS=174 / OOS=87`): `Prod (6-ag conf>=0.70, RSI<=75)` obteve **OOS `n=37, WR 59.5%, PF 1.56, +10.30R (+0.278R)`**, superando `Intraday+RSI75` (`PF 1.26, +4.52R`), `Intraday NoRSI` (`PF 1.06, +1.32R`), `RiskOnly+RSI75` (`PF 1.38, +9.64R`) e `RiskOnly` (`PF 1.21, +6.43R`). Pós-`ML15m` purgado (`65.683` IS rows, `OOS N=11`): `Prod` manteve **OOS `PF 1.18 (+0.41R)`**, enquanto `conf>=0.55` (`PF 0.78, -0.75R`), `NoRSI75` (`PF 0.77, -0.80R`) e `RiskOnly` (`PF 0.89, -0.66R`) colapsaram (`RSI > 75/< 25` OOS: `n=5, WR 20%, PF 0.31, -3.24R`) -> **NO-GO** (preservar gate atual).
- **`TRIAL_S6_02` (`DAY_TRADE 1h` Teto `RSI > 75 / < 25` — `2020–2026`, `IS<=2024 | 7d | OOS>=2025`, `SL 1.0x / TP 1.5x`, `16 bps` RT):**
  - Entradas em extremo `RSI > 75 (bull) / < 25 (bear)` registraram IS `PF 0.928 (-238.0R)` e **OOS `n=2.143, WR 39.6%, PF 0.730, -395.6R`**. Relaxar `25..75 -> 20..80 / 15..85 / 0..100` piorou o OOS em `-178.4R` a `-208.7R` -> **NO-GO** (manter `TechnicalAgent` intacto).
- **`TRIAL_S6_03` (`S6-2` Classificação Dinâmica TradFi `GOLDEN_POT` — GO):**
  - Auditoria ao vivo em `/fapi/v1/exchangeInfo` identificou `253` contratos não-cripto (`TradFi`/`EQUITY`/`COMMODITY`/`PREMARKET`/`INDEX`), incluindo `28+` ativos que vazavam da lista estática de `42` nomes. Implementada unificação em `NON_CRYPTO_BASE_EXCLUSIONS` (`85` bases) + cache TTL `6h` via `exchangeInfo` (backoff `5m`) mantendo 100% dos cripto/RWA `COIN` (`PAXG`, `XAUT`, `MANTRA`, `CFG`). Referência viva: `docs/PLANO_EVOLUTIVO_S6.md`.

---

# Experimento E_PATCH_S6_ONDA2 — Anti-Deadlock Época+`72h` em Calibração por Regime (`S6-3`) & Baseline ML Multi-Sym `Platt` (`S6-4`, `v1.0.13`, 2026-11-24)

- **`TRIAL_S6_04` (`S6-3` Deadlock `BEAR_TREND` em `LearningTracker` — GO para `V1_TTL_72H` + `pilot_start_utc`):**
  - **Replay Live 24h (`23/11 13:00 → 24/11 12:00 BRT`, `16 bps` RT, `SL 1.0x / TP 1.5x`):** `V0_PROD` gerou `1` sinal `>=0.70` / `30` `low_conf` (`+1.38R`); `V1_CLEAN_EPOCH` (`pilot_start_utc` com `S5-1` intacto) gerou **`27` sinais `>=0.70` / `0` `low_conf` (`6W/1L/2open`, `+8.10R`)**.
  - **Simulação Causal 5 Anos (`2022–2026`, `64.260` barras `2h`, `IS 2022–24 | 7d | OOS 2025–26`, `16 bps` RT):** `V0_PROD` travou permanentemente (`1.673,9d`, `OOS n=0`); **`V1_TTL_72H`** obteve o melhor `BEAR_TREND` OOS (`n=390, WR 41.3%, PF 0.77, -62.3R, max_dry=3.0d`), superando `48h` (`PF 0.65`), `120h` (`PF 0.49`), `168h` (`PF 0.69`), **`V2_TTL72_UNAN3` (`NO-GO`: `PF 0.69, -112.0R`)** e **`V3_NO_DAMPEN_TREND` (`NO-GO`: `PF 0.75, -297.1R`)**.
  - **Decisão:** Mantido `consensus_judge.py` 100% congelado; implementado isolamento estrito `pilot_start_utc` + TTL `72h` com quebra de cluster por silêncio apenas em `learning_tracker.py`.
- **`TRIAL_S6_05` (`S6-4` Head-to-Head OOS Champions Legados vs Multi-Sym `Platt` — GO `LGBM` & `RF`):**
  - **OOS 2 Anos Purgado (`IS <= 2024 [n=99.498] | 7d | OOS >= 2025-01-08 [n=49.353]`):** `LGBM` `LogLoss 1.0760 -> 1.0733` (`AUC 0.5847 -> 0.5870`, `emAcc 56.8% -> 57.8%`); `RF` `LogLoss 1.0764 -> 1.0753` (`AUC 0.5856 -> 0.5852`).
  - **Cauda Inédita `>= 2026-11-18` (`n=465`):** `LGBM` `LogLoss 1.0837 -> 1.0752` (`AUC 0.5691 -> 0.5745`); `RF` `LogLoss 1.0790 -> 1.0720` (`AUC 0.5817 -> 0.5809`). Base autorizada para troca única (`force_promote=True`) destravando `evaluate_challenger_joint()`.



---

# Experimento E_PATCH_S6_ABLACOES_SCALP_GP — Auditoria Zero Scalps/GP, Forensics W43/W44 & TapeReading (`2026-11-25`)

- **Motivação & Paridade Live (`2026-11-20 13:00Z .. 2026-11-25 01:46Z`):**
  - `SCALP`: `13` Squeezes no período (`4 BTC, 4 ETH, 5 SOL`), mas `11` (`84,6%`) caíram em `ML 15m < 45%` e `2` em `committee_disagree`. Em `2026-11-24`, `BTCUSDT` teve `3` Squeezes `SELL` barrados por `ML 15m` (`37,8%–39,1% < 45%`) e `4` horas seguidas `DIRECTIONAL` (`20:00Z–23:00Z`) barradas no `_flow_gate` (`no_flow_confluence`: `taker_imb_5m < 0` contra `book_imb_10 > 0`) antes do `ETHUSDT` preencher `MAX_SAME_DIR_MAJORS = 2` (`23:00:21Z`).
  - `GOLDEN_POT`: `~111` gatilhos em `8.060` avaliações (`1,38%`), estrangulados por `Macro BTC < -1,5%` (`94,4%` do tempo / `40` barras únicas `BUY`) + `Climax Exhaustion` (`38` barras únicas `SELL`) + `committee`/`veto`/`flow_gate`.
- **`TRIAL_0176..TRIAL_0183` (`SCALP 15m` OOS — `scripts/eval_scalp_s6_oos.py`, `datasets/reports/s6_scalp_oos_ablation.json`):**
  - Janela `2026-07-29 .. 2026-11-25` (`119,1d` OOS, entrada **taker**, `SL 1.4x / TP3 2.2x`, modelos `15m`/`1h` purgados pré-split, `learning_tracker=None`) vs `DIRECTIONAL 1h` (`4,30 t/d`, `n=512`, WR `39,1%`, PF `0,80`, `-74,96R`).
  - Todas `NO-GO` (`DSR ≈ 0`): `S0_PROD` (`n=11, 0,09/d, PF 3,39`), `S1_NO_ML15` (`n=37, 0,31/d, PF 1,47, IS PF 0,74`), `S2_ML15_DIR_ONLY`/`S3_ML15_THR38` (`n=28, 0,24/d, PF 1,62, IS PF 0,99`), `S4_SQZ1B_NO_ML15` (`n=43, 0,36/d, PF 1,57, IS PF 0,68`), `S5_SQZ2B_RELAX_VOL_NATR` (`n=51, 0,43/d, PF 1,32, IS PF 0,78`), `S6_MOM20_H1_NO_ML15` (`n=191, 1,60/d, PF 0,57, -68,51R`), `S7_META_LABEL_SQZ` (`n=13, 0,11/d, PF 1,96, IS PF 1,24`). A trava de mesmo par com `DIRECTIONAL` corta `~53%–73%` dos scalps (`S1: 37→13`).
- **`TRIAL_0184..TRIAL_0191` (`GOLDEN_POT 5m` OOS — `scripts/eval_golden_pot_s6_oos.py`, `datasets/reports/s6_golden_pot_oos_ablation.json`):**
  - `220` altcoins cripto puras (`54` TradFi excluídas), `597` segmentos contíguos, split `2026-11-05` (`20,1d` OOS, sendo `19,19d` em `OOS_BEAR`) vs `DIRECTIONAL 1h` (`4,77 t/d`, `n=96`, WR `50,0%`, PF `1,49`, `+26,20R`).
  - Todas `NO-GO`: `G0_PROD` (`n=54, 2,69/d [4,62 uncap], PF 1,01, IS PF 0,78`), `G1_RELAX_CLIMAX` (`n=59, 2,93/d [9,40 uncap], PF 1,01`), `G2_NO_CLIMAX` (`PF 0,85, -3,60R`), `G3/G4/G5` (`PF 0,91`), `G6_INTRADAY` (`PF 0,91`), `G7_EARLY_TRIGGER` (`n=60, 2,98/d [12,88 uncap], WR 66,7%, PF 1,09, Bear PF 0,97`). Caveat: chunks `5m` de ~400 barras pré-seleção no DB + sem cap Top-6 / `_flow_gate`.
- **Forensics `W43/W44` & `TapeReading` (`E10.10`):**
  - `GOLDEN_POT` em `W43/W44` (`+$89,06, n=7`) mostrou pistas de teto `ATR_exp <= 3,0x` (spikes `3x–6,35x` perderam) e erosão por `TP1 → BE` (`W45` perdeu metade dos trades em BE curto vs stops `-30/-34`), enquanto `SCALP` em `W43/W44` (`+$12,47, n=4`) era `3m` legado pós-prejuízo de `-$91,96` (`W42`).
  - Em `E10.10`, o ganho de `~+7..8,6 pp` de WR do `TapeReading` ocorreu no legado `SCALP_PROD_BASE` cortando `81%` das entradas (`70→13`); no `GOLDEN_POT`, `80,5%` do `flow_gate_log` (`33/41`) e `94%` no replay `29d` deram `UNAVAILABLE` devido a `top_n=3 < GP_MAX_CANDIDATES=6` e falta de fallback REST em `book_stale`/`tick_stale`/`tick_insufficient`.

---

# Pré-Registro E_PATCH_S6_ONDA3 — Expansão Majors (`T11`, Frente A) & Fix Tape Fallback GP (Frente B) (`2026-11-25`)

- **Handoff Executável:** [`docs/PROMPT_ONDA_S6_3_MAJORS_E_TAPE_FALLBACK.md`](file:///home/eudespatricio/ia_trader/docs/PROMPT_ONDA_S6_3_MAJORS_E_TAPE_FALLBACK.md).
- **Frente B (Fix Tape Fallback + Cobertura Coletor GP):**
  - Acionar `_evaluate_live_binance_fallback()` também em `book_stale`, `tick_stale` e `tick_insufficient` em `agents/tape_reading_agent.py` (preservando política de erro/spread); ampliar `top_n = 3 -> 6` em `scripts/microstructure_collector.py` para cobrir toda a fila `GP_MAX_CANDIDATES = 6`.
  - **Aceite Frente B:** Testes unitários/regressão independentes verdes (`100%` sem mocks permissivos), `0` regressões de latência/bloqueio indevido em majors, revisão cross-model `APPROVE`.
- **Frente A (`T11` Expansão Controlada de Majors `BTC/ETH/SOL` → `+5` Candidatas `XRPUSDT, NEARUSDT, DOGEUSDT, LINKUSDT, AAVEUSDT`):**
  - Ingestão `1h` (`2021–2026`), `15m` (`~1 ano`) e `funding_rates` com desduplicação canônica (`binance` prioridade, zero overlap Bitget UTC+8), preservando `BTC/ETH/SOL` como baseline.
  - Backtest OOS zero-leakage (embargo `7d` 1h / `10` barras 15m, custos estresse `16 bps` RT + funding real) por estratégia (`DIRECTIONAL 1h`, `SCALP 15m`, `SWING_4H`) e ablação de trava de correlação (`MAX_SAME_DIR_MAJORS = 2 vs 3 vs cap por beta/risco`).
  - **Aceite Frente A (`GO`):** Apenas pares/configurações que entreguem OOS conjunto `PF >= 1.25`, `PF_ex_top1 >= 1.10`, `expectância > 0`, `MaxDD_estresse < 15%`, `DSR` consistente com `N` trials acumulado e zero degradação no baseline `BTC/ETH/SOL` podem entrar no código de produção; caso contrário, manter `NO-GO` documentado.
- **Resultado Executado (`2026-11-25`, Revisão Cross-Model `Claude Opus 5.5` — `APPROVE`):**
  - **Frente B (`GO` / Implementada):** [`agents/tape_reading_agent.py`](file:///home/eudespatricio/ia_trader/agents/tape_reading_agent.py) aciona `_evaluate_live_binance_fallback()` (`LIVE_FALLBACK_TIMEOUT_S=2.5s`, cache OK `10s`, cooldown erro `15s`) em `book_stale`, `tick_stale`, `tick_insufficient` (`n_min`/`n_trades`/`sat_ratio > 0.50`) e `tick_zero_volume` apenas com `is_live_call=True` (zero rede em backtest), fechando SQLite RO antes do REST; [`scripts/microstructure_collector.py`](file:///home/eudespatricio/ia_trader/scripts/microstructure_collector.py) ampliado para `DEFAULT_GP_TOP_N = 6` (`MAX_ACTIVE_SYMBOLS = 16`, `log.error` explícito se falhar import TradFi). Suíte independente: `96 passed` ([test_tape_reading_gate.py](file:///home/eudespatricio/ia_trader/tests/test_tape_reading_gate.py) + [test_patch_s6_onda3_tape_fallback.py](file:///home/eudespatricio/ia_trader/tests/test_patch_s6_onda3_tape_fallback.py)).
  - **Frente A (`T11` Backfill OK / Promoção `NO-GO` Prudent Hold, `TRIAL_0196..TRIAL_0202`, [`s6_majors_expansion_oos.json`](file:///home/eudespatricio/ia_trader/datasets/reports/s6_majors_expansion_oos.json)):**
    - Ingestão idempotente ([`backfill_candidate_majors.py`](file:///home/eudespatricio/ia_trader/scripts/backfill_candidate_majors.py)): `52.828` velas `1h`, `40.338` velas `15m` e `6.604` `funding_rates` por candidato (`0` gaps, `0` duplicatas pós-dedup de `30` linhas Bitget `NEARUSDT:15m`, `PRAGMA quick_check = ok`, `pilot_start_utc` e posições abertas intactas).
    - Paridade zero-leakage: `enricher_max_diff = 0.0`, `ensemble_max_diff = 0.0` (`n=25` em `BASE_3 + XRPUSDT/NEARUSDT`), zero `.joblib` em disco.
    - **`DIRECTIONAL 1h` (`2025-01-08..2026-11-25`):** `M0_BASE_3` `PF 0.798` (`-406.34R`, `TRIAL_0196`) vs `M1_BEST` (`base3_cc2_nr2_beta`) `PF 0.855` (`-255.51R`, `MaxDD 94.93%`, `DSR 0.0000`, `TRIAL_0197`) → **`NO-GO`**.
    - **`SWING_4H 4h`:** Primária (`>= 2025-01-08`) `M0_BASE_3` `PF 1.478` (`+35.98R`, `MaxDD 12.06%`, `TRIAL_0200`) vs `M1_BEST` (`cc4_nrnone_unit`) `PF 1.340` (`+54.26R`, `MaxDD 21.75% >= 15%`, `DSR 0.6497`, `TRIAL_0201`); paridade (`>= 2025-11-20`) `M1_BEST` `PF 1.078` (`+7.42R`, `XRP/DOGE/LINK/AAVE < 1.0`) → **`NO-GO`**.
    - **`SCALP 15m` (`2026-07-29..2026-11-25`):** Standalone `M0_BASE_3` `n=11, PF 3.393` vs `M1_BEST` `n=33, PF 1.891, +13.61R, DSR 0.9174 < 0.95` (`XRP PF 0.582, AAVE PF 0.760`, `TRIAL_0202`) → **`NO-GO`**. Com `dirlock` (`TRIAL_0199`), `M1_BEST` atingiu `n=18, PF 6.159, +18.08R, DSR 0.9619`, porém é **contrafactual** porque `DIRECTIONAL 1h` permanece apenas em `BTC/ETH/SOL` (sem trava nos 5 pares extras, valeria o standalone `NO-GO`); `M3_SUBSET (NEAR/DOGE/LINK)` (`n=20, PF 3.770, DSR 0.9951`) é estritamente exploratório; `ML pooled8` degradou `M0` em `SCALP` (`PF 3.393 -> 2.491`). Produção preservada intacta (`NO-GO` hold). `G8_ATR_EXP_CAP_3X` fechou neutro (`n=55, sumR=-2.0126`).
