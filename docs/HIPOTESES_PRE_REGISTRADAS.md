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

