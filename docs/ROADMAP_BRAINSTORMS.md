# 🧠 Roadmap de Brainstorms & Planejamento Estratégico — IA Trader

> **Documento de Continuidade e Referência Rápida para Novas Sessões**  
> **Data de Atualização:** 18 de Outubro de 2026  
> **Status:** Ativo / Piloto em Andamento (Parâmetros Monitorados)  
> **Objetivo:** Consolidar todas as análises forenses, hipóteses de melhoria, benchmarks de interface e decisões de arquitetura debatidas nas sessões para uso imediato em novas janelas de chat.

---

## 📋 Índice
1. [Prompt de Inicialização Rápida (Copiar e Colar no Novo Chat)](#-prompt-de-inicialização-rápida-copiar-e-colar-no-novo-chat)
2. [Brainstorm 1: Enhancement do Gráfico & Interface Visual (Benchmark Exchanges)](#-brainstorm-1-enhancement-do-gráfico--interface-visual-benchmark-exchanges)
3. [Brainstorm 2: Diagnóstico Forense & Redesenho do Golden Pot (5m Momentum)](#-brainstorm-2-diagnóstico-forense--redesenho-do-golden-pot-5m-momentum)
4. [Brainstorm 3: Microestrutura & Tape Reading (Expansão e Refinamento)](#-brainstorm-3-microestrutura--tape-reading-expansão-e-refinamento)
5. [Brainstorm 4: Paridade Estrita Backtest vs Piloto & Modelos ML](#-brainstorm-4-paridade-estrita-backtest-vs-piloto--modelos-ml)
6. [Brainstorm 5: Governança de Execução, Circuit Breakers & Daemons](#-brainstorm-5-governança-de-execução-circuit-breakers--daemons)

---

## 🚀 Prompt de Inicialização Rápida (Copiar e Colar no Novo Chat)

> Ao iniciar uma nova sessão no Antigravity, envie a seguinte mensagem para sincronizar o contexto instantaneamente:

```text
Olá! Estou iniciando uma nova sessão do projeto IA Trader. O sistema está rodando o piloto de paper trading 24/7.
Por favor, consulte o documento docs/ROADMAP_BRAINSTORMS.md e o EVOLUTION_LOG.md para o estado atual da arquitetura.
Nossos tópicos prioritários de trabalho são:
1. [Enhancement do Gráfico]: Modernizar o componente de candles (benchmark TradingView/Lightweight Charts, zoom/pan intuitivo, linhas de SL/TP/Trailing e HUD).
2. [Correção do Golden Pot]: Aplicar travas de latência de barra, filtro de exaustão de clímax e limite de alavancagem 2x em altcoins voláteis.
3. [Monitoramento do Piloto]: Checar integridade dos daemons (supervisor, coletor, paper daemon) e métricas do ledger.
Por favor, verifique o status do sistema antes de iniciarmos.
```

---

## 🎨 Brainstorm 1: Enhancement do Gráfico & Interface Visual (Benchmark Exchanges) — [STATUS: CONCLUÍDO & HOMOLOGADO]

### Resumo da Implementação (18/10/2026):
* **TradingView Lightweight Charts v4.2.1:** Integrado nativamente no `templates/dashboard.html` com renderização a 60fps via canvas e fallback para Plotly mantido.
* **Multi-Timeframe Switcher:** `1m`, `3m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d` com auto-fetch on-demand na API `/api/chart` (Binance + fallback Bitget) quando o SQLite tem < 20 candles.
* **Smart Trade Exit Ladder Lines:** Projeção visual direta no canvas para Entry (azul), Stop Loss / Breakeven (vermelho/ciano), TP1, TP2, TP3 (verde), Trailing Stop (dourado) e Preço de Liquidação.
* **HUD Crosshair & Ferramentas:** Leitura instantânea de OHLCV, amplitude, volume e indicadores (SMA 20/50, EMA 9/21, BB) sem sobrecarga, com zoom centrado no cursor e arraste de escala no eixo direito.

---

## ⚡ Brainstorm 2: Diagnóstico Forense & Redesenho do Golden Pot (5m Momentum) — [STATUS: CONCLUÍDO & HOMOLOGADO]

### Resumo da Implementação (18/10/2026):
* **Trava de Latência de Barra:** Se `now - candle_close > 45s`, a oportunidade é descartada por obsolescência (`GP_STALE_BAR_MAX_AGE_MS = 45_000`). Confirmado em operação no paper daemon.
* **Filtro de Exaustão de Clímax:** Bloqueio de sinais se a variação de 3 velas excede 2.5 ATR com estiramento direcional, ou RSI < 15 (short) / RSI > 85 (long).
* **Cap de Alavancagem 2.0x:** Teto rígido de alavancagem em 2.0x para altcoins voláteis operadas pelo Golden Pot.
* **Stop Loss Adaptativo:** Calibrado com margem de segurança de 1.15x sobre a amplitude recente da barra para absorver pavios e retestes sem violação prematura.
* **Testes Automatizados:** Suíte dedicada `tests/test_golden_pot_forensic_protections.py` (6/6 passando); 162 testes de backend passando sem falhas.

---

## 🌊 Brainstorm 3: Microestrutura & Tape Reading (Expansão e Refinamento)

### Estado Atual & Dados Coletados
* **Volume de Dados:** Coletor dedicado (`scripts/microstructure_collector.py`) acumulou **60.270 snapshots de book**, **15.623 janelas de tick_agg** e **15.072 registros de derivativos** em SQLite de alta performance (`datasets/microstructure.db`).
* **Auditoria de Decisões:** 513 auditorias de fluxo registradas em `flow_gate_log` e 884 etapas monitoradas em `signal_funnel`.
* **Gate de Fluxo:** `agents/tape_reading_agent.py` monitorando taker flow 5m, book imbalance e microprice deviation.

### Diagnóstico Empírico de Gargalos (Auditoria de 513 Decisões):
1. **Rigidez Concomitante no Tape Reading (94,5% de Bloqueios):**
   * Das 513 chamadas do gate, **485 (94,5%) foram bloqueadas como `no_flow_confluence` (HOLD)**.
   * **Causa Raiz:** O gate exige conjunção booleana estrita: `taker >= 0.15` E `book >= 0.10` E `microprice >= 0.0`. Em futuros perpétuos, forte agressão de venda a mercado (`taker < -0.15`) frequentemente encontra ordens limite passivas de market makers absorvendo no book (`book > 0`).
   * **Dado Revelador:** 140 sinais de venda tinham forte agressão vendedora alinhada (`taker <= -0.15`), mas foram descartados porque o book estava temporariamente absorvendo. Apenas 5 de 485 satisfizeram ambos simultaneamente.
2. **Diluição Matemática no Comitê de Agentes (`ConsensusJudge`):**
   * Em `signal_funnel`: **96,4% de Day Trade (239/248)**, **96,1% de Golden Pot (49/51)** e **92,6% de Scalp (125/135)** foram rejeitados por `committee_disagree`.
   * **Causa Raiz:** O consenso calcula `weighted_score / total_weight`. Agentes de contexto (Macro, Sentiment, OnChain) votam `HOLD` por ausência de evento e inflam o denominador `total_weight` com peso $>0$ sem somar no numerador, diluindo o score abaixo do limiar de 0.30.

### Resultados Quantitativos da Simulação Offline dos 5 Itens (19/10/2026):
* **Item 1 (Abstenção no Comitê):** Elevou a vazão de Day Trade de 11 para 95 (+84), Golden Pot de 2 para 19 (+17) e Scalp de 11 para 51 (+40), desbloqueando ~4x a 10x mais oportunidades legítimas sem enfraquecer o veto do Sentinela de Risco.
* **Item 2 (Score Contínuo de Fluxo):** Em 514 decisões auditadas, a regra antiga aprovou apenas 8 (1,6%), bloqueando 506 (98,4%). O FlowScore contínuo ($|\text{FlowScore}| \ge 0.12$) aprovou 105 ordens alinhadas (20,4%, ganho de 13.1x em vazão segura) e bloqueou 180 fluxos contrários/armadilhas (35,0%).
* **Item 3 (Absorção Institucional):** Identificou 18 absorções de venda (bullish) e 23 de compra (bearish) nas 61.023 leituras de livro, criando novo gatilho causal de reversão.
* **Item 4 (Ollama / RTX 3060):** Teste real no gateway `172.30.176.1:11434` com `llama3.1:latest` validou tempo de resposta de 3,4s warm na GPU local, com 100% de conformidade JSON sem custo de nuvem.
* **Item 5 (Expansão Top 3 Altcoins):** Confirmou que a ausência de book nas altcoins do Golden Pot gerou os 2 únicos stops térmicos do piloto (`BTWUSDT` e `AGTUSDT`), responsáveis por 83,8% (-$33,53) do drawdown acumulado. Coletar o Top 3 altcoins fecha essa brecha.

### Status de Implementação em Produção (19/10/2026 - 100% Concluído):
* [x] **Item 1 (Abstenção no Comitê):** Implementado em `consensus_judge.py`. Agentes contextuais sem choque votam `HOLD` com peso 0.0 (sem diluir denominador).
* [x] **Item 2 (Score Contínuo de Fluxo):** Implementado em `agents/tape_reading_agent.py` ($\text{FlowScore} = 0.60 \times \text{TakerImb} + 0.30 \times \text{BookImb} + 0.10 \times \text{MicroDevNorm}$).
* [x] **Item 3 (Absorção Institucional):** Implementado em `agents/tape_reading_agent.py` (`absorption_buy_confirmed` e `absorption_sell_confirmed`).
* [x] **Item 4 (Desempate via GPU Local Ollama):** Conectado no `_committee_gate` em `scripts/run_paper_simulation.py` chamando `ChatGPTAssistant` (`llama3.1:latest` na RTX 3060 a R$ 0,00).
* [x] **Item 5 (Expansão Dinâmica Top 3 Altcoins):** Ativo em `scripts/microstructure_collector.py` monitorando Top 3 altcoins em tempo real e sincronizado no `TapeReadingAgent`.
* [x] **Homologação:** 228 testes passando (0 falhas); daemons rodando saudáveis (4/4 `HEALTHY`).

---

## ⚖️ Brainstorm 4: Paridade Estrita Backtest vs Piloto & Modelos ML

### Diretrizes "Laranjas com Laranjas"
* **Escada Unificada de Saída:** Implementada igualmente no `smart_trade.py` e `backtesting.py` com `pilot_parity=True`.
* **Timing de Entrada:** Convencionado em `open(t+1)` para evitar qualquer viés lookahead.
* **Modelo ML de Sinais (Argon):**
  * Pesquisa offline documentada em `docs/BUSCA_ESTRATEGIA_RESULTADOS.md`.
  * Filtro de padrões gráficos causais (Pin Bar, Engolfo, Inside Bar, Divergência RSI regular) implementado em `data/pattern_features.py`.
  * Preservar o Holdout intocado para homologação final antes de qualquer promoção para ambiente real.

---

## 🛡️ Brainstorm 5: Governança de Execução, Circuit Breakers & Daemons

### Arquitetura de Produção 24/7
* **Supervisor Robusto:** `scripts/system_supervisor.py` operando com healthcheck de socket (5s), re-adoção automática de PID e tolerância a 3 falhas antes de auto-heal.
* **Circuit Breakers Ativos:** Cooldown de 120 minutos após 6 perdas consecutivas (comportamento verificado e validado durante o piloto).
* **Reset Seguro:** `scripts/reset_pilot.py` disponível com backup pré-transacional íntegro para reinícios controlados quando houver alteração de modelo.
* **Regra de Congelamento:** Para validação estatística de alfa, os parâmetros do piloto devem permanecer congelados por 56 dias consecutivos após o início limpo.

---

## 🏆 Brainstorm 6: Homologação para Lucratividade Real (Prazo: 10 de Dezembro de 2026)

### Objetivo Estratégico
Transformar o sistema de um piloto funcional em uma **ferramenta institucional lucrativa e homologada** para operar capital real até 10 de dezembro de 2026 (52 dias / 7 semanas).

### A Tríade da Lucratividade Quantitativa
1. **Eficiência Drástica de Custos (Maker-First):**
   - Transição mandatória para ordens passivas (Post-Only) com spread reduzido e taxa maker (0.02% vs 0.06% taker), economizando até 60% do atrito em operações de alta frequência.
2. **Assimetria Matemática Estrita (R:R $\ge 2.0$):**
   - Garantir esperança matemática positiva $\mathbb{E}[R] > 0$ com alvos escalonados:
     - Scalp: SL 0.9x ATR | TP1 1.2x (40% parcial + BE) | TP2 2.0x (30% + trailing) | TP3 3.6x ATR (30% runner).
     - Golden Pot: SL 1.5x ATR | TP1 2.0x (40% + BE) | TP2 3.5x (30% + trailing) | TP3 6.5x ATR (30% fat-tail runner).
     - Directional: SL 1.0x ATR | TP1 1.8x | TP2 3.0x | TP3 5.0x ATR.
3. **Dimensionamento Dinâmico por Volatilidade (1% Risco da Banca):**
   - Eliminação de lotes fixos em dólar. O tamanho da posição é ponderado pela distância exata do Stop Loss:
     $$\text{Margin}_{\text{ideal}} = \frac{\text{Capital} \times 0.01}{(\text{Distância do SL em \%}) \times \text{Alavancagem}}$$
   - Reduz automaticamente a exposição em altcoins ruidosas e expande com segurança em estruturas limpas.
4. **Filtro de Regime de Mercado (Chop vs Trend):**
   - Detector de regime baseado em compressão de volatilidade e liquidez do book para blindar contra falsos rompimentos.

### Cronograma de Homologação em 7 Semanas

| Semana | Período | Foco Tático | Entregável Crítico | Status |
| :--- | :--- | :--- | :--- | :--- |
| **S1** | **19 a 25/Out** | **Microestrutura, Risco & Custos** | Execução Maker-First, Trava R:R $\ge 2.0$, Sizing Volatility-Adjusted (1% risco) + Piso de SL e Cooldown Pós-Stop. | 🟢 **CONCLUÍDO (24/10)** |
| **S2** | **26/Out a 01/Nov** | **Modelo Argon & Calibração de Agentes** | Treinamento LightGBM/RF nos datasets locais (50k velas) + Protocolo de Calibração (ECE, Brier, Regime Weights) e Otimização de Limiares em Backtest (PWF-CV, PBO < 20%, DSR > 0.95). | 🟡 **EM ANDAMENTO** |
| **S3** | **02 a 08/Nov** | **Classificador de Regime** | Detector de regime adaptativo (Tendência vs Consolidação Ruídosa) para seleção de estratégia. | ⚪ Planejado |
| **S4** | **09 a 15/Nov** | **Stress Test & Monte Carlo** | 10.000 iterações Monte Carlo sobre histórico de trades; validação de Max Drawdown < 6%. | ⚪ Planejado |
| **S5** | **16 a 22/Nov** | **Auditoria de Paper Trading** | 7 dias ininterruptos de telemetria; meta: Sharpe Ratio > 2.0, Profit Factor > 1.6. | ⚪ Planejado |
| **S6** | **23 a 29/Nov** | **Soft Launch (Micro-Banca Real)** | Conexão API Live com banca mínima (\$100–\$200) para medir slippage e latência real de fills. | ⚪ Planejado |
| **S7** | **30/Nov a 10/Dez** | **Homologação Final** | Calibração de slippage real, auditoria do ledger e liberação para capital pleno. | ⚪ Planejado |

### 📝 Anotações da Auditoria de 23/Out/2026 (ajustes mapeados por semana)

Base: 44 posições fechadas (15–23/Out) | Equity \$10.000 → \$9.316 (-6,8%) | Win 29,5% | PF 0,36 | Fees \$255 | MaxDD -8,8%.

| Semana | Ajuste previsto (evidência do piloto) |
| :--- | :--- |
| **S1** (em andamento) | **Cooldown por símbolo+direção após STOP_LOSS** (30–60 min): em 20/Out houve 4 shorts ETH em 9 min, todos stopados (-\$353). **Piso de distância do SL** (≥0,3% / ≥1x ATR): SCALP tinha SL mediano 0,08% (mín. 0,04%) < fees round-trip; scalps pagaram \$74 de fees p/ -\$69 bruto. **Teto de notional por símbolo** e revisão do teto de 4x (0/9 vitórias). Exigir TP1 ≥ 3x as fees round-trip. Decidir manter/desligar SCALP (PF 0,10, win 18%). |
| **S2** | **1) Migração / Dual-Timeframe do SCALP (`3m` → `15m`):** matriz ML provou que em `3m` as taxas consomem `0,42R–0,79R` por trade ($\mathbb{E}[R] < 0$), enquanto em `15m` o ATR sobe para `28–42 bps` (fee `0,16R–0,24R`) e o RF atinge **60,0%–66,7%** de acerto com $\mathbb{E}[R] = +0,17R \text{ a } +0,59R$. **2) Ensemble LGBM+RF 50k velas plugado ao vivo no `TechnicalAgent`** para `DIRECTIONAL (1h)` (`71,2%–93,8%` OOS em BTC/ETH/SOL). **3) Painel Pooled Multi-Altcoin para `GOLDEN_POT (5m)`** (combinar os ~20 pares com >= 1.200 velas num único dataset de ~25k barras + confirmação de absorção do `TapeReadingAgent`). **4) Inverter / recalibrar sinais de microestrutura** (`flow_confirmed` contrário vs `absorption_*` reversão) com mais amostras em `microstructure.db`. **5) Grid Search & Otimização de Limiares de Confiança (50.000 velas) no fim do ciclo S2:** testar sistematicamente de 0.05 em 0.05 (`0.60, 0.65, 0.70, 0.75, 0.80`) por regime e timeframe para encontrar o pico de $\mathbb{E}[R]$ e lucro líquido, substituindo saltos binários bruscos (0.80) por degraus graduais (+0.05/+0.10). *Régua defensiva da S2 calibrada provisoriamente em 0.65 para destravar trades e acumular dados.* |
| **S3** | Regime: 93% das leituras são `SIDEWAYS_RANGE`, mas entradas por momentum (preço < SMA200) são as que mais perdem. Investigar **assimetria de SHORT** (win 21% vs LONG 40%; net -\$579 vs -\$245). Seleção de estratégia por regime (mean-reversion em range). |
| **S4** | Monte Carlo sobre trades reais: baseline PF 0,36 / ganho médio +\$36 vs perda média -\$42; dia de pior drawdown 20/Out (-\$673). GOLDEN_POT: SL até 12% e perdas de -\$100 em microcaps (rever sizing 1% risco com piso de SL e liquidez mínima). |
| **S5** | Meta Sharpe>2 / PF>1,6 exige ≥ 7 dias contínuos de telemetria: depende de `performance_snapshots` e `agent_*` gravando (corrigido em 23/Out). Funil: 3.147 rejeições `breaker` e 1.191 `margin` no SCALP indicam gargalo operacional; revisar circuit breaker (13 cooldowns, 15 rollbacks, 26 resets manuais). |
| **S6/S7** | Contabilidade reconciliada em 24/Out: `paper_trades.pnl` desconta `close_fee`, enquanto `OPEN_*` tem `pnl=0` e debita `open_fee` direto do `balance` ($\text{Equity} = \$10.000 + \sum \text{pnl} - \sum \text{open\_fees}$ exato ao centavo). |

#### ✅ Item 7 corrigido em 23/Out e Fechamento da S1 + Matriz ML Multi-Estratégia em 24/Out
- **Causa (23/Out):** `LearningTracker.record_decision()` nunca era chamado no fluxo ao vivo (só `update_trade_outcome`, sem linhas a atualizar) e `PerformanceTracker.record_snapshot()` não era chamado pelo daemon (`scheduler.py` só roda via `launcher.py`, não pelo supervisor).
- **Correção, Alinhamento Learning/Performance & Validação (24/Out):**
  - Confirmado em produção: `performance_snapshots`, `agent_learning_memory` e `agent_regime_weights` ativos e sincronizados.
  - **Alinhamento do `LearningTracker`:** votos neutros/abstenção (`direction == "HOLD"`, como `RiskSentinelAgent` sem veto ou agentes macro/sentimento neutros) não são mais contabilizados como erro direcional em `recalibrate_agent_weights()`, preservando o peso de veto (`1.5`) do `RiskSentinelAgent` e evitando falsa acurácia `0%`.
  - **Alinhamento do `PerformanceTracker`:** `close_position()` agora força gravação imediata em `paper_portfolio` e `performance_snapshots` antes de acionar `compute_metrics_from_db()` / `rollback_guard`, e `_compute_strategy_breakdown()` calcula o `profit_factor` real por estratégia (`SCALP: 0.34`, `GOLDEN_POT: 0.20`, `DIRECTIONAL: 0.84`).
  - **Proteções S1 ativas:** amortecedor proporcional de micro-stops (`< 50 bps`) em `_lot_margin`, piso de ATR (`25 bps`) no Scalp S1, teto de alavancagem `3.0x` em Scalp/Swing S1, sizing dinâmico no Swing e cooldown dobrado após 2 `STOP_LOSS` consecutivos no mesmo par.
  - **Matriz ML Multi-Estratégia x Timeframe & Janela Ótima (50.000 velas, 2021–2026):**
    - Hardware calibrado para `i7-11800H`: `n_jobs=8` no LightGBM (evita contenção OpenMP/Hyper-Threading em 16 threads que elevava o treino de `1,5s` para `8,5s`).
    - **LightGBM (`lgbm_model_v1.joblib`, 50k velas `1h`) promovido:** acurácia de sinais emitidos **71,64%** ($\mathbb{E}[R] = +0,91R$ líquido de fees).
    - **Random Forest (`rf_model_v1.joblib`, 50k velas `1h`) promovido:** acurácia de sinais emitidos **89,55%–91,30%** ($\mathbb{E}[R] = +1,46R$ líquido de fees).

#### 🔬 Notas da S2: Protocolo de Validação em Backtest e Busca do Setup Ótimo (26/Out a 01/Nov/2026)
*Planejamento detalhado documentado em [`docs/S2_CALIBRACAO_AGENTES_BACKTEST.md`](file:///home/eudespatricio/ia_trader/docs/S2_CALIBRACAO_AGENTES_BACKTEST.md).*

1. **Os 3 Níveis de Calibração do Comitê Multi-Agente:**
   - **Nível 1 (Calibração de Probabilidade Individual):** Aplicação de *Isotonic Regression* e *Platt Scaling* nas confianças brutas dos agentes (`TechnicalAgent`, `TapeReadingAgent`, contextuais). Avaliação rigorosa via **Brier Score (BS)** e **Expected Calibration Error (ECE $\le 0.05$)** com Reliability Diagrams.
   - **Nível 2 (Ponderação de Consenso por Regime):** Substituição do multiplicador linear heurístico do `LearningTracker` por formulação de **Otimização Convexa Restrita (L2 / Ridge com Shrinkage Bayesiano)** por regime (`BULL_TREND`, `BEAR_TREND`, `SIDEWAYS_RANGE`, `HIGH_VOLATILITY_PANIC`), minimizando o erro quadrático direcional ponderado com preservação do veto soberano do `RiskSentinelAgent` ($w = 1.5$).
   - **Nível 3 (Limiares Operacionais & Decision Gates):** Otimização conjunta de limiar de consenso ($\theta_{\text{consensus}} \in [0.20, 0.40]$), gate de confiança ($\theta_{\text{gate}} \in [0.55, 0.80]$ em passos de $0.05$) e quórum direcional ($Q \in \{2, 3\}$) por estratégia/timeframe.

2. **Governança Anti-Overfitting em Backtest:**
   - **Purged Walk-Forward com Embargo (PWF-CV):** 12 meses de calibração rolante, 3 meses de teste OOS, com purga de posições com cruzamento de fronteira (*cross-boundary overlap*) e embargo de 7 a 10 dias de mercado.
   - **Combinatorially Symmetric Cross-Validation (CSCV) & PBO:** Varredura em 252 combinações de partições exigindo **Probability of Backtest Overfitting (PBO) < 20%** para aprovação do setup.
   - **Deflated Sharpe Ratio (DSR):** Todo ensaio registrado em `trial_log.csv`; o Sharpe OOS deve superar o limiar deflacionado pelo número total de tentativas ($N$), exigindo **DSR > 0.95**.
   - **Paridade Piloto Bitget:** Execução via `BacktestEngine(pilot_parity=True)` com taxas reais (taker $0.06\%$, maker $0.02\%$, slippage $0.02\%$, funding rates horários e parciais TP1/TP2/TP3).

3. **Função Objetivo Institucional (*Multi-Criterion Fitness*):**
   $$\mathcal{F} = \text{Sharpe}_{\text{OOS}} \times \left(1 - \frac{\text{MaxDD}_{\text{OOS}}}{0.10}\right) \times \mathbb{E}[R]_{\text{net}} \times (1 - \text{ECE}_{\text{ensemble}}) \times \sqrt{\min\left(1.0, \, \frac{N_{\text{trades}}}{50}\right)}$$
   *Hard Constraints:* $\mathbb{E}[R] \ge +0.25R$ líquido, Profit Factor $\ge 1.50$, Max Drawdown $\le 6.0\%$, Win Rate $\ge 45\%$, mínimo de 30 trades/trimestre.

4. **Integração às Ferramentas Existentes do Projeto (Validadas na Sexta 24/10):**
   - **Zero Retrabalho / Zero Redundância:** Reaproveitamento direto da infraestrutura já testada em 24/Out:
     - [`scripts/ab_consensus_experiment.py`](file:///home/eudespatricio/ia_trader/scripts/ab_consensus_experiment.py): Harness A/B offline com Purged Walk-Forward, embargo de 10 dias, taxas reais e cálculo de DSR.
     - [`scripts/parity_audit.py`](file:///home/eudespatricio/ia_trader/scripts/parity_audit.py): Auditoria de paridade 1:1 estrita ("laranjas com laranjas") backtest x paper trading.
     - [`model_trainer.py`](file:///home/eudespatricio/ia_trader/model_trainer.py): Treino e avaliação tabular nos 50.000 candles locais no WSL com otimização i7-11800H (`n_jobs=8`).
     - [`learning_tracker.py`](file:///home/eudespatricio/ia_trader/learning_tracker.py) & [`consensus_judge.py`](file:///home/eudespatricio/ia_trader/consensus_judge.py): Recalibração e consumo contínuo da matriz de pesos por regime.
   - **Encaixe na Rotina da S2 (26/Out a 01/Nov):**
     - *Etapa 1 (26-27/Out):* Scalp 15m & Ensemble 50k no `TechnicalAgent`.
     - *Etapa 2 (28-29/Out):* Pooled Altcoins Golden Pot (5m) & Calibração de Microestrutura (`microstructure.db`).
     - *Etapa 3 (30-31/Out):* Otimização de Consenso & Pesos por Regime.
     - *Etapa 4 (31/Out-01/Nov):* Grid Search de Limiares (50k Velas) em degraus de 0.05 com $\text{PBO} < 20\%$ e $\text{DSR} > 0.95$, promovendo o setup vencedor.

---

*Arquivo gerado e mantido pela equipe de engenharia do projeto IA Trader para garantir continuidade sem perda de contexto entre sessões de IA.*
