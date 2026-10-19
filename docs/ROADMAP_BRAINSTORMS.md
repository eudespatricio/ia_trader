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

*Arquivo gerado e mantido pela equipe de engenharia do projeto IA Trader para garantir continuidade sem perda de contexto entre sessões de IA.*
