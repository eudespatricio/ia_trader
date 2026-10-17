# 📋 Registro de Mudanças (Changelog) - IA Trader

Todas as alterações notáveis deste projeto são documentadas neste arquivo de acordo com as diretrizes do [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/) e aderentes ao [Semantic Versioning](https://semver.org/lang/pt-BR/).

---

## [Não lançado] - 2026-10-14/15 - Auditoria e Correções

### Alterado
- **Transição de Timeframes Intraday (2026-10-16):** Alinhamento operacional do Scalper (Majors BTC/ETH/SOL) para candles 3m com gatilho de momentum rápido e do Golden Pot (Radar Altcoins) para 5m com expansão de volatilidade calibrada; polling ágil a 3m para posições ativas; Smart Trade Exit Ladder mantida intacta; reinício a quente via supervisor com restauração total de estado (trade SOLUSDT preservado).
- **Funil de sinais (2026-10-15):** instrumentação pura (NÃO muda estratégia/decisões): tabela `signal_funnel` (hora UTC x estratégia x estágio, `reject:<motivo>`) alimentada por `PaperTradingEngine.funnel_inc` (dedupe por barra, flush ~30 s e no shutdown); `reset_pilot` a limpa.
- **E8 (2026-10-15):** gate de fluxo ativo (`TapeReadingAgent`, lê `microstructure.db` em modo RO) após a porta do comitê em SCALP/GOLDEN_POT/DAY TRADE: voto do tape deve confirmar a direção (oposto/HOLD bloqueia; dados ausentes = fail-open marcado); auditoria contrafactual em `flow_gate_log`. Piloto reiniciado. Ver ADDENDUM E8.
- **E7 (2026-10-15):** SCALP, GOLDEN_POT e DAY TRADE passam obrigatoriamente pelo comitê (scalp deixa de usar confidence 0.85/±0.80); GOLDEN_POT v3 = momentum + volatilidade extraordinária; **Smart Trade Exit Ladder** comum (TP1 40% + BE, TP2 30% + trailing por ATR sem lookahead, TP3/trailing no runner; perfis em `STRATEGY_PROFILES`). Piloto reiniciado. Ver ADDENDUM E7.
- **GOLDEN_POT v2 (E5, 2026-10-15):** entrada só com concordância do comitê completo (sem veto), `confidence`/`consensus`/`risk` reais, cooldown 16 barras, 1 GP/vela, filtro de faca-caindo, exclusão de ativos tokenizados, ≥400 candles com `sma_200` válida; lote = margem `paper_lot_usd` (padrão US$100) com alavancagem do Smart Trader (`leverage_policy.py`, teto 5x, liquidação). Piloto reiniciado; E4 invalidado para GOLDEN_POT.

### Corrigido
- **Auditoria Estrutural Argon — 6 Correções Validadas em Teste Offline A/B (2026-10-17):**
  1. **Deduplicação do `_flow_gate` por `(strategy, symbol, bar_ts)` (`scripts/run_paper_simulation.py`):** Impede a reavaliação da mesma vela fechada a cada 15s (redução de 99,2% na inflação da `flow_gate_log`) e elimina entradas atrasadas com preço antigo (*stale price*, como o trade `d3be50b1` que havia entrado 9m32s após o fechamento).
  2. **Alinhamento Temporal de Polling e Ratchet do Trailing (`smart_trade.py`, `scripts/run_paper_simulation.py`):** Posições `GOLDEN_POT` (5m) atualizam via velas fechadas de 5m e `SCALP`/`DIRECTIONAL` (3m/1h) via 3m, com inicialização imediata de `_last_tick_bar_ts` na abertura e evolução do `_ratchet_trailing` restrita ao fechamento do bucket nativo da estratégia (`prof.bar_ms`).
  3. **Priorização de Momentum no Scanner do Golden Pot (`agents/golden_pot_agent.py`, `scripts/run_paper_simulation.py`):** `SWING_BREAKOUT` avaliado antes de `SCALP_REVERSION` e fila `top_swings + top_scalps` priorizada para não sufocar rompimentos com ativos em exaustão de VWAP (+38,5% de candidatos de momentum detectados no backtest de 5m).
  4. **Bloqueio de Colisão de Ativo entre Estratégias e Calibração Fat-Tail do Golden Pot (`smart_trade.py`):** `_open_ladder_position` e `evaluate_swing_opportunities` vedam múltiplas posições simultâneas no mesmo símbolo independente da estratégia; `GOLDEN_POT.sl_atr_mult` calibrado de `1.0x` para `1.5x ATR` (alinhado ao `trail_atr_mult = 1.5x ATR`), absorvendo retestes naturais de rompimento em altcoins (Win Rate subiu de 33,3% para 50,0% no replay 5m).
  5. **Escalonamento de `ATR_PCT_EXTREME` por Timeframe e Sincronismo UTC no `restore_state` (`leverage_policy.py`, `smart_trade.py`):** Limiar de volatilidade extrema ajustado por timeframe (`3m`: `1.2%`, `5m`: `1.5%`, `15m`: `2.0%`, `1h`: `3.0%`, `4h`: `5.0%`) e `current_day_utc` sincronizado na restauração.
  6. **Consolidação por `position_id` no `PerformanceTracker` e `LearningTracker` (`performance_tracker.py`, `smart_trade.py`):** `LearningTracker` passa a receber `pos.realized_pnl` (acumulado das parciais) em vez do `net_pnl` apenas da última perna (`BREAKEVEN_STOP -$0.05`), e `PerformanceTracker.compute_metrics_from_db()` agrega por `position_id` (excluindo `FUNDING` isolado e posições abertas), impedindo que trades vencedores em `TP1 + Breakeven` sejam contabilizados como derrotas no `RollbackGuard`.
- **Isolamento de Velas Pré-Entrada (2026-10-16):** Em `smart_trade.py` (`update_market_tick`) e `scripts/run_paper_simulation.py` (`_update_symbol`), corrigido vazamento temporal onde candles anteriores à abertura da posição (`bts < entry_timestamp`) avaliavam a escada de saída e disparavam falso Stop Loss em trades de rompimento (ocorrido em `BTWUSDT`); barras pré-entrada agora são estritamente filtradas no polling e bloqueadas na ladder; testes de regressão adicionados; trava de cooldown do RollbackGuard liberada.
- **Lookahead V1-V8:** CPI no dia 20 do mês seguinte; FEDFUNDS no fim do mês+1; removido `bfill`; snapshots macro/order book só na última barra; `as_of_date` em VIX/DXY/SP500; candle em formação descartado. V9 corrigido com atualização de trailing apenas em barras fechadas.
- **Execução:** taxa de saída em dobro (`smart_trade.py`); cota diária reaberta na virada UTC; teto de margem no carry; race em `update_market_tick`; TP1 parcial duplicado; `entry_fees` em `funding_carry`.
- **Paper trading:** `restore_state()` (ledger, `MANUAL_RESET`); swing com 400 candles + warmup `sma_200` (viés comprador); A3 (sem SL/TP com dados pré-entrada); A1 (SL/BE/MANUAL como taker 0,06% + 0,03% slippage); win_rate por `position_id`; `position_size` em USD no swing.
- **Decisão:** NaN→confiança, peso de fallback, validação de LLM, divisão por zero em `signal_generator`.

### Adicionado
- `data/db_maintenance.py` (checkpoint WAL; purge de símbolos esparsos em dry-run).
- Supervisor: healthcheck 5s, 3 falhas, re-adoção de PID; `logs/flask.log`.
- `docs/CRITERIOS_APROVACAO_PILOTO.md` e `scripts/pilot_report.py`.
- Suíte backend: **544 testes passando** (`tests/test_pre_entry_regression.py` expandido com regressão das 6 correções Argon).

### Pendente / Propostas
- V9; ajustes de consenso (agentes neutros, vetos inertes, double counting) aguardam backtest A/B.
- Piloto pré-reset REPROVADO (35 trades, expectância bruta -0,995%, viesado); piloto limpo desde o reset de 2026-10-14/15, parâmetros congelados >= 56 dias. Alfa NÃO demonstrado.

---

## [1.0.0] - 2026-10-14 - Release Oficial (10 Fases Homologadas)

### Adicionado
- **Ponto de Entrada Unificado (`launcher.py`):**
  - CLI com suporte a flags operacionais: `--db`, `--host`, `--port`, `--headless`, `--no-trader`, `--paper-only`.
  - Orquestração simultânea e resiliente de threads (Paper Trading, TaskScheduler e Servidor Web Flask).
  - Encerramento gracioso via sinais `SIGINT` e `SIGTERM` com `PRAGMA wal_checkpoint(TRUNCATE)` e `PRAGMA optimize`.
- **Empacotamento Standalone com PyInstaller (`ia_trader.spec`):**
  - Binário executável independente autocontido (`dist/ia_trader/ia_trader` - 21.5 MB).
  - Coleta automática e empacotamento de templates web, licença e dependências analíticas (`lightgbm`, `sklearn`, `ta`, `aiohttp`, `flask`).
  - Suporte transparente a `sys._MEIPASS` para extração e renderização de templates em tempo de execução.
- **Suíte de Testes de Hardening e Concorrência (`tests/test_hardening_simulation.py`):**
  - Validação de estresse com 6 threads concorrentes realizando 120 escritas em SQLite WAL sem locks.
  - Testes de disparo de Circuit Breakers (cooldown por 4 perdas consecutivas) e recuperação supervisionada.
  - Elevação da suíte global para **208/208 testes com 100% de aprovação (Zero falhas, 0:01:12)**.

---

## [1.0.0-alpha.8] - 2026-10-14

### Adicionado
- **Módulo Scanner de Pares e Volatilidade Anômala (`pair_scanner.py`):**
  - Detecção precoce de anomalias de volatilidade (ATR relativo $\ge 2.0\times$) e surtos anormais de volume ($\ge 2.5\times$).
  - Detecção de colapso de preços e risco crítico de delistagem ($\le -35\%$), atribuindo isolamento e $0\%$ de alocação de capital.
  - Alocação isolada e independente de capital de risco para pares voláteis (teto de $5\%$).
- **Módulo Educacional Interativo e Didática de Trading (`educational_module.py`):**
  - Tradução e explicabilidade em linguagem simples de operações quantitativas ("Por que a IA comprou/vendeu?").
  - Análise pedagógica de risco, dimensionamento de Stop Loss por ATR e assimetria R:R.
  - Glossário quantitativo completo (`ORDER_BLOCK`, `FAIR_VALUE_GAP`, `LIQUIDITY_SWEEP`, `ATR`, `FUNDING_RATE`, `CIRCUIT_BREAKER`, `RISK_REWARD`).
- **Módulo de Comandos de Voz e Linguagem Natural (`voice_commands.py`):**
  - Roteador semântico para comandos textuais ou via voz ("status do robô", "mostrar últimas operações", "pausar operações", "retomar operações", "qual o regime do mercado", "explicar último trade").
  - Integração direta com `RollbackGuard` para bloqueio emergencial e `PerformanceTracker` para telemetria.
- **Módulo de Gestão de Usuários e Controle de Acesso (`auth_module.py`):**
  - Armazenamento com hash PBKDF2-HMAC-SHA256 (100.000 iterações + salt criptográfico de 16 bytes).
  - RBAC com três níveis de assinatura (`FREE`, `PRO`, `PREMIUM`) com controle estrito de pares, timeframes, golden pots e comandos de voz.
  - Rastreabilidade de consentimento dos termos de uso (LGPD / GDPR).
- **Módulo Legal & Compliance:**
  - `LICENSE.txt`: licença de software e termo de isenção de recomendação financeira.
  - `templates/terms_of_use.html` e `templates/privacy_policy.html`: páginas de termos e privacidade.
- **Rotas REST Integradas no Flask (`flask_interface.py`):**
  - `/terms`, `/privacy`, `/api/educational/glossary`, `/api/educational/explain`, `/api/voice/command`, `/api/scanner/opportunities`.
- **Suíte de Testes da Fase 9 (`tests/test_auxiliary_modules.py`):**
  - 21 novos testes automatizados elevando a suíte para **205/205 testes passando (100% Green)**.

---

## [1.0.0-alpha.7] - 2026-10-14

### Adicionado
- **Módulo de Memória Cognitiva & Aprendizado Adaptativo (`learning_tracker.py`):**
  - Tabela `agent_learning_memory`: armazena deliberações com voto de cada agente, regime de mercado e desfecho real de trade (Win, Loss, PnL).
  - Tabela `agent_regime_weights`: calibração dinâmica de pesos do comitê com base em acertos empíricos e Brier score por regime de mercado.
  - Classe `ModelEvolutionManager`: protocolo Champion-Challenger para retreinamento contínuo de modelos sem parada de sistema (hot-swap seguro com exigência de ganho $\ge 0.5\%$ em ROC-AUC/Accuracy).
- **Módulo Agendador de Tarefas Periódicas (`scheduler.py`):**
  - Loop assíncrono para recalibração de pesos de agentes (a cada 30 min).
  - Loop assíncrono para checagem de retreinamento de modelos (a cada 6h / diário).
  - Manutenção periódica do banco SQLite (`PRAGMA wal_checkpoint` e `optimize` diário).
- **Módulo de Telemetria Financeira (`performance_tracker.py`):**
  - Fonte única da verdade para PnL líquido, equity, pico de capital, Maximum Drawdown ($ e %), Win Rate, Profit Factor, Payoff Ratio, sequências (streaks) e índices de Sharpe, Sortino e Calmar.
  - Decomposição analítica de performance por estratégia (`SCALP`, `GOLDEN_POT`, `CARRY`).
  - Persistência de snapshots históricos na tabela `performance_snapshots`.
- **Módulo de Estabilidade e Circuit Breakers (`rollback_guard.py`):**
  - Circuit Breaker de Drawdown Total (10.0%): trava automática de novas ordens para preservação de capital.
  - Circuit Breaker de Perdas Consecutivas (4 stops): cooldown obrigatório de 2h contra overtrading e desvio de regime.
  - Auto-Rollback por Degradação: detecção de queda crítica de Win Rate (< 30%) e reversão para parâmetros ultraconservadores.
  - Parada manual de emergência e reset supervisionado com trilha de auditoria na tabela `circuit_breaker_events`.
- **Novos Endpoints REST no Dashboard Flask (`flask_interface.py`):**
  - `GET /api/telemetry/performance`: métricas financeiras completas.
  - `GET /api/telemetry/circuit-breaker`: estado ao vivo do guardião (`CLEAR`, `COOLDOWN`, `TRIPPED`).
  - `POST /api/telemetry/circuit-breaker/reset`: reset supervisionado de circuit breaker.
  - `POST /api/telemetry/emergency-stop`: acionamento imediato de parada de emergência.

### Modificado
- `smart_trade.py`: integrados os gates de segurança de circuit breaker em `can_execute_scalp()` e `can_execute_golden_pot()`; acionamento de telemetria e memória cognitiva ao encerrar trades em `close_position()`.
- `core.py`: incorporada a busca de pesos dinamicamente calibrados pelo regime de mercado no `TradingCore`.

---

## [1.0.0-alpha.6] - 2026-10-14

### Adicionado
- **Motor de Execução Smart Trade (`smart_trade.py`):**
  - Execução estrita em modo de simulação (Paper Trading) com bloqueio mandatório de ordens com capital real.
  - Separação e independência de cotas diárias: Scalps intradiários padrões (mínimo 2, teto 5/dia) e Golden Pots de alta convicção (máximo 3/dia) rodando de forma desacoplada sem canibalização.
  - Sistema de saídas parciais escalonadas (TP1 com realização de 50% + ajuste automático de Stop Loss para o Breakeven + TP2 para surfar expansões de tendência).
  - Otimização com compressão inteligente de snapshots de portfólio no SQLite (evita mais de 4.300 gravações redundantes por dia).
- **Daemon de Paper Trading Assíncrono (`scripts/run_paper_simulation.py`):**
  - Execução simultânea não-bloqueante (`asyncio.gather`) para múltiplos pares da Bitget/Binance.
  - Classificação macro de regime via `RegimeClassifier` (`BULL_TREND`, `BEAR_TREND`, `SIDEWAYS_RANGE`, `HIGH_VOLATILITY_PANIC`).

---

## [1.0.0-alpha.5] - 2026-10-14

### Adicionado
- **Simulador de Backtesting Institucional (`backtesting.py`):**
  - Modelagem estrita de microestrutura de custos: taxas maker (0.02%), taker (0.06%), slippage (0.02% a 0.03%) e rolagem de funding rate a cada 8h.
  - Colisão intra-barra conservadora (Stop Loss assumido primeiro em caso de overlap).
- **Módulo de Rigor Quantitativo (`statistical_rigor.py`):**
  - Cálculo do Deflated Sharpe Ratio (DSR - López de Prado) ajustado pelo número de testes múltiplos $N$.
  - Combinatorially Symmetric Cross-Validation (CSCV) para cálculo da Probability of Backtest Overfitting (PBO).
  - Registro atômico de ensaios e tentativas em `datasets/reports/trial_log.csv`.
- **Estratégias Institucionais Estruturais:**
  - Simulador de Funding Carry delta-neutro spot + perp (`funding_carry.py`).
  - Time Series Momentum com volatilidade alvo e banda de no-trade (`trend_following.py`).

---

## [1.0.0-alpha.4] - 2026-10-14

### Adicionado
- **Motor de Estrutura de Mercado & POI (`data/market_structure.py`):**
  - Algoritmo causal de identificação de fractais de Swing High / Swing Low (janela causal $N=3$).
  - Detecção de Fair Value Gaps (FVGs) e Order Blocks (OBs) de ignição institucional com deslocamento $\ge 1.5\times$ ATR.
  - Piscinas de liquidez Buy-Side (BSL) e Sell-Side (SSL) e detecção algorítmica de Liquidity Sweeps (varredura de stops com rejeição).
  - Distância percentual contínua para o suporte mais próximo (`dist_support_pct`) e resistência (`dist_resistance_pct`).
- **Comitê de Decisão Multi-Agente (`agents/`):**
  - `TechnicalAgent`: análise de Machine Learning tabular + confluência com POI e Sweeps.
  - `OrderflowAgent`: desbalanceamento de livro L2 e volume profile.
  - `SentimentAgent`: sentimento contrariano Fear & Greed e notícias CryptoPanic indexadas rigorosamente em D-1.
  - `MacroAgent`: correlações macroeconômicas DXY, VIX e S&P 500 em D-1.
  - `OnchainAgent`: fluxos de stablecoins e TVL DefiLlama em D-1.
  - `RiskSentinelAgent`: poder de veto absoluto por volatilidade anômala, iliquidez, spread alargado e proximidade de resistência/suporte imediato.
- **Juiz de Consenso Ponderado (`consensus_judge.py`):**
  - Ponderação matemática com precedência estrita de veto e timeout de 2.5s para árbitro LLM com fallback local.

---

## [1.0.0-alpha.3] - 2026-10-14

### Adicionado
- **Motor de Treinamento de Modelos (`model_trainer.py`):**
  - Modelos LightGBM e Random Forest com Triple-Barrier labeling (TP, SL e barreira temporal).
  - Validação cruzada temporal (*PurgedGroupTimeSeriesSplit*) com janela de embargo para impedir vazamento de autocorrelação.

---

## [1.0.0-alpha.2] - 2026-10-14

### Adicionado
- **Engenharia de Features Locais (`data/feature_engineering.py`):**
  - 100% vetorizado em `ta` e `numpy` para baixa latência (< 10ms por lote de 500 velas).
  - Indicadores de tendência (SMA 20/50/200, EMA 9/21, MACD), osciladores (RSI 14, Estocástico), volatilidade (ATR 14, Bandas de Bollinger, Bandwidth, NATR).
  - Microestrutura: OBV, Z-score de volume de 20 períodos, desvio de VWAP e imbalance de livro de ofertas.
  - Fatores macroeconômicos alinhados rigorosamente por release_date (anti-lookahead).

---

## [1.0.0-alpha.1] - 2026-10-14

### Adicionado
- **Camada de Dados 100% Gratuita (`data/data_integrations.py`):**
  - Conectores assíncronos independentes para Bitget v2 pública, Binance pública (data.binance.vision), CoinGlass scraping, FRED, Yahoo Finance REST, CryptoPanic e DefiLlama.
- **Gerenciador de Cache Local Multi-Cadência (`data/data_loader.py`):**
  - Cache persistente SQLite em modo WAL (`datasets/market_cache.db`).
  - TTL configurável por cadência de dados e ordenação temporal cronológica estrita.

---

## [1.0.0-alpha.0] - 2026-10-14

### Adicionado
- Setup inicial da arquitetura, governança e definição formal da **Regra de Ouro (Golden Rule)**.
- Documento de rastreabilidade [EVOLUTION_LOG.md](EVOLUTION_LOG.md) e cronograma de 10 fases [CRONOGRAMA_MODULAR.md](CRONOGRAMA_MODULAR.md).
