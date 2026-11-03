# 🤖 Revisão Arquitetural e Quantitativa do IA Trader — MiniMax M3.1 Flash (428B MoE)

> **Data da Análise:** 2026-10-21 18:09:33
> **Modelo Avaliador:** `minimax-m3.1-flash:cloud` (428B MoE, 1M Context)
> **Tempo de Execução:** 511.0s

---

# Parecer arquitetural e quantitativo — IA Trader

## Escopo desta revisão

A análise considera integralmente o código fornecido na conversa. Entretanto, há módulos citados na documentação que **não foram incluídos**, entre eles:

- `flask_interface.py`;
- `data/data_integrations.py`;
- `data/data_loader.py`;
- `data/feature_engineering.py`;
- `learning_tracker.py`;
- `model_trainer.py`;
- `db_conn.py`;
- `launcher.py`;
- `scripts/run_paper_simulation.py`;
- `scripts/microstructure_collector.py`;
- `chatgpt_assistant.py`;
- suíte de testes e artefatos do piloto.

Por isso, conclusões sobre esses componentes devem ser classificadas como **“não demonstradas no código fornecido”**, e não necessariamente como inexistentes.

Uma conclusão já pode ser feita com segurança: **o código fornecido é um sistema avançado de paper trading e pesquisa quantitativa, mas ainda não é um sistema comprovadamente apto a operar capital real**.

---

# 1. Parecer Executivo e Avaliação de Maturidade

## 1.1 Conclusão executiva

O projeto apresenta uma base de engenharia quantitativa substancial eSeveralGood practices:

- comitê multiagente com veto;
- backtesting temporal;
- políticas de alavancagem;
- circuit breakers;
- persistência de estado;
- instrumentação de sinais;
- tentativa explícita de paridade entre backtest e piloto;
- acknowledge formal de que **o alfa ainda não foi demonstrado**.

Entretanto, os bloqueadores abaixo são suficientes para um **NO-GO absoluto para capital real**:

1. `PaperTradingEngine.__init__()` bloqueia explicitamente `simulation_mode=False` com `PermissionError`; portanto, o código fornecido não possui executor live.
2. Não há, no código fornecido, implementação da transferência de lucros de futuros para spot, apesar de essa capacidade ser apresentada na README.
3. O comitê pode aprovar uma operação com **um único agente direcional** e chegar a confiança `0.99`.
4. `RollbackGuard.max_daily_drawdown_pct` é configurado, mas **não é efetivamente avaliado**.
5. O “auto rollback” apenas registra uma flag; ele não aplica pesos defensivos nem impede operações.
6. A persistência de posição e trade usa duas transações SQLite independentes, podendo gerar posições órfãs.
7. O reset manual preserva o histórico no banco, mas `PerformanceTracker` continua calculando métricas sobre o histórico anterior.
8. Não há OMS live, idempotência de ordens, reconciliação de交易所, state machine de ordens, tratamento de fills parciais ou recuperação de ordens órfãs.
9. O piloto anterior foi reprovado e a própria README declara: **“Alfa NÃO demonstrado”**.
10. O gate de Tape Reading não está dentro do núcleo de decisão nem no boundary final de execução mostrado em `smart_trade.py`.

### Veredito operacional

| Ambiente | Veredito | Condição |
|---|---:|---|
| Pesquisa/backtest | **GO** | Com correções de paridade e registro estatístico |
| Paper trading assistido | **GO com controles** | Parâmetros congelados, supervisão humana e ledger auditável |
| Shadow mode com sinais reais | **GO condicional** | Após freshness, gates e persistência atômica |
| Testnet | **NO-GO no código atual** | Falta o executorlive/testnet e o OMS |
| Capital real limitado | **NO-GO** | Todos os P0 obrigatórios e evidência OOS convincing |
| Produção institucional | **NO-GO** | Há maturidade de research, não de operação institucional |

A recomendação imediata é: **manter em paper trading assistido, congelar a estratégia e usar o período para construir evidência estatística e o OMS**, sem promover o sistema现有的 paper logic to live.

---

## 1.2 Avaliação de maturidade

Escala:

- **1 — conceitual**
- **2 — protótipo**
- **3 — paper/testnet**
- **4 — live limitado**
- **5 — institucional**

| Dimensão | Nota | Evidências positivas | Lacunas decisiva |
|---|---:|---|---|
| **Modelagem Quant** | **2,5/5** | Walk-forward, embargo, triple barrier, DSR, PBO, bootstrap, TSMOM, carry, custos e colisão intrabar conservadora | Alfa não demonstrado; piloto anterior negativo; inconsistências de paridade; confiança não calibrada; métricas de carry e TSMOM com defeitos |
| **Governança de Risco** | **2,5/5** | Veto do Risk Sentinel, modos de pausa, cotas, limites de margem, leverage cap, circuit breaker, logs de evolução | Drawdown diário não implementado; auto rollback sem efeito real; LLM com autoridade de direção; quorum ausente; reset manual insuficiente |
| **Integridade de Dados** | **2,0/5** | Cache local, WAL, histórico Binance Vision, validação cruzada declarada, controles de release date declarados | Freshness não é gate central; fail-open em Tape Reading; VWAP Bitget aproximado; classificação de 24h incorreta no scanner; connectors não fornecidos para auditoria |
| **Resiliência Operacional** | **1,5/5** | `restore_state()`, supervisor declarado, circuit breaker, manutenção WAL | Não há OMS, reconciliação, failover, heartbeat WS, backfill, tratamento de ordem órfã ou adapter live comprovado |
| **Concorrência/Execução** | **2,0/5** | `threading.RLock`, WAL, gravação por `position_id`, funil e flow gate logs | `RLock` não protege múltiplos processos; transações não atômicas; scheduler sem singleton visível; `asyncio.gather` não paraleliza I/O síncrono |
| **Segurança de Capital** | **1,5/5** | Simulação estrita, limites de margem, quotas, SL antes da liquidação, halt manual | Drawdown diário ausente; carry ignora breakers; liquidação aproximada; sem proteção contra perda de dados; sem execução real segura |
| **Maturidade global live** | **2,1/5** | Research-grade avançado | Ausência de prova de alfa e de infraestrutura operacional de execução |

### Observação sobre os testes

A documentação apresenta números conflitantes:

- `276/276` na tabela de fases da README;
- `544 passed` na seção de execução;
- `208/208` no `CRONOGRAMA_MODULAR.md`.

Mesmo que todos esses testes passem, número de testes não comprova:

- alfa;
- paridade live/backtest;
- robustez contra APIs reais;
- idempotência de ordens;
- reconciliação;
- comportamento em crash;
- segurança de capital.

É necessário vincular cada release a um commit, artefato de build, seed, dataset hash e relatório de cobertura.

---

## 1.3 A meta de 70% de acerto não é suficiente

A assertividade superior a 70% pode ser um objetivo de modelo, mas não é um gate de viabilidade econômica.

Para uma operação com payoff de 2,5:1, o ponto de equilíbrio puro é:

\[
p_{BE}=\frac{1}{1+2{,}5}\approx 28{,}6\%
\]

Antes de taxas, funding e slippage. Portanto:

- 70% de acerto com payoff de 0,8:1 pode ser lucrativo;
- 70% de acerto com payoff de 1,2:1 e custos relevantes pode ser prejuízo;
- a ladder com parciais altera a distribuição de retorno e não pode ser avaliada apenas por win rate.

O gate correto deve ser:

\[
E[\text{PnL líquido}] > 0
\]

com limite inferior do intervalo de confiança positivo, após:

- taxas;
- slippage;
- funding;
- market impact;
- latência;
- seleção de parâmetros;
- dependência serial;
- múltiplos regimes.

A amostra de 35 trades mentioned na README é insuficiente. Mesmo hipoteticamente 24 acertos em 35, o intervalo Wilson de 95% seria aproximadamente **53% a 83%**, não “70% comprovado”.

Para promoção, recomendo simultaneamente:

- mínimo de 200–300 trades OOS independentes;
- pelo menos 8–12 semanas de paper;
- pelo menos dois regimes de mercado;
- DSR ≥ 0,95;
- PBO ≤ 0,20, mais estrito que o threshold atual de 0,30;
- limite inferior do block bootstrap da expectancy > 0;
- profit factor OOS líquido consistentemente > 1,20;
- stress com slippage e taxas pelo menos 2× e 1,5×, respectivamente;
- nenhum ativo/regime individual respondendo por mais de 30% do PnL.

---

# 2. Pontos Fortes

## 2.1 Governança multiagente e veto absoluto

O veto é implementado em dois níveis:

- `TradingCore.evaluate_market_deterministic()` e `evaluate_market_async()` avaliam `RiskSentinelAgent` antes dos demais agentes;
- `ConsensusJudge.judge_deterministic()` e `judge_async()` verificam `sig.is_veto` antes do cálculo de consenso.

Isso é uma boa separação: o sentinela não é “outro voto com peso menor”, mas uma barreira absoluta.

O `RiskSentinelAgent` contém vetoes quantitativamente relevantes:

- spike de ATR;
- queda abrupta de volume;
- spread excessivo;
- janela de evento macro;
- NATR extremo;
- funding dislocation;
- proximidade estrutural de suporte/resistência.

A分離 entre `direction="HOLD"` e `is_veto=True` também é conceitualmente correta.

## 2.2 Separação entre modo backtest e execução

A documentação e o código Alinham several claims of rigor:

- `ConsensusJudge` não utiliza LLM no backtest;
- `evaluate_market_deterministic()` é reprodutível;
- `SignalGenerator` rejeita preço, ATR e direção inválidos;
- `BacktestEngine.run()` executa sinais no open da barra seguinte;
- colisões SL/TP intra-bar são resolvidas conservadoramente com SL primeiro;
- trailing de `BacktestEngine._update_trailing()` só usa uma barra já observada e vale a partir da seguinte.

Essa é uma base conceitual correta para mitigação de lookahead bias.

## 2.3 Statistical rigor工具

`statistical_rigor.py` inclui metodologias relevantes:

- `deflated_sharpe_ratio()`;
- `probability_of_backtest_overfitting()` por CSCV;
- `bootstrap_performance_metrics()`;
- block bootstrap;
- `trial_log`;
- `fcntl.flock()` para append concorrente.

A presença de DSR, PBO e registro de tentativas é um diferencial importante frente a sistemas baseados apenas em backtest e acerto total.

## 2.4 Gestão de risco e controle de alavancagem

Há várias proteções bem implementadas em paper:

- teto de 5x em `leverage_policy.MAX_LEVERAGE`;
- redução de alavancagem por ATR e desalinhamento de regime;
- `RiskSentinelAgent` com veto;
- `RollbackGuard` com drawdown total, cooldown e emergency halt;
- cotas separadas para SCALP e GOLDEN_POT;
- limite de posições simultâneas;
- uso de margem máxima de 85%;
- redução de alavancagem se o SL ficar além da liquidação;
- SL antes da liquidação em `_resolve_leverage()`;
- `EXIT_TP1_FRACTION`, `EXIT_TP2_FRACTION` e runner;
- consolidação de resultados por `position_id`.

A existência de `STRATEGY_PROFILES` centraliza melhor a mecânica de exit do que múltiplas implementações independentes.

## 2.5 Telemetria e auditabilidade

O funil `signal_funnel` e a tabela `flow_gate_log` são decisões de observabilidade bem conceived:

- `funnel_inc()` contabiliza stages sem interromper o trading;
- há flush periódico com UPSERT;
- `flow_gate_log` registra o voto de Tape Reading, features e decisão;
- `PerformanceTracker` consolida parciais e fechamentos por `position_id`;
- `RollbackGuard` registra eventos de circuit breaker;
- `EVOLUTION_LOG.md`, `trial_log.csv` e addenda应该有mm evidence de auditoria.

Esses mecanismos são úteis para a construção de um piloto realmente auditável.

## 2.6 Honestidade sobre a ausência de alfa

A README afirma explicitamente:

- “Alfa NÃO demonstrado”;
- piloto pré-reset reprovado;
- parâmetros congelados;
- V9 pendente;
- E4–E8 sem evidência prévia.

Essa é uma maturidade de governança positiva. O problema não é a admissão da ausência de alfa; é a coexistência com rótulos como “blindado”, “homologado” e “pronto”, que podem induzir uma decisão operacional indevida.

---

# 3. Pontos de Atenção, Vulnerabilidades e Riscos

## 3.1 Resumo dos bloqueadores críticos

| ID | Severidade | Achado | Evidência | Consequência |
|---|---:|---|---|---|
| C-01 | Crítica | Capacidade live e transferência para spot não existem no código fornecido | `PaperTradingEngine.__init__()` bloqueia live; nenhum executor spot/perp é mostrado | Incapacidade de operar capital real |
| C-02 | Crítica | Um único agente direcional pode aprovar uma operação | `ConsensusJudge.judge_deterministic()` | Falha da governança multiagente e confiança artificial |
| C-03 | Crítica | Drawdown diário configurado, mas não implementado | `RollbackGuard.evaluate_telemetry()` | Risco diário não limitado pelo guardião |
| C-04 | Crítica | Auto rollback é apenas cosmético | `RollbackGuard.is_rollback_active` não altera core/executor | Sistema continua operando após degradação |
| C-05 | Crítica | Posição e trade não são persistidos atomicamente | `_save_position()` e `_save_trade()` em commits separados | Posições órfãs e ledger inconsistente |
| C-06 | Crítica | Sem OMS live, idempotência ou reconciliação | Não há módulo de execução no código | Ordens duplicadas, fills perdidos, estado divergente |
| C-07 | Crítica | Reset manual não isola métricas históricas | `PerformanceTracker.compute_metrics_from_db()` não lê epochs | Piloto e dashboard podem reportar contaminated results |
| C-08 | Alta | Confiança de comitê usada como proxy de probabilidade real | `decide_leverage()` | Alavancagem baseada em score não calibrado |
| C-09 | Alta | Gate de Tape Reading ausente do núcleo e do boundary final | `TapeReadingAgent` não está em `TradingCore.agents`; `_open_ladder_position()` não exige gate | Entradas sem confirmação microestrutural |
| C-10 | Alta | Freshness/fail-closed não é regra central | Tape unavailable pode passar; Risk Sentinel com dados ausentes não veta | Decisões baseadas em dados antigos ou incompletos |
| C-11 | Alta | `trial_log` pode ser sobrescrito e contém linhas sintéticas | `initialize_trial_log(reset_if_missing=True)` | Auditoria de overfitting não confiável |
| C-12 | Alta | Carry live paper não possui pernas, basis e breakers completos | `execute_carry_allocation()` | Risco de basis, liquidação e concentração não contabilizado |
| C-13 | Alta | Segurança de credenciais apenas conceitual | README sugere token na URL do Git | Risco de vazamento de PAT e permissões |

---

## 3.2 Execução, ordens e conectividade

### 3.2.1 O “Smart Trade” fornecido é exclusivamente paper

`smart_trade.py` não contém:

- criação de ordens autenticadas;
- cancel/replace;
- consulta de status;
- fills parciais;
- reconciliação;
- cancelamento de ordens pendentes no restart;
- client order ID;
- treatment de rejected/expired orders;
- heartbeat WebSocket;
- gap recovery por sequência;
- backoff por `429`;
- proteção contra clock skew;
- segregação de permissões.

Consequentemente, os riscos classical de live execution **não foram resolvidos; eles ainda precisam de implementação**.

### 3.2.2 State machine de ordem necessária

A execução real deve passar obrigatoriamente por estados explícitos:

```text
SIGNAL_READY
  -> RISK_APPROVED
  -> ORDER_INTENT
  -> SENT
  -> ACKED
  -> OPEN / PARTIALLY_FILLED
  -> FILLED / CANCELED / REJECTED / EXPIRED
  -> CLOSED
  -> RECONCILED
```

Cada transição deve ser persistida com:

- `client_order_id` determinístico;
- `position_id`;
- `signal_id`;
- `strategy_id`;
- `account_id`;
- `venue`;
- `request_hash`;
- timestamp monotônico e timestamp de exchange;
- payload de request/response sanitizado;
- número de sequência;
- tentativa e idempotency key.

### 3.2.3 Ordens órfãs e reconciliação

No restart, o sistema deve:

1. listar posições abertas na exchange;
2. listar ordens abertas;
3. listar trades/fills desde o último cursor;
4. comparar com o ledger local;
5..cancelar ordens locais sem correspondência, se apropriado;
6. reconciliar fills recebidos enquanto o processo estava offline;
7. bloquear novas entradas até a reconciliação terminar.

Sem essa etapa, um restart após uma requisição HTTP ambígua pode duplicar a ordem.

### 3.2.4 Slippage não pode ser uma constante única

O projeto usa slippage fixo em `BacktestEngine` e `PaperTradingEngine`. Em produção, o custo deve depender de:

- notional;
- profundidade no nível;
- spread;
- volatilidade;
- side;
- hora;
- venue;
- latência;
- maker versus taker;
- taxa de fill.

Uma ordem de $500 em BTC não pode sofrer o mesmo slippage de $5.000 em altcoin ilíquido.

### 3.2.5 Maker fills Excessivamente otimistas

Em `_PilotSimulator._leg()` e `PaperTradingEngine._ladder_partial()`:

- TP maker preenche exatamente no target;
- não há probabilidade de fill;
- não há fila;
- não há partial fill;
- não há cancel por superseding;
- não há adverse selection.

Isso pode subestimar consistentemente o resultado. Para testnet/live, o modelo deve separar:

- fill taker;
- fill maker completo;
- fill maker parcial;
- miss por remoção da ordem;
- latency entre生成 ordem e chegada.

### 3.2.6 Transferência de lucros para spot

A README afirma que `smart_trade.py` transfere Internos de futures para spot antes da compra. O código fornecido **não contém essa transferência**.

Se implementada, a transferência deve:

- ser idempotente;
- ter reconciliação entre internal transfer e ledger;
- não ser contabilizada como PnL;
- preservar reserva de margem;
- evitar-transferências circulares;
- Possuir estado `TRANSFER_PENDING/RECONCILED/FAILED`;
- ser retry-safe;
- ter limite máximo por período;
- Não poder esvaziar a collateral needed for open positions.

---

## 3.3 Problemas quantitativos e de modelagem

### 3.3.1 Quorum inexistente: um único agente pode “aprovar” com 99% de confiança

Considere:

- `TechnicalAgent`: BUY, confiança `1,0`;
- `OrderFlowAgent`: HOLD;
- `RiskSentinelAgent`: HOLD, sem veto;
- agentes contextuais: HOLD/ABSTAIN.

Em `ConsensusJudge.judge_deterministic()`:

\[
score=\frac{1\cdot1{,}0}{1{,}0_{\text{sentinel}}+1{,}0_{\text{technical}}+0{,}8_{\text{flow}}}
\approx 0{,}556
\]

Como o threshold é apenas `0,30`, a operação é aprovada. A confidence formula produz:

\[
confidence \approx 0{,}99
\]

Isso viola diretamente o requisito de que nenhum trade seja decidido por agente isolado.

### Correção recomendada

O juiz deve exigir simultaneamente:

- `min_directional_agents >= 2`;
- `min_weighted_agreement`;
- participação mínima de agentes;
- directional score calculado apenas sobre agentes elegíveis;
- veto de evento/risco;
- freshness;
- Tape gate quando exigido;
- nenhuma diluição artificial por HOLD de agentes sem opinião.

Também é recomendável separar:

- `HOLD`: opinião contrária/neutra;
- `ABSTAIN`: sem dados ou fora de domínio.

Hoy `AgentSignal.__post_init__()` aceita apenas BUY/SELL/HOLD, tornando o ramo `ABSTAIN` do juiz efetivamente inalcançável.

### 3.3.2 LLM pode transformar HOLD em BUY

`ConsensusJudge.judge_async()` aceita uma resposta LLM com `direction="BUY"` mesmo quando o comitê local produziu HOLD.

Além disso:

- o timeout padrão é **8,5 segundos**, apesar de a documentação afirmar 2,5 s;
- o comentário menciona 4,5 s;
- o LLM pode receber rationale de conteúdo externo, sujeito a prompt injection;
- a confidence retornada pelo LLM não é calibrada;
- não há minimum independent votes.

Para uma estratégia de 3m/5m, 8,5 s é material. A recomendação é:

- remover o LLM do caminho autoritativo;
- deixá-lo como auditor/advisor assíncrono;
- timeout máximo de 250–500 ms se ele precisar participar;
- schema estrito e validação;
- nunca permitir que o LLM crie direção sem quorum quantitativo.

### 3.3.3 Confidence não é probabilidade de acerto

`ConsensusDecision.confidence` é uma fórmula de participação,Agreement e convicção. Não é:

- probabilidade de o trade ser lucrativo;
- probabilidade de acerto;
- confidence interval;
- score de calibração.

Mesmo assim, `decide_leverage()` mapeia esse número diretamente para:

- 2x a partir de 0,60;
- 3x a partir de 0,70;
- 4x a partir de 0,80;
- 5x a partir de 0,90.

É preciso calibrar a confidence em fold OOS, por exemplo via isotonic regression ou Platt scaling, e usar o expected shortfall real para dimensionar.

### 3.3.4 Dimensionamento com dimensionalidade inconsistente

Em `SignalGenerator.generate()`:

```python
desired_size = dollar_risk / sl_distance
```

`position_size` é quantidade do ativo base.

Em `PaperTradingEngine._lot_margin()`:

```python
signal.position_size / lev_req
```

é interpretado como margem em USD. Isso mistura:

- quantidade de ativo;
- notional;
- margem.

O bug fica mascarado pelo fato de `paper_lot_usd` normalmente limitar o resultado. Deve existir um contrato único, por exemplo:

```python
RiskOrder:
    side
    quantity
    notional
    margin
    stop_price
    expected_loss_including_costs
    leverage
    liquidation_price
```

### 3.3.5 R:R documentado não corresponde ao default

`SignalGenerator` usa:

- TP: `1.5 × ATR`;
- SL: `1.0 × ATR`;

portanto R:R default de **1,5:1**, não 2,5:1.

A ladder de `STRATEGY_PROFILES` tem outra economics. Isso precisa ser reconciliado em uma tabela única, distinguish:

- payoff do TP1;
- payoff do TP2;
- payoff do TP3;
- perda no stop;
- probability-weighted expected payoff.

### 3.3.6 Drawdown diário ausente

Em `RollbackGuard.evaluate_telemetry()`:

- existe `current_dd`;
- existe `max_dd`;
- existe `max_daily_drawdown_pct`;
- mas não existe baseline de equity no início do dia;
- não há branch de daily drawdown.

Portanto, o limite de 5% diário prometido na documentação não funciona.

É necessário registrar:

- `day_start_equity`;
- `high_water_equity_day`;
- `realized_pnl_day`;
- `unrealized_pnl_day`;
- `fees_day`;
- drawdown intraday e desde o início do dia.

### 3.3.7 Auto rollback não altera a operação

Após detectar degradação, `RollbackGuard` apenas:

```python
self.status.is_rollback_active = True
```

e grava log. Não:

- muda `ConsensusJudge.min_consensus_threshold`;
- reduz pesos;
- reduz lote;
- desliga GOLDEN_POT;
- reduz leverage;
- bloqueia novos trades.

A função `can_open_new_position()` sequer trata `is_rollback_active` como bloqueio.

O rollback deve emitir um objeto de política versionado, por exemplo:

```python
RiskPolicy(
    min_confidence=0.80,
    max_leverage=2.0,
    lot_multiplier=0.50,
    allowed_strategies={"DIRECTIONAL"},
    require_tape_gate=True,
)
```

O executor deve consumir esse objeto, e não apenas registrar a intenção de rollback.

### 3.3.8 Carry não é realmente delta-neutro no engine

`execute_carry_allocation()` cria um `PaperPosition` com:

- side `DELTA_NEUTRAL_CARRY`;
- apenas `quantity=spot_qty`;
- sem `perp_quantity`;
- sem basis;
- sem preço de liquidação;
- sem reconciliação das duas pernas;
- sem circuit breaker;
- sem limite de posições simultâneas;
- sem PnL de basis.

No fechamento, o gross PnL é apenas `pos.funding_collected`.

Isso modela um receipt de funding, não um carry delta-neutro institucional.

### 3.3.9 Carry simulator usa stress test hardcoded

`FundingCarrySimulator._evaluate_stress_shock_30pct()` assume:

- capital de $10.000;
- collateral de $5.000;
- notional de $5.000.

Se `initial_capital` for outro, o stress continua usando $5.000. O stress:

- não usa o estado real da posição;
- ignora `leverage_perp`;
- não testa queda de -30%, apesar da documentação;
- não modela manutenção e liquidação completas;
- não considera basis adverso.

### 3.3.10 TSMOM não registra trades short

`DailyTSMOMStrategy.run_backtest()` só rastreia posições quando `w > 0`. Em `mode="long_short"`, posições negativas produzem retornos, mas não entram em `TSMOMTradeRecord`.

Consequentemente:

- win rate;
- quantidade de trades;
- duração;
- payoff;

ficam incorretos para a estratégia short.

### 3.3.11 Regime classifier usa médias incompletas como SMA200

`MarketRegimeClassifier.classify_from_df()` aceita DataFrames com apenas 20 barras e calcula:

```python
sma200 = close.rolling(200, min_periods=20).mean()
```

Assim, com 20 barras, a “SMA200” é na verdade uma média de até 20 observações.

Isso distorce regimes em backtests curtos. Deve exigir histórico mínimo real ou marcar a classificação como insuficiente.

### 3.3.12 Tape Reading: escala de microprice incorreta e fallback stale

Em `TapeReadingAgent`:

```python
mdev_norm = max(-1.0, min(1.0, mdev))
```

`mdev` está em bps. Assim, apenas 1 bp de microprice deviation satura o componente em `1.0`,_memory contributing `0.10` ao flow score. Isso é equivalente a uma ponderação excessiva para desvios muito pequenos.

Deve-se normalizar, por exemplo:

\[
micro\_z = \frac{microprice-mid}{spread}
\]

ou por volatilidade esperada.

No fallback Binance:

- `aggTrades limit=500` pode não cobrir cinco minutos;
- quando isso ocorre, o código usa as últimas 50 trades sem validar idade;
- ainda assim retorna status `OK` e timestamp atual.

Isso pode transformar ticks velhos em confirmação contemporânea.

---

## 3.4 Backtesting e rigor estatístico

### 3.4.1 Pontos positivos

`BacktestEngine.run()` apresenta:

- sinal em `t`;
- entrada no open de `t+1`;
- colisão SL/TP conservadora;
- ATR conhecido no momento da decisão;
- funding;
- fees;
- slippage;
- trailing sem usar o futuro;
- semanticamente explícita de `pilot_parity`.

Essas são foundations importantes.

### 3.4.2 Limitações que ainda impedemLIVE

O backtest não modela adequadamente:

- fills maker condicionais;
- ordens parciais;
- latência variável;
- rejeição por rate limit;
- market impact;
- slippage por liquidez;
- disconnect/reconnect;
- imbalance do livro no momento exato;
- spread widening;
- manutenção tier da exchange;
- bankrupt/liquidation fee;
- borrow ou settlement;
- token collateral haircuts;
- difference de venue;
- disponibilidade de funding;
- circuit breaker temporal;
- exatamente o mesmo gate de Tape Reading;
- exatamente o mesmo singleton/lock operacional.

O próprio `_PilotSimulator` declara que `RollbackGuard` não é modelado. Portanto, a cauda de drawdown do paper/backtest pode ser subestimada.

### 3.4.3 O threshold de leverage por timeframe não é usado no backtest

Em `_PilotSimulator.try_open()`:

```python
lev_req = lp.decide_leverage(
    float(conf),
    atr_pct=atr_pct,
    regime_alignment=regime_ok,
)
```

Não é passado `timeframe`. Assim, `ATR_PCT_EXTREME_BY_TF` não é aplicado no backtest, embora o live caller possa aplicá-lo.

É uma quebra de paridade entre backtest e execução.

### 3.4.4 DSR/PBO ainda não cobrem toda a governança

`statistical_rigor.py` oferece as ferramentas, mas há deficiências:

- `calculate_annualized_sharpe()` não ajusta autocorrelação;
- o bootstrap iid é o default;
- retornos com `inf` não são filtrados;
- não há convenção explícita entre retornos decimais e percentuais;
- `variance_of_trials` pode estar em unidade diferente da observada;
- não há CPCV, White Reality Check ou SPA;
- não há integração obrigatória do DSR/PBO ao gate de promoção.

### 3.4.5 `trial_log.csv` compromete a auditoria

`initialize_trial_log(reset_if_missing=True)`:

1. ignora o nome “reset_if_missing”;
2. se o arquivo já existe, pode sobrescrevê-lo;
3. cria 56 linhas sintéticas com PnL, Sharpe e win rate fixos;
4. marca essas linhas como `is_oos=True`.

Isso não é um registro empírico de tentativas. Um trial log institucional deve ser append-only e conter:

- hash do dataset;
- hash do código;
- configuração;
- seed;
- período exato;
- número de features;
- data de início/fim;
- são ou não OOS;
- todos os experimentos, inclusive negativas;
- decisão de promoção;
- operador/agente responsável.

### 3.4.6 Métricas artificiais

Em `BacktestEngine` e `PerformanceTracker`, quando não há perdas, o Profit Factor é representado por `99.0`. Isso é enganoso: o valor correto é infinito ou `None`.

O campo `f1_score` de `BacktestEngine` permanece em `0.0` por falta de labels, o que é correto conceitualmente, mas deve ser `Optional[float] = None`, não zero.

---

## 3.5 Concorrência, estado e SQLite

### 3.5.1 `RLock` é correto dentro do processo, mas insuficiente entre processos

`PaperTradingEngine` utiliza `threading.RLock` em:

- `equity`;
- `margin_used`;
- `available_margin`;
- `can_execute_scalp()`;
- `can_execute_golden_pot()`;
- `_open_ladder_position()`;
- `close_position()`;
- `update_market_tick()`.

Isso protege o estado em memória do engine dentro de uma única instância.

Não protege:

- dois processos do scheduler;
- dashboard com escrita indevida;
- training job;
- supervisor que reinicia o daemon;
- outro launcher apontando para o mesmo DB.

O lock de arquivo mencionado na README não aparece em `scheduler.py`. Ele pode estar em `launcher.py`, não fornecido. Isso precisa ser provado por um teste de dois processos.

### 3.5.2 O scheduler não implementa o comportamento documentado

No `scheduler.py` fornecido:

- não há `fcntl.flock()`;
- não há lease no banco;
- não há prevented duplicate;
- não há rotina de ingestão de velas;
- `_routine_model_evolution_check()` apenas verifica uma condição e grava log; não executa retreinamento;
- não há persistência do timestamp real do último retreinamento;
- `stop()` cancela tasks, mas não aguarda sua finalização;
- operações síncronas de SQLite/treinamento podem bloquear o event loop.

### 3.5.3 `asyncio.gather()` não cria concorrência real para os agentes mostrados

`BaseAgent.evaluate_async()` apenas chama:

```python
return self.evaluate(context)
```

Como `evaluate()` é síncrono, uma operação de I/O bloqueante dentro do agente impede o event loop de executar as outras tasks.

O mesmo ocorre com `TapeReadingAgent._evaluate_live_binance_fallback()`, que usa `urllib.request.urlopen()` síncrono, com timeout de até 3,5 s por chamada e duas chamadas.

A opção correta é:

- connectors natively async;
- `asyncio.to_thread()` apenas para blocking calls inevitáveis;
- timeout por agente;
- circuit breaker por fonte;
- `asyncio.TaskGroup()` ou equivalente para cancelamento estruturado.

### 3.5.4 Transações SQLite não são atômicas

Em `_open_ladder_position()`:

```python
self._save_position(pos)
self._save_trade(trade)
```

Cada método abre uma conexão e faz `commit()` separadamente.

Se a segunda gravação falhar:

- `paper_positions` contém a posição;
- `paper_trades` não contém a abertura;
- a posição não é adicionada à memória;
- no restart, a posição é ressuscitada como órfã.

Em `close_position()` o problema é ainda mais grave:

- a posição é persistida como `CLOSED`;
- o trade de saída pode não ser persistido;
- o rollback em memória não reverte o banco;
- o restore não consegue reconstruir o PnL corretamente.

A correção é uma única transação:

```sql
BEGIN IMMEDIATE;
-- position/event/outbox
COMMIT;
```

### 3.5.5 `INSERT OR REPLACE` não é adequado para ledger

O uso de `INSERT OR REPLACE` pode:

- deletar e reinserir a linha;
- alterar `rowid`;
- interferir nos epochs usados por `manual_reset()`;
- dificultar auditoria e.event sourcing.

Prefira:

- `INSERT ... ON CONFLICT DO UPDATE`;
- chave de negócio estável;
- `version`/optimistic locking;
- tabela separada de eventos imutáveis.

### 3.5.6 Reset manual não isola o performance tracker

`PaperTradingEngine.manual_reset()` grava:

- `ledger_epoch_trade_rowid`;
- `ledger_epoch_pos_rowid`;
- `quota_epoch_trade_rowid`.

Mas `PerformanceTracker.compute_metrics_from_db()` lê:

```sql
SELECT * FROM paper_trades ORDER BY timestamp ASC
```

e:

```sql
SELECT timestamp, equity FROM paper_portfolio ORDER BY timestamp ASC
```

sem qualquer epoch.

Consequências:

- win rate continua incluyendo trades anteriores;
- drawdown inclui equity curve anterior;
- Sharpe e Sortino misturam sesiones;
- a afirmação de “piloto reiniciado” pode não aparecer no dashboard;
- E4/E5 podem serPresentation como invalidados, mas a telemetria não os isola.

### 3.5.7 Dashboard read-only não é garantido pela classe de performance

A `PerformanceTracker` é descrita como read-only, mas:

- usa `sqlite3.connect()` normal, não `mode=ro`;
- executa `PRAGMA journal_mode=WAL`;
- cria tabelas em `_ensure_telemetry_tables()`.

Se o Flask importar/inicializar essa classe, ele escribe no banco. O painel deve usar:

- URI `file:...?mode=ro`;
- migrations executadas antes do start do servidor;
- snapshot/replica de leitura;
- zero schema changes no request path.

### 3.5.8 Snapshot de portfolio usa timestamp como primary key

`paper_portfolio.timestamp` é `PRIMARY KEY`, e `_record_portfolio_snapshot()` usa `INSERT OR REPLACE`.

Barras com timestamps repetidos, testes ou processões simultâneos podem sobrescrever snapshots. O ideal é:

- `snapshot_id` autoincremental;
- `observed_at` indexado;
- `event_sequence`;
- `source`;
- `ingested_at`.

---

## 3.6 Vulnerabilidades do Free Data Mandate

“Zero data cost” é um objetivo válido, mas não significa:

- zero indisponibilidade;
- zero latência;
- zero risco de ToS;
- zero stale data;
- zero mudança de schema;
- zero bias;
- zero custo operacional.

### 3.6.1 Falha deve ter política explícita

O Tape Reading documenta `UNAVAILABLE` como fail-open. Para uma estratégia que exige confirmação de fluxo, isso é contraditório.

A recomendação é classificar a decisão:

| Estado | Ação para entrada |
|---|---|
| `FRESH` | Elegível |
| `STALE` | Depende da criticidade; default HOLD |
| `UNAVAILABLE` | HOLD/BLOCK |
| `INVALID` | BLOCK + alerta |
| `DEGRADED` | Apenas modo close-only |

Para scalps de 3m/5m, dados de 2–3 minutos já podem ser materialmente stale.

### 3.6.2 Scraping não é sinônimo de fallback robusto

A README inclui scraping de CoinGlass, Twitter/X e CoinMarketCal. É necessário:

- validar termos de uso e `robots.txt`;
- não contornar autenticação ou bloqueios;
- respeitar `Retry-After`;
- usar cache negative;
- jitter;
- circuit breaker por host;
- backoff exponencial;
- validação de schema;
- fallback para múltiplas fontes;
- reconciliação de timestamp.

Uma API pública que responde HTTP 200 com HTML ou JSON de formato diferente também precisa ser rejeitada como `INVALID`, não ingerida.

### 3.6.3 Fallback entre exchanges não preserva semântica

`GoldenPotAgent.fetch_market_universe_tickers()`:

- Binance fornece `weightedAvgPrice`;
- Bitget usa:

```python
vwap = (high_p + low_p + last_p) / 3.0
```

Isso **não é VWAP**. A feature `dist_vwap_pct` passa a significar coisas diferentes conforme a fonte.

O mesmo problema ocorre com Tape Reading Binance para uma execução Bitget:

- venue de dados diferente;
- basis diferente;
- latência diferente;
- profundidade diferente.

Para scalping, o fallback deve manter a mesma venue ou ser usado apenas para discovery, nunca para confirmação final de microestrutura.

### 3.6.4 Scanner de 24h usa 12,5 horas

`PairScanner.analyze_pair_candles()` usa `closes[0]` como “price 24h ago”. Em `scan_watchlist()`, são carregadas 50 velas de 15m:

\[
50 \times 15m = 12{,}5h
\]

Logo:

- `change_24h_pct` não representa 24h;
- thresholds de volume e delisting estão errados;
- a detecção de delisting é tardia e heurística.

Para 24h em 15m, são necessárias 96 barras.

### 3.6.5 Risco de delisting precisa de fonte primária

Uma queda de 35% não é evidência suficiente de delisting. Pode ser:

- crash;
- depeg;
- falha de oracle;
- dumping;
- exchange incident.

O sentinel precisa de:

- status oficial da exchange;
- announcements;
- leverage/token changes;
- trading suspension;
- index/delisting notices;
- cross-check com mais de uma fonte.

`pair_scanner` pode gerar heurística, mas não deve ser a única autoridade de veto.

---

## 3.7 Segurança de credenciais e rollback

### 3.7.1 Credenciais

A recomendação da README:

```bash
git remote set-url origin https://<USUARIO>:<TOKEN>@github.com/...
```

não deve ser usada. O token pode ficar:

- em `~/.git/config`;
- no histórico do shell;
- em logs;
- em screenshots;
- em processes;
- em backups.

Prefira:

- SSH keys;
- GitHub CLI;
- credential helper;
- secret manager;
- token de escopo mínimo;
- rotação;
- revogação de qualquer token que tenha sido inserido em URL.

Para a exchange:

- subaccount exclusiva;
- API key sem permissão de withdrawal;
- IP allowlist;
- read/trade-only;
- chaves de paper/testnet separadas;
- secret store do SO;
- rotação programada;
- logs com redaction;
- validação de conta e venue;
-禁止 de rodar a mesma chave em ambientes distintos.

### 3.7.2 Rollback de software e de estratégia precisam ser separados

Hoje há概念 de:

- rollback de configuração defensiva;
- emergency halt;
- manual reset;
- champion/challenger conceitual.

O que falta:

- rollback verificável de artifact/version;
- hash do model/config;
- promotion transaction;
- teste automático de rollback;
- compatibilidade de schema;
- snapshot de configuração;
- aprovação para promoção;
- audit log imutável;
- política de “safe state” após crash.

Um rollback de estratégia não deve ser confundido com rollback de software.

---

# 4. Recomendações Práticas e Priorizadas

## P0 — Imediatas: segurança de capital e blockers de operação

| Ação | Arquivos/métodos relevantes | Critério objetivo de conclusão |
|---|---|---|
| **1. Manter live bloqueado e corrigir o escopo da documentação** | `PaperTradingEngine.__init__`, README | Nenhum caminho capaz de enviar ordem autenticada sem adapter explicitamente revisado |
| **2. Corrigir quorum do consenso** | `ConsensusJudge.judge_deterministic()` | Um único BUY nunca aprova; teste obrigatório com Technical BUY e todos os demais HOLD |
| **3. Remover autoridade de direção do LLM** | `ConsensusJudge.judge_async()` | LLM apenas audita; timeout ≤500 ms; output estruturado; não pode criar BUY sem votes |
| **4. Introduzir quorum e estado ABSTAIN** | `AgentSignal`, `BaseAgent` | BUY/SELL exige mínimo de agentes direcionais e concordância ponderada |
| **5. Tornar o Risk Gate o boundary final** | `PaperTradingEngine._open_ladder_position()` | Toda entrada verifica confidence mínima do perfil, Tape gate, veto, freshness, drawdown, margem e exclusões |
| **6. Aplicar `STRATEGY_PROFILES.min_committee_confidence`** | `smart_trade.py`, `backtesting.py` | Teste prova que confidence abaixo do perfil não abre posição |
| **7. Fazer o leverage ser decidido no executor** | `_open_ladder_position()`, `leverage_policy.decide_leverage()` | Caller não consegue enviar leverage arbitrária; limite e reduções são sempre aplicados |
| **8. Implementar daily drawdown real** | `RollbackGuard.evaluate_telemetry()` | Teste com baseline diário bloqueia novas posições e executa política de flattening definida |
| **9. Tornar auto rollback efetivo** | `RollbackGuard`, `TradingCore`, executor | Um objeto `RiskPolicy` versionado reduz confidence, leverage, lote e estratégias permitidas |
| **10. Aplicar breakers também ao carry** | `execute_carry_allocation()` | Carry não abre posição com breaker, stale data, margem excedida ou concentration limit |
| **11. Corrigir duas fases do risco** | `TradingCore`, `RiskSentinelAgent` | Checagens dependentes de direção são refeitas depois do candidato do comitê |
| **12. Tornar ledger atômico** | `_save_position()`, `_save_trade()`, `_ladder_partial()`, `_apply_funding()` | Position, trade, funding e outbox são gravados em uma única transação |
| **13. Introduzir idempotência** | IDs de ordem e eventos | Retry da mesma requisição não cria duas ordens |
| **14. Corrigir epochs de métricas** | `PerformanceTracker.compute_metrics_from_db()` | Métricas do piloto excluem todos os registros anteriores ao reset |
| **15. Implementar OMS/testnet adapter** | Novo módulo, antes de live | State machine, retries, WS recovery, fills parciais, cancel/replace e reconciliação |
| **16. Tornar stale data fail-closed** | Tape, Risk Sentinel, data layer | Feature sem `as_of`, `event_time` e freshness é rejeitada antes da entrada |
| **17. Corrigir point-in-time macro** | `feature_engineering.py`, não fornecido | Teste demonstra que CPI/FOMC usa release date e vintage real |
| **18. Corrigir bugs de modelagem** | Ver lista abaixo | Testes de regressão para regime, carry, tape, leverage e TSMOM |
| **19. Corrigir trial log** | `statistical_rigor.initialize_trial_log()` | Log append-only, imutável, com hashes; nenhuma linha sintética apresentada como OOS |
| **20. Endurecer credenciais** | README, config, secrets store | Nenhum segredo no repositório, Git config, logs ou artifacts |
| **21. Definir kill switch externo** | Supervisor/OMS | Kill switch local e remoto bloqueia novas entradas e aciona flattening seguro |

### Bugs de modelagem que devem entrar no P0

1. `MarketRegimeClassifier.classify_from_df()`: exigir SMA200 real.
2. `_PilotSimulator.try_open()`: passar `timeframe` a `decide_leverage()`.
3. `TapeReadingAgent`: normalizar microprice por spread e rejeitar fallback stale.
4. `PairScanner`: usar 96 barras de 15m para 24h.
5. `GoldenPotAgent`: não chamar `(H+L+C)/3` de VWAP.
6. `FundingCarrySimulator`: stress test com capital, leverage e posições reais.
7. `DailyTSMOMStrategy`: registrar trades short.
8. `SignalGenerator`/`_lot_margin()`: separar quantity, notional e margin.
9. `PaperTradingEngine`: centralizar cooldown por estratégia/vela.
10. `PerformanceTracker`: agrupar estratégia por coluna estruturada, não por busca textual em rationale.

---

## P1 — Curto/médio prazo: robustez, observabilidade e pipeline

### P1.1 Arquitetura operacional

Migrar o caminho live para uma arquitetura event-driven:

```text
Feeds
  -> Raw/Immutable Store
  -> Data Quality + As-of Barrier
  -> Feature Store
  -> Decision Engine
  -> Deterministic Risk Kernel
  -> Order Intent / Outbox
  -> OMS
  -> Exchange
  -> Fills/Events
  -> Ledger
  -> Reconciliation
```

- SQLite/WAL pode permanecer no paper e em deployment single-node.
- Para multiprocesso/live, usar um store transacional com locking robusto, por exemplo PostgreSQL.
- Não usar “state in memory + SQLite best effort” como única fonte de verdade em live.

### P1.2 Observabilidade

Adicionar métricas e tracing:

- decision latency;
- data age por fonte;
- agent execution latency;
- signal funnel;
- reject reason;
- order ack latency;
- fill latency;
- slippage realizado versus modelado;
- rejection por venue;
- WebSocket gaps/reconnects;
- reconciliation divergence;
- margin e open risk;
- PnL liquidado e não liquidado;
- funding paid/received;
- error budget.

Usar `signal_id`, `position_id` e `client_order_id` como correlation IDs de ponta a ponta.

### P1.3 Validação quantitativa

- Separar切 strictly datasets de treino, validação, calibração e holdout.
- Usar purged walk-forward e embargo em todas as tentativas.
- Integrar DSR/PBO automaticamente ao promotion gate.
- Usar block bootstrap e cluster bootstrap por dia/símbolo.
- Calibração de probabilidades por regime.
- Teste de stationarity e structural break.
- Comparação com benchmarks líquidos de custo: Buy & Hold, SMA200, momentum e carry.
- Análise de concentração de PnL.
- Substituir accuracy isolated por expected utility, Brier score, calibration curve, precision/recall e confusion matrix por classe.
- Modelar os gates de Tape Reading em backtest com dados históricos gratuitos, quando disponíveis.

### P1.4 Pipeline de dados

Criar um contrato único:

```python
DataEnvelope:
    source
    instrument
    event_time
    receive_time
    as_of_time
    revision
    schema_version
    quality_flags
    payload_hash
```

Regras:

- joins exclusivamente por `as_of_time`;
- release dates e revisions;
- checksums de arquivos Binance Vision;
- detecção de gaps;
- validação OHLC;
- timezone UTC;
- clock synchronization;
- token bucket por host;
- `Retry-After`;
- circuit breaker por fonte;
- fallback com a mesma semântica;
- stale data como `BLOCK`, não como observação normal.

### P1.5 Risco de portfólio

Além do risco por trade:

- open risk total;
- risco por símbolo;
- correlação entre ativos;
- concentração por exchange;
- concentração por stablecoin collateral;
- expected shortfall;
- stress de gap;
- stress de funding;
- stress de basis;
- stress de outage;
- reserva de margem;
- kill switch por threshold de capital.

### P1.6 Governança do aprendizado adaptativo

`LearningTracker` e champion/challenger devem:

- congelar pesos durante o piloto;
- exigir amostra mínima;
- usar apenas OOS;
- aplicar limites de mudança;
- detectorar overfitting;
- manter challenger separado;
- promover apenas com evidência;
- reverter automaticamente;
- registrar versão erazão da promoção.

### P1.7 Transferência para spot

Construir módulo separado com:

- state machine;
- idempotency;
- reconciliação;
- reserva de collateral;
- ledger de transferências;
- política de reinvestimento;
- controle de slippage da transferência;
- taxa/IVA contábil se aplicável;
- prohibiting transferência que jeopardize margem de posições abertas.

### P1.8 Testes

Adicionar:

- property-based tests para invariantes de risco;
- fault injection entre writes;
- contract tests contra sandbox da exchange;
- replay de WebSocket;
- chaos tests de rede/DB;
- testes com clock skew;
- testes de restart em cada state machine stage;
- mutation testing;
- cobertura de branchesyncio/risk paths;
- soak test superior a 24 horas;
- build reproduzível e SBOM.

---

## P2 — Longo prazo: otimização de alpha e expansão institucional

1. **Multi-venue execution** com venue-specific features e smart order routing.
2. **Market making e quote skewing** apenas após microestrutura e latency maduros.
3. **Execution algorithms**: VWAP/TWAP/POV, participation rate e impact model.
4. **Collateral management** cross-margin, isolated margin, token haircuts e rebalancing automático.
5. **Portfolio optimizer** com risco, correlação, funding e capital efficiency.
6. **Research platform** com feature store, lineage, experiment registry e reproducibility.
7. **Model registry** com signatures, drift monitoring, champion/challenger e rollback.
8. **Custódia e segregação de capital** com hardware-backed keys ou MPC.
9. **Controles institucionais** de auditoria, compliance, incident response e change management.
10. **Expansão de strategies** apenas após validação individual e combined-portfolio.

---

# Critério objetivo para a próxima revisão

O sistema deve permanecer em paper até que, no mínimo:

1. o quorum do comitê estiver corrigido;
2. o LLM não tiver autoridade de decisão;
3. drawdown diário estiver funcional;
4. rollback for efetivo;
5. o ledger for atômico e reset-aware;
6. freshness for fail-closed;
7. Tape gate estiver no boundary final;
8. a estratégia passar por OOS com expectancy, DSR, PBO e stress;
9. o OMS passar por restart, retry, duplicidade e reconciliação;
10. o piloto atingir número estatisticamente suficiente de trades.

Para testnet:

- zero ordem órfã;
- 100% de reconciliação;
- kill switch testado;
- sem incidente de capital;
- slippage realizado dentro de envelope;
- comportamento de disconnect recovery validado.

Para live real:

- todos os P0 concluídos;
- revisão separada do adapter live;
- credenciais segregadas;
- capital inicial pequeno e whitelist de venues;
- aprovação formal do comitê de risco;
- rollback e flattening testados;
- paper/testnet evidence suficiente.

---

# Veredito final

O projeto tem **fundações quantitativas e de governança bem acima da média de um bot pessoal**, especialmente em:

- veto;
- anti-lookahead;
- backtesting;
- circuit breakers;
- leverage policy;
- persistência;
- rastreabilidade.

Contudo, a **maturidade do research stack é maior do que a maturidade do trading stack**. Os maiores riscos não são a quantidade de agentes ou a sofisticação dos indicadores; são:

1. quorum incorreto;
2. breakers incompletos;
3. ledger não atômico;
4. ausência de OMS/reconciliação;
5. freshness insuficiente;
6. ausência de alfa demonstrado;
7. divergência entre paper, backtest e futura execução real.

A decisão correta é: **manter em paper assistido, executar o P0 de forma-binary e só promover para testnet após uma revisão focada em OMS, persistência atômica, risk gate e paridade backtest/live.** Nenhum capital real deve ser exposto enquanto o sistema continuar registrando apenas paper orders e sem reconciliação verificável com uma venue.