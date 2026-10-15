# Critérios de Aprovação do Piloto (Paper Trading) — Objetivos e Falsificáveis

**Autoridade:** `PARECER_CHIEF_TRADING_SCIENTIST.md` §5 e `PROTOCOLO_EXPERIMENTAL.md` §1.8/§2.
**Executor:** `.venv/bin/python scripts/pilot_report.py` (exit 0 = APROVADO, 1 = REPROVADO, 2 = INCONCLUSIVO; saída em `datasets/reports/pilot_report.json`).
**Congelamento:** limiares abaixo são fixos a partir desta data. Alterá-los após ver resultados invalida o piloto.

## 0. Princípios (cinismo obrigatório)

1. **Assimetria:** APROVAR exige *todos* os critérios A1–A14. REPROVAR exige *uma* condição R1–R7. Ausência de dado = critério falho.
2. **Custo estresse, não custo contábil.** O simulador debita maker (0,02%) em entradas/saídas de `GOLDEN_POT`/`SCALP` (ver `smart_trade.py` L685/L805/L976) e trata *stop-loss* como maker — fantasia: stop é ordem a mercado. Todas as métricas decisórias são recalculadas com **0,06% taker + 0,02% slippage = 0,08% por perna sobre o notional de cada fill** (0,16% round-trip).
3. **Reset não apaga história.** O ledger inteiro conta para refutação; resets/alterações de parâmetro só *reiniciam o relógio de aprovação* (A1), nunca o de refutação. Resetar após perdas é viés de sobrevivência.
4. **Só vale o que estava pré-registrado.** Estratégia sem hipótese em `HIPOTESES_PRE_REGISTRADAS.md` (ou S1–S3 do Parecer) não pode ser aprovada, independentemente do P&L (A3).

## 1. Critérios de APROVAÇÃO (todos obrigatórios)

| ID | Critério | Limiar |
|---|---|---|
| A1 | Duração **limpa** (desde último reset/alteração de parâmetros) | ≥ 56 dias (alvo 84) |
| A2 | Trades fechados | ≥ 100 (Protocolo §1.4) |
| A3 | Estratégia pré-registrada | 100% dos `strategy_type` negociados |
| A4 | Expectância líquida estresse por trade | > 0 **e** limite inferior bootstrap (5º pct, 5000 reamostras) > 0 |
| A5 | Sharpe líquido anualizado (retornos diários, ×√365, capital $10k) | ≥ 1,0 (e ≥ 30 obs. diárias) |
| A6 | Deflated Sharpe (`statistical_rigor.deflated_sharpe_ratio`, N = tentativas do `trial_log.csv` + 1, variância dos Sharpes do log) | ≥ 0,95 |
| A7 | Max drawdown (equity em custo estresse, MtM) | ≤ 15% e ≤ 1,5× o MaxDD do backtest de referência |
| A8 | Hit rate vs breakeven: $WR_{BE}=\frac{\bar L+\bar C}{\bar W+\bar L}$ (W, L = ganho/perda bruta média, C = custo estresse médio) | limite inferior Wilson 95% do WR > $WR_{BE}$ |
| A9 | Divergência paper × backtest (exige `docs/pilot_backtest_reference.json` com `expectancy_net_pct`, `sharpe_net`, `trades_per_day`, `max_dd_pct`) | expectância ≥ 50% da do backtest; Sharpe ≥ 50%; frequência de trades ±30%; DD ≤ 1,5×. **Sem referência → falha** |
| A10 | Semanas líquidas positivas (≥ 8 semanas) | ≥ 60% |
| A11 | Integridade experimental | 0 `MANUAL_RESET` e 0 alterações em `simulation_settings` na janela |
| A12 | Cobertura de snapshots 15 min / maior gap | ≥ 98% / ≤ 60 min |
| A13 | Saúde operacional: daemon `HEALTHY`, DB `HEALTHY`, reinícios do daemon ≤ 2, último snapshot ≤ 30 min | todos |
| A14 | Reconciliação ledger (Σ`paper_trades.pnl` = Σ`paper_positions.realized_pnl`) | \|Δ\| < $0,01 |

Nota estatística: o erro-padrão do Sharpe anualizado com T observações diárias é ≈ √365·√((1+SR_d²/2)/T) ≈ 2,1 para T = 84. **12 semanas de paper não provam Sharpe 1; só refutam estratégias ruins e validam operação.** Aprovação = licença para capital real mínimo (valor que o usuário aceita perder por inteiro), nunca prova de alfa.

## 2. Condições de PARADA / REFUTAÇÃO (qualquer uma → REPROVADO, interromper piloto)

| ID | Condição |
|---|---|
| R1 | MaxDD estresse ≥ 20% |
| R2 | N ≥ 30 e limite **superior** IC95 unilateral da expectância líquida estresse < 0 (futilidade) |
| R3 | N ≥ 30 e limite **superior** Wilson do hit rate < $WR_{BE}$ |
| R4 | ≥ 3 disparos de `CONSECUTIVE_LOSS_COOLDOWN` na janela (o *circuit breaker* é a confissão operacional de edge negativo) |
| R5 | ≥ 28 dias limpos com P&L líquido estresse < 0 |
| R6 | N ≥ 30 e expectância **bruta** ≤ 0 (sem alfa antes mesmo dos custos — nenhuma otimização de execução salva) |
| R7 | Prazo de 56 dias atingido com Sharpe < 1,0 ou DSR < 0,95 |

Reprovado ⇒ **proibido** "ajustar e reiniciar" a mesma estratégia sem nova hipótese pré-registrada, +1 no `trial_log.csv` e novo holdout (Protocolo §1.1–1.2, 1.7).

## 3. INCONCLUSIVO

Nenhuma refutação disparada, mas ao menos um critério A* não satisfeito (tipicamente tempo/amostra/referência ausente). Estado padrão nos primeiros 56 dias.

## 4. Achados sobre o piloto atual (auditoria 2026-10-15)

- **A estratégia em teste (`GOLDEN_POT`, tickers como SOXSUSDT/WALUSDT/BTWUSDT) não é S1/S2/S3 do Parecer nem consta em `HIPOTESES_PRE_REGISTRADAS.md`.** É o ML/heurística direcional de curto prazo que o Parecer §2.2 condenou. Este piloto não valida a tese do Plano Mestre.
- **Protocolo violado:** 11 `MANUAL_RESET`, parâmetros em `simulation_settings` alterados durante a janela, relógio de 8–12 semanas inexistente (≈ 8 h de dados).
- **Bug contábil** [`smart_trade.py` L993–1028](file:///home/eudespatricio/ia_trader/smart_trade.py#L993-L1028): `net_pnl = gross − exit_fee` e depois `balance += net_pnl − exit_fee` → taxa de saída debitada **duas vezes** no saldo (e `paper_portfolio.total_fees_paid` ignora parte das taxas). Equity reportada é pessimista em ~ as taxas de saída; não afeta o veredito (usamos o ledger `paper_trades`), mas A14 deve ser estendido a equity antes do piloto oficial.
- **Cobertura operacional:** snapshots com gaps de até 123 min (esperado: 15 min); supervisor registra reinícios do `flask_dashboard` (não conta no A13, mas indica instabilidade).
- Sem backtest de referência da estratégia → A9 inatingível por construção.

## 5. Como rodar um piloto válido

1. Pré-registrar hipótese + `docs/pilot_backtest_reference.json`; congelar `simulation_settings`.
2. Definir `--start` oficial, um único reset, sem intervenção por ≥ 56 dias.
3. Rodar `pilot_report.py` semanalmente (cron); decisão só nas datas de checkpoint, mas R1–R6 podem encerrar a qualquer momento.
