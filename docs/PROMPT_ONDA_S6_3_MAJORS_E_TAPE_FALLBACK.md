# Prompt Pronto — Onda S6.3: Expansão de Majors (Frente A) & Fix TapeReading GP (Frente B)

> **Como usar:** Abra uma **nova conversa limpa** (`Ctrl + N`) no workspace `/home/eudespatricio/ia_trader` e cole o bloco abaixo na íntegra. `[skip-interview]` já fecha o escopo com base nas ablações OOS registradas em `docs/PLANO_EVOLUTIVO_S6.md`, `docs/HIPOTESES_PRE_REGISTRADAS.md` (`E_PATCH_S6_ONDA3`) e `docs/gaps.md`.

```markdown
[skip-interview] Executar a **Onda S6.3** do IA Trader em duas frentes independentes (Frente B cirúrgica + Frente A `T11` Expansão de Majors), seguindo estritamente as regras de governança Multi-LLM, testes independentes em janela limpa, revisão cross-model e validação OOS zero-leakage antes de promover qualquer alteração para produção.

## Contexto Já Validado (NÃO refazer do zero — apenas verificar no código/DB)
- Ablações OOS `S6-5a`/`S6-5b` (`TRIAL_0176..0191` em `datasets/reports/s6_scalp_oos_ablation.json` e `datasets/reports/s6_golden_pot_oos_ablation.json`) fecharam **todas `NO-GO`**: afrouxar gatilhos do `SCALP` 15m (nos 3 majors atuais) ou `GOLDEN_POT` 5m não atinge `PF OOS >= 1.25`.
- **Frente B (`TapeReadingAgent` + Coletor L2):** Em `flow_gate_log` de `GOLDEN_POT`, `80,5%` (`33/41`) deram `UNAVAILABLE_PASSED` porque:
  1. `scripts/microstructure_collector.py` (`get_active_golden_pot_symbols(top_n=3)`, L111 e L486) monitora apenas 3 altcoins enquanto `scripts/run_paper_simulation.py` avalia `GP_MAX_CANDIDATES = 6`.
  2. `agents/tape_reading_agent.py` (L358 `book_stale`, L395 `tick_stale`, L397/L401 `tick_insufficient`) devolve `_unavailable` sem acionar `_evaluate_live_binance_fallback()` quando o símbolo já está em `covered_symbols`.
- **Frente A (`T11` Expansão Controlada de Majors `3 → 8`):**
  - Baseline atual: `BTCUSDT`, `ETHUSDT`, `SOLUSDT` (60k velas 1h / 28k–40k velas 15m).
  - Candidatas Tier 1 selecionadas por liquidez (`$93M–$704M`/d), maturidade (`>= 6a`), spread (`1,07–3,49 bps < 5 bps`), NATR 15m (`48–74 bps`) e diversificação: **`XRPUSDT`, `NEARUSDT`, `DOGEUSDT`, `LINKUSDT`, `AAVEUSDT`** (manter `BNBUSDT`, `1000PEPEUSDT`, `SUIUSDT`, `ZECUSDT` como reserva/controle; descartar `ADAUSDT` por spread `6,98 bps` e tokens `Alpha < 1,5a`).
  - Hoje essas 5 candidatas têm **`0` velas `1h`** e `0–557` velas `15m` em `datasets/market_cache.db`.

---

## Regras Críticas de Segurança & Governança
1. **Banco ao vivo (`datasets/market_cache.db`, `datasets/microstructure.db`):** O daemon `run_paper_simulation.py` e `microstructure_collector.py` podem estar rodando ao vivo!
   - Faça **backup** antes de qualquer escrita em `market_cache.db` ou escreva primeiro/valide em modo seguro sem travar o SQLite (`timeout`, transações curtas, zero duplicatas Bitget UTC+8, rodar `scripts/audit_and_dedup_candles.py --check` se aplicável).
   - **Proibido** resetar `pilot_start_utc` ou fechar/perder posições abertas (`ETHUSDT`/`SOLUSDT`). Qualquer restart eventual só após `GO` explícito e preservando o ledger.
2. **Multi-LLM + Testers Independentes + Cross-Model Review:**
   - Faça Quick Pull (`swarm(action="models", all=True)` + `/home/eudespatricio/.gemini/antigravity/bin/llm-inventory`) e distribua em `swarm` (`workspace="inherit"` com arquivos disjuntos ou `"branch"`/`"share"`).
   - O autor do código **nunca** escreve os testes críticos sozinho nem faz auto-aprovação: delegue suítes independentes a um Tester em janela limpa e revisão final a um modelo Tier S/A distinto.
3. **Módulos editados/novos:** Respeite teto `<= 500` linhas em arquivos novos, sem `except: pass` / fail-silent.

---

## Escopo Técnico — Frente B (Fix TapeReading Fallback + Cobertura L2 GP)
1. **`agents/tape_reading_agent.py`:**
   - Quando `is_live_call` for `True` e o caminho local falhar por `book_stale` (L358), `tick_stale` (L395) ou `tick_insufficient` (L397/L401, inclusive `sat_ratio > 0.50`), acionar `_evaluate_live_binance_fallback(sym, now_ms, reason)` antes de desistir com `_unavailable` — mantendo a proteção anti-stale dentro de `_evaluate_live_binance_fallback` (L148 `stale_tick_fallback`) e o teto de spread (`SPREAD_MAX_BPS` para `COVERED_SYMBOLS`, `15.0 bps` para altcoins).
   - Garantir que majors (`BTC/ETH/SOL` + futuras majors) **não** percam precisão nem caiam em loop de latência REST síncrono excessivo (respeitar timeout curto e cache/cooldown curto se necessário, sem degradar o ciclo de 15s do daemon).
2. **`scripts/microstructure_collector.py`:**
   - Ampliar `get_active_golden_pot_symbols(top_n=6)` (alinhando com `GP_MAX_CANDIDATES = 6`), preservando `BASE_SYMBOLS` + cap de carga segura para WS/REST.
3. **Testes Independentes (Frente B):**
   - Cobrir explicitamente: `book_stale` → fallback REST OK; `tick_stale` / `tick_insufficient` / `sat_ratio > 0.50` → fallback REST OK; fallback com dados stale → continua `UNAVAILABLE`; `is_live_call=False` (backtest determinístico) **nunca** faz rede e mantém `_unavailable`; `get_active_golden_pot_symbols(top_n=6)` respeita deduplicação e limite `6`.

---

## Escopo Técnico — Frente A (`T11` Backfill + Validação OOS Zero-Leakage de Expansão de Majors)
1. **Etapa A1 — Ingestão Histórica Limpa (Script dedicado idempotente, ex. `scripts/backfill_candidate_majors.py`):**
   - Baixar via Binance Futures REST (`/fapi/v1/klines` + `/fapi/v1/fundingRate`) para `XRPUSDT, NEARUSDT, DOGEUSDT, LINKUSDT, AAVEUSDT` (e opcionalmente `BNBUSDT, 1000PEPEUSDT, SUIUSDT` para ablação comparativa se rápido):
     - `1h`: `2021-01-01` (ou `onboardDate`) até agora (`candles`, `exchange='binance'`, sem candle em formação).
     - `15m`: último `~1 ano` (alinhado com janela 15m de `BTC/ETH/SOL`).
     - `funding_rates`: histórico completo compatível para cálculo real de funding `8h`.
   - Verificar integridade pós-carga: `0` gaps críticos inesperados, `0` duplicatas, `PRAGMA integrity_check = ok`.
2. **Etapa A2 — Ablação OOS Zero-Leakage (`scripts/eval_majors_expansion_s6_oos.py` → `datasets/reports/s6_majors_expansion_oos.json`):**
   - Avaliar **por estratégia baseline** (`DIRECTIONAL 1h`, `SCALP 15m`, `SWING_4H 4h`) sem tocar nas regras centrais, comparando:
     - `M0_BASE_3`: `BTC, ETH, SOL` (controle)
     - `M1_TIER1_8`: `M0 + XRP, NEAR, DOGE, LINK, AAVE`
     - `M2_PER_SYMBOL`: contribuição marginal OOS individual de cada nova moeda (`IS <= 2024 | embargo 7d | OOS >= 2025` no `1h`/`4h`; `SPLIT = 2026-07-29 | embargo 10b` no `15m`).
   - **ML Purgado:** Treinar modelos `1h` e `15m` **estritamente pré-split** (sem carregar `.joblib` de disco!), comparando modelo treinado só em `BTC/ETH/SOL` vs pooled nos `8` pares (testar se há ganho OOS real sem diluição).
   - **Trava de Correlação & Portfólio (`MAX_SAME_DIR_MAJORS`):** Com `8` majors (`corr 0.77–0.90`), testar `corr_cap = 2` vs `3` vs `4` e cap por risco líquido agregado (`max_directional_net_risk_r`), medindo `MaxDD_estresse`, `PF`, `PF_ex_top1`, `Sharpe`, `DSR` (com `N` trials atualizado de `trial_log.csv`) e `trades/dia`.
   - **Pista complementar opcional (somente se rápida e purgada):** checar em OOS `GOLDEN_POT` se `ATR_exp <= 3,0x` (pista `W43/W44`) melhora OOS, registrando em `trial_log.csv`.
3. **Etapa A3 — Gate de Decisão (`GO` vs `NO-GO`):**
   - **Critérios `GO` para promover um novo par ao universo live (`CORE_SYMBOLS` / `SWING_4H` / `SCALP` / `BASE_SYMBOLS`):**
     1. Portfólio expandido melhora `sum_r` e `trades/dia` **sem** reduzir `PF OOS` abaixo de `1.25` (ou abaixo do baseline `M0_BASE_3` se este for `< 1.25`), `PF_ex_top1 >= 1.10`, `MaxDD_estresse < 15%`.
     2. Contribuição marginal da moeda no OOS (ou no comitê conjunto) é não-destrutiva (`avg_r >= 0` ou melhoria de `Sharpe`/`Calmar` do portfólio).
     3. Se **nenhum** ou apenas um **subconjunto** (ex.: só `XRP+NEAR+AAVE`) passar nos critérios, promover **somente** o subconjunto aprovado (ou documentar `NO-GO` e NÃO alterar o código de produção das estratégias!).

---

## Entregáveis & Verificação Final Obrigatória
1. Registrar todos os experimentos em `datasets/reports/trial_log.csv` (`statistical_rigor.log_trial`), `docs/PLANO_EVOLUTIVO_S6.md`, `docs/HIPOTESES_PRE_REGISTRADAS.md` e `docs/gaps.md`.
2. Rodar testes direcionados (`tests/test_tape_reading_gate.py`, novos testes da Onda S6.3, `tests/test_strategy_profiles_invariants.py`, `tests/test_paper_trading_engine.py`) e só aceitar com `100%` pass + revisão cross-model `APPROVE`.
3. Antes de reiniciar qualquer serviço em produção (se houver promoção `GO`), mostrar o resumo exato dos ganhos OOS vs `M0_BASE_3` e confirmar preservação das posições abertas.
```
