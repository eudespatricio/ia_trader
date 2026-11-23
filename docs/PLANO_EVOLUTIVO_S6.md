# Plano Evolutivo S6 — Pacote Evolutivo Contínuo

**Data de abertura:** `2026-11-23`  
**Status:** **Onda S6.1 (`v1.0.12`) IMPLEMENTADA & EM HOMOLOGAÇÃO**  
**Regra Mandatória:** *Toda alteração de lógica ou filtro exige prova empírica zero-leakage em OOS (custos estresse + embargo) antes de ir para produção, testes independentes em janela limpa e revisão cross-model.*

---

## 1. Onda S6.1 (`v1.0.12`) — Entregas Aprovadas (`GO`)

### Item `S6-2` — `GOLDEN_POT`: Filtro Dinâmico de Perps TradFi / Não-Cripto (`exchangeInfo` + Fallback Estático) **[IMPLEMENTADO]**

- **Diagnóstico ao Vivo (`2026-11-23 12:17 BRT`):**
  - A lista estática `GP_EXCLUDED_BASE_ASSETS` (`42` tickers) não acompanhou novas listagens TradFi/Stock Perps na Binance Futures e Bitget: o endpoint `/fapi/v1/exchangeInfo` marca **`253` contratos não-cripto** (`EQUITY`, `KR_EQUITY`, `CN_EQUITY`, `HK_EQUITY`, `COMMODITY`, `PREMARKET`, `INDEX`, `underlyingSubType=['TradFi']`), dos quais `28+` vazavam para a tabela `golden_pots` e para avaliação 5m do daemon (`MUUSDT`, `MUUUSDT`, `ANTUSDT`, `DRAMUSDT`, `EWYUSDT`, `NBISUSDT`, `ZHIPUUSDT`, `ZHONGJIUSDT`, `BEUSDT`, `MRVLUSDT`, `NVDLUSDT`, `CRCLUSDT`, `IRENUSDT`, `NATGASUSDT`, `CBRSUSDT`, `SPCXUSDT`, `MINIMAXUSDT`, etc.).
- **Implementação Cirúrgica:**
  1. [golden_pot_agent.py](file:///home/eudespatricio/ia_trader/agents/golden_pot_agent.py#L47-L182): Fonte única `NON_CRYPTO_BASE_EXCLUSIONS` (`85` ativos-base estáticos, preservando tokens `COIN`/`RWA` cripto como `PAXG`, `XAUT`, `MANTRA`, `CFG`, `AIGENSYN`), classificadores `extract_base_asset()` e `classify_non_crypto_from_exchange_info()`, cache dinâmico via `/fapi/v1/exchangeInfo` com TTL `6h` e backoff de falha `5m` em `refresh_non_crypto_symbols()`, e filtro `is_non_crypto_symbol()` aplicado tanto no loop Binance quanto no fallback Bitget em `fetch_market_universe_tickers()`.
  2. [run_paper_simulation.py](file:///home/eudespatricio/ia_trader/scripts/run_paper_simulation.py#L159-L162): `GP_EXCLUDED_BASE_ASSETS = NON_CRYPTO_BASE_EXCLUSIONS` (eliminando duplicação) e `_is_gp_symbol_excluded()` na entrada de `evaluate_golden_pot_opportunities()`.

---

## 2. Itens Investigados & Refutados em OOS (**NO-GO** — Manter Produção Intacta)

### Item `S6-1a` — Remoção/Relaxamento do `_committee_gate` e Teto `RSI > 75` no `SCALP 15m` **[NO-GO]**

- **Motivação:** Durante o pump de `2026-11-23 13:45Z` (`BTC +4.18%`, `ETH +4.97%`, `SOL +5.54%`), o gatilho Squeeze `15m` disparou `BUY`, mas `ETH`/`SOL` foram barrados pelo ML `15m < 45%` e `BTC` foi barrado pelo comitê (`RSI=84.2 > 75` neutralizou `TechnicalAgent`, enquanto `Macro+OnChain` votaram `SELL`). *(Nota live: entrar em `BTC` às `13:45Z` em `63,833.30` teria acionado `SL 63,276.44` na barra seguinte `14:00Z`.)*
- **Ablação Purgada Zero-Leakage (`98.116` barras `15m` BTC+ETH+SOL, custo estresse `16 bps` RT, `SL 1.4x / TP 2.2x`, split `2026-07-29`, embargo `10` barras, LGBM+RF purgado em `65.683` linhas IS):**

| Variante Testada | Full Squeeze IS (`N=174`) | Full Squeeze OOS (`N=87`) | Squeeze + `ML15m >= 0.45` OOS (`N=11`) | Veredicto |
|---|---|---|---|---|
| **Produção (`6-ag conf>=0.70, RSI<=75`)** | `n=66, PF 0.64, -17.67R` | **`n=37, WR 59.5%, PF 1.56, +10.30R (+0.278R)`** | **`n=4, WR 50.0%, PF 1.18, +0.41R`** | **MANTER** |
| `V0b` Produção `conf >= 0.55` | — | — | `n=5, WR 40.0%, PF 0.78, -0.75R` | **NO-GO** |
| `V1` Sem teto `RSI > 75` (`6-ag`) | — | — | `n=5, WR 40.0%, PF 0.77, -0.80R` | **NO-GO** |
| `V2b` Intraday `Tech+OF` com `RSI<=75` (sem Macro/OC) | `n=74, PF 0.61, -20.70R` | `n=31, WR 54.8%, PF 1.26, +4.52R` | `n=2, WR 100%, +2.60R` | **NO-GO** (corta `>50%` do OOS) |
| `V2` Intraday `Tech+OF` sem `RSI` (sem Macro/OC) | `n=87, PF 0.59, -26.05R` | `n=36, WR 50.0%, PF 1.06, +1.32R` | `n=6, WR 50.0%, PF 1.15, +0.52R` | **NO-GO** |
| `V3b` `RiskOnly + RSI<=75/25` | `n=98, PF 0.56, -32.06R` | `n=48, WR 56.2%, PF 1.38, +9.64R` | `n=4, WR 75.0%, PF 3.02, +2.58R` | Inferior no Full Squeeze (`PF 1.38 < 1.56`) |
| `V3` `RiskOnly` (sem comitê, sem RSI) | `n=111, PF 0.53, -39.94R` | `n=53, WR 52.8%, PF 1.21, +6.43R` | `n=9, WR 44.4%, PF 0.89, -0.66R` | **NO-GO** |
| Isolado: `RSI > 75 (BUY) / < 25 (SELL)` | `n=20, PF 0.74, -3.82R` | `n=9, WR 33.3%, PF 0.59, -2.86R` | `n=5, WR 20.0%, PF 0.31, -3.24R` | **Confirmado nocivo** |
| Isolado: `RSI 25..75` normal | `n=154, PF 0.60, -47.40R` | `n=78, WR 52.6%, PF 1.18, +8.04R` | `n=5, WR 80.0%, PF 4.02, +3.86R` | Cap protetor validado |

### Item `S6-1b` — Relaxamento do Teto `RSI > 75 / < 25` no `TechnicalAgent` (`DAY_TRADE 1h`) **[NO-GO]**

- **Backtest 6 Anos (`2020–2026`, `BTC+ETH+SOL`, IS `<=2024` | embargo `7d` | OOS `>=2025`, `SL 1.0x / TP 1.5x`, `16 bps` RT):**
  - Trades isolados em extremo (`RSI > 75` acima da SMA200 / `RSI < 25` abaixo da SMA200): IS `n=5.337, PF 0.928, -238.0R` | **OOS `n=2.143, WR 39.6%, PF 0.730, -395.6R` (`-0.1846R`/trade)**.
  - Relaxar o teto `25..75` para `20..80`, `15..85` ou `0..100` degradou o OOS de `-3.894,3R` para `-4.072,7R` / `-4.103,0R` / `-4.076,2R`. Trava mantida intacta.

---

## 3. Backlog Evolutivo S6 (Próximas Ondas — Aguardando Coleta / Execução Dedicada)

1. **`S6-3` / `S5-5` Ativação Controlada (`shadow -> active`):** Avaliar promoção de `Swing4HMetaSizer` (`LogReg-L2` + `EWMA_VolTarget`) após amostra limpa de trades `SWING_4H` em produção.
2. **`T2` Carry Treasury:** Reavaliar ativação apenas quando funding anualizado OOS superar custo + hurdle institucional (hoje `STANDBY`).
3. **`T11` Expansão de Universo (`Top 15–20 Perps Líquidos`) + Sleeve Beta-Neutro (`XSMOM/Carry`):** Ondas futuras após ingestão multi-ano `2021–2026` sem viés de sobrevivência.
4. **Wave P4b (`~dez/2026–jan/2027`):** Estudo OOS zero-leakage com `30–60` dias limpos de `liquidations` WS + `tick_agg` (`sat=0`).
5. **Meta-Labeling ML sobre Sinais Filtrados:** Pesquisar meta-rotulagem binária `P[win | sinal]` especificamente sobre candidatos já filtrados (ex.: `SCALP` e `DAY_TRADE`) quando houver série estendida.
