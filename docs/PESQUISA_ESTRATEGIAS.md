# Pesquisa de estratégias – IA Trader (Bitget perp, varejo, dados grátis)

> **AVISO DE CONFIABILIDADE:** este relatório foi produzido por um subagente SEM acesso à internet. Tudo vem de
> conhecimento prévio do modelo: nenhuma busca ao vivo foi feita. Os papers e obras citados são reais e conhecidos,
> mas links e números (retornos, Sharpe, profundidades de API) devem ser VERIFICADOS antes de serem tratados como
> fatos. Os números são ordens de grandeza, não medições. O que for testável deve ser medido nos nossos próprios
> dados (por exemplo, a profundidade real dos endpoints de funding da Bitget e da Binance).

## Resumo cínico
- Previsão direcional de curto prazo com indicadores técnicos (RSI, MACD etc.) em cripto: evidência fraca. Com taker 0,06% (round-trip ~0,12% + slippage), exige edge bruto por trade bem acima disso. Resultado nulo OOS é o esperado.
- O que tem evidência são prêmios estruturais/de risco (funding carry, trend/momentum de baixa rotatividade) e rigor metodológico (CV purgado, DSR, PBO). Mesmo esses estão comprimidos desde ~2022 e são arriscados.
- Regra: qualquer backtest com Sharpe > 2 em cripto de varejo, sem custos realistas e sem correção de múltiplos testes, é marketing até prova em contrário.

## Ranking (evidência × viabilidade com nossos dados/custos)
| # | Abordagem | Evidência | Viabilidade | Confiança |
|---|---|---|---|---|
| 1 | Trend following/TSMOM diário (BTC/ETH + top-N), vol targeting | Média-alta | Alta (turnover baixo) | Média |
| 2 | Funding carry delta-neutro (spot long + perp short) | Média | Média-alta (precisa spot; capital pequeno) | Média |
| 3 | Rigor ML: CV purgado/CPCV, DSR, PBO, triple-barrier, meta-labeling como FILTRO de uma regra base (trend) | Média (melhora robustez, não cria alfa) | Alta | Média |
| 4 | Momentum cross-sectional cripto (long/short top/bottom) | Média-baixa | Média (custos, liquidez de altcoins, short squeezes) | Baixa-média |
| 5 | Execução maker/redução de turnover | Alta como redutor de custo; não gera alfa | Alta | Média |
| 6 | Regime detection (HMM) como filtro | Baixa-média; propenso a overfit | Alta | Baixa |
| 7 | Mean reversion intradiária | Baixa; taxas matam; funciona em pares/stat-arb em microestrutura, não varejo | Baixa | Baixa |
| 8 | Directional ML em indicadores técnicos 1m–1h | Muito baixa | — | Alta de que não funciona após custos |

## 1. Funding carry / basis
- Mecânica: long spot + short perp mesmo notional; recebe funding quando positivo (padrão 0,01%/8h = ~11%/ano base; em bull sobe muito).
- Evidência: Schmeling, Schrimpf, Todorov (BIS WP 1087, 2023) "Crypto carry" – carry grande, varia no tempo, com crashes e limites de arbitragem. Retornos 2020–21 de 20–60%+ a.a.; 2023–25 tipicamente ~5–15% a.a., com períodos de funding ~0 ou negativo (números a verificar). Realista líquido para varejo: ~4–12% a.a. com flutuação; não é "renda passiva".
- Custos: entrada+saída spot e perp (~4 pernas): ~0,2–0,25% taker; só vale se segurar semanas/meses.
- Riscos reais: liquidação da perna short em squeeze (alavancagem ≤2–3x, margem sobrando, rebalanço); funding negativo; risco de exchange/contraparte (FTX); basis spot/perp e diferenças de índice; capital preso em dois locais; ADL; custos fixos em capital pequeno.
- Dados: histórico de funding da Bitget (endpoint history-funding-rate, paginado, profundidade a verificar); basis = close perp − close spot computável. Para profundidade longa, complementar com Binance (fundingRate desde 2019).
- Experimento viável: backtest com funding real + taxas, sem alavancagem exótica, regra "entra se funding médio 7d > X, sai se < Y".

## 2. Trend / momentum
- Evidência: Moskowitz, Ooi, Pedersen (2012); Liu & Tsyvinski (2021, RFS); Liu, Tsyvinski, Wu (2022 JF); Hurst, Ooi, Pedersen (AQR); Barroso & Santa-Clara (2015); Moreira & Muir (2017, replicação contestada por Cederburg et al. 2020). Preprints otimistas de trend em BTC: cautela.
- Realismo: Sharpe líquido plausível 0,7–1,2 em BTC/ETH diário, dominado por poucos mercados de alta (beta de cripto disfarçado); drawdowns de 30–50%. Comparar SEMPRE com buy&hold ajustado a risco.
- Implementação: sinais diários/4h (ensemble de retornos 20/50/100/200d, Donchian, EMA cross), alvo de vol (15–25% a.a.), rebalanço com banda para reduzir turnover; 10–40 trades/ano por ativo.
- Cross-sectional: efeito existe mas é instável e concentrado em small caps ilíquidas. Mean reversion de curto prazo é consumida por spread/taxas.

## 3. Técnicas de ML (López de Prado)
- Purged K-fold + embargo e CPCV: essencial.
- Triple-barrier + sample weights por unicidade: melhora a definição de label; não cria sinal.
- Meta-labeling: filtra/dimensiona sinais de uma regra primária; só funciona se a regra primária já tem edge fraco positivo; não resgata sinal nulo.
- Fractional differentiation: ganhos modestos; baixa prioridade.
- Deflated Sharpe Ratio e PBO: diagnóstico obrigatório. Registrar N de tentativas.
- Harvey, Liu, Zhu (2016): limiar t>3.
- Regime HMM: fácil de overfit; só com fit walk-forward e comparando com filtro simples (vol realizada, SMA 200).
- Com poucos ativos e ~5–8 anos, modelos complexos perdem para regras simples. A RTX 3060 não é gargalo nem vantagem.

## 4. Execução
- Maker 0,02% vs taker 0,06%, mas há seleção adversa; simular fill conservador (só se o preço atravessar o limite) e medir em paper trading.
- Reduzir turnover é a alavanca mais confiável: bandas de no-trade, sinais lentos, rebalanço só com desvio > limiar.
- Custo total: taxa + spread + slippage + funding pago/recebido.

## 5. Dados gratuitos
- Bitget REST pública: candles perp/spot, funding history (profundidade limitada), OI atual, long/short ratios (profundidade curta). Sem trades/L2 históricos.
- Binance: klines desde 2017–2019; fundingRate desde 2019; data.binance.vision tem dumps diários/mensais grátis (klines, aggTrades, funding, metrics com OI e L/S desde ~2021).
- Bybit: public.bybit.com (trades históricos). OKX: histórico curto, dumps em okx.com/data-download.
- Liquidações: histórico profundo grátis é raro; coletar via websocket daqui para frente.
- Outros: Deribit (DVOL), Coin Metrics community (on-chain diário), alternative.me Fear&Greed.
- Cuidado com survivorship bias e mudanças de metodologia de funding.

## Top 3 experimentos
**E1 – Trend diário com vol targeting (BTC, ETH, + líquidos)**
- H: ensemble TSMOM (20/50/100/200d), vol alvo 20%, longo/flat (short opcional) tem Sharpe líquido > buy&hold em Sharpe e max drawdown menor, após taxa 0,06% + slippage 0,03% + funding real.
- Refutação: Sharpe líquido OOS (walk-forward, ≥2 anos retidos) < 0,5; ou não supera B&H em Sharpe E em drawdown; DSR < 0,95 considerando todas as variantes; dependente de um único ano/ativo (sem o melhor ano, Sharpe < 0,3); sensível: ±30% nos lookbacks derruba Sharpe > 50%; custos > 30% do retorno bruto.

**E2 – Funding carry delta-neutro (spot+perp)**
- H: capturar funding com alavancagem ≤2x rende líquido > 5% a.a. (acima de T-bill ~4–5%) com max drawdown < 5% e sem liquidação em stress de ±30% em 1 dia.
- Refutação: retorno líquido OOS (2023+) < taxa livre de risco + 3% a.a.; PnL concentrado em poucos episódios (ex.: 2021); funding negativo/baixo > 40% do tempo; custos de rotação > 30% do carry; margem mínima < 30% no stress.
- Nota: considerar risco de exchange e capital preso; teste de viabilidade econômica, não de "alfa".

**E3 – Meta-labeling sobre a regra E1 com CPCV + DSR/PBO**
- H: LightGBM com features simples (vol realizada, retornos defasados, funding, OI, basis, distância à média) como filtro/dimensionador da regra E1, treinado com triple-barrier e sample weights, melhora Sharpe líquido OOS e reduz drawdown ou turnover frente à regra pura.
- Refutação: p5 da distribuição de Sharpe OOS (CPCV) ≤ regra pura; PBO > 0,3; DSR < 0,95; ganho de Sharpe < 0,2; AUC OOS < 0,53; ganho desaparece com embargo maior ou outro período.
- Pré-registrar o número de configurações testadas.

## Regras gerais de refutação/higiene
1. Reservar holdout final intocado (últimos 12–18 meses), aberto uma única vez.
2. Custos conservadores: taker em tudo, + slippage, funding real; maker só após medir em paper trading.
3. Contabilizar todos os testes feitos (incluindo os do comitê de agentes) para DSR.
4. Comparar com baselines tolos (buy&hold, SMA200, carry constante).
5. Paper trade ≥ 3 meses antes de capital real; começar pequeno.
6. Expectativa honesta: retorno ajustado a risco modesto; chance real de que nenhum edge sobreviva, e isso também é resultado válido.

## Principais fontes (verificar links)
- Schmeling, Schrimpf, Todorov, "Crypto Carry", BIS WP 1087: https://www.bis.org/publ/work1087.htm
- Liu & Tsyvinski, "Risks and Returns of Cryptocurrency", RFS 2021: https://doi.org/10.1093/rfs/hhaa113
- Liu, Tsyvinski, Wu, "Common Risk Factors in Cryptocurrency", JF 2022
- Moskowitz, Ooi, Pedersen, "Time Series Momentum", JFE 2012
- Hurst, Ooi, Pedersen, "A Century of Evidence on Trend-Following Investing"
- Barroso & Santa-Clara, "Momentum has its moments", JFE 2015; Moreira & Muir, JF 2017; Cederburg et al., JFE 2020
- Bailey & López de Prado, "The Deflated Sharpe Ratio", 2014 (SSRN 2460551); Bailey et al., "The Probability of Backtest Overfitting" (SSRN 2326253)
- López de Prado, Advances in Financial Machine Learning (2018)
- Joubert, "Meta-Labeling: Theory and Framework", JFDS 2022
- Harvey, Liu, Zhu, "...and the Cross-Section of Expected Returns", RFS 2016
- Binance public data: https://github.com/binance/binance-public-data / https://data.binance.vision
- Bitget API docs: https://www.bitget.com/api-doc/contract/market/Get-History-Funding-Rate
