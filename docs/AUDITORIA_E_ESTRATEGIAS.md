# Auditoria da documentação original e fila de estratégias

> Comparação item a item entre `IA_Trader_Documentacao_FINAL_VERSAO_1_0.txt` e o estado atual do repositório
> (inspecionado em 2026-10-14). Legenda: ✅ feito · 🟡 parcial · ❌ não feito · ⛔ removido/alterado por decisão
> (precisa da confirmação do usuário).

## Parte 1: o que se perdeu ou ficou pela metade

### Módulos
| Item da documentação original | Estado | Observação |
| :--- | :---: | :--- |
| core.py (orquestrador) | 🟡 | Existe, só simulação. Faltam: re-treino diário, múltiplos pares/timeframes em operação, exportação do sinal para o painel e para os bots de execução. |
| feature_engineering / model_trainer / signal_generator / data_loader | ✅ | Feitos e testados. Lacuna: o doc pede **grid search** no LightGBM; confirmar se foi feito (não encontrei evidência). |
| `liquidity_zones.py`, `macro_data.py`, `twitter_sentiment.py` (cronograma, item 2) | ❌ | Nunca criados como módulos. A funcionalidade de macro existe dentro de data_integrations; liquidez e Twitter, não. |
| backtesting.py | ✅ | Falta o item "base para comparação entre versões de modelo" (registro de versões lado a lado). |
| smart_trade.py (futuros, alavancagem por confiança/volatilidade, lote mínimo, trail stop, limite de drawdown, máx. perdas consecutivas, conversão em spot) | ❌ | Nada feito. Ver nota de risco abaixo. |
| chatgpt_assistant.py | 🟡 | Só um stub desligado. Faltam: acionamento automático em baixa confiança, uso manual via painel, ajuda em parâmetros. **Conflito com a restrição de dados gratuitos:** a API da OpenAI é paga. |
| flask_interface.py: endpoints `/predict`, `/set_params`, `/report`, `/ask_chatgpt` | ❌ | O painel atual usa `/api/*` e não tem esses endpoints. |
| scheduler.py (APScheduler, log por ciclo) | ❌ | |
| learning_tracker.py (histórico de sinais e resultados, análise de erros, sugestões de mutação de código) | ❌ | |
| rollback_guard.py | ❌ | |
| performance_tracker.py (futuros e spot, exportação CSV/JSON) | ❌ | |
| pair_scanner.py (volatilidade, delistagem, capital reservado) | ❌ | |
| educational_module.py (texto e áudio) | ❌ | |
| voice_commands.py | ❌ | |
| Gestão de usuários: `/auth/register`, `/auth/login`, `/auth/user-info`, `/auth/validate-token`, `/auth/renew`; tabela de usuários; planos Free/Pro/Premium; Flask-Limiter | ❌ | As dependências estão no requirements.txt, mas nada foi implementado. O painel **não tem login**. |
| Legal: `/terms`, `/privacy`, `legal/*.html`, `LICENSE.txt`, middleware de aceite | ❌ | |
| `log_handler.py` (loguru centralizado) | ❌ | |
| `/health` | ✅ | |
| Empacotamento PyInstaller, hash SHA256 do build, remoção do `.env` do .exe | ❌ | |
| Pasta `/config` com parâmetros em yaml/json | ❌ | Está **vazia**: os parâmetros estão fixos no código. |
| Estrutura de pastas do doc | 🟡 | Os módulos de dados foram para `/data` e os agentes para `/agents`, diferente do doc. Aceitável, mas documentar. |

### Fontes de dados (item 11)
| Fonte | Estado | Observação |
| :--- | :---: | :--- |
| Binance | 🟡 | Em implementação agora (conector, histórico, validação cruzada). Antes só existia como fallback de OI. |
| TradingView (tvdatafeed) | ⛔ | Removi por instabilidade (API não oficial). Era para validação externa de indicadores. **Decisão sua:** manter removido ou reintroduzir como validação opcional. |
| CoinGlass | 🟡 | Conector com fallback para Bitget/Binance; scraping real do CoinGlass não é confiável. |
| Bookmap | ⛔ | Removido (sem API gratuita). Substituído por order book da Bitget/Binance. |
| CryptoPanic, CoinMarketCal | 🟡 | Conectores existem, mas sem histórico gratuito profundo: só servem ao vivo. |
| Twitter/Tweepy | ❌ | Não implementado. O `.env` tem os campos. A API gratuita do X tende a ser inviável para leitura (a verificar), o que conflita com a restrição de dados gratuitos. |
| FRED, Yahoo Finance, DefiLlama | ✅ | |
| CryptoQuant (free tier) | ❌ | Não implementado: o agente on-chain usa só DefiLlama. |
| Glassnode | ⛔ | Removido (sem API aberta). |
| Fallback via scraping | 🟡 | Parcial. |

### Problemas encontrados nesta revisão
1. **O `.env` não estava protegido.** Não existia `.gitignore`. Criei um, incluindo `.env`, `.venv`, bancos SQLite e screenshots. O `.env` atual não tem chaves preenchidas, pelo que vi.
2. **Nomes de variáveis divergentes:** o `.env` original usa `BITGET_SECRET_KEY`, o painel de configurações grava `BITGET_API_SECRET`. É preciso unificar antes de qualquer uso real das chaves.
3. **Painel sem autenticação,** escutando em `0.0.0.0`. O doc previa login JWT. Enquanto não houver, o painel só deve ser usado em rede local confiável.
4. **Risco no `smart_trade.py` como descrito:** "alavancagem variável com base em confiança" amplifica perdas quando não há vantagem comprovada. Se vier a ser construído, deve nascer limitado a paper trading, alavancagem fixa baixa e travas rígidas (drawdown e perdas consecutivas).
5. **`/config` vazia** e **CHANGELOG.md vazio.**

### Decisões que preciso de você
- tvdatafeed: manter removido ou reintroduzir como validação opcional?
- Assistente de LLM: a restrição "só dados gratuitos" exclui a API paga da OpenAI. Alternativas gratuitas: um modelo local (a RTX 3060 de 6 GB roda modelos de 7–8B quantizados), ou o plano gratuito de alguma API de LLM. Qual prefere?
- Twitter/X: tentar mesmo assim, ou aceitar que sentimento ao vivo venha só de CryptoPanic e Fear&Greed?
- CryptoQuant: vale implementar o tier gratuito?
- Ordem de prioridade dos módulos que faltam (sugestão: config, log_handler, performance_tracker, scheduler, depois auth/legal).

---

## Parte 2: fila de estratégias para os testes

Princípio: cada estratégia nova aumenta o número de tentativas e piora a correção estatística (Deflated Sharpe).
Por isso priorizamos poucas, com razão econômica clara, e descartamos as que dependem de dados que não temos.

### Em andamento (Fase R)
| ID | Estratégia | Estado |
| :--- | :--- | :--- |
| E1 | Trend/TSMOM diário com vol targeting | Em teste |
| E2 | Funding carry delta-neutro (spot + perp) | Em teste |
| E3 | Meta-labeling sobre E1 | Condicionada a E1 |

### Candidatas para a fila (ordenadas por evidência × viabilidade)
| ID | Estratégia | Razão econômica | Dados necessários | Evidência | Hipótese de refutação (resumo) |
| :--- | :--- | :--- | :--- | :---: | :--- |
| E4 | **Portfólio trend + carry** (combinar E1 e E2) | Duas fontes de retorno pouco correlacionadas melhoram o Sharpe sem criar sinal novo. | Já temos | Média | Sharpe da combinação não supera o melhor componente isolado ajustado por risco, ou a correlação em stress sobe perto de 1. |
| E5 | **Filtro de regime sobre trend** (SMA200, volatilidade realizada, DXY/VIX/Fear&Greed) | Trend sofre em mercados laterais; um filtro simples pode reduzir drawdown. | Já temos | Média-baixa | Não reduz drawdown nem melhora Sharpe em walk-forward frente ao trend puro; ou depende de um único ano. |
| E6 | **Momentum cross-sectional semanal em 10–15 moedas líquidas** (long top quartil; versão long-short como variante) | Efeito documentado em cripto, mas instável. | Binance (em andamento) | Média-baixa | Sharpe líquido < 0,5 OOS; concentrado em poucas moedas ou num só ano; custo > 30% do bruto; survivorship bias explica o resultado. |
| E7 | **Compressão de volatilidade → rompimento** (Bollinger squeeze, ATR percentil baixo) em 4h/1d | Volatilidade agrupa; rompimento após compressão tem razão comportamental. Pouco turnover. | Já temos | Baixa-média | Sem vantagem sobre rompimento aleatório com o mesmo turnover; win rate abaixo do equilíbrio. |
| E8 | **Funding extremo como sinal contrário** (funding muito positivo + OI subindo → reversão/long squeeze) | Posicionamento alavancado unilateral é frágil. | Funding Bitget/Binance, OI (profundidade curta) | Baixa-média | Poucos eventos (< 100) para conclusão → rotular amostra insuficiente; sem vantagem após custos. |
| E9 | **Pairs/stat-arb BTC–ETH** (spread cointegrado, entradas por z-score) | Relação de longo prazo entre os dois; baixa exposição direcional. | Já temos | Baixa | Cointegração instável fora da amostra; custos de duas pernas consomem o retorno. |
| E10 | **Sazonalidade** (fim de semana, virada de mês, horário de liquidação do funding) | Efeitos de liquidez e fluxo. | Já temos | Baixa | Não sobrevive à correção por múltiplas comparações; **testar com poucas hipóteses fixas, não varrendo horários**. |

### Não testar agora (sem dados ou sem base)
- **Market making / execução passiva como fonte de lucro:** exige L2 e trades históricos, que não temos. Ação útil: **começar a gravar L2, trades, OI e liquidações por WebSocket desde já**, para ter dados daqui a alguns meses.
- **Fade de cascatas de liquidação:** mesmo problema (sem histórico gratuito profundo).
- **ML direcional em indicadores técnicos de 1m a 1h:** já testado, sem vantagem; só reavaliar com dados novos.
- **Deep learning (LSTM/Transformers):** poucos ativos e poucos anos; a chance de overfit supera a de ganho.
- **Sentimento do Twitter:** inviável sem acesso gratuito e sem histórico.

### Regras para a fila
1. Máximo de 3 estratégias novas por ciclo, cada uma com hipótese e refutação escritas **antes** do teste.
2. Todas contam no `trial_log.csv` e na correção do Deflated Sharpe.
3. Comparar sempre com buy&hold, SMA200 e carry constante.
4. O holdout final continua trancado.
5. A decisão final sobre dinheiro real só depois de 8–12 semanas de paper trading.
