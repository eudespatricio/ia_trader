# Protocolo Experimental: como buscar lucratividade sem enganar a nós mesmos

> Regra: "testar até ficar lucrativo" produz resultados falsos. Se testarmos centenas de variações, algumas
> vão parecer lucrativas por sorte. Este protocolo existe para que um resultado positivo, se aparecer, seja real.

## 1. Regras obrigatórias para qualquer experimento

1. **Hipótese antes do teste.** Cada experimento tem, escrita antes de rodar: a hipótese, a razão econômica
   pela qual deveria funcionar, os parâmetros, e o critério objetivo de refutação.
2. **Contador de tentativas.** Todo backtest executado entra em `datasets/reports/trial_log.csv`
   (data, estratégia, parâmetros, resultado). O número total de tentativas é usado nas correções estatísticas.
3. **Correção estatística.** Resultados são julgados com Deflated Sharpe Ratio e correção de múltiplas
   comparações (Bonferroni/Holm) sobre o total de tentativas. Probabilidade de overfitting (PBO/CSCV)
   quando houver seleção entre variantes.
4. **Amostra mínima.** Nenhuma conclusão com menos de 100 trades fora da amostra (ou equivalente em períodos).
5. **Walk-forward com embargo** para tudo que tenha parâmetros ajustados. Parâmetros nunca são ajustados no teste.
6. **Custos reais sempre.** Taxa taker 0,06% e slippage 0,02% por lado, funding real. Maker só como cenário
   separado e só com fill rate e seleção adversa modelados.
7. **Holdout final trancado.** Um período recente fica fora de TODA exploração. Só é aberto UMA vez, para a
   estratégia final congelada. O holdout já usado na Fase 4 (BTC 1h, 2026-04-14 a 2026-10-14) está consumido
   para aquela configuração; novas estratégias precisam de um período novo, ou do teste a seguir.
8. **Teste à frente (paper trading).** O único teste que não pode ser enganado: operar em simulação ao vivo,
   com dados que ainda não existiam, por pelo menos 8–12 semanas, sem alterar a estratégia.

## 2. Critérios para chamar algo de "lucrativo"

Todos precisam ser verdadeiros, depois dos custos:

- Lucro líquido positivo OOS com pelo menos 100 trades ou 2 anos de dados;
- Deflated Sharpe Ratio com probabilidade ≥ 95% de o Sharpe verdadeiro ser positivo;
- Resultado positivo na maioria dos folds do walk-forward (não só na média);
- Resultado positivo em mais de um ativo OU com razão econômica clara para um só;
- Drawdown máximo compatível com o capital que o usuário aceita perder;
- Sobrevive ao holdout trancado e a 8–12 semanas de paper trading.

## 3. Critério de parada

Se após os experimentos priorizados em `docs/PESQUISA_ESTRATEGIAS.md` nenhuma estratégia atender aos
critérios, o relatório final dirá que **não encontramos vantagem implementável com dados gratuitos e custos
de varejo**. Isso é um resultado válido e evita perda de dinheiro real.

## 4. Dinheiro real

Fora do escopo deste protocolo. Só pode ser discutido com o usuário depois de a estratégia passar em todos os
critérios acima, e mesmo assim começando com um valor que o usuário aceite perder por inteiro.
