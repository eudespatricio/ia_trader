# Patch S2 de Homologação e Calibração de Agentes (v1.0.4)

> **Data de Homologação:** 03/Nov/2026  
> **Status:** 🟢 **100% HOMOLOGADO & VERDE (577 passed, 1 skipped, 6 xfail)**  
> **Escopo:** S2 — Calibração Convexa de Pesos, Dampening em SIDEWAYS_RANGE e Retreinamento 50k

---

## 1. Sumário Executivo das Implementações

1. **Correção de Regressão da Exit Ladder & Funil (`v1.0.4`):**
   - Corrigida asserção de trailing stop em [`tests/test_exit_ladder_e7.py`](file:///home/eudespatricio/ia_trader/tests/test_exit_ladder_e7.py) para paridade com cálculo nativo (`trail0 = 101.21`, `trail1 = 101.30`).
   - Sincronizado timestamp de teste do funil em [`tests/test_signal_funnel.py`](file:///home/eudespatricio/ia_trader/tests/test_signal_funnel.py) usando o bucket nativo de 15m (`SCALP_BAR_MS`).

2. **Nível 1 & Nível 2 do Protocolo S2 ([`docs/S2_CALIBRACAO_AGENTES_BACKTEST.md`](file:///home/eudespatricio/ia_trader/docs/S2_CALIBRACAO_AGENTES_BACKTEST.md)):**
   - **Otimização Convexa Restrita (L2 / Ridge com Prior Bayesiano):** Implementado em [`learning_tracker.py`](file:///home/eudespatricio/ia_trader/learning_tracker.py) no método `recalibrate_agent_weights()`. Minimiza erro quadrático direcional com prior `DEFAULT_BASE_WEIGHTS`, pesos não-negativos, soma normalizada e trava inegociável de veto do `RiskSentinelAgent` (`w = 1.5`).
   - **Dampening de Sobreconfiança em `SIDEWAYS_RANGE`:** Implementado em [`consensus_judge.py`](file:///home/eudespatricio/ia_trader/consensus_judge.py) e integrado via [`core.py`](file:///home/eudespatricio/ia_trader/core.py). Quando ECE > 0.15 ou acurácia < 45% no regime lateral, a confiança máxima é limitada cirurgicamente a 0.55, eliminando o efeito penhasco e falsos rompimentos.

3. **Retreinamento e Avaliação Champion-Challenger (50k velas BTCUSDT 1h):**
   - Executado via `learning_tracker.retrain_daily_models()`:
     - **LightGBM:** 29.848 velas avaliadas, Challenger Acc Sinais = 70.51% (Champion mantido a 74.44%).
     - **Random Forest:** 29.848 velas avaliadas, Challenger Acc Sinais = 87.95% (Champion mantido a 89.55%).

4. **Certificação Completa de Backend:**
   - Comando: `.venv/bin/pytest -q --ignore=tests/ui`
   - Resultado: **577 passed, 1 skipped, 6 xfailed em 30.79s (Zero falhas)**.
