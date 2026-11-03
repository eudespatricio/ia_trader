# 📈 Handover de Arquitetura & Operação — IA Trader

> **Status:** ✅ 100% Consolidado & Isolado no WSL Nativo | **Versão:** 1.0.0 (Fase 10 Homologada)  
> **Data de Transição:** 20/10/2026 | **Responsável:** Engenharia Quantitativa & Sistemas

---

## 🏛️ 1. Local Oficial & Repositório Git

O projeto **IA Trader** é 100% autônomo e isolado:

- **Diretório Oficial:** `/home/eudespatricio/ia_trader`
- **Repositório Git:** `https://github.com/eudespatricio/ia_trader.git` (branch `main`)
- **Ambiente Virtual:** `/home/eudespatricio/ia_trader/.venv` (Python 3.12 nativo)
- **Base de Dados:** 
  - `data/ia_trader.db` (Paper Trading, Histórico, Ordens e PnL)
  - `data/microstructure.db` (Order Book, Tape Reading, CVD e LOB)
  - `market_cache.db` (Cache de baixa frequência / macro)

---

## 🧹 2. Limpeza e Consolidação de Documentações

Resquícios soltos que residiam na pasta compartilhada do Windows (`D:\Projeto`) foram consolidados dentro do IA Trader:
- `docs/IA_Trader_Documentacao_FINAL_VERSAO_1_0.txt`: Documentação institucional completa.
- `.agents/`: Skills de engenharia quantitativa e risco integradas.
- O IA Trader não possui nenhuma dependência ou vínculo com discos externos NTFS.

---

## 🚀 3. Processos Operacionais e Monitoramento

O ecossistema do IA Trader opera em regime contínuo com múltiplos subsistemas:

| Subsistema | Comando / Script | Função / Porta |
| :--- | :--- | :--- |
| **Interface Web & REST** | `flask_interface.py 5000` | Dashboard Plotly & API REST (`http://localhost:5000`) |
| **Paper Trading Engine** | `scripts/run_paper_simulation.py 0 15` | Execução ao vivo com cotas independentes e Exit Ladder |
| **Coletor Microestrutura** | `scripts/microstructure_collector.py` | Coleta em tempo real de LOB, CVD e fluxo institucional |
| **Cloudflare Tunnel** | `bin/cloudflared tunnel --url http://127.0.0.1:5000` | Exposição segura para monitoramento externo |

Para checar os processos em execução:
```bash
ps aux | grep -E "flask_interface|run_paper_simulation|microstructure_collector" | grep -v grep
```

---

## 🌐 4. Healthcheck & Telemetria

- **Status Geral:** `curl -s http://localhost:5000/api/status`
- **Paper Trading:** `curl -s http://localhost:5000/api/paper-trading/status`
- **Telemetria de Performance:** `curl -s http://localhost:5000/api/telemetry/performance`
- **Comitê Multi-Agente:** `curl -s http://localhost:5000/api/committee/deliberate`

---

## 🧪 5. Testes e Governança

Para validar o funcionamento completo da suíte de 228 testes backend:
```bash
cd /home/eudespatricio/ia_trader
./.venv/bin/pytest tests/ -q
```
Regra de Ouro (Golden Rule): **100% dos testes devem permanecer verdes**, sem *lookahead bias* em dados macro ou de microestrutura.
