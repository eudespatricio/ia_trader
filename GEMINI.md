# Workspace Token Guard & Local-First Hardware Acceleration

## 1. Local-First & Potencial Máximo do Hardware Local (WSL / CPU / GPU)
- **Computação Local Antes de Tokens**: Esgotar o poder computacional local do WSL (CPU i7 16 threads, GPU RTX 3060, bash, scripts python) antes de delegar para inferência em nuvem.
- **Filtragem Cirúrgica no Shell**: Nunca despeje logs ou dumps de bancos SQLite (`market_cache.db`, etc.) no contexto. Utilize comandos locais (`sqlite3`, `grep`, `awk`, `tail -n 20`, scripts pontuais) para filtrar estritamente a informação exata.
- **Offloading para GPU Local (Ollama)**: Sempre que uma tarefa de LLM for necessária no trading/árbitro, utilize a inferência local na RTX 3060 (`http://localhost:11434`) a custo zero de API.
- **Processos Locais vs Subagentes**: Execute tarefas pesadas via background processes no WSL em vez de disparar subagentes de nuvem.

## 2. Monitoramento de Contexto & Alerta de /compact
- **Vigilância de Contexto**: Em conversas com mais de 10 a 12 interações ou após inspeção de logs, sugira ao final o `/compact` ou abertura de nova sessão (`Ctrl + N`).

## 3. Concisão e Leituras Cirúrgicas
- Respostas técnicas e diretas, sem repetições de código intacto.
- Leitura de arquivos restrita aos intervalos de linhas necessários (`StartLine` / `EndLine`).
