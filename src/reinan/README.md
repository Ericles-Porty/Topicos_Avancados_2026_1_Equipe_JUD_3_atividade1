# Reinan

## Execução

### Pré-requisitos

- Python 3.12+
- uv 0.10+

### Instalação

```bash
# opcional: criar e ativar venv
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\activate

# dependências mínimas
uv sync

# executar
uv run python main.py
```

## Referências

- [Best Practices and Methods for LLM Evaluation](https://www.databricks.com/br/blog/best-practices-and-methods-llm-evaluation)
- [LLM Evaluation Metrics: Everything You Need for LLM Evaluation](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
- [LLM Auto-Eval Best Practices for RAG](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG)
- [LLM Evaluation: A Comprehensive Survey](https://arxiv.org/html/2504.21202v1)
- [uv - Python package manager](https://docs.astral.sh/uv/)
- [Ruff - An extremely fast Python linter](https://docs.astral.sh/ruff/)