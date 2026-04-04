# Estrutura do projeto

```
.
├── LICENSE                          # Licenca MIT
├── README.md                        # Documentacao principal
├── .gitignore                       # Arquivos ignorados pelo Git
├── docs/                            # Documentacao detalhada
│   ├── intro.md                     # Introducao e visao geral
│   ├── references.md                # Referencias bibliograficas
│   ├── getting-started/             # Primeiros passos
│   │   ├── prerequisites.md
│   │   ├── installation.md
│   │   ├── quick-start.md
│   │   └── project-structure.md
│   ├── datasets/                    # Documentacao dos datasets
│   │   ├── overview.md
│   │   ├── oab-bench.md
│   │   ├── oab-exams.md
│   │   └── team-distribution.md
│   ├── curation/                    # Curadoria automatizada
│   │   ├── overview.md
│   │   ├── difficulty-level.md
│   │   ├── basic-legislation.md
│   │   └── prompts.md
│   ├── inference/                   # Inferencia com LLMs
│   │   ├── overview.md
│   │   ├── hardware.md
│   │   ├── models.md
│   │   └── prompts.md
│   ├── evaluation/                  # Estrategias de avaliacao
│   │   ├── overview.md
│   │   ├── open-questions.md
│   │   ├── multiple-choice.md
│   │   ├── comparative.md
│   │   ├── cross-metrics.md
│   │   └── metrics/
│   │       ├── accuracy-precision.md
│   │       ├── bleu.md
│   │       ├── rouge.md
│   │       └── bertscore.md
│   └── results/
│       └── overview.md
└── src/                             # Codigo-fonte
    ├── load_dataset.py              # Carregamento e preparacao dos datasets
    ├── run_models.py                # Inferencia com LLMs + curadoria
    ├── evaluation.py                # Avaliacao e geracao de leaderboard
    ├── templates/                   # Templates de prompts (Jinja)
    │   ├── multiple_choice.jinja
    │   ├── multiple_choice_system.jinja
    │   ├── judge_open_question.jinja
    │   ├── judge_comparative.jinja
    │   ├── curator_difficulty.jinja
    │   └── curator_legislation.jinja
    ├── dataset/                     # Datasets (ignorado pelo .gitignore)
    └── results/                     # Resultados (ignorado pelo .gitignore)
```

## Descricao dos arquivos principais

### Scripts Python

| Arquivo | Descricao |
|---|---|
| `src/load_dataset.py` | Baixa os datasets OAB Bench e OAB Exams, extrai o subconjunto designado e salva como CSV |
| `src/run_models.py` | Executa inferencia com os tres modelos para questoes abertas e multipla escolha, alem de tarefas de curadoria |
| `src/evaluation.py` | Avalia respostas usando rubrica, comparacao qualitativa, metricas automatizadas e gera o leaderboard final |

### Templates de prompts

| Template | Uso |
|---|---|
| `multiple_choice.jinja` | Formata questao + alternativas para o modelo |
| `multiple_choice_system.jinja` | System prompt para MC — solicita resposta em JSON |
| `judge_open_question.jinja` | Prompt do modelo juiz para avaliacao por rubrica |
| `judge_comparative.jinja` | Prompt do modelo juiz para avaliacao comparativa |
| `curator_difficulty.jinja` | Prompt de curadoria — classificacao de dificuldade |
| `curator_legislation.jinja` | Prompt de curadoria — identificacao de legislacao base |
