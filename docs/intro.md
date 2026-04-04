# Introducao

Bem-vindo a documentacao do projeto da **Equipe 3 (Juridica)** para a disciplina **Topicos Avancados em Engenharia de Software e Sistemas de Informacao I** — UFS, semestre 2026.1.

## Objetivo

Este projeto realiza a **curadoria de datasets juridicos** e a **inferencia basica com Modelos de Linguagem (LLMs)**, com foco em questoes do Exame da Ordem dos Advogados do Brasil (OAB).

## Contribuicao individual

Este repositorio contem as contribuicoes realizadas pelo aluno **Ericles dos Santos**, incluindo:

- Carregamento e preparacao dos datasets (OAB Bench + OAB Exams)
- Curadoria automatizada: classificacao de dificuldade e identificacao de legislacao base
- Inferencia com tres modelos (Mistral, Llama 3, Gemma) via Ollama
- Avaliacao por rubrica, comparativa e metricas automatizadas (BLEU, ROUGE, BERTScore)
- Geracao de leaderboard consolidado com metricas de classificacao (Precision, Recall, F1)

## Estrutura da documentacao

| Secao | Descricao |
|---|---|
| [Primeiros passos](getting-started/prerequisites.md) | Pre-requisitos, instalacao e execucao rapida |
| [Datasets](datasets/overview.md) | Descricao dos datasets utilizados |
| [Curadoria](curation/overview.md) | Classificacao de dificuldade e legislacao base |
| [Inferencia](inference/overview.md) | Hardware, modelos e prompts utilizados |
| [Avaliacao](evaluation/overview.md) | Estrategias e metricas de avaliacao |
| [Resultados](results/overview.md) | Resultados obtidos e analise |

## Datasets utilizados

| Dataset | Tipo | Quantidade total | Subconjunto utilizado |
|---|---|---|---|
| **J1 — OAB Bench** | Questoes Abertas | 210 | 12 questoes (indices 153–164) |
| **J2 — OAB Exams** | Multipla Escolha | 2210 | 123 questoes (indices 1600–1722) |
