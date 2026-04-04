# Inferencia — Visao geral

## O que e inferencia

Inferencia e o processo de submeter as questoes do dataset aos modelos de linguagem para obter respostas. Neste projeto, cada questao e processada por **tres modelos distintos**, gerando respostas que serao posteriormente avaliadas.

## Pipeline de inferencia

```
Dataset preparado
     ↓
┌─────────────────────────────────────────────────┐
│  Questoes Abertas                               │
│  system prompt (dataset) + enunciado + turns    │
│     → Mistral → resposta                        │
│     → Llama 3 → resposta                        │
│     → Gemma   → resposta                        │
├─────────────────────────────────────────────────┤
│  Multipla Escolha                               │
│  system prompt (JSON) + enunciado + alternativas│
│     → Mistral → {"resposta": "X"}               │
│     → Llama 3 → {"resposta": "X"}               │
│     → Gemma   → {"resposta": "X"}               │
├─────────────────────────────────────────────────┤
│  Curadoria                                      │
│  prompts especializados                         │
│     → Llama 3 → dificuldade + legislacao        │
└─────────────────────────────────────────────────┘
     ↓
Resultados em src/results/
```

## Implementacao

O script `src/run_models.py` executa:

1. **`run_open_questions()`** — Inferencia de questoes abertas com os 3 modelos
2. **`run_multiple_choice_questions()`** — Inferencia de multipla escolha com os 3 modelos
3. **`run_curator_tasks()`** — Curadoria automatizada com llama3.2:3b

## Detalhes

- [Hardware](hardware.md) — Configuracao de hardware utilizada
- [Modelos](models.md) — Modelos selecionados e justificativa
- [Prompts](prompts.md) — Templates de prompts utilizados
