# Curadoria — Visao geral

## O que e curadoria

Curadoria e o processo de **enriquecimento dos dados** com metadados juridicos adicionais, agregando valor analitico ao dataset original. Neste projeto, cada questao e anotada automaticamente com:

1. **Nivel de dificuldade** — Classificacao em Facil (1), Medio (2) ou Dificil (3)
2. **Area de especialidade** — Classificacao da area juridica (Direito Civil, Penal, Constitucional, Trabalhista, etc.)
3. **Legislacao base** — Identificacao da legislacao principal que fundamenta a questao

## Abordagem automatizada

A curadoria e realizada de forma automatizada usando um **modelo de linguagem como curador** (LLM-as-Curator). O modelo `llama3.2:3b` e utilizado como curador, recebendo prompts especializados para cada tarefa.

### Vantagens da abordagem automatizada

- **Reprodutibilidade:** Os mesmos prompts com `temperature=0` produzem resultados consistentes
- **Escalabilidade:** Permite classificar centenas de questoes sem intervencao manual
- **Padronizacao:** Criterios aplicados uniformemente a todas as questoes

## Pipeline de curadoria

```
Questao → Prompt de Dificuldade    → LLM (llama3.2:3b) → JSON {dificuldade, nivel}
                                                              ↓
Questao → Prompt de Especialidade  → LLM (llama3.2:3b) → JSON {area_especialidade}
                                                              ↓
Questao → Prompt de Legislacao     → LLM (llama3.2:3b) → JSON {legislacao_base}
                                                           ↓
                                                 curator_annotations.json
```

## Implementacao

A funcao `run_curator_tasks()` em `src/run_models.py` processa todas as questoes (abertas e multipla escolha), gerando as anotacoes de curadoria e salvando em `src/results/curator_annotations.json`.

## Detalhes

- [Nivel de dificuldade](difficulty-level.md) — Criterios e classificacao
- [Area de especialidade](specialty-area.md) — Classificacao por area juridica
- [Legislacao base](basic-legislation.md) — Identificacao de legislacao
- [Prompts](prompts.md) — Templates utilizados
