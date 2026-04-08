# Curadoria — Visao geral

## O que e curadoria

Curadoria e o processo de **enriquecimento dos dados** com metadados que avaliam cada questao sob a otica da **Complexidade de Raciocinio (Reasoning)** e do **Aterramento (Grounding)** exigidos da IA. Neste projeto, cada questao e anotada automaticamente com:

1. **Nivel de dificuldade (Complexidade do Raciocinio do LLM)** — Classificacao do tipo de processamento cognitivo que o LLM precisa realizar:
    - Nivel 1: Recuperacao Factual Direta (*Fact Retrieval*)
    - Nivel 2: Raciocinio Logico-Dedutivo (*Logical Deduction*)
    - Nivel 3: Hermeneutica Juridica Complexa (*Complex Hermeneutics*)
2. **Subdominio Semantico** — Area de especialidade juridica (ex: Direito Civil, Direito Penal, Direito Constitucional)
3. **Corpus de Referencia (Ground Truth)** — Legislacao ou fonte normativa onde a resposta correta deve estar ancorada para evitar alucinacoes

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
Questao → Prompt de Subdominio     → LLM (llama3.2:3b) → JSON {subdominio_semantico}
                                                              ↓
Questao → Prompt de Corpus         → LLM (llama3.2:3b) → JSON {corpus_referencia}
                                                           ↓
                                                 curator_annotations.json
```

## Implementacao

A funcao `run_curator_tasks()` em `src/run_models.py` processa todas as questoes (abertas e multipla escolha), gerando as anotacoes de curadoria e salvando em `src/results/curator_annotations.json`.

## Detalhes

- [Nivel de dificuldade](difficulty-level.md) — Complexidade do raciocinio do LLM
- [Subdominio semantico](specialty-area.md) — Area de especialidade juridica
- [Corpus de referencia](basic-legislation.md) — Ground truth (legislacao base)
- [Prompts](prompts.md) — Templates utilizados
