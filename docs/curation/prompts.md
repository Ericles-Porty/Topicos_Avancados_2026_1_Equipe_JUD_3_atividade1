# Prompts de curadoria

## Motor de templates

O projeto utiliza **MiniJinja** como motor de templates, permitindo a insercao dinamica de dados das questoes nos prompts. Os templates estao em `src/templates/`.

## Template: Classificacao de dificuldade

**Arquivo:** `src/templates/curator_difficulty.jinja`

O prompt apresenta os criterios de classificacao (Facil/Medio/Dificil) e solicita que o modelo analise o enunciado completo, incluindo subitens quando existentes.

**Variaveis injetadas:**
- `question_id` — Identificador da questao
- `statement` — Enunciado completo
- `turns` — Lista de subitens (pode ser nula para MC)

**Saida esperada:**
```json
{"question_id": "...", "dificuldade": 0, "nivel": ""}
```

## Template: Legislacao base

**Arquivo:** `src/templates/curator_legislation.jinja`

O prompt solicita a identificacao da legislacao principal que fundamenta a questao, com regras claras sobre quando citar artigos especificos e quando retornar "Inconclusivo".

**Variaveis injetadas:**
- `question_id` — Identificador da questao
- `statement` — Enunciado completo
- `turns` — Lista de subitens (pode ser nula para MC)

**Saida esperada:**
```json
{"question_id": "...", "legislacao_base": ""}
```

## Padronizacao

Ambos os templates:
- Solicitam retorno **exclusivamente em JSON valido**, sem markdown
- Utilizam o modelo `llama3.2:3b` com `temperature=0`
- Processam tanto questoes abertas quanto de multipla escolha
