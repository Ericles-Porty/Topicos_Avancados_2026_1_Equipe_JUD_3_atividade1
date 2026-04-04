# Avaliacao por rubrica — Questoes abertas

## Objetivo

Avaliar as respostas dos modelos as questoes discursivas utilizando a **rubrica oficial** do Exame da OAB, simulando a correcao que seria feita por um avaliador humano.

## Como funciona

1. Para cada resposta de cada modelo, o sistema monta um prompt contendo:
   - O enunciado da questao
   - As subperguntas (`turns`)
   - Os valores de cada item (`values`)
   - A resposta do modelo
   - Os criterios de correcao oficiais (`rubric`, extraidos do subset `guidelines`)

2. O prompt e enviado ao modelo juiz (`llama3.2:3b`) com `temperature=0`

3. O modelo juiz retorna um JSON com notas para cada subquestao:
   ```json
   {"scores": [0.5, 0.6], "total": 1.1}
   ```

## Template do prompt

**Arquivo:** `src/templates/judge_open_question.jinja`

O prompt configura o modelo como "corretor da prova da OAB" e apresenta:
- Pergunta completa
- Subquestoes e valores
- Resposta do candidato (modelo)
- Criterios de correcao

### Regras de correcao

- Cada nota deve respeitar o valor maximo definido em `values`
- Nao usar formato fracionario (ex: `0.00/0.10`)
- Retornar apenas numeros

## Saida

Os resultados sao salvos em `src/results/eval_open_questions.csv` com as colunas:
- `question_id` — Identificador da questao
- `model` — Nome do modelo avaliado
- `scores` — Lista de notas por subquestao
- `total_score` — Soma das notas

## Tratamento de erros

Se o modelo juiz retornar uma resposta que nao pode ser parseada como JSON, os campos `scores` e `total_score` sao preenchidos com `None`. A funcao `_clean_score()` lida com formatos variados de nota (virgula vs ponto, formato fracionario).
