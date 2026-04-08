# Nivel de dificuldade — Complexidade do Raciocinio do LLM

## Abordagem

Em vez de classificar as questoes como Facil/Medio/Dificil, esta curadoria classifica o **tipo de processamento cognitivo** que um LLM precisa realizar para acertar a questao.

## Niveis de complexidade

| Nivel | Nome | Tipo (EN) | Criterios |
|---|---|---|---|
| **1** | Recuperacao Factual Direta | *Fact Retrieval* | A resposta depende apenas da memorizacao de um artigo de lei especifico ou conceito exato. A IA so precisa "lembrar" o conteudo legislativo |
| **2** | Raciocinio Logico-Dedutivo | *Logical Deduction* | A questao apresenta um caso concreto. A IA precisa extrair os fatos do texto e aplicar uma regra juridica clara (Se A, entao B) |
| **3** | Hermeneutica Juridica Complexa | *Complex Hermeneutics* | A questao exige interpretacao profunda, cruzamento de multiplas leis, analise de jurisprudencia ou lida com ambiguidades legais |

### Regra especial

Pecas pratico-profissionais devem ser classificadas como **nivel 3** (Complex Hermeneutics), independentemente de outros criterios, devido a sua complexidade inerente.

## Formato de saida

O modelo retorna um JSON estruturado:

```json
{
  "question_id": "41_direito_administrativo_questao_1",
  "dificuldade": 2,
  "nivel": "Raciocínio Lógico-Dedutivo"
}
```

## Implementacao

A classificacao e feita pelo modelo `llama3.2:3b` com `temperature=0` para garantir determinismo. O prompt completo esta em `src/templates/curator_difficulty.jinja`.
