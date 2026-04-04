# Avaliacao comparativa

## Objetivo

Comparar as respostas de **diferentes modelos** a uma mesma questao aberta, avaliando qualidade relativa em dimensoes especificas.

## Como funciona

1. Para cada questao, todas as respostas dos modelos sao agrupadas
2. Um prompt e montado contendo a questao e todas as respostas (identificadas por modelo)
3. O modelo juiz (`llama3.2:3b`) avalia cada resposta em tres dimensoes

## Dimensoes de avaliacao

| Dimensao | Escala | O que avalia |
|---|---|---|
| **Argumentacao** | 0-5 | Qualidade da argumentacao juridica |
| **Precisao** | 0-5 | Precisao e correcao das referencias legais |
| **Coesao** | 0-5 | Coesao e clareza textual |

## Score final

O score final de cada modelo em cada questao e calculado como media ponderada:

```
final_score = 0.4 * argumentacao + 0.4 * precisao + 0.2 * coesao
```

Os pesos refletem a maior importancia da argumentacao e precisao no contexto juridico.

## Template do prompt

**Arquivo:** `src/templates/judge_comparative.jinja`

O prompt configura o modelo como "professor de direito avaliando respostas da prova da OAB" e instrui a retornar apenas JSON, sem explicacoes ou markdown.

Formato de saida esperado:

```json
{
  "llama3.2:3b": {"argumentacao": 4, "precisao": 3, "coesao": 4},
  "gemma2:2b":   {"argumentacao": 2, "precisao": 3, "coesao": 3},
  "qwen2.5:3b":  {"argumentacao": 3, "precisao": 4, "coesao": 3}
}
```

## Saida

Os resultados sao salvos em `src/results/eval_comparative.csv` com as colunas:
- `question_id`, `model`
- `argumentacao`, `precisao`, `coesao`
- `final_score`
