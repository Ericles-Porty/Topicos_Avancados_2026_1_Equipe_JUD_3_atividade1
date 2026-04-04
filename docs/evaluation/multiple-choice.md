# Avaliacao — Multipla escolha

## Objetivo

Avaliar as respostas dos modelos as questoes objetivas comparando com o **gabarito oficial** do dataset OAB Exams.

## Como funciona

### Extracao da resposta

A resposta do modelo e processada em duas etapas:

1. **Tentativa JSON:** Limpa blocos markdown (` ```json `) e tenta parsear como JSON, extraindo o campo `"resposta"`
2. **Fallback regex:** Se o JSON falhar, busca a primeira ocorrencia de uma letra A-D no texto

```python
# Etapa 1: JSON
json_str = _extract_json(model_answer)
if json_str:
    resp = json.loads(json_str)
    extracted = resp.get("resposta", "").strip().upper()[:1]

# Etapa 2: Regex fallback
if not extracted or extracted not in "ABCD":
    match = re.search(r"\b([A-D])\b", model_answer.upper())
    extracted = match.group(1) if match else model_answer[:1].upper()
```

### Comparacao com gabarito

A resposta extraida e comparada diretamente com o campo `answerKey` do dataset. O resultado e um booleano `is_correct`.

## Metricas calculadas

### No leaderboard

| Metrica | Descricao |
|---|---|
| **Acuracia** | Percentual de respostas corretas |
| **Precision (macro)** | Media da precisao por classe (A, B, C, D) |
| **Recall (macro)** | Media do recall por classe |
| **F1 (macro)** | Media harmonica de Precision e Recall por classe |

As metricas Precision, Recall e F1 sao calculadas via `sklearn.metrics` com `average="macro"` e `zero_division=0`, tratando cada alternativa como uma classe de classificacao.

Para mais detalhes sobre as metricas, consulte [Acuracia, Precision, Recall, F1](metrics/accuracy-precision.md).

## Saida

Os resultados sao salvos em `src/results/eval_multiple_choice.csv` com as colunas:
- `question_id` — Identificador da questao
- `model` — Nome do modelo
- `answer` — Resposta extraida do modelo
- `correct` — Gabarito oficial
- `is_correct` — Se a resposta esta correta
