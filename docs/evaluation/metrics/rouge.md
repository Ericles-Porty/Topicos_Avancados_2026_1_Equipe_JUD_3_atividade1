# ROUGE

## O que e

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) e uma familia de metricas que avalia a qualidade de texto gerado com foco em **recall** — ou seja, quanto do conteudo da referencia e coberto pelo texto gerado.

## Variantes utilizadas

| Variante | O que mede |
|---|---|
| **ROUGE-1** | Overlap de unigramas (palavras individuais) |
| **ROUGE-2** | Overlap de bigramas (pares de palavras consecutivas) |
| **ROUGE-L** | Maior subsequencia comum (LCS) entre os textos |

### ROUGE-1

Mede quantas palavras da referencia aparecem no texto gerado. Captura cobertura de vocabulario geral.

### ROUGE-2

Mede quantos pares de palavras consecutivas da referencia aparecem no texto gerado. Captura similaridade de estrutura frasal.

### ROUGE-L

Mede a maior sequencia de palavras que aparece em ambos os textos (nao necessariamente consecutivas). Captura similaridade de estrutura de longo alcance.

## Escala

- **0:** Nenhuma cobertura do conteudo de referencia
- **1:** Cobertura completa

## Uso no projeto

```python
import evaluate as hf_evaluate

rouge_metric = hf_evaluate.load("rouge")
result = rouge_metric.compute(predictions=preds, references=refs)
# result["rouge1"], result["rouge2"], result["rougeL"]
```

## Diferenca em relacao ao BLEU

| Aspecto | BLEU | ROUGE |
|---|---|---|
| Foco | Precisao (o que o modelo gerou esta na referencia?) | Recall (o que a referencia contem esta no texto gerado?) |
| Uso tipico | Traducao automatica | Sumarizacao, geracao de texto |
| Penalidade | Textos curtos (brevidade) | — |

Ambas as metricas sao baseadas em overlap de n-gramas e nao capturam similaridade semantica. Para isso, utilizamos o [BERTScore](bertscore.md).
