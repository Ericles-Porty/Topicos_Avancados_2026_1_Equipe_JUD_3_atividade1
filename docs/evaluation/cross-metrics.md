# Metricas automatizadas (Cross-Model)

## Objetivo

Comparar as respostas entre **pares de modelos** usando metricas automatizadas de NLP, fornecendo medidas **reprodutiveis e objetivas** de similaridade textual e semantica.

## Metricas utilizadas

| Metrica | O que mede | Escala |
|---|---|---|
| **BLEU** | Overlap de n-gramas (precisao) | 0-1 |
| **ROUGE-1** | Overlap de unigramas (recall) | 0-1 |
| **ROUGE-2** | Overlap de bigramas (recall) | 0-1 |
| **ROUGE-L** | Maior subsequencia comum (recall) | 0-1 |
| **BERTScore F1** | Similaridade semantica via embeddings | 0-1 |

Para detalhes de cada metrica, consulte:
- [BLEU](metrics/bleu.md)
- [ROUGE](metrics/rouge.md)
- [BERTScore](metrics/bertscore.md)

## Como funciona

1. Gera todos os pares possiveis de modelos usando `itertools.combinations`
2. Para cada par, coleta as respostas as mesmas questoes
3. Calcula as metricas usando o framework HuggingFace Evaluate
4. Salva os resultados

### Pares avaliados

Com 3 modelos (Mistral, Llama 3, Gemma), sao gerados 3 pares:
- Mistral vs Llama 3
- Mistral vs Gemma
- Llama 3 vs Gemma

## Implementacao

```python
import evaluate as hf_evaluate

bleu_metric = hf_evaluate.load("bleu")
rouge_metric = hf_evaluate.load("rouge")
bertscore_metric = hf_evaluate.load("bertscore")

# Para cada par de modelos:
bleu = bleu_metric.compute(predictions=preds, references=refs)
rouge = rouge_metric.compute(predictions=preds, references=refs)
bert = bertscore_metric.compute(predictions=preds, references=refs, lang="pt")
```

O BERTScore utiliza `lang="pt"` para selecionar o modelo multilingual adequado (`bert-base-multilingual-cased`).

## Saida

Os resultados sao salvos em `src/results/eval_cross_metrics.csv` com as colunas:
- `pair` — Par de modelos (ex: "llama3.2:3b vs gemma2:2b")
- `bleu`, `rouge1`, `rouge2`, `rougeL`, `bertscore_f1`

## Complementaridade com a avaliacao por juiz

As metricas automatizadas **complementam** (nao substituem) a avaliacao comparativa por juiz:

| Aspecto | Juiz LLM | Metricas automatizadas |
|---|---|---|
| Reprodutibilidade | Baixa (varia entre execucoes) | Alta (deterministicas) |
| Subjetividade | Alta (interpreta qualidade) | Baixa (mede similaridade) |
| O que mede | Qualidade juridica | Similaridade textual/semantica |
