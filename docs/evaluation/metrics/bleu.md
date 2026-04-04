# BLEU

## O que e

**BLEU** (Bilingual Evaluation Understudy) e uma metrica que avalia a qualidade de texto gerado comparando o **overlap de n-gramas** entre o texto gerado e um texto de referencia.

Originalmente desenvolvida para avaliacao de traducao automatica, e amplamente utilizada para comparar textos gerados por modelos de linguagem.

## Como funciona

1. Extrai n-gramas (sequencias de N palavras consecutivas) do texto gerado e da referencia
2. Calcula a **precisao** de n-gramas: quantos n-gramas do texto gerado aparecem na referencia
3. Aplica uma **penalidade de brevidade** para textos muito curtos
4. Combina scores de diferentes ordens de n-gramas (1-grama, 2-gramas, 3-gramas, 4-gramas)

## Escala

- **0:** Nenhum overlap de n-gramas
- **1:** Overlap perfeito (textos identicos)

Na pratica, scores BLEU para textos longos e criativos tendem a ser baixos (0.1–0.3), pois existem muitas formas validas de expressar a mesma ideia.

## Uso no projeto

No projeto, o BLEU e calculado entre respostas de **pares de modelos** a uma mesma questao:

```python
import evaluate as hf_evaluate

bleu_metric = hf_evaluate.load("bleu")
result = bleu_metric.compute(predictions=preds, references=refs)
# result["bleu"] → score BLEU
```

## Limitacoes

- Baseado apenas em overlap exato de palavras (nao captura sinonimos)
- Penaliza variacao lexical, mesmo que semanticamente correta
- Mais adequado para textos curtos e factuais do que para textos argumentativos longos
