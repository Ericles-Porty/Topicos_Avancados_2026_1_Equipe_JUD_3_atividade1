# BERTScore

## O que e

**BERTScore** e uma metrica que avalia a similaridade **semantica** entre textos usando embeddings contextuais de modelos BERT. Ao contrario de BLEU e ROUGE, que comparam palavras exatas, o BERTScore captura **sinonimos e paráfrases**.

## Como funciona

1. Cada token do texto gerado e da referencia e convertido em um **embedding contextual** usando um modelo BERT pre-treinado
2. Para cada token do texto gerado, encontra-se o token mais similar na referencia (e vice-versa) usando **similaridade do cosseno**
3. Calcula-se:
   - **Precision:** Media da similaridade maxima de cada token gerado com a referencia
   - **Recall:** Media da similaridade maxima de cada token de referencia com o texto gerado
   - **F1:** Media harmonica de Precision e Recall

## Escala

- **0:** Sem similaridade semantica
- **1:** Semanticamente identicos

Na pratica, scores BERTScore tendem a ser mais altos que BLEU/ROUGE (0.6–0.9), pois capturam similaridade semantica mesmo com vocabulario diferente.

## Uso no projeto

```python
import evaluate as hf_evaluate

bertscore_metric = hf_evaluate.load("bertscore")
result = bertscore_metric.compute(
    predictions=preds,
    references=refs,
    lang="pt"
)
# result["f1"] → lista de scores F1 por par de textos
```

### Parametro `lang="pt"`

O parametro `lang="pt"` seleciona automaticamente o modelo `bert-base-multilingual-cased`, adequado para textos em portugues.

## Por que usar BERTScore para questoes juridicas

Textos juridicos frequentemente expressam o mesmo conceito com **vocabulario variado**:
- "rescindir o contrato" vs "resolver o vinculo contratual"
- "dano moral" vs "lesao a direitos da personalidade"

BLEU e ROUGE nao capturam essas equivalencias, enquanto o BERTScore identifica a similaridade semantica entre as expressoes.

## Warnings esperados

Ao executar o BERTScore, warnings como `Some weights of the model checkpoint were not used` sao **normais e esperados**. Eles indicam que o modelo BERT tem camadas de classificacao que nao sao utilizadas para o calculo de embeddings — apenas as camadas de encoding sao relevantes.

## Comparacao com outras metricas

| Aspecto | BLEU/ROUGE | BERTScore |
|---|---|---|
| Base | Overlap exato de palavras | Similaridade semantica |
| Sinonimos | Nao captura | Captura |
| Parafrases | Nao captura | Captura |
| Velocidade | Rapido | Mais lento (requer GPU) |
| Dependencia | Apenas texto | Modelo BERT pre-treinado |
