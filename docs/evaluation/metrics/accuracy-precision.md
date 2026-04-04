# Acuracia, Precision, Recall e F1

## Contexto

Estas metricas sao utilizadas para avaliar as respostas de **multipla escolha**, tratando cada alternativa (A, B, C, D) como uma classe de classificacao.

## Acuracia

A acuracia mede a proporcao de respostas corretas sobre o total de questoes:

```
Acuracia = respostas corretas / total de questoes
```

**Exemplo:** Se um modelo acerta 55 de 123 questoes:

```
Acuracia = 55 / 123 = 0.4472 (44.72%)
```

### Limitacoes

A acuracia isolada pode ser enganosa quando as classes estao desbalanceadas. Se a maioria das respostas corretas for "A", um modelo que sempre responda "A" teria uma acuracia artificialmente alta.

## Precision (macro)

Para cada classe (A, B, C, D), a precision mede quantas das respostas que o modelo atribuiu aquela classe estao corretas:

```
Precision(classe) = VP(classe) / (VP(classe) + FP(classe))
```

A **macro precision** e a media aritmetica da precision de todas as classes:

```
Precision(macro) = (Precision(A) + Precision(B) + Precision(C) + Precision(D)) / 4
```

## Recall (macro)

Para cada classe, o recall mede quantas das questoes cuja resposta correta e aquela classe foram identificadas pelo modelo:

```
Recall(classe) = VP(classe) / (VP(classe) + FN(classe))
```

A **macro recall** e a media aritmetica do recall de todas as classes.

## F1 Score (macro)

O F1 Score e a media harmonica de Precision e Recall, equilibrando ambas as metricas:

```
F1(classe) = 2 * (Precision(classe) * Recall(classe)) / (Precision(classe) + Recall(classe))
```

A **macro F1** e a media aritmetica do F1 de todas as classes.

## Implementacao

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Converter letras para inteiros
letter_to_int = {"A": 1, "B": 2, "C": 3, "D": 4}
y_true = [letter_to_int.get(c, 0) for c in grp["correct"]]
y_pred = [letter_to_int.get(a, 0) for a in grp["answer"]]

precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
```

O parametro `zero_division=0` evita erros quando alguma classe nao aparece nas predicoes.
