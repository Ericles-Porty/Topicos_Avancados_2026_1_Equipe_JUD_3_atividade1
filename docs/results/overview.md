# Resultados — Visao geral

## Arquivos gerados

Apos a execucao completa do pipeline, os seguintes arquivos sao gerados em `src/results/`:

### Inferencia

| Arquivo | Descricao |
|---|---|
| `open_questions.json` | Respostas dos 3 modelos as 12 questoes abertas |
| `multiple_choice.json` | Respostas dos 3 modelos as 123 questoes de multipla escolha |
| `curator_annotations.json` | Anotacoes de curadoria (dificuldade + legislacao) para todas as questoes |

### Avaliacao

| Arquivo | Descricao |
|---|---|
| `eval_open_questions.csv` | Notas por rubrica para questoes abertas |
| `eval_multiple_choice.csv` | Respostas e acertos para multipla escolha |
| `eval_comparative.csv` | Notas comparativas (argumentacao, precisao, coesao) |
| `eval_cross_metrics.csv` | Metricas automatizadas entre pares de modelos |

### Consolidacao

| Arquivo | Descricao |
|---|---|
| `leaderboard.csv` | Leaderboard consolidado com todas as metricas por modelo |
| `model_comparison.png` | Grafico de comparacao (3x2 grid) |

## Leaderboard

O leaderboard consolida as seguintes metricas por modelo:

| Coluna | Origem | Descricao |
|---|---|---|
| `open_score` | Rubrica | Pontuacao media nas questoes abertas |
| `mc_accuracy_%` | MC | Acuracia em multipla escolha (%) |
| `mc_precision` | MC (sklearn) | Precision macro |
| `mc_recall` | MC (sklearn) | Recall macro |
| `mc_f1` | MC (sklearn) | F1 macro |
| `argumentacao` | Comparativa | Media da nota de argumentacao (0-5) |
| `precisao` | Comparativa | Media da nota de precisao (0-5) |
| `coesao` | Comparativa | Media da nota de coesao (0-5) |
| `final_score` | Comparativa | Score final ponderado |

## Grafico de comparacao

O grafico `model_comparison.png` apresenta uma grade 3x2:

1. **Linha 1, esquerda:** Rubrica — pontuacao media por modelo
2. **Linha 1, direita:** Avaliacao comparativa — argumentacao, precisao e coesao
3. **Linha 2:** Metricas cross-model — BLEU, ROUGE-1, ROUGE-L e BERTScore F1 por par
4. **Linha 3:** Multipla escolha — Acuracia, Precision, Recall e F1 por modelo
