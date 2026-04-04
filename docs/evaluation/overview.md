# Avaliacao — Visao geral

## Estrategias de avaliacao

O projeto utiliza **quatro estrategias complementares** para avaliar as respostas dos modelos:

| Estrategia | Tipo de questao | Abordagem |
|---|---|---|
| [Rubrica](open-questions.md) | Abertas | Modelo juiz avalia com base nos criterios oficiais |
| [Comparativa](comparative.md) | Abertas | Modelo juiz compara respostas entre modelos |
| [Metricas automatizadas](cross-metrics.md) | Abertas | BLEU, ROUGE, BERTScore entre pares |
| [Multipla escolha](multiple-choice.md) | MC | Comparacao exata com gabarito + sklearn |

## Por que multiplas estrategias?

Cada abordagem captura aspectos diferentes da qualidade das respostas:

- **Rubrica:** Avalia aderencia aos criterios oficiais da OAB (mais proxima da avaliacao real)
- **Comparativa:** Avalia qualidade relativa entre modelos em dimensoes especificas (argumentacao, precisao, coesao)
- **Metricas automatizadas:** Medem similaridade textual/semantica de forma reprodutivel e objetiva
- **Multipla escolha:** Avaliacao exata e deterministica com metricas de classificacao padronizadas

## Modelo juiz

As avaliacoes por rubrica e comparativa utilizam o modelo `llama3.2:3b` como **juiz**, com `temperature=0` para garantir determinismo.

## Implementacao

Todas as funcoes de avaliacao estao em `src/evaluation.py`:

| Funcao | Descricao |
|---|---|
| `evaluate_open_questions()` | Avaliacao por rubrica |
| `evaluate_comparative()` | Avaliacao comparativa |
| `evaluate_cross_metrics()` | Metricas automatizadas (BLEU, ROUGE, BERTScore) |
| `evaluate_multiple_choice()` | Avaliacao de multipla escolha |
| `generate_leaderboard()` | Consolidacao e visualizacao |

## Metricas detalhadas

- [Acuracia, Precision, Recall, F1](metrics/accuracy-precision.md)
- [BLEU](metrics/bleu.md)
- [ROUGE](metrics/rouge.md)
- [BERTScore](metrics/bertscore.md)
