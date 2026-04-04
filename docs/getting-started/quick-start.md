# Inicio rapido

Este guia descreve o fluxo minimo para reproduzir os experimentos do projeto.

## Fluxo de execucao

O pipeline completo consiste em tres etapas sequenciais:

### Etapa 1 — Carregar datasets

```bash
python src/load_dataset.py
```

Este script:
- Baixa o dataset OAB Bench (questoes abertas) do GitHub da Maritaca AI
- Baixa o dataset OAB Exams (multipla escolha) do HuggingFace
- Extrai o subconjunto de questoes designadas (12 abertas + 123 multipla escolha)
- Salva os arquivos CSV em `src/dataset/`

### Etapa 2 — Executar inferencia + curadoria

```bash
python src/run_models.py
```

Este script:
- Submete as questoes abertas aos tres modelos (Mistral, Llama 3, Gemma)
- Submete as questoes de multipla escolha aos tres modelos
- Executa tarefas de curadoria (dificuldade + legislacao) com o modelo juiz
- Salva resultados em `src/results/`

**Tempo estimado:** Varia conforme o hardware. Em GPU GTX 1650, cada modelo leva alguns minutos por tipo de questao.

### Etapa 3 — Avaliar e gerar leaderboard

```bash
python src/evaluation.py
```

Este script:
- Avalia questoes abertas por rubrica (modelo juiz)
- Avalia questoes abertas por comparacao qualitativa entre modelos
- Calcula metricas automatizadas (BLEU, ROUGE, BERTScore) entre pares de modelos
- Avalia multipla escolha (acuracia + Precision + Recall + F1)
- Gera o leaderboard consolidado e graficos de comparacao

## Saidas

Apos a execucao completa, os seguintes arquivos serao gerados em `src/results/`:

| Arquivo | Descricao |
|---|---|
| `open_questions.json` | Respostas dos modelos as questoes abertas |
| `multiple_choice.json` | Respostas dos modelos as questoes de multipla escolha |
| `curator_annotations.json` | Anotacoes de curadoria (dificuldade + legislacao) |
| `eval_open_questions.csv` | Avaliacoes por rubrica das questoes abertas |
| `eval_multiple_choice.csv` | Avaliacoes das questoes de multipla escolha |
| `eval_comparative.csv` | Avaliacoes comparativas entre modelos |
| `eval_cross_metrics.csv` | Metricas automatizadas entre pares de modelos |
| `leaderboard.csv` | Leaderboard consolidado |
| `model_comparison.png` | Grafico de comparacao dos modelos |
