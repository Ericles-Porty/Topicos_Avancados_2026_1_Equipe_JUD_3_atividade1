# Pre-requisitos

## Requisitos obrigatorios

| Requisito | Versao minima | Descricao |
|---|---|---|
| **Python** | 3.12+ | Linguagem de programacao principal |
| **pip** | — | Gerenciador de pacotes Python |
| **Ollama** | — | Runtime local para execucao de LLMs |
| **Git** | — | Controle de versao |

## Requisitos recomendados

| Requisito | Descricao |
|---|---|
| **GPU NVIDIA** | Acelera significativamente a inferencia dos modelos |
| **Drivers CUDA** | Necessarios para execucao em GPU |
| **4 GB+ VRAM** | Minimo recomendado para os modelos selecionados |

## Verificacao

```bash
# Verificar Python
python --version

# Verificar pip
pip --version

# Verificar Ollama
ollama --version

# Verificar Git
git --version
```

## Modelos Ollama necessarios

Os seguintes modelos devem ser baixados antes da execucao:

```bash
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull qwen2.5:3b
```

## Dependencias Python

As principais dependencias do projeto sao:

| Pacote | Uso |
|---|---|
| `pandas` | Manipulacao de dados tabulares |
| `ollama` | Comunicacao com o servidor Ollama |
| `minijinja` | Renderizacao de templates de prompts |
| `matplotlib` | Geracao de graficos |
| `scikit-learn` | Metricas de classificacao (Precision, Recall, F1) |
| `datasets` | Carregamento de datasets do HuggingFace |
| `requests` | Download de arquivos do OAB Bench |
| `evaluate` | Framework de metricas (BLEU, ROUGE) |
| `rouge-score` | Implementacao da metrica ROUGE |
| `bert-score` | Metrica BERTScore para similaridade semantica |
