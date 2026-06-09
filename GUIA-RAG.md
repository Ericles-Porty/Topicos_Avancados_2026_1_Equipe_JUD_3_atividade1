# Guia de execução da inferência com RAG

Este guia descreve como reexecutar a inferência da Atividade 1 com **RAG
(Retrieval-Augmented Generation)** e como gerar a linha de base **sem RAG** para o
comparativo. A implementação é um port fiel da arquitetura do Reinan.

## Visão geral do pipeline

1. **Indexação** (`rag-populate`): cada lei em `database/rag/*.html` é quebrada
   por artigo (chunking), transformada em embeddings (`qwen3-embedding:8b`) e
   persistida no **ChromaDB** (`.chroma/`, coleção `legislacao`).
2. **Recuperação + inferência** (`--rag`): para cada questão, o sistema reescreve
   a consulta, faz **busca híbrida** (vetorial + lexical), aplica **rerank
   jurídico** e uma **porta de confiança** (se a recuperação for fraca, não injeta
   contexto). Os artigos recuperados viram um bloco `[LEGISLAÇÃO DE SUPORTE]`
   prefixado ao prompt, e o LLM responde.
3. **Comparação**: respostas com RAG são salvas em `*_rag.json`; sem RAG, em
   `*.json`. Cada resposta com RAG carrega `used_rag: true` e `rag_info` (os
   scores de cada trecho recuperado).

## Pré-requisitos

| Ferramenta | Versão | Observação |
|---|---|---|
| Python | 3.12+ | linguagem do projeto |
| Ollama | 0.3+ | runtime local dos modelos |
| Git | 2.x | controle de versão |

> **Hardware:** o modelo de embedding `qwen3-embedding:8b` é grande (~8B). Exige
> VRAM/RAM suficiente. A indexação roda só uma vez; depois fica em cache no
> `.chroma/`.

## Passo a passo

```bash
# 0) Dependências Python
pip install -r requirements.txt

# 1) Modelos no Ollama (o embedding também é baixado automaticamente)
ollama pull qwen3-embedding:8b
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull qwen2.5:3b

# 2) Indexar a legislação no ChromaDB (uma única vez)
python main.py --stage rag-populate

# 3) Inferência COM RAG  ->  src/results/multiple_choice_rag.json e open_questions_rag.json
python main.py --stage mc   --rag --top-k 10
python main.py --stage open --rag --top-k 10

# 4) Linha de base SEM RAG  ->  src/results/multiple_choice.json e open_questions.json
python main.py --stage mc
python main.py --stage open
```

> Se já houver a base sem RAG da entrega anterior, o passo 4 é desnecessário.

## Tabela de comandos

| Comando | O que faz |
|---|---|
| `python main.py --stage rag-populate` | Indexa `database/rag/*.html` no ChromaDB. |
| `python main.py --stage mc --rag` | Múltipla escolha **com** RAG → `multiple_choice_rag.json`. |
| `python main.py --stage open --rag` | Questões abertas **com** RAG → `open_questions_rag.json`. |
| `python main.py --stage mc` | Múltipla escolha **sem** RAG → `multiple_choice.json`. |
| `python main.py --stage open` | Questões abertas **sem** RAG → `open_questions.json`. |
| `--top-k N` | Quantos trechos de lei injetar (padrão: 10). |

## Notas

- **Retomável:** a inferência pula pares `(questão, modelo)` já salvos. Pode
  interromper e retomar sem reprocessar.
- **Recuperação por modelo:** a reescrita da consulta usa o próprio modelo
  candidato (fiel ao Reinan), então a etapa de RAG é mais lenta que a inferência
  pura.
- **Porta de confiança:** quando a recuperação é fraca, o sistema não injeta
  legislação (evita "injetar textos desconexos", como alerta o enunciado da
  Atividade 3).
- **Base de conhecimento:** `database/rag/` reúne a legislação curada por Reinan,
  Éricles e Fernanda (23 documentos). Para incluir novas leis, basta adicionar o
  HTML e rodar `rag-populate` novamente.
- **`.chroma/`** é gerado e **não** versionado (está no `.gitignore`).

## Próxima etapa (Atividade 2)

Os resultados `*_rag.json` alimentam a Atividade 2, onde o juiz avalia as
respostas com e sem RAG (`usou_rag`) e a análise estatística compara o ganho.
