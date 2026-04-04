# Modelos de linguagem

## Modelos selecionados

| # | Modelo | Desenvolvedor | Comando Ollama |
|---|---|---|---|
| 1 | **Llama 3.2 3B** | Meta | `ollama run llama3.2:3b` |
| 2 | **Gemma 2 2B** | Google | `ollama run gemma2:2b` |
| 3 | **Qwen 2.5 3B** | Alibaba | `ollama run qwen2.5:3b` |

## Justificativa da escolha

### Diversidade de origem

Os tres modelos provem de organizacoes distintas (Meta, Google e Alibaba), permitindo comparar diferentes abordagens de treinamento e arquiteturas em um mesmo conjunto de questoes juridicas.

### Suporte multilingue

Os tres modelos oferecem suporte ao idioma portugues, requisito essencial para inferencia em questoes do Exame da OAB, redigidas inteiramente em portugues brasileiro.

### Compatibilidade com Ollama

Todos os modelos estao disponiveis no ecossistema Ollama, garantindo:
- Execucao local padronizada
- Mesma interface de API para todos os modelos
- Facilidade de reproducao dos experimentos

## Modelo juiz

Alem da inferencia, o modelo **Llama 3.2 3B** (`llama3.2:3b`) e utilizado como **modelo juiz** para:
- Avaliacao por rubrica de questoes abertas
- Avaliacao comparativa entre respostas dos modelos
- Tarefas de curadoria (dificuldade + legislacao)

A escolha do Llama 3.2 3B como juiz se deve ao seu bom desempenho em tarefas de compreensao e avaliacao de texto em portugues.

## Instalacao

```bash
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull qwen2.5:3b
```
