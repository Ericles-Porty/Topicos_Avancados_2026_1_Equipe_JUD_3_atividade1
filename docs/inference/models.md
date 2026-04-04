# Modelos de linguagem

## Modelos selecionados

| # | Modelo | Desenvolvedor | Comando Ollama |
|---|---|---|---|
| 1 | **Mistral** | Mistral AI | `ollama run mistral` |
| 2 | **Llama 3** | Meta | `ollama run llama3` |
| 3 | **Gemma** | Google | `ollama run gemma` |

## Justificativa da escolha

### Diversidade de origem

Os tres modelos provem de organizacoes distintas (Mistral AI, Meta e Google), permitindo comparar diferentes abordagens de treinamento e arquiteturas em um mesmo conjunto de questoes juridicas.

### Suporte multilingue

Os tres modelos oferecem suporte ao idioma portugues, requisito essencial para inferencia em questoes do Exame da OAB, redigidas inteiramente em portugues brasileiro.

### Compatibilidade com Ollama

Todos os modelos estao disponiveis no ecossistema Ollama, garantindo:
- Execucao local padronizada
- Mesma interface de API para todos os modelos
- Facilidade de reproducao dos experimentos

## Modelo juiz

Alem da inferencia, o modelo **Llama 3** (`llama3`) e utilizado como **modelo juiz** para:
- Avaliacao por rubrica de questoes abertas
- Avaliacao comparativa entre respostas dos modelos
- Tarefas de curadoria (dificuldade + legislacao)

A escolha do Llama 3 como juiz se deve ao seu bom desempenho em tarefas de compreensao e avaliacao de texto em portugues.

## Instalacao

```bash
ollama pull mistral
ollama pull llama3
ollama pull gemma
```
