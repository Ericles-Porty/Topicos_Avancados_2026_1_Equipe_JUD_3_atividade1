# OAB Bench (J1)

## Origem

O dataset **OAB Bench** foi desenvolvido pela [Maritaca AI](https://github.com/maritaca-ai/oab-bench) como benchmark para avaliacao de modelos de linguagem em tarefas de escrita juridica. Contem **210 questoes discursivas** da 2a fase do Exame da OAB, incluindo questoes dissertativas e pecas pratico-profissionais.

> **Artigo:** [ACM Digital Library — OAB Bench](https://dl.acm.org/doi/pdf/10.1145/3769126.3769227)

## Estrutura dos campos

| Campo | Tipo | Descricao |
|---|---|---|
| `question_id` | `string` | Identificador unico. Codifica a edicao do exame, area do Direito e numero da questao (ex: `41_direito_constitucional_questao_2`) |
| `category` | `string` | Categoria tematica, agrupando por exame e area juridica (ex: `41_direito_constitucional`) |
| `statement` | `string` | Enunciado completo da questao, incluindo contexto fatico, narrativa juridica e comando da tarefa |
| `turns` | `array[string]` | Subperguntas ou desdobramentos. Em pecas pratico-profissionais, pode conter item vazio (`""`) |
| `values` | `array[number]` | Pesos ou pontuacoes de cada item de `turns` (ex: `[0.65, 0.6]` para subperguntas) |
| `system` | `string` | Instrucao de sistema para o modelo, definindo o papel do candidato e regras da prova |

## Subsets

O dataset possui dois subsets:

| Subset | Registros | Descricao |
|---|---|---|
| `questions` | 105 | Enunciados das questoes |
| `guidelines` | 105 | Gabaritos e criterios de correcao |

## Exemplo de registro

```json
{
  "question_id": "41_direito_administrativo_questao_1",
  "category": "41_direito_administrativo",
  "statement": "Esglobenia, servidora publica federal estavel, acreditava ter preenchido os respectivos requisitos do Regime Proprio de Previdencia...",
  "turns": [
    "O ato aposentadoria de Esglobenia estava perfeito, ou seja, completou o seu ciclo de formacao, antes do pronunciamento da Corte de Contas? Justifique.",
    "Para negar o registro da aposentadoria de Esglobenia, o Tribunal de Contas precisa observar a ampla defesa e o contraditorio? Justifique."
  ],
  "values": [0.6, 0.65],
  "system": "Voce e um bacharel em direito que esta realizando a segunda fase da prova da OAB..."
}
```

## Carregamento no projeto

O script `src/load_dataset.py` baixa os arquivos JSONL diretamente do repositorio GitHub da Maritaca AI, converte para CSV e extrai o subconjunto designado (indices 153–164).
