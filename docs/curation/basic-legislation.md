# Corpus de referencia — Ground Truth

## Objetivo

Identificar o **corpus de referencia (ground truth)** onde a resposta correta de cada questao deve estar ancorada, evitando alucinacoes do LLM. Em vez de apenas "Legislacao Base", este campo usa a terminologia de IA para indicar a fonte normativa que fundamenta a questao.

## O que e identificado

O modelo analisa o enunciado da questao e identifica a legislacao principal com seu nome oficial e numero da lei quando possivel:

- Constituicao Federal de 1988
- Codigo Civil (Lei 10.406/2002)
- Codigo Penal (Decreto-Lei 2.848/1940)
- Codigo de Processo Civil (Lei 13.105/2015)
- Codigo de Defesa do Consumidor (Lei 8.078/1990)
- Consolidacao das Leis do Trabalho - CLT (Decreto-Lei 5.452/1943)
- Leis especificas (ex: Lei 8.112/1990)
- Artigos especificos, quando ha certeza

## Regras

- Identificar apenas a legislacao principal
- Incluir o nome oficial e numero da lei quando possivel
- Citar artigos especificos **somente quando houver certeza**
- Nao fazer referencia a normas ficticias
- Retornar "Inconclusivo" quando nao for possivel determinar com seguranca

## Formato de saida

```json
{
  "question_id": "41_direito_administrativo_questao_1",
  "corpus_referencia": "Constituição Federal de 1988, art. 71; Lei 9.784/1999"
}
```

## Implementacao

A identificacao e feita pelo modelo `llama3.2:3b` com `temperature=0`. O prompt completo esta em `src/templates/curator_legislation.jinja`.
