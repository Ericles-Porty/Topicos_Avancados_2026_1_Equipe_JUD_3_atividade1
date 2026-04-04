# Legislacao base

## Objetivo

Enriquecer cada questao com a **referencia normativa principal** que a fundamenta, adicionando metadados uteis para analise tematica do dataset.

## O que e identificado

O modelo analisa o enunciado da questao e identifica a legislacao principal, como:

- Constituicao Federal
- Codigo Civil
- Codigo Penal
- Codigo de Processo Civil
- Codigo de Defesa do Consumidor
- Leis especificas (ex: Lei n. 8.112/90)
- Artigos especificos, quando ha certeza

## Regras

- Identificar apenas a legislacao principal
- Citar artigos especificos **somente quando houver certeza**
- Nao fazer referencia a normas ficticias
- Retornar "Inconclusivo" quando nao for possivel determinar com seguranca

## Formato de saida

```json
{
  "question_id": "41_direito_administrativo_questao_1",
  "legislacao_base": "Constituicao Federal, art. 71; Lei n. 9.784/99"
}
```

## Implementacao

A identificacao e feita pelo modelo `llama3` com `temperature=0`. O prompt completo esta em `src/templates/curator_legislation.jinja`.
