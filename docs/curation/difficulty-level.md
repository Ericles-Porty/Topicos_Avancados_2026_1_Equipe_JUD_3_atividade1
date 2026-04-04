# Nivel de dificuldade

## Criterios de classificacao

A classificacao de dificuldade segue tres niveis, baseados em criterios objetivos:

| Nivel | Valor | Criterios |
|---|---|---|
| **Facil** | 1 | Enunciado curto, linguagem direta, aplicacao imediata de um unico dispositivo legal, sem necessidade de interpretacao complexa |
| **Medio** | 2 | Tamanho intermediario, presenca de termos tecnicos, exige raciocinio juridico moderado e conhecimento de mais de um dispositivo |
| **Dificil** | 3 | Textos longos, exige interpretacao aprofundada, combina mais de um tema juridico, alternativas muito similares |

### Regra especial

Pecas pratico-profissionais devem ser classificadas como **dificuldade 3** (Dificil), independentemente de outros criterios, devido a sua complexidade inerente.

## Fonte dos criterios

Os criterios foram definidos com base em:
- Padroes de classificacao da **Estrategia OAB**
- Analise de padroes das provas organizadas pela **FGV**

## Formato de saida

O modelo retorna um JSON estruturado:

```json
{
  "question_id": "41_direito_administrativo_questao_1",
  "dificuldade": 2,
  "nivel": "Medio"
}
```

## Implementacao

A classificacao e feita pelo modelo `llama3` com `temperature=0` para garantir determinismo. O prompt completo esta em `src/templates/curator_difficulty.jinja`.
