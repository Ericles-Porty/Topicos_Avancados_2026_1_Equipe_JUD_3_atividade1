# Area de especialidade

## Categorias de classificacao

Cada questao e classificada em uma das seguintes areas de especialidade juridica:

| Area | Descricao |
|---|---|
| **Direito Constitucional** | Questoes sobre principios, direitos fundamentais e organizacao do Estado |
| **Direito Civil** | Obrigacoes, contratos, familia, sucessoes e propriedade |
| **Direito Penal** | Crimes, penas, imputabilidade e legislacao penal |
| **Direito Trabalhista** | Relacoes de trabalho, CLT e direitos do trabalhador |
| **Direito Administrativo** | Administracao publica, licitacoes e atos administrativos |
| **Direito Tributario** | Tributos, obrigacoes fiscais e legislacao tributaria |
| **Direito Processual Civil** | Procedimentos e recursos no ambito civil |
| **Direito Processual Penal** | Procedimentos e recursos no ambito penal |
| **Direito Empresarial** | Sociedades, titulos de credito e falencia |
| **Direito Ambiental** | Protecao ambiental e legislacao ecologica |
| **Direito do Consumidor** | Relacoes de consumo e CDC |
| **Direitos Humanos** | Tratados internacionais e direitos fundamentais |
| **Etica Profissional e Estatuto da OAB** | Deontologia e regulamentacao da advocacia |
| **Direito Internacional** | Direito internacional publico e privado |
| **Direito Previdenciario** | Seguridade social e beneficios previdenciarios |

## Regras de classificacao

- Escolha a area que **melhor representa** o tema central da questao
- Se a questao envolve mais de uma area, escolha a **predominante**
- Caso nenhuma categoria se aplique, retorne "Outra"

## Formato de saida

O modelo retorna um JSON estruturado:

```json
{
  "question_id": "41_direito_administrativo_questao_1",
  "area_especialidade": "Direito Administrativo"
}
```

## Implementacao

A classificacao e feita pelo modelo `llama3.2:3b` com `temperature=0` para garantir determinismo. O prompt completo esta em `src/templates/curator_specialty.jinja`.
