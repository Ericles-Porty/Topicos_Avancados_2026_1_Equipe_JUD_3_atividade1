# Subdominio semantico — Area de especialidade

## Abordagem

As areas do direito sao tratadas como **subdominios semanticos**, ou seja, subconjuntos de dados estruturados que permitem agrupar e analisar o desempenho dos LLMs por area tematica.

## Subdominios disponiveis

| Subdominio Semantico | Descricao |
|---|---|
| **Direito Constitucional** | Principios, direitos fundamentais e organizacao do Estado |
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

- Escolha o subdominio que **melhor representa** o tema central da questao
- Se a questao envolve mais de um subdominio, escolha o **predominante**
- Caso nenhum subdominio se aplique, retorne "Outro"

## Formato de saida

O modelo retorna um JSON estruturado:

```json
{
  "question_id": "41_direito_administrativo_questao_1",
  "subdominio_semantico": "Direito Administrativo"
}
```

## Implementacao

A classificacao e feita pelo modelo `llama3.2:3b` com `temperature=0` para garantir determinismo. O prompt completo esta em `src/templates/curator_specialty.jinja`.
