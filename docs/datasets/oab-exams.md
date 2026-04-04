# OAB Exams (J2)

## Origem

O dataset **OAB Exams** foi compilado por [Eduardo Garcia](https://huggingface.co/datasets/eduagarcia/oab_exams) e reune **2210 questoes objetivas de multipla escolha** da 1a fase do Exame da OAB, contemplando edicoes de 2010 a 2018.

## Estrutura dos campos

| Campo | Tipo | Descricao |
|---|---|---|
| `id` | `string` | Identificador unico (ex: `2016-21_52` = questao 52 do exame `2016-21`) |
| `question_number` | `integer` | Numero ordinal da questao na prova |
| `exam_id` | `string` | Identificador da edicao do exame |
| `exam_year` | `string` | Ano de realizacao do exame |
| `question_type` | `string \| null` | Classificacao tematica (ex: `ETHICS`, `CONSTITUTIONAL`). Pode ser `null` |
| `nullified` | `boolean` | Se a questao foi anulada pela banca |
| `question` | `string` | Enunciado principal da questao |
| `choices` | `object` | Alternativas: `label` (A, B, C, D) + `text` (texto de cada opcao) |
| `answerKey` | `string` | Gabarito oficial (A, B, C ou D) |

## Exemplo de registro

```json
{
  "id": "2016-21_52",
  "question_number": 52,
  "exam_id": "2016-21",
  "exam_year": "2016",
  "question_type": null,
  "nullified": false,
  "question": "Bernardino adquiriu de Lorena acoes preferenciais escriturais da companhia Campos Logistica S/A...",
  "choices": {
    "text": [
      "pelo extrato a ser fornecido pela instituicao custodiante...",
      "pela inscricao do nome de Bernardino no livro de Registro...",
      "pelo lancamento efetuado pela instituicao depositaria...",
      "por termo lavrado no livro de Transferencia..."
    ],
    "label": ["A", "B", "C", "D"]
  },
  "answerKey": "C"
}
```

## Carregamento no projeto

O script `src/load_dataset.py` utiliza a biblioteca `datasets` do HuggingFace para baixar o dataset, expande a coluna `choices` em colunas separadas (`choice_a`, `choice_b`, `choice_c`, `choice_d`), e extrai o subconjunto designado (indices 1600–1722).
