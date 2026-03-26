# Contribuições Individuais — Reinan Gabriel

Este documento descreve as contribuições realizadas pelo aluno **Reinan Gabriel** no contexto da **Atividade Avaliativa 1** da disciplina **Tópicos Avançados em Engenharia de Software e Sistemas de Informação I**, ministrada na Universidade Federal de Sergipe (UFS), semestre 2026.1.

O trabalho consiste na curadoria de datasets jurídicos e na realização de inferência básica com Modelos de Linguagem (LLMs), aplicados a questões do Exame da Ordem dos Advogados do Brasil (OAB).

---

## 1. Ambiente de execução

### 1.1 Configuração de hardware

Os experimentos de inferência foram executados em uma máquina local com a seguinte configuração de GPU:

| Componente                | Especificação           |
|---------------------------|-------------------------|
| **GPU**                   | NVIDIA GeForce GTX 1050 |
| **VRAM dedicada**         | 4,0 GB                  |
| **Memória compartilhada** | 8,0 GB                  |
| **Versão do driver**      | 32.0.15.8228            |
| **Data do driver**        | 20/01/2026              |
| **Versão do DirectX**     | 12 (FL 12.1)            |

Dado o limite de **4 GB de VRAM dedicada**, a seleção dos modelos foi direcionada para LLMs compactos (até ~3B de parâmetros) com versões quantizadas que cabem confortavelmente na memória disponível da GPU.

### 1.2 Modelos de linguagem selecionados

Foram escolhidos **três modelos de linguagem** de diferentes organizações para garantir diversidade de arquiteturas e bases de treinamento na comparação. Todos os modelos são executados localmente via [Ollama](https://ollama.com/).

| # | Modelo       | Desenvolvedor | Parâmetros | Quantização | Tamanho (download) | Contexto máximo | Comando Ollama           |
|---|--------------|---------------|------------|-------------|--------------------|-----------------|--------------------------|
| 1 | Llama 3.2 3B | Meta          | 3,21B      | Q4_K_M      | ~2,0 GB            | 128K tokens     | `ollama run llama3.2:3b` |
| 2 | Gemma 2 2B   | Google        | 2,61B      | Q4_0        | ~1,6 GB            | 8K tokens       | `ollama run gemma2:2b`   |
| 3 | Qwen 2.5 3B  | Alibaba Cloud | 3,09B      | Q4_K_M      | ~1,9 GB            | 32K tokens      | `ollama run qwen2.5:3b`  |

#### 1.3 Justificativa da escolha

- **Compatibilidade com o hardware:** Todos os modelos possuem tamanho inferior a 2 GB em suas versões quantizadas, garantindo execução integral na VRAM dedicada de 4 GB sem necessidade de *offloading* para a memória compartilhada.
- **Diversidade de origem:** Os três modelos provêm de organizações distintas (Meta, Google e Alibaba Cloud), o que permite comparar diferentes abordagens de treinamento e arquiteturas em um mesmo conjunto de questões jurídicas.
- **Suporte multilíngue:** Os três modelos oferecem suporte ao idioma português, requisito essencial para a inferência em questões do Exame da OAB redigidas em português brasileiro.

#### 1.4 Instalação dos modelos

```bash
# Instalar o Ollama (caso ainda não instalado)
curl -fsSL https://ollama.com/install.sh | sh
# Windows (PowerShell)
# irm https://ollama.com/install.ps1 | iex

# Baixar os três modelos
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull qwen2.5:3b

# Verificar os modelos instalados
ollama list
```

---

## 2. Instruções de execução

### 2.1 Pré-requisitos

Para reproduzir os experimentos, é necessário ter instalado:

- **Python** 3.12 ou superior
- **uv** 0.10 ou superior (gerenciador de pacotes e runner Python)

### 2.2 Instalação e execução

```bash
# (Opcional) Criar e ativar um ambiente virtual
python -m venv .venv

# Ativação no Linux/macOS
source .venv/bin/activate

# Ativação no Windows (PowerShell)
# .venv\Scripts\activate

# Instalar as dependências do projeto
uv sync

# Executar o script principal
uv run python main.py

# Baixar os datasets
uv run python main.py pull oab_bench
uv run python main.py pull oab_exams

# Executar inferência oab_bench
uv run python main.py run oab_bench --model llama3.2:3b
uv run python main.py run oab_bench --model gemma2:2b
uv run python main.py run oab_bench --model qwen2.5:3b

# Executar inferência oab_exams
uv run python main.py run oab_exams --model llama3.2:3b
uv run python main.py run oab_exams --model gemma2:2b
uv run python main.py run oab_exams --model qwen2.5:3b

# Avaliar inferência
uv run python main.py evaluate oab_bench
uv run python main.py evaluate oab_exams
```

---

## 3. Distribuição e mapeamento das questões

### 3.1 Dataset J1 — Questões abertas (`maritaca-ai/oab-bench`)

Conforme as orientações da atividade, o dataset **J1** (`maritaca-ai/oab-bench`) contém **210 registros** distribuídos em dois subsets:

| Subset       | Intervalo (contagem geral) | Quantidade |
|--------------|----------------------------|------------|
| `guidelines` | 1 a 105                    | 105        |
| `questions`  | 106 a 210                  | 105        |

As questões designadas para esta análise correspondem ao intervalo **177 a 188** (12 questões). Como esse intervalo está inteiramente contido na segunda metade da contagem geral (106 a 210), todas as 12 questões pertencem exclusivamente ao subset `questions`. Nenhuma questão deste lote pertence ao subset `guidelines`.

**Filtragem via código:** Na implementação em Python, a indexação é baseada em zero. Dessa forma, para acessar as questões de número 177 a 188, os dois subsets são concatenados (preservando a ordem `guidelines` + `questions`) e os registros são extraídos pelos **índices 176 a 187** (inclusive).

### 3.2 Dataset J2 — Questões objetivas de múltipla escolha (`eduagarcia/oab_exams`)

O dataset **J2** (`eduagarcia/oab_exams`) é composto por **2210 questões objetivas de múltipla escolha**, provenientes da 1ª fase do Exame da OAB. Diferentemente do dataset J1 (questões discursivas), este dataset não possui divisão em subsets.

As questões designadas para esta análise correspondem ao intervalo **1846 a 1968**, totalizando **122 questões objetivas**. Na implementação em Python, considerando que a indexação inicia em zero, o intervalo correspondente é **1845 a 1967** (inclusive).

---

## 4. Estrutura dos datasets

A seguir, apresenta-se a descrição dos campos que compõem cada um dos datasets utilizados nesta atividade.

### 4.1 Dataset `maritaca-ai/oab-bench` — Subset `questions`

Este dataset contém os enunciados das questões discursivas da 2ª fase do Exame da OAB, acompanhados de metadados e instruções de sistema para os modelos de linguagem.

| Campo         | Tipo            | Descrição                                                                                                                                                                                                                                                       |
|---------------|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `question_id` | `string`        | Identificador único da questão. Codifica a edição do exame, a área do Direito e o número da questão (ex.: `41_direito_constitucional_questao_2`).                                                                                                               |
| `category`    | `string`        | Categoria temática, agrupando a questão por exame e área jurídica (ex.: `41_direito_constitucional`).                                                                                                                                                           |
| `statement`   | `string`        | Enunciado completo da questão, incluindo o contexto fático, a narrativa jurídica e o comando introdutório da peça ou questão discursiva.                                                                                                                        |
| `turns`       | `array[string]` | Lista de subperguntas ou desdobramentos da tarefa. Em questões discursivas, cada elemento corresponde a uma pergunta específica. Em peças prático-profissionais, pode conter um único item vazio (`""`), indicando que a resposta esperada é a peça na íntegra. |
| `values`      | `array[number]` | Pesos ou pontuações atribuídas a cada item de `turns`. Os valores refletem a distribuição de pontos do exame (ex.: `[0.65, 0.6]` para subperguntas ou `[5.0]` para o valor total de uma peça).                                                                  |
| `system`      | `string`        | Instrução de sistema (*system prompt*) para o modelo de linguagem, definindo o papel do candidato, as regras da prova e as restrições de formatação exigidas pelo exame.                                                                                        |

#### 4.1.1 Exemplo de registro

```json
{
  "question_id": "41_direito_administrativo_questao_1",
  "category": "41_direito_administrativo",
  "statement": "[ver exemplo completo abaixo]",
  "turns": [
    "O ato aposentadoria de Esglobênia estava perfeito, ou seja, completou o seu ciclo de formação, antes do pronunciamento da Corte de Contas? Justifique.",
    "Para negar o registro da aposentadoria de Esglobênia, o Tribunal de Contas precisa observar a ampla defesa e o contraditório? Justifique."
  ],
  "values": [0.6, 0.65],
  "system": "[ver exemplo completo abaixo]"
}
```

<details>
<summary><strong>Exemplo do campo <code>statement</code></strong></summary>

**QUESTÃO**

**Esglobênia**, servidora pública federal estável, acreditava ter preenchido os respectivos requisitos do **Regime Próprio de Previdência** no cargo que ocupava, razão pela qual pleiteou e obteve, junto ao órgão de origem, a **aposentadoria voluntária**.

Ato contínuo, o processo foi encaminhado ao **Tribunal de Contas da União**, o qual verificou algumas inconsistências no deferimento do pedido, de modo que está tendente a **negar o registro da aposentadoria**, sendo certo que o processo chegou à **Corte de Contas** há apenas **um ano**.

Diante dessa situação hipotética, responda, como **advogado(a)**, fundamentadamente, aos questionamentos a seguir.

</details>

<details>
<summary><strong>Exemplo do campo <code>system</code></strong></summary>

Você é um bacharel em direito que está realizando a segunda fase da prova da Ordem dos Advogados do Brasil (OAB), organizada pela FGV. Sua tarefa é responder às questões discursivas e elaborar uma peça processual, demonstrando seu conhecimento jurídico, capacidade de raciocínio e habilidade de aplicar a legislação e jurisprudência pertinentes ao caso apresentado.

**ATENÇÃO**

Na elaboração dos textos da peça prático-profissional e das respostas às questões discursivas, você deverá incluir todos os dados que se façam necessários, sem, contudo, produzir qualquer identificação ou informações além daquelas fornecidas e permitidas nos enunciados contidos no caderno de prova. A omissão de dados que forem legalmente exigidos ou necessários para a correta solução do problema proposto acarretará em descontos na pontuação atribuída a você nesta fase. Você deve estar atento para não gerar nenhum dado diferente que dê origem a uma marca identificadora.

A detecção de qualquer marca identificadora no espaço destinado à transcrição dos textos definitivos acarretará a anulação da prova prático-profissional e a eliminação de você. Assim, por exemplo, no fechamento da peça, você deve optar por utilizar apenas "reticências" ou "XXX", ou seja:

- data: `...` ou `XXX`
- local: `...` ou `XXX`
- advogado: `...` ou `XXX`
- inscrição OAB: `...` ou `XXX`

Destacando-se que, no corpo das respostas, você não deverá criar nenhum dado gerador de marca de identificação.

**OBSERVAÇÕES**

**PEÇA PRÁTICO-PROFISSIONAL:** A peça deve abranger todos os fundamentos de Direito que possam ser utilizados para dar respaldo à pretensão. A simples menção ou transcrição do dispositivo legal não confere pontuação.

**QUESTÃO:** Você deve fundamentar suas respostas. A mera citação do dispositivo legal não confere pontuação.

A partir de agora, todas as suas respostas comporão o texto definitivo (não o caderno de rascunhos).

</details>

### 4.2 Dataset `eduagarcia/oab_exams` — Questões objetivas

Este dataset reúne questões objetivas de múltipla escolha da 1ª fase do Exame da OAB, contemplando diversas edições da prova. Cada registro contém o enunciado da questão, as alternativas de resposta e o respectivo gabarito oficial, além de metadados como o ano e a edição do exame.

| Campo             | Tipo             | Descrição                                                                                                                                                                                                       |
|-------------------|------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `id`              | `string`         | Identificador único da questão. Combina o identificador do exame com o número da questão (ex.: `2016-21_52` representa a questão 52 do exame `2016-21`).                                                        |
| `question_number` | `integer`        | Número ordinal da questão dentro da prova, indicando sua posição sequencial no caderno do exame.                                                                                                                |
| `exam_id`         | `string`         | Identificador da edição do exame, utilizado para agrupar as questões pertencentes a uma mesma aplicação da prova da OAB.                                                                                        |
| `exam_year`       | `string`         | Ano de realização do exame, representando o recorte temporal da questão.                                                                                                                                        |
| `question_type`   | `string \| null` | Classificação temática da questão, quando disponível. Pode indicar áreas como `ETHICS`, `INTERNATIONAL` ou `CONSTITUTIONAL`. Quando o valor é `null`, a questão não possui classificação atribuída neste campo. |
| `nullified`       | `boolean`        | Indicador de anulação da questão. O valor `false` denota que a questão permanece válida; o valor `true` indica que foi anulada pela banca examinadora.                                                          |
| `question`        | `string`         | Enunciado principal da questão, contendo a situação-problema e o comando a ser analisado pelo candidato.                                                                                                        |
| `choices`         | `object`         | Objeto que agrupa as alternativas da questão objetiva, composto pelos subcampos `text` e `label`.                                                                                                               |
| `choices.text`    | `array[string]`  | Lista contendo o texto completo de cada alternativa de resposta. Cada posição do array corresponde a uma opção da questão.                                                                                      |
| `choices.label`   | `array[string]`  | Lista com os rótulos identificadores das alternativas (ex.: `A`, `B`, `C`, `D`). Cada rótulo está alinhado posicionalmente ao texto correspondente em `choices.text`.                                           |
| `answerKey`       | `string`         | Gabarito oficial da questão, indicando o rótulo da alternativa considerada correta (ex.: `A`, `B`, `C` ou `D`).                                                                                                 |

#### 4.2.1 Exemplo de registro

```json
{
  "id": "2016-21_52",
  "question_number": 52,
  "exam_id": "2016-21",
  "exam_year": "2016",
  "question_type": null,
  "nullified": false,
  "question": "Bernardino adquiriu de Lorena ações preferenciais escriturais da companhia Campos Logística S/A e recebeu do(a) advogado(a) orientação de como se dará a formalização da transferência da propriedade.\nA resposta do(a) advogado(a) é a de que a transferência das ações se opera",
  "choices": {
    "text": [
      "pelo extrato a ser fornecido pela instituição custodiante, na qualidade de proprietária fiduciária das ações.",
      "pela inscrição do nome de Bernardino no livro de Registro de Ações Nominativas em poder da companhia.",
      "pelo lançamento efetuado pela instituição depositária em seus livros, a débito da conta de ações de Lorena e a crédito da conta de ações de Bernardino.",
      "por termo lavrado no livro de Transferência de Ações Nominativas, datado e assinado por Lorena e por Bernardino ou por seus legítimos representantes."
    ],
    "label": ["A", "B", "C", "D"]
  },
  "answerKey": "C"
}
```

---

## 5. Metodologia

> **Nota:** Esta seção será complementada à medida que os experimentos forem executados e os resultados consolidados.

### 5.1 Curadoria (Classificação criativa)

Cada questão do lote atribuído será classificada de acordo com os parâmetros definidos em conjunto pela equipe, incluindo nível de dificuldade, área de especialidade jurídica e legislação de referência.

### 5.2 Inferência com LLMs

Serão selecionados **três modelos de linguagem** para submeter as questões abertas do lote. A escolha dos modelos, bem como os parâmetros de inferência utilizados, serão documentados nesta seção após a execução dos experimentos.

### 5.3 Avaliação e comparação

Por se tratar do domínio jurídico, as questões abertas **não possuem gabarito oficial**. A avaliação será realizada por meio da comparação entre as respostas dos três modelos, considerando critérios como argumentação jurídica, precisão técnica e coesão textual. As métricas adotadas (quantitativas e/ou qualitativas) serão detalhadas após a definição em equipe.

---

## 6. Referências

- Databricks. [Best Practices and Methods for LLM Evaluation](https://www.databricks.com/br/blog/best-practices-and-methods-llm-evaluation).
- Confident AI. [LLM Evaluation Metrics: Everything You Need for LLM Evaluation](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation).
- Databricks. [LLM Auto-Eval Best Practices for RAG](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG).
- Zhao, H. *et al.* [LLM Evaluation: A Comprehensive Survey](https://arxiv.org/html/2504.21202v1). arXiv, 2025.
- Astral. [uv — Python package manager](https://docs.astral.sh/uv/).
- Astral. [Ruff — An extremely fast Python linter](https://docs.astral.sh/ruff/).
- Typer. [Typer - Typer is a library for building CLIs in Python](https://typer.tiangolo.com/)
