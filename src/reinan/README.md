# Contribuições Individuais — Reinan Gabriel

Este documento descreve as contribuições realizadas pelo aluno **Reinan Gabriel** no contexto da **Atividade Avaliativa 1** da disciplina **Tópicos Avançados em Engenharia de Software e Sistemas de Informação I**, ministrada na Universidade Federal de Sergipe (UFS), semestre 2026.1.

O trabalho consiste na curadoria de datasets jurídicos e na realização de inferência básica com Modelos de Linguagem (LLMs), aplicados a questões do Exame da Ordem dos Advogados do Brasil (OAB).

---

## 1. Instruções de execução

### 1.1 Pré-requisitos

Para reproduzir os experimentos, é necessário ter instalado:

- **Python** 3.12 ou superior
- **uv** 0.10 ou superior (gerenciador de pacotes e runner Python)

### 1.2 Instalação e execução

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
```

---

## 2. Distribuição e mapeamento das questões

Conforme as orientações da atividade, o dataset **J1** (`maritaca-ai/oab-bench`) contém **210 registros** distribuídos em dois subsets:

| Subset       | Intervalo (contagem geral) | Quantidade |
|--------------|----------------------------|------------|
| `guidelines` | 1 a 105                    | 105        |
| `questions`  | 106 a 210                  | 105        |

As questões designadas para esta análise correspondem ao intervalo **177 a 188** (12 questões). Como esse intervalo está inteiramente contido na segunda metade da contagem geral (106 a 210), todas as 12 questões pertencem exclusivamente ao subset `questions`. Nenhuma questão deste lote pertence ao subset `guidelines`.

### 2.1 Filtragem via código

Na implementação em Python, a indexação é baseada em zero. Dessa forma, para acessar as questões de número 177 a 188, os dois subsets são concatenados (preservando a ordem `guidelines` + `questions`) e os registros são extraídos pelos **índices 176 a 187** (inclusive).

---

## 3. Estrutura dos datasets

A seguir, apresenta-se a descrição dos campos que compõem o dataset utilizado nas questões abertas.

### 3.1 Dataset `maritaca-ai/oab-bench` — Subset `questions`

Este dataset contém os enunciados das questões discursivas da 2ª fase do Exame da OAB, acompanhados de metadados e instruções de sistema para os modelos de linguagem.

| Campo         | Tipo            | Descrição                                                                                                                                                                                                                                                       |
|---------------|-----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `question_id` | `string`        | Identificador único da questão. Codifica a edição do exame, a área do Direito e o número da questão (ex.: `41_direito_constitucional_questao_2`).                                                                                                               |
| `category`    | `string`        | Categoria temática, agrupando a questão por exame e área jurídica (ex.: `41_direito_constitucional`).                                                                                                                                                           |
| `statement`   | `string`        | Enunciado completo da questão, incluindo o contexto fático, a narrativa jurídica e o comando introdutório da peça ou questão discursiva.                                                                                                                        |
| `turns`       | `array[string]` | Lista de subperguntas ou desdobramentos da tarefa. Em questões discursivas, cada elemento corresponde a uma pergunta específica. Em peças prático-profissionais, pode conter um único item vazio (`""`), indicando que a resposta esperada é a peça na íntegra. |
| `values`      | `array[number]` | Pesos ou pontuações atribuídas a cada item de `turns`. Os valores refletem a distribuição de pontos do exame (ex.: `[0.65, 0.6]` para subperguntas ou `[5.0]` para o valor total de uma peça).                                                                  |
| `system`      | `string`        | Instrução de sistema (*system prompt*) para o modelo de linguagem, definindo o papel do candidato, as regras da prova e as restrições de formatação exigidas pelo exame.                                                                                        |

### 3.2 Exemplo de registro

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

---

## 4. Metodologia

> **Nota:** Esta seção será complementada à medida que os experimentos forem executados e os resultados consolidados.

### 4.1 Curadoria (Classificação criativa)

Cada questão do lote atribuído será classificada de acordo com os parâmetros definidos em conjunto pela equipe, incluindo nível de dificuldade, área de especialidade jurídica e legislação de referência.

### 4.2 Inferência com LLMs

Serão selecionados **três modelos de linguagem** para submeter as questões abertas do lote. A escolha dos modelos, bem como os parâmetros de inferência utilizados, serão documentados nesta seção após a execução dos experimentos.

### 4.3 Avaliação e comparação

Por se tratar do domínio jurídico, as questões abertas **não possuem gabarito oficial**. A avaliação será realizada por meio da comparação entre as respostas dos três modelos, considerando critérios como argumentação jurídica, precisão técnica e coesão textual. As métricas adotadas (quantitativas e/ou qualitativas) serão detalhadas após a definição em equipe.

---

## 5. Referências

- Databricks. [Best Practices and Methods for LLM Evaluation](https://www.databricks.com/br/blog/best-practices-and-methods-llm-evaluation).
- Confident AI. [LLM Evaluation Metrics: Everything You Need for LLM Evaluation](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation).
- Databricks. [LLM Auto-Eval Best Practices for RAG](https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG).
- Zhao, H. *et al.* [LLM Evaluation: A Comprehensive Survey](https://arxiv.org/html/2504.21202v1). arXiv, 2025.
- Astral. [uv — Python package manager](https://docs.astral.sh/uv/).
- Astral. [Ruff — An extremely fast Python linter](https://docs.astral.sh/ruff/).
