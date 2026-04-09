<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/1/1c/Ufs_principal_positiva-nova.png" alt="ufs-logo" width="20%">

<h1>Tópicos Avançados ES e SI</h1>

<h3>Atividade Avaliativa 1 — Curadoria de Datasets e Inferência Básica com LLMs</h3>

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Ericles-Porty/Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1?machine=standardLinux2gb)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue.svg" alt="Python 3.12+">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="Licença MIT">
  </a>
  <a href="https://github.com/Ericles-Porty/Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1/commits/main">
    <img src="https://img.shields.io/github/last-commit/Ericles-Porty/Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1.svg" alt="Último commit">
  </a>
  <a href="https://github.com/Ericles-Porty/Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1/stargazers">
    <img src="https://img.shields.io/github/stars/Ericles-Porty/Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1.svg?style=social" alt="Stars">
  </a>
</p>

</div>

## Sobre

Repositório individual de **Éricles dos Santos** para a primeira atividade avaliativa da disciplina **Tópicos Avançados em Engenharia de Software e Sistemas de Informação I** (UFS — 2026.1). O projeto consiste na curadoria de datasets jurídicos e na realização de inferência básica utilizando Modelos de Linguagem (LLMs), com foco em questões do Exame da OAB (Ordem dos Advogados do Brasil).

## Onde está a documentação

A documentação completa do projeto está disponível na pasta [`docs/`](docs/), e a leitura deve começar por [`docs/intro.md`](docs/intro.md).

## Domínio de atuação

Este projeto atua no **Domínio Jurídico**, trabalhando com os seguintes datasets:

| Dataset | Tipo | Quantidade | Fonte |
|---|---|---|---|
| **J1 — OAB Bench** | Questões Abertas | 210 questões | [maritaca-ai/oab-bench](https://github.com/maritaca-ai/oab-bench) |
| **J2 — OAB Exams** | Múltipla Escolha | 2210 questões | [eduagarcia/oab_exams](https://huggingface.co/datasets/eduagarcia/oab_exams) |

> **Artigo de referência:** [ACM Digital Library — OAB Bench](https://dl.acm.org/doi/pdf/10.1145/3769126.3769227)

## Vídeo Demonstrativo

> **Link do vídeo:** [https://youtu.be/1QMsz2wOlt0](https://youtu.be/1QMsz2wOlt0)

## Colaboradores

<div align="center">
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/Ericles-Porty">
        <img src="https://github.com/Ericles-Porty.png" height="64" width="64" alt="Éricles dos Santos"/>
      </a><br/>
      <a href="https://github.com/Ericles-Porty">Éricles dos Santos</a>
    </td>
  </tr>
</table>
</div>

---

## 1. Ambiente de execução

### 1.1 Configuração de hardware

Os experimentos de inferência foram executados em uma máquina local com a seguinte configuração:

| Componente | Especificação |
|---|---|
| **GPU** | NVIDIA GeForce RTX 4050 |
| **VRAM dedicada** | 6,0 GB |
| **RAM** | 32 GB |
| **SO** | Windows 11 |

### 1.2 Modelos de linguagem selecionados

Foram escolhidos **três modelos de linguagem** de diferentes organizações, executados localmente via [Ollama](https://ollama.com/):

| # | Modelo | Desenvolvedor | Comando Ollama |
|---|---|---|---|
| 1 | Llama 3.2 3B | Meta | `ollama run llama3.2:3b` |
| 2 | Gemma 2 2B | Google | `ollama run gemma2:2b` |
| 3 | Qwen 2.5 3B | Alibaba | `ollama run qwen2.5:3b` |

### 1.3 Justificativa da escolha

- **Diversidade de origem:** Os três modelos provêm de organizações distintas (Meta, Google e Alibaba), permitindo comparar diferentes abordagens de treinamento e arquiteturas.
- **Modelos compactos:** Todos possuem até 3B de parâmetros, viabilizando execução em hardware modesto.
- **Suporte multilíngue:** Os três modelos oferecem suporte ao idioma português, requisito essencial para inferência em questões da OAB.
- **Compatibilidade com Ollama:** Todos os modelos estão disponíveis no ecossistema Ollama, facilitando a execução local padronizada.

### 1.4 Instalação dos modelos

```bash
# Instalar o Ollama (Windows: baixar de https://ollama.com/download)

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

- **Python** 3.12 ou superior
- **Ollama** com os modelos `llama3.2:3b`, `gemma2:2b` e `qwen2.5:3b` instalados
- **pip** para instalação de dependências

### 2.2 Instalação e execução

```bash
# Clonar o repositório
git clone https://github.com/Ericles-Porty/Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1.git
cd Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1

# Criar e ativar ambiente virtual
python -m venv src/.venv

# Ativação no Windows (PowerShell)
src\.venv\Scripts\activate

# Ativação no Linux/macOS
# source src/.venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# 1. Carregar e preparar os datasets
python src/load_dataset.py

# 2. Executar inferência com os três modelos + curadoria
python src/run_models.py

# 3. Avaliar resultados e gerar leaderboard
python src/evaluation.py
```

---

## 3. Mapeamento das questões

### 3.1 Dataset J1 — Questões abertas (`maritaca-ai/oab-bench`)

O dataset J1 contém **210 registros**. As questões designadas para minha análise correspondem ao intervalo de índices **153 a 164** (Python, base zero), totalizando **12 questões abertas**.

### 3.2 Dataset J2 — Questões objetivas (`eduagarcia/oab_exams`)

O dataset J2 contém **2210 questões objetivas**. As questões designadas correspondem ao intervalo de índices **1600 a 1722** (Python, base zero), totalizando **122 questões de múltipla escolha**.

---

## 4. Estrutura dos datasets

### 4.1 Dataset `maritaca-ai/oab-bench`

| Campo | Tipo | Descrição |
|---|---|---|
| `question_id` | `string` | Identificador único da questão |
| `category` | `string` | Categoria temática (exame + área jurídica) |
| `statement` | `string` | Enunciado completo da questão |
| `turns` | `array[string]` | Subperguntas ou desdobramentos |
| `values` | `array[number]` | Pesos de cada item de `turns` |
| `system` | `string` | System prompt para o modelo |

### 4.2 Dataset `eduagarcia/oab_exams`

| Campo | Tipo | Descrição |
|---|---|---|
| `id` | `string` | Identificador único da questão |
| `question_number` | `integer` | Número da questão na prova |
| `exam_id` | `string` | Identificador da edição do exame |
| `exam_year` | `string` | Ano de realização do exame |
| `question` | `string` | Enunciado da questão |
| `choices` | `object` | Alternativas (`label` + `text`) |
| `answerKey` | `string` | Gabarito oficial (A, B, C ou D) |

---

## 5. Metodologia

### 5.1 Curadoria

A curadoria avalia cada questão sob a ótica da **Complexidade de Raciocínio (Reasoning)** e do **Aterramento (Grounding)** exigidos da IA. Cada questão é enriquecida automaticamente com:

- **Nível de Dificuldade — Complexidade do Raciocínio do LLM:**
  - Nível 1: Recuperação Factual Direta (*Fact Retrieval*)
  - Nível 2: Raciocínio Lógico-Dedutivo (*Logical Deduction*)
  - Nível 3: Hermenêutica Jurídica Complexa (*Complex Hermeneutics*)
- **Subdomínio Semântico** — Área de especialidade jurídica (Direito Civil, Direito Penal, Direito Constitucional, etc.)
- **Corpus de Referência** — Ground truth onde a resposta deve estar ancorada (Constituição Federal de 1988, Código Civil - Lei 10.406/2002, etc.)

A classificação é realizada pelo modelo `llama3.2:3b` com `temperature=0` via prompts especializados.

### 5.2 Inferência com LLMs

As questões são submetidas aos três modelos selecionados. Questões abertas utilizam system prompt do dataset original. Questões de múltipla escolha utilizam system prompt estruturado que solicita resposta em JSON (`{"resposta": "letra"}`).

### 5.3 Avaliação e comparação

A avaliação utiliza múltiplas estratégias:

- **Questões abertas — Rubrica:** Modelo juiz (`llama3.2:3b`) avalia com base nos critérios oficiais
- **Questões abertas — Comparativa:** Modelo juiz avalia argumentação, precisão e coesão (0-5)
- **Questões abertas — Métricas automatizadas:** BLEU, ROUGE-1/2/L e BERTScore F1 entre pares de modelos
- **Múltipla escolha:** Acurácia, Precision, Recall e F1 (macro) via sklearn

---

## 6. Resultados

### 6.1 Leaderboard Consolidado

| Modelo | Rubrica (%) | Acurácia MC (%) | Precision MC | Recall MC | F1 MC | Argumentação | Precisão | Coesão | Score Comparativo |
|---|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | 25.00 | 46.34 | 0.4096 | 0.3662 | 0.3632 | 4.00 | 3.92 | 4.00 | 3.97 |
| gemma2:2b | 21.45 | 40.65 | 0.4124 | 0.3979 | 0.3884 | 4.83 | 4.83 | 4.83 | 4.83 |
| qwen2.5:3b | 16.67 | 50.41 | 0.5250 | 0.4903 | 0.4742 | 4.67 | 4.67 | 4.67 | 4.67 |

### 6.2 Avaliação Cruzada — Questões Abertas

| Par de Modelos | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---|---|---|---|---|---|
| llama3.2:3b vs gemma2:2b | 0.1550 | 0.4997 | 0.2249 | 0.2651 | 0.7568 |
| llama3.2:3b vs qwen2.5:3b | 0.1288 | 0.4796 | 0.2081 | 0.2405 | 0.7534 |
| gemma2:2b vs qwen2.5:3b | 0.1145 | 0.5003 | 0.1855 | 0.2222 | 0.7502 |

### 6.3 Avaliação Exata — Múltipla Escolha

| Modelo | Acurácia (%) | Precision | Recall | F1 |
|---|---|---|---|---|
| llama3.2:3b | 46.34 | 0.4096 | 0.3662 | 0.3632 |
| gemma2:2b | 40.65 | 0.4124 | 0.3979 | 0.3884 |
| qwen2.5:3b | 50.41 | 0.5250 | 0.4903 | 0.4742 |

---

## 7. Referências

- Databricks. [Best Practices and Methods for LLM Evaluation](https://www.databricks.com/br/blog/best-practices-and-methods-llm-evaluation).
- Confident AI. [LLM Evaluation Metrics](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation).
- Zhao, H. *et al.* [LLM Evaluation: A Comprehensive Survey](https://arxiv.org/html/2504.21202v1). arXiv, 2025.
- Maritaca AI. [OAB Bench](https://github.com/maritaca-ai/oab-bench).
- HuggingFace. [OAB Exams](https://huggingface.co/datasets/eduagarcia/oab_exams).
- Ollama. [Ollama](https://ollama.com/).

## Licença

Este projeto está licenciado sob a Licença MIT — veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Inspiração

Este projeto teve grande inspiração nos conceitos e tecnologias aplicadas no repositório de [Reinan Gabriel](https://github.com/ReinanHS):
- [ReinanHS/Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1](https://github.com/ReinanHS/Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1)

---

<div align="center">
  <sub>Desenvolvido por Éricles dos Santos — Domínio Jurídico | UFS — 2026.1</sub>
</div>
