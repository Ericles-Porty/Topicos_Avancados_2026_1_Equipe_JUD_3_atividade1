<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/1/1c/Ufs_principal_positiva-nova.png" alt="ufs-logo" width="20%">

<h1>Topicos Avancados ES e SI</h1>

<h3>Atividade Avaliativa 1 — Curadoria de Datasets e Inferencia Basica com LLMs</h3>

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Ericles-Porty/Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1?machine=standardLinux2gb)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%2B-blue.svg" alt="Python 3.12+">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="Licenca MIT">
  </a>
  <a href="https://github.com/Ericles-Porty/Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1/commits/main">
    <img src="https://img.shields.io/github/last-commit/Ericles-Porty/Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1.svg" alt="Ultimo commit">
  </a>
  <a href="https://github.com/Ericles-Porty/Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1/stargazers">
    <img src="https://img.shields.io/github/stars/Ericles-Porty/Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1.svg?style=social" alt="Stars">
  </a>
</p>

</div>

## Sobre

Repositorio da **Equipe 3 (Juridica)** para a primeira atividade avaliativa da disciplina **Topicos Avancados em Engenharia de Software e Sistemas de Informacao I**. O projeto consiste na curadoria de datasets juridicos e na realizacao de inferencia basica utilizando Modelos de Linguagem (LLMs), com foco em questoes do Exame da OAB (Ordem dos Advogados do Brasil).

## Onde esta a documentacao

A documentacao completa do projeto esta disponivel na pasta [`docs/`](docs/), e a leitura deve comecar por [`docs/intro.md`](docs/intro.md).

## Dominio de atuacao

A Equipe 3 atua no **Dominio Juridico**, trabalhando com os seguintes datasets:

| Dataset | Tipo | Quantidade | Fonte |
|---|---|---|---|
| **J1 — OAB Bench** | Questoes Abertas | 210 questoes | [maritaca-ai/oab-bench](https://github.com/maritaca-ai/oab-bench) |
| **J2 — OAB Exams** | Multipla Escolha | 2210 questoes | [eduagarcia/oab_exams](https://huggingface.co/datasets/eduagarcia/oab_exams) |

> **Artigo de referencia:** [ACM Digital Library — OAB Bench](https://dl.acm.org/doi/pdf/10.1145/3769126.3769227)

## Colaboradores

<div align="center">
<table align="center">
  <tr>
    <td align="center">
      <a href="https://github.com/Ericles-Porty">
        <img src="https://github.com/Ericles-Porty.png" height="64" width="64" alt="Ericles dos Santos"/>
      </a><br/>
      <a href="https://github.com/Ericles-Porty">Ericles dos Santos</a>
    </td>
    <td align="center">
      <a href="https://github.com/ReinanHS">
        <img src="https://github.com/reinanhs.png" height="64" width="64" alt="Reinan Gabriel"/>
      </a><br/>
      <a href="https://github.com/ReinanHS">Reinan Gabriel</a>
    </td>
  </tr>
</table>
</div>

---

## 1. Ambiente de execucao

### 1.1 Configuracao de hardware

Os experimentos de inferencia foram executados em uma maquina local com a seguinte configuracao:

| Componente | Especificacao |
|---|---|
| **GPU** | NVIDIA GeForce RTX 4050 |
| **VRAM dedicada** | 6,0 GB |
| **RAM** | 32 GB |
| **SO** | Windows 11 |

### 1.2 Modelos de linguagem selecionados

Foram escolhidos **tres modelos de linguagem** de diferentes organizacoes, executados localmente via [Ollama](https://ollama.com/):

| # | Modelo | Desenvolvedor | Comando Ollama |
|---|---|---|---|
| 1 | Mistral | Mistral AI | `ollama run mistral` |
| 2 | Llama 3 | Meta | `ollama run llama3` |
| 3 | Gemma | Google | `ollama run gemma` |

### 1.3 Justificativa da escolha

- **Diversidade de origem:** Os tres modelos provem de organizacoes distintas (Mistral AI, Meta e Google), permitindo comparar diferentes abordagens de treinamento e arquiteturas.
- **Suporte multilingue:** Os tres modelos oferecem suporte ao idioma portugues, requisito essencial para inferencia em questoes da OAB.
- **Compatibilidade com Ollama:** Todos os modelos estao disponiveis no ecossistema Ollama, facilitando a execucao local padronizada.

### 1.4 Instalacao dos modelos

```bash
# Instalar o Ollama (Windows: baixar de https://ollama.com/download)

# Baixar os tres modelos
ollama pull mistral
ollama pull llama3
ollama pull gemma

# Verificar os modelos instalados
ollama list
```

---

## 2. Instrucoes de execucao

### 2.1 Pre-requisitos

- **Python** 3.12 ou superior
- **Ollama** com os modelos `mistral`, `llama3` e `gemma` instalados
- **pip** para instalacao de dependencias

### 2.2 Instalacao e execucao

```bash
# Clonar o repositorio
git clone https://github.com/Ericles-Porty/Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1.git
cd Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1

# Criar e ativar ambiente virtual
python -m venv src/.venv

# Ativacao no Windows (PowerShell)
src\.venv\Scripts\activate

# Ativacao no Linux/macOS
# source src/.venv/bin/activate

# Instalar dependencias
pip install pandas ollama minijinja matplotlib scikit-learn requests datasets evaluate rouge-score bert-score

# 1. Carregar e preparar os datasets
python src/load_dataset.py

# 2. Executar inferencia com os tres modelos + curadoria
python src/run_models.py

# 3. Avaliar resultados e gerar leaderboard
python src/evaluation.py
```

---

## 3. Distribuicao e mapeamento das questoes

### 3.1 Dataset J1 — Questoes abertas (`maritaca-ai/oab-bench`)

O dataset J1 contem **210 registros**. As questoes designadas para esta analise correspondem ao intervalo de indices **153 a 164** (Python, base zero), totalizando **12 questoes abertas**.

### 3.2 Dataset J2 — Questoes objetivas (`eduagarcia/oab_exams`)

O dataset J2 contem **2210 questoes objetivas**. As questoes designadas correspondem ao intervalo de indices **1600 a 1722** (Python, base zero), totalizando **123 questoes de multipla escolha**.

---

## 4. Estrutura dos datasets

### 4.1 Dataset `maritaca-ai/oab-bench`

| Campo | Tipo | Descricao |
|---|---|---|
| `question_id` | `string` | Identificador unico da questao |
| `category` | `string` | Categoria tematica (exame + area juridica) |
| `statement` | `string` | Enunciado completo da questao |
| `turns` | `array[string]` | Subperguntas ou desdobramentos |
| `values` | `array[number]` | Pesos de cada item de `turns` |
| `system` | `string` | System prompt para o modelo |

### 4.2 Dataset `eduagarcia/oab_exams`

| Campo | Tipo | Descricao |
|---|---|---|
| `id` | `string` | Identificador unico da questao |
| `question_number` | `integer` | Numero da questao na prova |
| `exam_id` | `string` | Identificador da edicao do exame |
| `exam_year` | `string` | Ano de realizacao do exame |
| `question` | `string` | Enunciado da questao |
| `choices` | `object` | Alternativas (`label` + `text`) |
| `answerKey` | `string` | Gabarito oficial (A, B, C ou D) |

---

## 5. Metodologia

### 5.1 Curadoria

Cada questao do lote e enriquecida automaticamente com:
- **Nivel de dificuldade** (1=Facil, 2=Medio, 3=Dificil)
- **Legislacao base** (Constituicao Federal, Codigo Civil, etc.)

A classificacao e realizada pelo modelo `llama3` via prompts especializados.

### 5.2 Inferencia com LLMs

As questoes sao submetidas aos tres modelos selecionados. Questoes abertas utilizam system prompt do dataset original. Questoes de multipla escolha utilizam system prompt estruturado que solicita resposta em JSON (`{"resposta": "letra"}`).

### 5.3 Avaliacao e comparacao

A avaliacao utiliza multiplas estrategias:

- **Questoes abertas — Rubrica:** Modelo juiz (`llama3`) avalia com base nos criterios oficiais
- **Questoes abertas — Comparativa:** Modelo juiz avalia argumentacao, precisao e coesao (0-5)
- **Questoes abertas — Metricas automatizadas:** BLEU, ROUGE-1/2/L e BERTScore F1 entre pares de modelos
- **Multipla escolha:** Acuracia, Precision, Recall e F1 (macro) via sklearn

---

## 6. Resultados

### 6.1 Avaliacao Cruzada — Questoes Abertas

| Par de Modelos | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 |
|---|---|---|---|---|---|
| mistral vs llama3 | — | — | — | — | — |
| mistral vs gemma | — | — | — | — | — |
| llama3 vs gemma | — | — | — | — | — |

### 6.2 Avaliacao Exata — Multipla Escolha

| Modelo | Acuracia | Precision | Recall | F1 |
|---|---|---|---|---|
| mistral | — | — | — | — |
| llama3 | — | — | — | — |
| gemma | — | — | — | — |

> **Nota:** Os valores serao preenchidos apos a execucao completa dos experimentos.

---

## 7. Referencias

- Databricks. [Best Practices and Methods for LLM Evaluation](https://www.databricks.com/br/blog/best-practices-and-methods-llm-evaluation).
- Confident AI. [LLM Evaluation Metrics](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation).
- Zhao, H. *et al.* [LLM Evaluation: A Comprehensive Survey](https://arxiv.org/html/2504.21202v1). arXiv, 2025.
- Maritaca AI. [OAB Bench](https://github.com/maritaca-ai/oab-bench).
- HuggingFace. [OAB Exams](https://huggingface.co/datasets/eduagarcia/oab_exams).
- Ollama. [Ollama](https://ollama.com/).

## Licenca

Este projeto esta licenciado sob a Licenca MIT — veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

<div align="center">
  <sub>Desenvolvido pela Equipe 3 — Dominio Juridico | UFS — 2026.1</sub>
</div>
