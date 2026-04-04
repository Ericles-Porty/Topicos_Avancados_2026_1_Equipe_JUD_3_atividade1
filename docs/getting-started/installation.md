# Instalacao

## 1. Clonar o repositorio

```bash
git clone https://github.com/Ericles-Porty/Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1.git
cd Topicos_Avancados_2026_1_Equipe_JUD_3_atividade1
```

## 2. Criar ambiente virtual

```bash
python -m venv src/.venv
```

Ativacao:

```bash
# Windows (PowerShell)
src\.venv\Scripts\activate

# Linux/macOS
source src/.venv/bin/activate
```

## 3. Instalar dependencias

```bash
pip install pandas ollama minijinja matplotlib scikit-learn requests datasets evaluate rouge-score bert-score
```

## 4. Baixar modelos do Ollama

Certifique-se de que o servidor Ollama esta em execucao e baixe os modelos:

```bash
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull qwen2.5:3b
```

## 5. Verificacao

Para verificar que tudo esta configurado corretamente:

```bash
# Verificar que os modelos estao disponiveis
ollama list

# Verificar que as dependencias Python estao instaladas
python -c "import pandas, ollama, minijinja, matplotlib, sklearn; print('OK')"
```
