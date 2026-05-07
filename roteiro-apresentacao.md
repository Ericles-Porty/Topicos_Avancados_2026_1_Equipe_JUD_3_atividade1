# Roteiro de Apresentacao — Curadoria de Datasets e Inferencia com LLMs (Dominio Juridico)

## 1. Abertura (~1 min)

> "Nosso projeto e uma avaliacao de modelos de linguagem (LLMs) aplicados ao dominio juridico brasileiro. A ideia central e: sera que modelos pequenos, rodando localmente, conseguem responder questoes do Exame da OAB?"

- Disciplina: Topicos Avancados em Eng. de Software — UFS, 2026.1
- Equipe 3 (Dominio Juridico): Reinan, Fernanda, Ericles, Mikaela e Victor
- Cada membro ficou responsavel por um lote de questoes da OAB

---

## 2. O Problema e a Motivacao (~2 min)

> "O Exame da OAB e um dos exames mais dificeis do Brasil. Ele exige interpretacao de legislacao, argumentacao juridica e conhecimento tecnico profundo. Queriamos entender como LLMs compactos se comportam nesse cenario."

- O exame tem dois tipos de questao: **multipla escolha** e **dissertativas**
- Usamos dois datasets publicos como fonte:
  - **OAB Bench** (Maritaca AI) — questoes abertas com rubricas oficiais
  - **OAB Exams** (HuggingFace) — questoes de multipla escolha com gabarito
- Cada membro curou um subconjunto dessas questoes (cerca de 12 abertas + 122 objetivas)

---

## 3. Os Modelos Escolhidos (~1 min)

> "Selecionamos tres modelos de familias diferentes para evitar vies arquitetural. Todos rodam localmente via Ollama."

| Modelo | Params | Desenvolvedor |
|---|---|---|
| **Gemma 2** | 2B | Google |
| **Llama 3.2** | 3B | Meta |
| **Qwen 2.5** | 3B | Alibaba |

- Criterios: compatibilidade com hardware limitado (GPUs de 4-6 GB VRAM), suporte ao portugues e diversidade de arquitetura

---

## 4. O Pipeline — Fluxo do Projeto (~3 min)

> "O projeto segue um pipeline automatizado em tres etapas."

### Etapa 1 — Carga e Curadoria dos Dados

- Os datasets sao baixados automaticamente (HuggingFace e GitHub da Maritaca AI)
- As questoes sao filtradas para o lote de cada membro
- Resultado: CSVs prontos para inferencia

### Etapa 2 — Inferencia com os LLMs

- Cada questao e enviada aos 3 modelos via Ollama (execucao local)
- Para questoes abertas: o modelo recebe o enunciado + turnos de pergunta com prompt de sistema contextualizado
- Para multipla escolha: usamos templates Jinja que formatam a questao e as alternativas, pedindo resposta em JSON
- Um modulo de **curadoria automatica** classifica cada questao quanto a dificuldade, legislacao base e subdominio juridico (usando o proprio LLM como anotador)

### Etapa 3 — Avaliacao e Leaderboard

- Questoes objetivas: comparacao direta com gabarito (Acuracia, Precisao, Recall, F1-Score)
- Questoes abertas — tres formas de avaliacao:
  1. **Rubrica oficial**: modelo juiz avalia a resposta contra os criterios da banca
  2. **Avaliacao comparativa**: modelo juiz pontua argumentacao, precisao e coesao
  3. **Metricas cross-model**: BLEU, ROUGE e BERTScore entre pares de modelos e contra a guideline

---

## 5. Conceitos e Metricas Utilizados (~2 min)

> "Usamos uma combinacao de metricas lexicais e semanticas para capturar diferentes aspectos da qualidade das respostas."

- **Acuracia / Precisao / Recall / F1**: metricas classicas de classificacao, usadas nas questoes objetivas
- **BLEU**: mede sobreposicao de n-gramas — captura similaridade literal
- **ROUGE** (1, 2, L): avalia recall de n-gramas — util para ver se o modelo cobriu os pontos da referencia
- **BERTScore**: usa embeddings contextuais para medir similaridade semantica — crucial porque respostas juridicas podem estar corretas sem usar as mesmas palavras
- **Modelo Juiz (LLM-as-a-Judge)**: um LLM avalia as respostas de outros LLMs com base em rubrica ou criterios qualitativos — conceito importante da literatura recente de avaliacao de LLMs

---

## 6. Principais Resultados (~2 min)

> "Nenhum modelo foi o melhor em tudo. Cada um se destacou em uma dimensao diferente."

- **Multipla escolha**: desempenho entre 36% e 50% — abaixo do minimo para aprovacao na OAB, mas esperado para modelos tao pequenos
- **Questoes abertas**: os modelos concordam semanticamente entre si (BERTScore ~0.75), mas se distanciam da rubrica oficial (BERTScore ~0.65)
- **Inversao de ranking**: dependendo da metrica, o melhor modelo muda — ex: Qwen lidera em acuracia objetiva, mas fica em ultimo na aderencia a rubrica; Gemma fica em ultimo em objetivas, mas lidera em qualidade de escrita
- Essa ausencia de vencedor absoluto reforca que **avaliacoes baseadas em uma unica metrica podem ser enganosas**

---

## 7. Limitacoes e Licoes (~1 min)

- Modelos com menos de 4B de parametros tem dificuldade com raciocinio juridico complexo
- Usar o proprio modelo avaliado como juiz pode introduzir vies de auto-preferencia
- Resultados variam conforme o subconjunto de questoes — sensibilidade a composicao do dataset
- A abordagem Docs-as-Code permitiu versionar codigo e documentacao juntos

---

## 8. Encerramento (~30s)

> "O projeto mostra que, mesmo com recursos limitados, e possivel montar um pipeline completo de avaliacao de LLMs. Os modelos pequenos ja conseguem produzir texto juridico coeso, mas ainda falham na precisao tecnica exigida pela OAB. A avaliacao multidimensional — combinando metricas automaticas e modelo juiz — se mostrou essencial para uma analise justa."

---

**Tempo total estimado: ~12 minutos**
