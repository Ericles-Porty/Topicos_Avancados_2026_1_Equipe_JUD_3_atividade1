# Datasets — Visao geral

## Dominio juridico

A Equipe 3 atua no **Dominio Juridico**, utilizando dois datasets de questoes do Exame da OAB (Ordem dos Advogados do Brasil):

| Dataset | Tipo | Total | Subconjunto | Fonte |
|---|---|---|---|---|
| **J1 — OAB Bench** | Questoes discursivas | 210 | 12 questoes | [maritaca-ai/oab-bench](https://github.com/maritaca-ai/oab-bench) |
| **J2 — OAB Exams** | Multipla escolha | 2210 | 123 questoes | [eduagarcia/oab_exams](https://huggingface.co/datasets/eduagarcia/oab_exams) |

## Papel de cada dataset

### OAB Bench (J1) — Questoes abertas

Utilizado para avaliar a capacidade dos modelos em:
- Producao de texto juridico discursivo
- Argumentacao e fundamentacao legal
- Elaboracao de pecas pratico-profissionais

A avaliacao e feita por **rubrica oficial**, **comparacao qualitativa** entre modelos e **metricas automatizadas** (BLEU, ROUGE, BERTScore).

### OAB Exams (J2) — Multipla escolha

Utilizado para avaliar a capacidade dos modelos em:
- Compreensao de enunciados juridicos
- Selecao da alternativa correta entre 4 opcoes

A avaliacao e feita por **comparacao exata** com o gabarito oficial, utilizando metricas de classificacao (Acuracia, Precision, Recall, F1).

## Fontes

- **OAB Bench:** Desenvolvido pela Maritaca AI como benchmark para avaliacao de LLMs em tarefas de escrita juridica. Contem questoes da 2a fase do Exame da OAB.
- **OAB Exams:** Dataset compilado por Eduardo Garcia com questoes objetivas da 1a fase do Exame da OAB (edicoes de 2010 a 2018).
