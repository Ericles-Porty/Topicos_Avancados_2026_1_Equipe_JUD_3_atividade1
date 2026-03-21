import json
import pandas as pd
import ollama
import ast
import re

with open("results/respostas.json", encoding="utf-8") as f:
    answers = json.load(f)

answers_df = pd.DataFrame(answers)

questions_df = pd.read_csv("dataset/minhas_questoes.csv")
guidelines_df = pd.read_csv("dataset/oab_guidelines.csv")

judge_model = "llama3"

results = []


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def clean_score(value):
    """
    Converte formatos como:
    '0.00/0,10'
    '0.00/0,60/0,70'
    para float
    """
    if isinstance(value, (int, float)):
        return float(value)

    value = str(value)

    # pegar primeiro número
    match = re.search(r"\d+[.,]?\d*", value)

    if match:
        return float(match.group(0).replace(",", "."))

    return 0.0


for _, row in answers_df.iterrows():

    question_id = row["question_id"]
    model = row["model"]
    answer = row["answer"]

    q = questions_df[questions_df["question_id"] == question_id].iloc[0]
    g = guidelines_df[guidelines_df["question_id"] == question_id].iloc[0]

    statement = q["statement"]
    turns = ast.literal_eval(q["turns"])
    values = ast.literal_eval(q["values"])

    choices = ast.literal_eval(g["choices"])
    rubric = choices[0]["turns"][0]

    judge_prompt = f"""
Você é corretor da prova da OAB.

PERGUNTA:
{statement}

SUBQUESTÕES:
{turns}

VALOR DE CADA ITEM:
{values}

RESPOSTA DO CANDIDATO:
{answer}

CRITÉRIOS DE CORREÇÃO:
{rubric}

Avalie cada subquestão separadamente.

Retorne APENAS JSON:

{{
"scores":[nota1,nota2,...],
"total":nota_total
}}

Regras:
- cada nota deve respeitar o valor máximo em VALOR DE CADA ITEM
- não use formato 0.00/0.10
- retorne apenas números
"""

    response = ollama.chat(
        model=judge_model,
        options={"temperature": 0},
        messages=[{"role": "user", "content": judge_prompt}]
    )

    content = response["message"]["content"]

    try:

        json_text = extract_json(content)

        result = json.loads(json_text)

        raw_scores = result.get("scores", [])

        scores = [clean_score(s) for s in raw_scores]

        total = sum(scores)

    except Exception as e:

        print("Erro avaliando questão", question_id)
        print("Resposta do judge:\n", content)

        scores = None
        total = None

    results.append({
        "question_id": question_id,
        "model": model,
        "scores": scores,
        "total_score": total
    })

results_df = pd.DataFrame(results)

print("\nScore médio por modelo:\n")

print(results_df.groupby("model")["total_score"].mean())

results_df.to_csv(
    "results/evaluation.csv",
    index=False
)