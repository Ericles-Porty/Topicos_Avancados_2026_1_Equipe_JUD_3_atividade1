import json
import pandas as pd
import ollama
import re

with open("results/respostas.json", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

questions = df["question_id"].unique()

judge_model = "llama3"

results = []


def extract_json(text):
    """
    Extrai JSON da resposta do modelo
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return None


for q in questions:

    subset = df[df["question_id"] == q]

    answers = {}

    for _, row in subset.iterrows():
        answers[row["model"]] = row["answer"]

    answers_text = "\n\n".join(
        [f"Modelo {m}:\n{a}" for m, a in answers.items()]
    )

    prompt = f"""
Você é professor de direito avaliando respostas da prova da OAB.
Retorne APENAS JSON.
Não escreva explicações.
Não use markdown.

Pergunta:
{subset.iloc[0]["question"]}

Respostas:

{answers_text}

Avalie cada resposta em:

1 Argumentação jurídica (0-5)
2 Precisão legal (0-5)
3 Coesão textual (0-5)

Retorne APENAS JSON no formato:

{{
"mistral": {{"argumentacao":0,"precisao":0,"coesao":0}},
"llama3": {{"argumentacao":0,"precisao":0,"coesao":0}},
"gemma": {{"argumentacao":0,"precisao":0,"coesao":0}}
}}
"""

    response = ollama.chat(
        model=judge_model,
        options={"temperature": 0},
        messages=[{"role": "user", "content": prompt}]
    )

    content = response["message"]["content"]

    try:

        json_text = extract_json(content)

        scores = json.loads(json_text)

        for model in scores:

            r = scores[model]

            final = (
                0.4 * r["argumentacao"] +
                0.4 * r["precisao"] +
                0.2 * r["coesao"]
            )

            results.append({
                "question_id": q,
                "model": model,
                "argumentacao": r["argumentacao"],
                "precisao": r["precisao"],
                "coesao": r["coesao"],
                "final_score": final
            })

    except Exception as e:

        print("Erro avaliando questão", q)
        print("Resposta do judge:\n", content)


results_df = pd.DataFrame(results)

print("\nResultado médio por modelo:\n")

print(
    results_df.groupby("model")["final_score"].mean()
)

results_df.to_csv(
    "results/evaluation_comparative.csv",
    index=False
)
