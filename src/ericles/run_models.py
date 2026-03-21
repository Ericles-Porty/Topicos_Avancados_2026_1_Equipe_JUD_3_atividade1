import ollama
import pandas as pd
import json
import ast

df = pd.read_csv("dataset/minhas_questoes.csv")

models = [
    "mistral",
    "llama3",
    "gemma"
]

results = []


for index, row in df.iterrows():

    question = row["statement"]
    system = row["system"]

    turns = ast.literal_eval(row["turns"])

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question}
    ]

    for turn in turns:
        if turn.strip():
            messages.append({
                "role": "user",
                "content": turn
            })

    print("\nPergunta:", question)

    for model in models:

        response = ollama.chat(
            model=model,
            messages=messages
        )

        answer = response["message"]["content"]

        results.append({
            "question_id": row["question_id"],
            "model": model,
            "question": question,
            "answer": answer
        })

        print("\nModelo:", model)
        print(answer[:200])

with open("results/respostas.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("\nRespostas salvas.")