import ollama
import pandas as pd
import json

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

    prompt = f"""
		{system}

		Pergunta:
		{question}
		"""
    
    prompt_especializado = f"""
		Você é um especialista em direito brasileiro.

		Responda a seguinte questão da OAB de forma técnica.

		Estruture a resposta com:
		1. explicação jurídica
		2. base legal (lei ou artigo)
		3. conclusão

		Pergunta:
		{question}
		"""

    print("\nPergunta:", question)

    for model in models:

        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt_especializado
                }
            ]
        )

        answer = response["message"]["content"]

        results.append({
            "question_id": index,
            "model": model,
            "question": question,
            "answer": answer
        })

        print("\nModelo:", model)
        print(answer[:200])

with open("results/respostas.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("\nRespostas salvas.")