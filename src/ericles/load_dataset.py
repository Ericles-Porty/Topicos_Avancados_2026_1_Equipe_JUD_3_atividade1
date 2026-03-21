from datasets import load_dataset
import pandas as pd
import os

os.makedirs("dataset", exist_ok=True)

questions = load_dataset("maritaca-ai/oab-bench", "questions", split="train")
guidelines = load_dataset("maritaca-ai/oab-bench", "guidelines", split="train")

df_questions = pd.DataFrame(questions)
df_guidelines = pd.DataFrame(guidelines)

print("questions:", len(df_questions))
print("guidelines:", len(df_guidelines))

df_questions.to_csv("dataset/oab_questions.csv", index=False)
df_guidelines.to_csv("dataset/oab_guidelines.csv", index=False)

print("Datasets salvos")