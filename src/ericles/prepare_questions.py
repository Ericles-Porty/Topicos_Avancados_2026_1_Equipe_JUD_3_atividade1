import pandas as pd
import os

os.makedirs("dataset", exist_ok=True)

df = pd.read_csv("dataset/oab_questions.csv")

subset = df.iloc[47:59]

subset.to_csv("dataset/minhas_questoes.csv", index=False)

print(subset[["question_id","statement"]])