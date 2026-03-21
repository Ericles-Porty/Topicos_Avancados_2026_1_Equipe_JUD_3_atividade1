import json
import pandas as pd

with open("results/respostas.json", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print(df.groupby("model").count())