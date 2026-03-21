import pandas as pd
import matplotlib.pyplot as plt

# carregar dados
df = pd.read_csv("results/evaluation_comparative.csv")

# média das métricas por modelo
model_means = df.groupby("model")[["argumentacao", "precisao", "coesao", "final_score"]].mean()

print(model_means)

# gráfico comparativo
model_means.plot(kind="bar")

plt.title("Comparação de Modelos - Avaliação Jurídica")
plt.ylabel("Pontuação média")
plt.xlabel("Modelo")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("results/model_comparison.png")

plt.show()