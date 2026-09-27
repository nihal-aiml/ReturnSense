import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/prob_comparison.csv")
df = df[df["total_purchases"] >= 50].copy()
df = df.sort_values("model_return_prob", ascending=False).head(20)

plt.figure(figsize=(12, 6))
plt.barh(df["Description"].str[:35], df["model_return_prob"]*100, color="#4C72B0")
plt.xlabel("Predicted Return Probability (%)")
plt.title("Top 20 Products by Predicted Return Risk")
plt.tight_layout()
plt.savefig("eda_return_by_category.png", dpi=150, bbox_inches='tight')
print("Saved eda_return_by_category.png")