import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

df = pd.read_csv("iris_with_class.txt", sep="\t", header=None,
                 names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
X = df.iloc[:, :4].values
y = df["class"].values

colors = {"Iris-setosa": "red", "Iris-versicolor": "green", "Iris-virginica": "blue"}

# Skirtingos perplexity reikšmės
perplexity_values = [5, 15, 30, 50]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, perp in zip(axes, perplexity_values):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_2d = tsne.fit_transform(X)

    for label, color in colors.items():
        mask = y == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=label, alpha=0.7)

    ax.set_title(f"perplexity={perp}")
    ax.legend(fontsize=7)

plt.suptitle("Perplexity įtaka – Iris duomenys")
plt.tight_layout()
plt.show()