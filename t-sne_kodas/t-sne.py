import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#IRIS WITH CLASS TSNE


df = pd.read_csv('iris_with_class.txt', sep="\t", header=None,
                 names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
print(df.head())

X = df.iloc[:, :4].values
y = df["class"].values

tsne = TSNE() #čia nurodyti hiperparametrus
X_2d = tsne.fit_transform(X)

for label in df["class"].unique():
    plt.scatter(X_2d[y == label, 0], X_2d[y == label, 1], label=label)

plt.title("t-SNE projekcija – Iris duomenys")
plt.legend()
plt.show()


# MNIST TRAIN TSNE
df = pd.read_csv("mnist_train.csv")
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Sumažinta imtis, ištrynus krauna bent 10 min.
idx = np.random.RandomState(42).choice(len(X), size=5000, replace=False)
X, y = X[idx], y[idx]

tsne = TSNE(perplexity=50, random_state=1)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(8, 6))
for label in np.unique(y):
    mask = y == label
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=8, label=str(label))

plt.legend(title="Class", markerscale=2)
plt.title("t-SNE projekcija – MNIST duomenys")
plt.tight_layout()
plt.show()