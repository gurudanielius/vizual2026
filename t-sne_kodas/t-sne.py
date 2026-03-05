#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

df = pd.read_csv('../iris_with_class.txt', sep="\t", header=None,
                 names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

X = df.iloc[:, :4].values
y = df["class"].values
print(len(X))

tsne = TSNE(random_state=42)
tsne.fit_transform(X)
print(tsne.learning_rate_)
#%%
#SKIRTINGI PERPLEKSITY
perplexity_values = [2, 5, 30, 50, 100]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

# Originalūs duomenys
for label in df["class"].unique():
    mask = y == label
    axes[0].scatter(X[mask, 0], X[mask, 1], label=label)
axes[0].set_title("Originalūs duomenys be t-SNE", style="italic")

# t-SNE su skirtingais perplexity
for ax, perp in zip(axes[1:], perplexity_values):
    X_2d = TSNE(perplexity=perp, random_state=42).fit_transform(X)
    for label, group in df.groupby("class"):
        idx = group.index
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1], label=label)
    ax.set_title(f"Perpleksiškumas: {perp}")

# Rėmeliai visiems grafikiems
for ax in axes:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("gray")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.02))
plt.subplots_adjust(wspace=0.3)
plt.show()

#%%
#SKIRTINGI LEARNING RATE
learning_rates = [10, 50, 200, 500, 1000, 2000, 'auto']

fig, axes = plt.subplots(2, 4, figsize=(12, 8))
axes = axes.flatten()

# Originalūs duomenys
for label in df["class"].unique():
    mask = y == label
    axes[0].scatter(X[mask, 0], X[mask, 1], label=label, s=8)
axes[0].set_title("Originalūs duomenys be t-SNE", style="italic")

# t-SNE su skirtingais learning rate
for ax, lr in zip(axes[1:], learning_rates):
    X_2d = TSNE(learning_rate=lr, random_state=42).fit_transform(X)
    for label in df["class"].unique():
        mask = y == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label, s=8)
    ax.set_title(f"Mokymosi greitis: {lr}")

# Rėmeliai
for ax in axes:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("gray")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.02))

plt.subplots_adjust(wspace=0.3)
plt.show()
#%%
iter_values = [250, 500, 750, 1000, 2000]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

# Originalūs duomenys
for label in df["class"].unique():
    mask = y == label
    axes[0].scatter(X[mask, 0], X[mask, 1], label=label)
axes[0].set_title("Originalūs duomenys be t-SNE", style="italic")


# t-SNE su skirtingais iteracijų skaičiais
for ax, n_iter in zip(axes[1:], iter_values):
    X_2d = TSNE(max_iter=n_iter, random_state=42).fit_transform(X)
    for label in df["class"].unique():
        mask = y == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label)
    ax.set_title(f"Iteracijos: {n_iter}")

# Rėmeliai
for ax in axes:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("gray")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.02))

plt.subplots_adjust(wspace=0.3)
plt.show()

#%%
ee_values = [1, 4, 12, 50, 100]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

# Originalūs duomenys
for label in df["class"].unique():
    mask = y == label
    axes[0].scatter(X[mask, 0], X[mask, 1], label=label)
axes[0].set_title("Originalūs duomenys be t-SNE", style="italic")


# t-SNE su skirtingais early exaggeration
for ax, ee in zip(axes[1:], ee_values):
    X_2d = TSNE(early_exaggeration=ee, random_state=42).fit_transform(X)
    for label in df["class"].unique():
        mask = y == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label)
    ax.set_title(f"Early exaggeration: {ee}")

# Rėmeliai
for ax in axes:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("gray")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.02))

plt.subplots_adjust(wspace=0.3)
plt.show()

#%%
X_2d = TSNE(perplexity=30, learning_rate='auto', early_exaggeration=12, max_iter=750).fit_transform(X)

for label in df["class"].unique():
    plt.scatter(X_2d[y == label, 0], X_2d[y == label, 1], label=label)

plt.title("t-SNE projekcija – Iris duomenys")
plt.legend()
plt.show()

#%%
#MNIST TRAIN TSNE
df = pd.read_csv("../mnist_train.csv")
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Sumažinta imtis, ištrynus krauna bent 10 min.
idx = np.random.RandomState(42).choice(len(X), size=1000, replace=False)
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

df = pd.read_csv("../mnist_train.csv")
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Sumažinta imtis eksperimentams
idx = np.random.RandomState(42).choice(len(X), size=1000, replace=False)
X, y = X[idx], y[idx]

perplexity_values = [2, 5, 30, 50, 100]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

# Originalūs duomenys (pirmieji 2 požymiai)
for label in np.unique(y):
    mask = y == label
    axes[0].scatter(X[mask, 0], X[mask, 1], s=8, label=str(label))
axes[0].set_title("Originalūs duomenys be t-SNE", style="italic")

# t-SNE su skirtingais perplexity
for ax, perp in zip(axes[1:], perplexity_values):
    X_2d = TSNE(perplexity=perp, random_state=42).fit_transform(X)
    for label in np.unique(y):
        mask = y == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=8, label=str(label))
    ax.set_title(f"Perpleksiškumas: {perp}")

for ax in axes:
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("gray")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
           bbox_to_anchor=(0.5, 0.02), title="Klasė", markerscale=2)

plt.subplots_adjust(wspace=0.3)
plt.show()