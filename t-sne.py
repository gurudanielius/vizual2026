#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
#%%
df = pd.read_csv('./iris_with_class.txt', sep="\t", header=None,
                 names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

X = df.iloc[:, :4].values
y = df["class"].values

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

tsne = TSNE(random_state=42)
tsne.fit_transform(X)
print(tsne.learning_rate_)
#%%
perplexity_values = [5, 30, 50, 100]

fig, axes = plt.subplots(2, 2,figsize=(12, 8))	
axes = axes.flatten()

for ax, perp in zip(axes, perplexity_values):
    X_2d = TSNE(perplexity=perp, random_state=42).fit_transform(X)
    for label in df["class"].unique():
        mask = y == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label, alpha=0.7)
    ax.set_title(f"Perpleksiškumas: {perp}")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.05))
plt.tight_layout()
plt.show()
#%%
learning_rates = [50, 500, 1000, 2000]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for ax, lr in zip(axes, learning_rates):
    X_2d = TSNE(learning_rate=lr, random_state=42).fit_transform(X)
    for label in df["class"].unique():
        mask = y == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label,alpha=0.7)
    ax.set_title(f"Mokymosi greitis: {lr}")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,)
plt.show()
#%%
iter_values = [250, 500, 1000, 2000]
fig, axes = plt.subplots(2,2,figsize=(12, 8))
axes = axes.flatten()
for ax, n_iter in zip(axes, iter_values):
	X_2d = TSNE(max_iter=n_iter, random_state=42).fit_transform(X)
	for label in df["class"].unique():
		mask = y == label
		ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label, alpha=0.7)
	ax.set_title(f"Iteracijos: {n_iter}")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.show()


#%%
ee_values = [4, 12, 50, 100]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, ee in zip(axes[0:], ee_values):
    X_2d = TSNE(early_exaggeration=ee, random_state=42).fit_transform(X)
    for label in df["class"].unique():
        mask = y == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label)
    ax.set_title(f"Early exaggeration: {ee}")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.02))

plt.show()

X_2d = TSNE(perplexity=5, learning_rate=50, early_exaggeration=12, max_iter=500, ).fit_transform(X)

for label in df["class"].unique():
    plt.scatter(X_2d[y == label, 0], X_2d[y == label, 1], label=label)

plt.title("Iris duomenų projekcija naudojant t-SNE su pasirinktais parametrais")
plt.legend()
plt.show()
#%%

df = pd.read_csv("mnist_train.csv")
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

idx = np.random.RandomState(42).choice(len(X), size=5000, replace=False)
X, y = X[idx], y[idx]
np.unique(y)

tsne = TSNE(
    perplexity=3000,
    learning_rate=50,
    init="pca",
    max_iter=2500,
    early_exaggeration=12
)

X_2d = tsne.fit_transform(X)
min_val = min(X_2d[:, 0].min(), X_2d[:, 1].min())
max_val = max(X_2d[:, 0].max(), X_2d[:, 1].max())
plt.figure(figsize=(8,6))

for label in np.unique(y):
    mask = y == label
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label, s=8)

plt.legend()
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.axis('equal')
plt.show()
#%%
X = np.loadtxt("ellipsoid.50d10c.8.txt")
tsne = TSNE(
    perplexity=3000,
    learning_rate=500,
    early_exaggeration=12,
    max_iter=1000
)

X_2d = tsne.fit_transform(X)
min_val = min(X_2d[:, 0].min(), X_2d[:, 1].min())
max_val = max(X_2d[:, 0].max(), X_2d[:, 1].max())
plt.figure(figsize=(7,6))
plt.scatter(X_2d[:,0], X_2d[:,1], s=8)
plt.title("t-SNE projekcija ellipsoid duomenims (perpleksiškumas=3000)")
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.axis('equal')
plt.show()
#%%
X = np.loadtxt("iris.txt")
tsne = TSNE(
    perplexity=5,
    learning_rate=50,
    early_exaggeration=12,
    max_iter=500
)

X_2d = tsne.fit_transform(X)    
plt.figure(figsize=(7,6))
plt.scatter(X_2d[:,0], X_2d[:,1], s=8)
plt.title("t-SNE projekcija iris duomenims")
plt.show()
#%%
X = pd.read_csv("swiss_roll_example.csv", sep=";")

tsne = TSNE(
    perplexity=350,
    learning_rate=400,
    early_exaggeration=4,
    max_iter=5000
)

X_2d = tsne.fit_transform(X)
min_val = min(X_2d[:, 0].min(), X_2d[:, 1].min())
max_val = max(X_2d[:, 0].max(), X_2d[:, 1].max())
plt.figure(figsize=(8,6))
plt.scatter(X_2d[:,0], X_2d[:,1], s=8)
plt.title("t-SNE projekcija swiss roll duomenims")
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.axis('equal')
plt.show()

#%%