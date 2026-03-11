#%%
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler
import umap

data = pd.read_csv('INV12.csv')
data.describe()

data['Timestamp']= pd.to_datetime(data['Timestamp'])
data["month_day_hour"]=data['Timestamp'].dt.strftime('%m-%d-%H')

data.groupby(data["month_day_hour"]).sum(numeric_only=True)
data.columns = ['Timestamp'] + [f'string_{i}' for i in range(1, 11)] + ['month_day_hour']

## NUSPRENDEME NA REIKSMES TIESIOG PASALINTI
data=data.dropna(subset=[f'string_{i}' for i in range(1, 11)])
#%%
filtered_data = data[data['Timestamp'].dt.hour.between(10, 17)]
data_hourly=filtered_data.groupby(filtered_data["month_day_hour"]).sum(numeric_only=True)
data_hourly
# %%
#Standartizavimas
X = filtered_data.drop(columns=['Timestamp', 'month_day_hour'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#%%
#PCA
pca = PCA(n_components=2, random_state=1)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=8)
plt.title("PCA (2 komponenčių) projekcija")
plt.axis('equal')
plt.show()

#%%
#t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=1)
X_2d = tsne.fit_transform(X_scaled)
min_val = min(X_2d[:, 0].min(), X_2d[:, 1].min())
max_val = max(X_2d[:, 0].max(), X_2d[:, 1].max())
plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], s=8)
plt.title("t-SNE projekcija")
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.axis('equal')
plt.show()
#%%
#MDS - NEREKOMENDUOJU LEISTI, NES LABAI ILGAI KRAUNA
mds = MDS(n_components=2, random_state=1, dissimilarity="euclidean")
X_mds = mds.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_mds[:, 0], X_mds[:, 1], s=8)
plt.title("MDS projekcija")
plt.axis("equal")
plt.show()

#%%
#UMAP
# Note: setting random_state forces UMAP to run single-threaded (n_jobs=1).
# If you want parallelism, remove random_state (but results will not be deterministic).
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=1, n_jobs=1)
X_umap = reducer.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], s=8)
plt.title("UMAP projekcija")
plt.axis("equal")
plt.show()
# %%
