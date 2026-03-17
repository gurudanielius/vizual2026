# %%
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import matplotlib.dates as mdates
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler
import umap
import numpy as np

# %%
data = pd.read_csv('INV12.csv')
data.columns = ['Timestamp'] + [f'string_{i}' for i in range(1, 11)] 
data['Timestamp']= pd.to_datetime(data['Timestamp'])
data["month_day"]=data['Timestamp'].dt.strftime('%m-%d')
mapping=data.groupby(data["month_day"]).sum(numeric_only=True)
mapping["sum"]=mapping.sum(axis=1)
mapping["category"]=np.where(mapping["sum"] > mapping["sum"].quantile(0.75), "High",np.where(mapping["sum"] < mapping["sum"].quantile(0.25), "Low", "Medium"))
data["category"]=data["month_day"].map(mapping["category"])
data["hour"]=data['Timestamp'].dt.strftime('%H:%M')

# Melt to convert string columns into rows
melted = data.melt(
    id_vars=['month_day', 'hour'],
    value_vars=[f'string_{i}' for i in range(1, 11)],
    var_name='string',
    value_name='value'
)

# Pivot with month_day and string as index, hour as columns
result = melted.pivot_table(
    index=['month_day', 'string'],
    columns='hour',
    values='value',
    aggfunc='sum'
)

result["month_day"] = result.index.get_level_values('month_day')
result["string"] = result.index.get_level_values('string')
result.reset_index(drop=True, inplace=True)

cols = result.columns.tolist()
cols = ['month_day', 'string'] + [col for col in cols if col not in ['month_day', 'string']]
result = result[cols]
result=result.sort_values(by=['month_day', 'string']).reset_index(drop=True)
result.index.name = 'id'
data_final=result

# %%
data["month_day_hour"]=data['Timestamp'].dt.strftime('%m-%d-%H')

data.groupby(data["month_day_hour"]).sum(numeric_only=True)
data.columns = ['Timestamp'] + [f'string_{i}' for i in range(1, 11)] + ['month_day_hour']

## NUSPRENDEME NA REIKSMES TIESIOG PASALINTI
data=data.dropna(subset=[f'string_{i}' for i in range(1, 11)])
filtered_data = data[data['Timestamp'].dt.hour.between(10, 17)]

# %%
#Standartizavimas
X = filtered_data.drop(columns=['Timestamp', 'month_day_hour'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
#PCA
pca = PCA(n_components=2, random_state=1)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=8)
plt.title("PCA projekcija")
plt.axis('equal')
plt.show()


# %%
#t-SNE
# Color points by one of the original features (columns) in X_scaled.
# Change `color_col` to 0..9 to color by a different source column.
color_col = 0

tsne = TSNE(n_components=2, random_state=1)
X_2d = tsne.fit_transform(X_scaled)
min_val = min(X_2d[:, 0].min(), X_2d[:, 1].min())
max_val = max(X_2d[:, 0].max(), X_2d[:, 1].max())
plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], s=8, c=X_scaled[:, color_col], cmap='viridis')
plt.colorbar(label=f"Feature {color_col} value")
plt.title("t-SNE projekcija")
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.axis('equal')
plt.show()

# %%
#MDS - NEREKOMENDUOJU LEISTI, NES LABAI ILGAI KRAUNA
mds = MDS(n_components=2, random_state=1, dissimilarity="euclidean")
X_mds = mds.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_mds[:, 0], X_mds[:, 1], s=8)
plt.title("MDS projekcija")
plt.axis("equal")
plt.show()


# %%
#UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=1)
X_umap = reducer.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], s=8)
plt.title("UMAP projekcija")
plt.axis("equal")
plt.show()


