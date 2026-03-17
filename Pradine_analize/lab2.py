# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
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



# %%
string_cols = [f'string_{i}' for i in range(1, 11)]
data_minmax_by_category = data.groupby('category')[string_cols].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
data[string_cols] = data_minmax_by_category



# %%

data["hour"]=data['Timestamp'].dt.strftime('%H:%M')
data=data.dropna(subset=[f'string_{i}' for i in range(1, 11)])
melted = data.melt(
    id_vars=['month_day', 'hour', 'category'],
    value_vars=[f'string_{i}' for i in range(1, 11)],
    var_name='string',
    value_name='value'
)
result = melted.pivot_table(
    index=['month_day', 'string', 'category'],
    columns="hour",
    values='value',
    aggfunc='sum'
)
result["month_day"] = result.index.get_level_values('month_day')
result["string"] = result.index.get_level_values('string')
result["category"] = result.index.get_level_values('category')
result.reset_index(drop=True, inplace=True)

cols = result.columns.tolist()
cols = ['month_day', 'string', 'category'] + [col for col in cols if col not in ['month_day', 'string', 'category']]
result = result[cols]
result=result.sort_values(by=['month_day', 'string']).reset_index(drop=True)
result.index.name = 'id'


# %%
data_high = result[result['category'] == 'High'].drop(columns=['category', 'month_day'])
data_medium = result[result['category'] == 'Medium'].drop(columns=['category', 'month_day'])
data_low = result[result['category'] == 'Low'].drop(columns=['category', 'month_day'])


# %%
X_high, X_medium, X_low = data_high.drop(columns=['string']).fillna(0), data_medium.drop(columns=['string']).fillna(0), data_low.drop(columns=['string']).fillna(0)



# %%
tsne_high = TSNE(
    n_components=2,
    random_state=80085,
    init='pca',
    perplexity=30,
    learning_rate='auto',
    metric='euclidean',
    max_iter=1000
)
tsne_high_result = tsne_high.fit_transform(X_high)




# %%
X_pca_df = pd.DataFrame(tsne_high_result, columns=['PC1', 'PC2'])
X_pca_df['string'] = data_high['string'].values
plt.figure(figsize=(10, 6))
sns.scatterplot(data=X_pca_df, x='PC1', y='PC2', hue='string', palette='husl', s=75, hue_order=[f'string_{i}' for i in range(1, 11)])
plt.legend(title='Grandinės', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('t-SNE Projekcija (High)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()



# %%
tsne_high = TSNE(
    random_state=80085,
    perplexity=30,
    max_iter=20000,
    learning_rate='auto'
)
tsne_high_result = tsne_high.fit_transform(X_high)




# %%
X_pca_df = pd.DataFrame(tsne_high_result, columns=['PC1', 'PC2'])
X_pca_df['string'] = data_high['string'].values
plt.figure(figsize=(10, 6))
sns.scatterplot(data=X_pca_df, x='PC1', y='PC2', hue='string', palette='husl', s=75, hue_order=[f'string_{i}' for i in range(1, 11)])
plt.legend(title='Grandinės', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('t-SNE Projekcija (High)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()



# %%
#TSNE medium

tsne_medium = TSNE(
    random_state=80085,
    perplexity=12,
    learning_rate='auto',
    max_iter=2000,
)

tsne_medium_result = tsne_medium.fit_transform(X_medium)
X_pca_df = pd.DataFrame(tsne_medium_result, columns=['PC1', 'PC2'])
X_pca_df['string'] = data_medium['string'].values
plt.figure(figsize=(10, 6))
sns.scatterplot(data=X_pca_df, x='PC1', y='PC2', hue='string', palette='husl', s=75, hue_order=[f'string_{i}' for i in range(1, 11)])
plt.legend(title='Grandinės', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('t-SNE Projekcija (Medium)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()




# %%
#TSNE low

tsne_low = TSNE(
    random_state=80085,
    perplexity=50,
    learning_rate='auto',
    max_iter=10000
)

tsne_low_result = tsne_low.fit_transform(X_low)
X_pca_df = pd.DataFrame(tsne_low_result, columns=['PC1', 'PC2'])
X_pca_df['string'] = data_low['string'].values
plt.figure(figsize=(10, 6))
sns.scatterplot(data=X_pca_df, x='PC1', y='PC2', hue='string', palette='husl', s=75, hue_order=[f'string_{i}' for i in range(1, 11)])
plt.legend(title='Grandinės', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('t-SNE Projekcija (Low)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()



# %%
#UMAP high
umap_high = umap.UMAP(
    n_components=2,
    random_state=80085,
    n_neighbors=5, #Svarbus, kaitalioti
    min_dist=0.9, #Svarbus
    metric='euclidean', #Galima keisti
    n_epochs=2000, #Galima keisti
)
umap_high_result = umap_high.fit_transform(X_high)
X_umap_df = pd.DataFrame(umap_high_result, columns=['UMAP1', 'UMAP2'])
X_umap_df['string'] = data_high['string'].values
plt.figure(figsize=(10, 6))
sns.scatterplot(data=X_umap_df, x='UMAP1', y='UMAP2', hue='string', palette='husl', s=75, hue_order=[f'string_{i}' for i in range(1, 11)])
plt.legend(title='Grandinės', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('UMAP Projekcija (High)')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()



# %%
#UMAP medium



# %%
#UMAP low




# %%
#MDS high
mds_high = MDS(
	n_components=2,
	random_state=80085,
	metric=True,
	max_iter=3000,
	n_init=
)

# %%
#MDS medium




# %%
#MDS low









