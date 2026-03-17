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
data_final=result

data_high = data_final[data_final['category'] == 'High'].drop(columns=['category', 'month_day'])
data_medium = data_final[data_final['category'] == 'Medium'].drop(columns=['category', 'month_day'])
data_low = data_final[data_final['category'] == 'Low'].drop(columns=['category', 'month_day'])


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
    n_components=2,
    random_state=80085,
    # init='pca',
    perplexity=30,
    metric='euclidean',
    max_iter=20000,
    learning_rate=100
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
    n_components=2,
    random_state=80085,
    init='pca',
    perplexity=5,
    learning_rate='auto',
    metric='euclidean',
    max_iter=1000
)

tsne_medium_result = tsne_medium.fit_transform(X_medium)
plot_tsne_projection(tsne_medium_result, X_medium.to_numpy(), color_col=0, title="t-SNE projekcija (Medium)")



# %%
#TSNE low

tsne_low = TSNE(
    n_components=2,
    random_state=80085,
    init='pca',
    perplexity=5,
    learning_rate='auto',
    metric='euclidean',
    max_iter=1000
)

tsne_low_result = tsne_low.fit_transform(X_low)
plot_tsne_projection(tsne_low_result, X_low.to_numpy(), color_col=0, title="t-SNE projekcija (Low)")


# %%
#UMAP high


# %%
#UMAP medium

#UMAP low


# %%
#MDS high


# %%
#MDS medium


# %%
#MDS low



