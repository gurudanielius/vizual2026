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

#%%
def plot_tsne_panel(tsne_results, hyperparameter, title, labels, ncols=3, figsize=(14, 8)):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_plots = len(tsne_results)
    nrows = (n_plots + ncols - 1) // ncols

    labels = np.array(labels) 
    unique_labels = sorted(labels, key=lambda x: int(x.split('_')[1]))
    unique_labels = list(dict.fromkeys(unique_labels))
    label_names = unique_labels

    palette = sns.color_palette("husl", len(unique_labels))
    color_map = dict(zip(label_names, palette))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True, sharex=False, sharey=False)
    axes = np.atleast_1d(axes).flatten()

    for i, (value, X_2d) in enumerate(tsne_results.items()):
        ax = axes[i]

        df = pd.DataFrame({
            "Dim1": X_2d[:, 0],
            "Dim2": X_2d[:, 1],
            "string": labels  
        })

        sns.scatterplot(
            data=df,
            x="Dim1",
            y="Dim2",
            hue="string",
            palette=color_map,
            hue_order=label_names,
            s=30,
            legend=False,
            ax=ax
        )

        ax.set_title(f"{hyperparameter} = {value}")
        ax.set_box_aspect(1)
        ax.set_aspect('equal', adjustable='datalim')

        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    handles = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=color_map[name],
            markersize=6,
            linestyle=''
        )
        for name in label_names
    ]

    fig.legend(
        handles,
        label_names,
        title="Grandinės",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )

    plt.suptitle(title)
    # plt.tight_layout(rect=[0, 0, 0.9, 1]) 
    plt.savefig(f"{title.replace(' ', '_')}_{hyperparameter.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()
# %%
perplexity_values = [5, 15, 30, 45, 60, 75]
learning_rate_values = [10, 100, 200, 500, 1000, 1500]
early_exaggeration_values = [1, 4, 12, 24, 32, 48]
max_iter_values = [250, 500, 1000, 1500, 2000, 3000]

tsne_perplexity_results = {}
tsne_learning_rate_results = {}
tsne_early_exaggeration_results = {}
tsne_max_iter_results = {}

#%%
for perplexity in perplexity_values:
    tsne_high = TSNE(
        n_components=2,
        random_state=80085,
        init='pca',
        perplexity=perplexity,
        learning_rate='auto',
        metric='euclidean',
        max_iter=1000
    )
    tsne_perplexity_results[perplexity] = tsne_high.fit_transform(X_high)

plot_tsne_panel(tsne_perplexity_results, title = "t-SNE projekcija skirtingiems perpleksiškumams (aukšta elektros gamyba)", hyperparameter="Perpleksiškumas", labels=data_high['string'].values, ncols=3, figsize=(15, 10))

#%%
for learning_rate in learning_rate_values:
    tsne_high = TSNE(
        n_components=2,
        random_state=80085,
        init='pca',
        perplexity=45,
        learning_rate=learning_rate,
        metric='euclidean',
        max_iter=1000
    )
    tsne_learning_rate_results[learning_rate] = tsne_high.fit_transform(X_high)

plot_tsne_panel(tsne_learning_rate_results, title = "t-SNE projekcija skirtingiems mokymosi greičiams (aukšta elektros gamyba)", hyperparameter="Mokymosi greitis", labels=data_high['string'].values, ncols=3, figsize=(15, 10))

#%%
for early_exaggeration in early_exaggeration_values:
    tsne_high = TSNE(
        n_components=2,
        random_state=80085,
        init='pca',
        perplexity=45,
        learning_rate='auto',
        early_exaggeration=early_exaggeration,
        metric='euclidean',
        max_iter=1000
    )
    tsne_early_exaggeration_results[early_exaggeration] = tsne_high.fit_transform(X_high)

plot_tsne_panel(tsne_early_exaggeration_results, title = "t-SNE projekcija skirtingiems early exaggeration reikšmėms (aukšta elektros gamyba)", hyperparameter="Early Exaggeration", labels=data_high['string'].values, ncols=3, figsize=(15, 10))
# %%
for max_iter in max_iter_values:
    tsne_high = TSNE(
        n_components=2,
        random_state=80085,
        init='pca',
        perplexity=45,
        learning_rate='auto',
        early_exaggeration=12,
        metric='euclidean',
        max_iter=max_iter
    )
    tsne_max_iter_results[max_iter] = tsne_high.fit_transform(X_high)

plot_tsne_panel(tsne_max_iter_results, title = "t-SNE projekcija skirtingiems iteracijų skaičiams (aukšta elektros gamyba)", hyperparameter="Iteracijų skaičius", labels=data_high['string'].values, ncols=3, figsize=(15, 10))

#%%
tsne_medium_results = {}
#%% 
#TSNE medium
for perplexity in perplexity_values:
    tsne_medium = TSNE(
        n_components=2,
        random_state=80085,
        init='pca',
        perplexity=perplexity,
        learning_rate='auto',
        metric='euclidean',
        max_iter=1000
    )

    tsne_medium_results[perplexity] = tsne_medium.fit_transform(X_medium)

plot_tsne_panel(tsne_medium_results, title = "t-SNE projekcija skirtingiems perpleksiškumams (vidutinė elektros gamyba)", hyperparameter="Perpleksiškumas", labels=data_medium['string'].values, ncols=3, figsize=(15, 10))
#realiai jokio skirtumo galima imti sakykime perplexity = 30
#%%
# %%
#UMAP high
umap_high = umap.UMAP(
    n_components=2,
    random_state=80085,
    n_neighbors=5, #Svarbus, kaitalioti
    min_dist=0.9, #Svarbus
    metric='euclidean', #Galima keisti
    n_epochs=2000, #Galima keisti
    n_jobs=4
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
plt.axis('equal')
plt.show()

# %%
#UMAP medium



# %%
#UMAP low




# %%
#MDS high
from sklearn.manifold import MDS
mds_high = MDS(
	n_components=2,
    n_init=20,
    random_state=80085,
    init='classical_mds',
	metric='euclidean',
	max_iter=4000,
    n_jobs=-1)
mds_high_result = mds_high.fit_transform(X_high)
X_mds_df = pd.DataFrame(mds_high_result, columns=['MDS1', 'MDS2'])
X_mds_df['string'] = data_high['string'].values
plt.figure(figsize=(10, 6))
sns.scatterplot(data=X_mds_df, x='MDS1', y='MDS2', hue='string', palette='husl', s=75, hue_order=[f'string_{i}' for i in range(1, 11)])
plt.legend(title='Grandinės', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('MDS Projekcija')
plt.xlabel('MDS1')
plt.ylabel('MDS2')
plt.axis('equal')
plt.show()

# %%
#MDS medium
from sklearn.manifold import MDS
mds_medium = MDS(
	n_components=2,
    n_init=20,
    random_state=80085,
    init='classical_mds',
	metric='euclidean',
	max_iter=4000,
    n_jobs=-1)
mds_medium_result = mds_medium.fit_transform(X_medium)
X_mds_df = pd.DataFrame(mds_medium_result, columns=['MDS1', 'MDS2'])
X_mds_df['string'] = data_medium['string'].values
plt.figure(figsize=(10, 6))
sns.scatterplot(data=X_mds_df, x='MDS1', y='MDS2', hue='string', palette='husl', s=75, hue_order=[f'string_{i}' for i in range(1, 11)])
plt.legend(title='Grandinės', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('MDS Projekcija')
plt.xlabel('MDS1')
plt.ylabel('MDS2')
plt.axis('equal')
plt.show()

# %%
#MDS low
from sklearn.manifold import MDS
mds_low = MDS(
	n_components=2,
	n_init=20,
	random_state=80085,
	init='classical_mds',
	metric='euclidean',
	max_iter=4000,
	n_jobs=-1)
mds_low_result = mds_low.fit_transform(X_low)
X_mds_df = pd.DataFrame(mds_low_result, columns=['MDS1', 'MDS2'])
X_mds_df['string'] = data_low['string'].values
plt.figure(figsize=(10, 6))
sns.scatterplot(data=X_mds_df, x='MDS1', y='MDS2', hue='string', palette='husl', s=75, hue_order=[f'string_{i}' for i in range(1, 11)])
plt.legend(title='Grandinės', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('MDS Projekcija')
plt.xlabel('MDS1')
plt.ylabel('MDS2')
plt.axis('equal')
plt.show()


