# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, MDS
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
#%%

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
tsne_high_final = TSNE(
    n_components=2,
    random_state=80085,
    init='pca',
    perplexity=45,
    learning_rate=1000,
    early_exaggeration=12,
    metric='euclidean',
    max_iter=1000
)

tsne_high_final_result = tsne_high_final.fit_transform(X_high)

#%%
labels=data_high['string'].values
labels = np.array(labels)
unique_labels = sorted(set(labels), key=lambda x: int(x.split('_')[1]))

palette = sns.color_palette("husl", len(unique_labels))
color_map = dict(zip(unique_labels, palette))

df = pd.DataFrame({
    "Dim1": tsne_high_final_result[:, 0],
    "Dim2": tsne_high_final_result[:, 1],
    "string": labels
})

fig, ax = plt.subplots(figsize=(7, 6))

sns.scatterplot(
    data=df,
    x="Dim1",
    y="Dim2",
    hue="string",
    palette=color_map,
    hue_order=unique_labels,
    s=30,
    legend=False,
    ax=ax
)

ax.set_box_aspect(1)
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlabel("")
ax.set_ylabel("")

handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_map[name], markersize=6, linestyle='')
    for name in unique_labels
]
ax.legend(handles, unique_labels, title="Grandinės",
          loc="center left", bbox_to_anchor=(1.02, 0.5))

plt.suptitle("t-SNE projekcija (aukšta elektros energijos gamyba)")
plt.savefig(f"t-SNE projekcija (aukšta elektros energijos gamyba).png", dpi=300, bbox_inches='tight')
plt.show()
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
for learning_rate in learning_rate_values:
    tsne_medium = TSNE(
        n_components=2,
        random_state=80085,
        init='pca',
        perplexity=30,
        learning_rate=learning_rate,
        metric='euclidean',
        max_iter=1000
    )
    tsne_learning_rate_results[learning_rate] = tsne_medium.fit_transform(X_medium)

plot_tsne_panel(tsne_learning_rate_results, title = "t-SNE projekcija skirtingiems mokymosi greičiams (vidutinė elektros gamyba)", hyperparameter="Mokymosi greitis", labels=data_medium['string'].values, ncols=3, figsize=(15, 10))

#galima imti 1000, nes jau su 1500 atrodo, kad persimoko

#%%
for early_exaggeration in early_exaggeration_values:
    tsne_medium = TSNE(
        n_components=2,
        random_state=80085,
        init='pca',
        perplexity=30,
        learning_rate='auto',
        early_exaggeration=early_exaggeration,
        metric='euclidean',
        max_iter=1000
    )
    tsne_early_exaggeration_results[early_exaggeration] = tsne_medium.fit_transform(X_medium)

plot_tsne_panel(tsne_early_exaggeration_results, title = "t-SNE projekcija skirtingiems early exaggeration reikšmėms (vidutinė elektros gamyba)", hyperparameter="Early Exaggeration", labels=data_medium['string'].values, ncols=3, figsize=(15, 10))

#cia tas early exaggeration irgi nelabai skirias, tai galim imti 12 sakykime

#%%
for max_iter in max_iter_values:
    tsne_medium = TSNE(
        n_components=2,
        random_state=80085,
        init='pca',
        perplexity=30,
        learning_rate='auto',
        early_exaggeration=12,
        metric='euclidean',
        max_iter=max_iter
    )
    tsne_max_iter_results[max_iter] = tsne_medium.fit_transform(X_medium)

plot_tsne_panel(tsne_max_iter_results, title = "t-SNE projekcija skirtingiems iteracijų skaičiams (vidutinė elektros gamyba)", hyperparameter="Iteracijų skaičius", labels=data_medium['string'].values, ncols=3, figsize=(15, 10))

#%%
#galutiniai t-SNE medium, perplexity 30, learning_rate 1000, ee 12, max_iter 1000
tsne_medium_final = TSNE(
    n_components=2,
    random_state=80085,
    init='pca',
    perplexity=30,
    learning_rate=1000,
    early_exaggeration=12,
    metric='euclidean',
    max_iter=1000
)

tsne_medium_final_result = tsne_medium_final.fit_transform(X_medium)

#%%
labels=data_medium['string'].values
labels = np.array(labels)
unique_labels = sorted(set(labels), key=lambda x: int(x.split('_')[1]))

palette = sns.color_palette("husl", len(unique_labels))
color_map = dict(zip(unique_labels, palette))

df = pd.DataFrame({
    "Dim1": tsne_medium_final_result[:, 0],
    "Dim2": tsne_medium_final_result[:, 1],
    "string": labels
})

fig, ax = plt.subplots(figsize=(7, 6))

sns.scatterplot(
    data=df,
    x="Dim1",
    y="Dim2",
    hue="string",
    palette=color_map,
    hue_order=unique_labels,
    s=30,
    legend=False,
    ax=ax
)

ax.set_box_aspect(1)
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlabel("")
ax.set_ylabel("")

handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_map[name], markersize=6, linestyle='')
    for name in unique_labels
]
ax.legend(handles, unique_labels, title="Grandinės",
          loc="center left", bbox_to_anchor=(1.02, 0.5))

plt.suptitle("t-SNE projekcija (vidutinė elektros energijos gamyba)")
plt.savefig(f"t-SNE projekcija (vidutinė elektros energijos gamyba).png", dpi=300, bbox_inches='tight')
plt.show()

#%%
#tsne low
tsne_low_results = {}
for perplexity in perplexity_values:
    tsne_low = TSNE(
        n_components=2,
        random_state=80085,
        init='pca',
        perplexity=perplexity,
        learning_rate='auto',
        metric='euclidean',
        max_iter=1000
    )
    tsne_low_results[perplexity] = tsne_low.fit_transform(X_low)

plot_tsne_panel(tsne_low_results, title = "t-SNE projekcija skirtingiems perpleksiškumams (maža elektros gamyba)", hyperparameter="Perpleksiškumas", labels=data_low['string'].values, ncols=3, figsize=(15, 10))
#nelabai yra skirtumu, galime imti 30
#%%
for learning_rate in learning_rate_values:
    tsne_low = TSNE(
        n_components=2,
        random_state=80085,
        init='pca',
        perplexity=30,
        learning_rate=learning_rate,
        metric='euclidean',
        max_iter=1000
    )
    tsne_learning_rate_results[learning_rate] = tsne_low.fit_transform(X_low)

plot_tsne_panel(tsne_learning_rate_results, title = "t-SNE projekcija skirtingiems mokymosi greičiams (maža elektros gamyba)", hyperparameter="Mokymosi greitis", labels=data_low['string'].values, ncols=3, figsize=(15, 10))
#cia jau matom skirtumus, tai galim imti 500, nes 100 jau atrodo truputeli per didelis

#%%
for early_exaggeration in early_exaggeration_values:
    tsne_low = TSNE(
        n_components=2,
        random_state=80085,
        init='pca',
        perplexity=30,
        learning_rate=500,
        early_exaggeration=early_exaggeration,
        metric='euclidean',
        max_iter=1000
    )
    tsne_early_exaggeration_results[early_exaggeration] = tsne_low.fit_transform(X_low)

plot_tsne_panel(tsne_early_exaggeration_results, title = "t-SNE projekcija skirtingiems early exaggeration reikšmėms (maža elektros gamyba)", hyperparameter="Early Exaggeration", labels=data_low['string'].values, ncols=3, figsize=(15, 10))
#realiai nelabai matom skirtumus, tai galim imti 24

#%%
for max_iter in max_iter_values:
    tsne_low = TSNE(
        n_components=2,
        random_state=80085,
        init='pca',
        perplexity=30,
        learning_rate=500,
        early_exaggeration=24,
        metric='euclidean',
        max_iter=max_iter
    )
    tsne_max_iter_results[max_iter] = tsne_low.fit_transform(X_low)

plot_tsne_panel(tsne_max_iter_results, title = "t-SNE projekcija skirtingiems iteracijų skaičiams (maža elektros gamyba)", hyperparameter="Iteracijų skaičius", labels=data_low['string'].values, ncols=3, figsize=(15, 10))

#velgi nesvarbu, po 1000 iteraciju grafikai identiski

#%%
tsne_low_final = TSNE(
    n_components=2,
    random_state=80085,
    init='pca',
    perplexity=30,
    learning_rate=500,
    early_exaggeration=24,
    metric='euclidean',
    max_iter=1000
)
tsne_low_final_result = tsne_low_final.fit_transform(X_low)

labels=data_low['string'].values
labels = np.array(labels)
unique_labels = sorted(set(labels), key=lambda x: int(x.split('_')[1]))

palette = sns.color_palette("husl", len(unique_labels))
color_map = dict(zip(unique_labels, palette))

df = pd.DataFrame({
    "Dim1": tsne_low_final_result[:, 0],
    "Dim2": tsne_low_final_result[:, 1],
    "string": labels
})

fig, ax = plt.subplots(figsize=(7, 6))

sns.scatterplot(
    data=df,
    x="Dim1",
    y="Dim2",
    hue="string",
    palette=color_map,
    hue_order=unique_labels,
    s=30,
    legend=False,
    ax=ax
)

ax.set_box_aspect(1)
ax.set_aspect('equal', adjustable='datalim')
ax.set_xlabel("")
ax.set_ylabel("")

handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=color_map[name], markersize=6, linestyle='')
    for name in unique_labels
]
ax.legend(handles, unique_labels, title="Grandinės",
          loc="center left", bbox_to_anchor=(1.02, 0.5))

plt.suptitle("t-SNE projekcija (žema elektros energijos gamyba)")
plt.savefig(f"t-SNE projekcija (žema elektros energijos gamyba).png", dpi=300, bbox_inches='tight')
plt.show()

#%%
#UMAP high n_neighbors palyginimas
n_neighbors_values = [2, 10, 15, 30, 40, 50]
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

results = []
handles, labels = None, None

for i, nn in enumerate(n_neighbors_values):
    umap_model = umap.UMAP(
        n_components=2,
        random_state=80085,
        init='random',
        n_neighbors=nn,
        metric='euclidean',
    )

    umap_result = umap_model.fit_transform(X_high)

    X_umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
    X_umap_df['string'] = data_high['string'].values

    ax = axes[i]
    lims = [
        min(np.floor(X_umap_df['UMAP1'].min()), np.floor(X_umap_df['UMAP2'].min()))-1,
        max(np.ceil(X_umap_df['UMAP1'].max()), np.ceil(X_umap_df['UMAP2'].max()))+1
    ]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

    sns.scatterplot(
        data=X_umap_df,
        x='UMAP1',
        y='UMAP2',
        hue='string',
        palette='husl',
        s=75,
        hue_order=[f'string_{j}' for j in range(1, 11)],
        ax=ax,
        legend=(i == 0),
    )

    ax.set_title(f'UMAP Projekcija (n_neighbors={nn})')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')

    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend_.remove()

    results.append({
        'n_neighbors': nn
    })

fig.legend(handles, labels, title='Grandinės', loc='right')
fig.suptitle('UMAP Projekcija. Skirtingos n_neighbors reikšmės', fontsize=20)
plt.show()

pd.DataFrame(results)
#%%
#UMAP high min_dist palyginimas
min_dist_values = [0.0, 0.1, 0.25, 0.5, 0.8, 0.99]
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

results = []
handles, labels = None, None

for i, md in enumerate(min_dist_values):
    umap_model = umap.UMAP(
        n_components=2,
        random_state=80085,
        init='random',
        min_dist=md,
        metric='euclidean',
    )

    umap_result = umap_model.fit_transform(X_high)

    X_umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
    X_umap_df['string'] = data_high['string'].values

    ax = axes[i]
    lims = [
        min(np.floor(X_umap_df['UMAP1'].min()), np.floor(X_umap_df['UMAP2'].min()))-0.2,
        max(np.ceil(X_umap_df['UMAP1'].max()), np.ceil(X_umap_df['UMAP2'].max()))+0.2
    ]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')

    sns.scatterplot(
        data=X_umap_df,
        x='UMAP1',
        y='UMAP2',
        hue='string',
        palette='husl',
        s=75,
        hue_order=[f'string_{j}' for j in range(1, 11)],
        ax=ax,
        legend=(i == 0),
    )

    ax.set_title(f'UMAP Projekcija (min_dist={md})')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')

    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend_.remove()

    results.append({
        'min_dist': md
    })

fig.legend(handles, labels, title='Grandinės', loc='right')
fig.suptitle('UMAP Projekcija. Skirtingos min_dist reikšmės', fontsize=20)
plt.show()

pd.DataFrame(results)

#%%
umap_model = umap.UMAP(
    n_components=2,
    random_state=80085,
    init='random',
    n_neighbors=12,
    min_dist=0.8,
    metric='euclidean'
)

umap_result = umap_model.fit_transform(X_high)

X_umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
X_umap_df['string'] = data_high['string'].values

plt.figure(figsize=(10, 7))

lims = [
    min(np.floor(X_umap_df['UMAP1'].min()), np.floor(X_umap_df['UMAP2'].min()))-1,
    max(np.ceil(X_umap_df['UMAP1'].max()), np.ceil(X_umap_df['UMAP2'].max()))+1
]

plt.xlim(lims)
plt.ylim(lims)
plt.gca().set_aspect('equal', adjustable='box')

sns.scatterplot(
    data=X_umap_df,
    x='UMAP1',
    y='UMAP2',
    hue='string',
    palette='husl',
    s=75,
    hue_order=[f'string_{j}' for j in range(1, 11)]
)

plt.title('UMAP projekcija (n_neighbors=12, min_dist=0.8)')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')

plt.legend(title='Grandinės', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# %%
#UMAP medium
umap_model = umap.UMAP(
    n_components=2,
    random_state=80085,
    init='random',
    n_neighbors=30,
    min_dist=0.2,
    metric='euclidean'
)

umap_result = umap_model.fit_transform(X_medium)

X_umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2'])
X_umap_df['string'] = data_medium['string'].values

plt.figure(figsize=(10, 7))

lims = [
    min(np.floor(X_umap_df['UMAP1'].min()), np.floor(X_umap_df['UMAP2'].min()))-1,
    max(np.ceil(X_umap_df['UMAP1'].max()), np.ceil(X_umap_df['UMAP2'].max()))+1
]

plt.xlim(lims)
plt.ylim(lims)
plt.gca().set_aspect('equal', adjustable='box')

sns.scatterplot(
    data=X_umap_df,
    x='UMAP1',
    y='UMAP2',
    hue='string',
    palette='husl',
    s=75,
    hue_order=[f'string_{j}' for j in range(1, 11)]
)

plt.title('UMAP projekcija (n_neighbors=12, min_dist=0.8)')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')

plt.legend(title='Grandinės', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()




# %%
#UMAP low





# %%
niter_values = [50, 100, 250, 500, 750, 1000]
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

results = []
handles, labels = None, None

for i, niter in enumerate(niter_values):
    mds = MDS(
        n_components=2,
        n_init=1,
        random_state=80085,
        init='random',
        metric=True,
        dissimilarity='euclidean',
        max_iter=niter,
        n_jobs=-1,
        eps=1e-10,
        normalized_stress=True
    )

    mds_result = mds.fit_transform(X_high)

    X_mds_df = pd.DataFrame(mds_result, columns=['MDS1', 'MDS2'])
    X_mds_df['string'] = data_high['string'].values

    ax = axes[i]
    lims=[min(np.floor(X_mds_df['MDS1'].min()), np.floor(X_mds_df['MDS2'].min())),max(np.ceil(X_mds_df['MDS1'].max()), np.ceil(X_mds_df['MDS2'].max()))]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    sns.scatterplot(
        data=X_mds_df,
        x='MDS1',
        y='MDS2',
        hue='string',
        palette='husl',
        s=75,
        hue_order=[f'string_{j}' for j in range(1, 11)],
        ax=ax,
        legend=(i == 0),
        
    )

    ax.set_title(f'MDS Projekcija (max_iter={niter}). Stress: {mds.stress_:.4f}')
    ax.set_xlabel('MDS1')
    ax.set_ylabel('MDS2')

    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend_.remove()

    results.append({
        'max_iter': niter,
        'stress': mds.stress_,
        'used_iter': mds.n_iter_
    })

fig.legend(handles, labels, title='Grandinės',loc="right")
fig.suptitle('MDS Projekcija. max_iter skirtingos reikšmės, n_init=1, init=random', fontsize=20)
plt.show()

pd.DataFrame(results)

# %%


# %%
n_init_values = [1,2,4,8,16,20] 
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

results = []
handles, labels = None, None

for i, n_init in enumerate(n_init_values):
    mds = MDS(
        n_components=2,
        n_init=n_init,
        random_state=80085,
        init='random',
        metric=True,
        dissimilarity='euclidean',
        max_iter=100,
        n_jobs=-1,
        eps=1e-10,
        normalized_stress=True
    )

    mds_result = mds.fit_transform(X_high)

    X_mds_df = pd.DataFrame(mds_result, columns=['MDS1', 'MDS2'])
    X_mds_df['string'] = data_high['string'].values

    ax = axes[i]
    lims=[min(np.floor(X_mds_df['MDS1'].min()), np.floor(X_mds_df['MDS2'].min())),max(np.ceil(X_mds_df['MDS1'].max()), np.ceil(X_mds_df['MDS2'].max()))]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'MDS Projekcija (n_init={n_init}). Stress: {mds.stress_:.6f}')
    ax.set_xlabel('MDS1')
    ax.set_ylabel('MDS2')
    sns.scatterplot(
        data=X_mds_df,
        x='MDS1',
        y='MDS2',
        hue='string',
        palette='husl',
        s=75,
        hue_order=[f'string_{j}' for j in range(1, 11)],
        ax=ax,
        
        legend=(i == 0),
        
    )
	

    if i == 0:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend_.remove()

    results.append({
        'n_init': n_init,
        'stress': mds.stress_,
        'used_iter': mds.n_iter_
    })

fig.legend(handles, labels, title='Grandinės',loc="right")
#set title
fig.suptitle('MDS Projekcija. n_init skirtingos reikšmės, max_iter=100, init=random', fontsize=20)
plt.show()

pd.DataFrame(results)

# %%
#MDS high
from sklearn.manifold import MDS
mds_high = MDS(
	n_components=2,
    n_init=16,
    random_state=80085,
    init='classical_mds',
	metric='euclidean',
	max_iter=500,
    n_jobs=-1,
    normalized_stress=True)
mds_high_result = mds_high.fit_transform(X_high)
X_mds_df = pd.DataFrame(mds_high_result, columns=['MDS1', 'MDS2'])
X_mds_df['string'] = data_high['string'].values
plt.figure(figsize=(10, 8))
sns.scatterplot(data=X_mds_df, x='MDS1', y='MDS2', hue='string', palette='husl', s=75, hue_order=[f'string_{i}' for i in range(1, 11)])
ax = plt.gca()
lims = [
    min(-3, -3),
    max(3, 3)
]
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal', adjustable='box')
plt.legend(title='Grandinės', bbox_to_anchor=(1.05, 0.25), loc='lower left')
plt.title(f'MDS Projekcija. Stress: {mds_high.stress_:.4f}, max_iter=500, n_init=16, init=classical_mds')
plt.xlabel('MDS1')
plt.ylabel('MDS2')
plt.show()


# %%
#MDS medium
from sklearn.manifold import MDS
mds_medium = MDS(
	n_components=2,
    n_init=16,
    random_state=80085,
    init='classical_mds',
	metric='euclidean',
	max_iter=500,
    n_jobs=-1,
    normalized_stress=True)
mds_medium_result = mds_medium.fit_transform(X_medium)
X_mds_df = pd.DataFrame(mds_medium_result, columns=['MDS1', 'MDS2'])
X_mds_df['string'] = data_medium['string'].values
plt.figure(figsize=(10, 8))
sns.scatterplot(data=X_mds_df, x='MDS1', y='MDS2', hue='string', palette='husl', s=75, hue_order=[f'string_{i}' for i in range(1, 11)])
ax = plt.gca()
lims = [
    min(ax.get_xlim()[0], ax.get_ylim()[0]),
    max(ax.get_xlim()[1], ax.get_ylim()[1])
]
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal', adjustable='box')
plt.legend(title='Grandinės', bbox_to_anchor=(1.05, 0.25), loc='lower left')
plt.title(f'MDS Projekcija. Stress: {mds_medium.stress_:.4f}, max_iter=500, n_init=16, init=classical_mds')
plt.xlabel('MDS1')
plt.ylabel('MDS2')
plt.show()


# %%
#MDS low
from sklearn.manifold import MDS
mds_low = MDS(
	n_components=2,
	n_init=16,
	random_state=80085,
	init='classical_mds',
	metric='euclidean',
	max_iter=500,
	n_jobs=-1,
    normalized_stress=True)
mds_low_result = mds_low.fit_transform(X_low)
X_mds_df = pd.DataFrame(mds_low_result, columns=['MDS1', 'MDS2'])
X_mds_df['string'] = data_low['string'].values
plt.figure(figsize=(10,8))
sns.scatterplot(data=X_mds_df, x='MDS1', y='MDS2', hue='string', palette='husl', s=75, hue_order=[f'string_{i}' for i in range(1, 11)])
ax = plt.gca()
lims = [
    min(ax.get_xlim()[0], ax.get_ylim()[0]),
    max(ax.get_xlim()[1], ax.get_ylim()[1])
]
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect('equal', adjustable='box')
plt.legend(title='Grandinės', bbox_to_anchor=(1.05, 0.25), loc='lower left')
plt.title(f'MDS Projekcija. Stress: {mds_low.stress_:.4f}, max_iter=500, n_init=16, init=classical_mds')
plt.xlabel('MDS1')
plt.ylabel('MDS2')
plt.show()



# %%
import sklearn
from sklearn.manifold import MDS
print(sklearn.__version__)

# %%
