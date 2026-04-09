# %%
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, trustworthiness
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, pairwise_distances, silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors






# %%
data_raw = pd.read_csv('Elektrines_duomenys_2023-2024m.csv', sep=';', decimal=',')
data_selected_features= data_raw[["timestamp"] + [f"Total_active_power_INV-{i}" for i in range(1, 9)]]
inv_cols = [c for c in data_selected_features.columns if c != "timestamp"]
data_selected_features["timestamp"] = pd.to_datetime(data_selected_features["timestamp"])
data_selected_features






# %%
mask_all_na = data_selected_features[inv_cols].isna().all(axis=1)
all_empty=data_selected_features[mask_all_na]
all_empty["day"] = all_empty["timestamp"].dt.date
all_empty = all_empty[(all_empty["timestamp"].dt.hour >= 2) & (all_empty["timestamp"].dt.hour < 19)]
all_empty["count_per_day"] = all_empty.groupby("day")["timestamp"].transform("size")
all_empty






# %% [markdown]
#      Inverteris 2024 metais išsijungia random nuo 19:00 iki 02:00, todėl stebėjome keistumus, bet čia problemų yra ir kitų - skaityk duomenų kiekis atitinkantis 17 dienų yra tušti;

# %%
data_selected_features=data_selected_features[data_selected_features["timestamp"].dt.year == 2023]






# %% [markdown]
#   <span style="color: rgb(244, 12, 105);">  Daug geriau yra su praleistomis reikšmėmis -- čia yra tik viena diena kur visi inverteriai, jei imame tik 2023 metus, čia problema yra tik su 3 inverteriu, NA reikšmes čia užpildydami vidurkiu visai gerą aproksimacija gaunasi mano galva;

# %%
mask_all_na_2023 = data_selected_features[inv_cols].isna().all(axis=1)
all_empty_2023=data_selected_features[mask_all_na_2023]
all_empty_2023






# %%
data_raw = data_selected_features[~mask_all_na_2023]
data_raw






# %% [markdown]
#       Turime su 3 inverteriu daug praleistų reikšmelių (56 dienas) siūlau trinti, kol kas užpildau vidurkiu pagal eilutes

# %%
data_raw[inv_cols] = data_raw[inv_cols].apply(lambda row: row.fillna(row.mean()), axis=1)
data_raw






# %%
data_raw["Total_active_power"] = data_raw[[f"Total_active_power_INV-{i}" for i in range(1, 9)]].sum(axis=1)
data_summed= data_raw[["timestamp", "Total_active_power"]]
data_summed






# %%
data_summed.isna().sum()






# %%
data_summed["Day"] = data_summed["timestamp"].dt.date
data_summed["Hour"] = data_summed["timestamp"].dt.floor("h").dt.strftime("%H:%M")
data_summed=data_summed[(data_summed["Hour"] <= "19:00") & (data_summed["Hour"] >= "01:00")]
sum_of_inv = (
    data_summed.groupby(["Day", "Hour"], as_index=False)["Total_active_power"]
    .sum()
)
sum_of_inv_wide = sum_of_inv.pivot(index="Day", columns="Hour", values="Total_active_power")
sum_of_inv_wide.columns.name = None  
sum_of_inv_wide = sum_of_inv_wide.reset_index()

final_dataset = sum_of_inv_wide[["Day"] + sorted(sum_of_inv_wide.columns[1:])]
data_summed






# %%
final_dataset["month"] = pd.to_datetime(final_dataset["Day"]).dt.month






# %%
season_map = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn"
}

final_dataset["season"] = final_dataset["month"].map(season_map)






# %%
season_order = ["Winter", "Spring", "Summer", "Autumn"]
season_colors = ["#4C78A8", "#59A14F", "#F28E2B", "#9C755F"]

id_cols = ["Day", "month", "season"]
value_cols = [c for c in final_dataset.columns if c not in id_cols]

final_dataset_melted = final_dataset.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="time",
    value_name="power"
)

data_by_season = [
    final_dataset_melted.loc[final_dataset_melted["season"] == s, "power"].values
    for s in season_order
]
sezonai=["Žiema", "Pavasaris", "Vasara", "Ruduo"]

plt.figure(figsize=(10, 6))
bp = plt.boxplot(
    data_by_season,
    labels=sezonai,
    patch_artist=True,
    medianprops={"color": "black", "linewidth": 1.4},
    boxprops={"linewidth": 1.2},
    whiskerprops={"linewidth": 1.2},
    capprops={"linewidth": 1.2}
)

for box, color in zip(bp["boxes"], season_colors):
    box.set_facecolor(color)

plt.title("Galios pasiskirstymas pagal sezoną")
plt.xlabel("Sezonas")
plt.ylabel("Galia")
plt.tight_layout()
plt.show()






# %%
final_dataset_melted["time_dt"] = pd.to_datetime(final_dataset_melted["time"], format="%H:%M", errors="coerce")
line_df = (
    final_dataset_melted
    .groupby(["season", "time", "time_dt"], as_index=False)["power"]
    .sum()
    .sort_values("time_dt")
)
season_labels_lt = {
    "Winter": "Žiema",
    "Spring": "Pavasaris",
    "Summer": "Vasara",
    "Autumn": "Ruduo",
}
season_order = ["Winter", "Spring", "Summer", "Autumn"]
plt.figure(figsize=(12, 6))
for s in season_order:
    part = line_df[line_df["season"] == s]
    plt.plot(
        part["time"],
        part["power"],
        label=season_labels_lt.get(s, s),
        linewidth=1.8
    )
plt.title("Galios generavimas pagal laiką ir sezoną")
plt.xlabel("Laikas")
plt.ylabel("Galia")
plt.xticks(ticks=range(0, len(part["time"])), rotation=45, ha="right") 
plt.legend()
plt.tight_layout()
plt.show()









# %% [markdown]
#       Patriminau laiką;

# %%
print(final_dataset.head())
print("#" * 50)
print(final_dataset_melted.head())






# %%
id_cols = ["Day", "month", "season"]
value_cols = [c for c in final_dataset.columns if c not in id_cols]
scaler = RobustScaler()
final_dataset_scaled = final_dataset.copy()
final_dataset_scaled[value_cols] = scaler.fit_transform(final_dataset[value_cols])
final_dataset_melted_scaled = final_dataset_scaled.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="time",
    value_name="power"
)






# %%
final_dataset_scaled






# %%
final_dataset_melted_scaled






# %%
data_by_season = [
    final_dataset_melted_scaled.loc[final_dataset_melted_scaled["season"] == s, "power"].values
    for s in season_order
]
sezonai=["Žiema", "Pavasaris", "Vasara", "Ruduo"]

plt.figure(figsize=(10, 6))
bp = plt.boxplot(
    data_by_season,
    labels=sezonai,
    patch_artist=True,
    medianprops={"color": "black", "linewidth": 1.4},
    boxprops={"linewidth": 1.2},
    whiskerprops={"linewidth": 1.2},
    capprops={"linewidth": 1.2}
)

for box, color in zip(bp["boxes"], season_colors):
    box.set_facecolor(color)

plt.title("Galios pasiskirstymas pagal sezoną - normuota")
plt.xlabel("Sezonas")
plt.ylabel("Galia")
plt.tight_layout()
plt.show()






# %%
final_dataset_melted_scaled["time_dt"] = pd.to_datetime(final_dataset_melted_scaled["time"], format="%H:%M", errors="coerce")
line_df = (
    final_dataset_melted_scaled	
    .groupby(["season", "time", "time_dt"], as_index=False)["power"]
    .sum()
    .sort_values("time_dt")
)
season_labels_lt = {
    "Winter": "Žiema",
    "Spring": "Pavasaris",
    "Summer": "Vasara",
    "Autumn": "Ruduo",
}
season_order = ["Winter", "Spring", "Summer", "Autumn"]
plt.figure(figsize=(12, 6))
for s in season_order:
    part = line_df[line_df["season"] == s]
    plt.plot(
        part["time"],
        part["power"],
        label=season_labels_lt.get(s, s),
        linewidth=1.8
    )
plt.title("Galios generavimas pagal laiką ir sezoną - normuota")
plt.xlabel("Laikas")
plt.ylabel("Galia")
plt.xticks(ticks=range(0, len(part["time"])), rotation=45, ha="right") 
plt.legend()
plt.tight_layout()
plt.show()









# %%
print(final_dataset_melted[["power","season"]].groupby("season").describe())







# %%
print(final_dataset_melted_scaled[["power","season"]].groupby("season").describe())







# %%

heatmap_by_season = (
    final_dataset_melted.groupby(["season", "time"], as_index=False)["power"]
    .sum()
    .pivot(index="season", columns="time", values="power")
    .reindex(season_order)
)

season_labels_lt = {
    "Winter": "Ziema",
    "Spring": "Pavasaris",
    "Summer": "Vasara",
    "Autumn": "Ruduo"
}
heatmap_by_season.index = [season_labels_lt.get(season, season) for season in heatmap_by_season.index]

plt.figure(figsize=(18, 6))
sns.heatmap(heatmap_by_season, cmap="YlOrRd", cbar_kws={"label": "Energijos kiekis"})
ax = plt.gca()

xtick_positions = range(0, len(heatmap_by_season.columns))
ax.set_xticks(xtick_positions)
ax.set_xticklabels(
    [heatmap_by_season.columns[i] for i in xtick_positions],
    rotation=45,
    ha="right"
)

plt.title("Šilumos energijos kiekis per dieną pagal sezoną")
plt.xlabel("Laikas")
plt.ylabel("Metų laikas")
plt.tight_layout()
plt.show()







# %%

heatmap_by_season_scaled = (
    final_dataset_melted_scaled.groupby(["season", "time"], as_index=False)["power"]
    .sum()
    .pivot(index="season", columns="time", values="power")
    .reindex(season_order)
)

season_labels_lt = {
    "Winter": "Ziema",
    "Spring": "Pavasaris",
    "Summer": "Vasara",
    "Autumn": "Ruduo"
}
heatmap_by_season_scaled.index = [season_labels_lt.get(season, season) for season in heatmap_by_season_scaled.index]

plt.figure(figsize=(18, 6))
sns.heatmap(heatmap_by_season_scaled, cmap="YlOrRd", cbar_kws={"label": "Energijos kiekis"})
ax = plt.gca()

xtick_positions = range(0, len(heatmap_by_season_scaled.columns))
ax.set_xticks(xtick_positions)
ax.set_xticklabels(
    [heatmap_by_season_scaled.columns[i] for i in xtick_positions],
    rotation=45,
    ha="right"
)

plt.title("Šilumos energijos kiekis per dieną pagal sezoną - normuota")
plt.xlabel("Laikas")
plt.ylabel("Metų laikas")
plt.tight_layout()
plt.show()







# %%
final_dataset
id_cols





# %%
X = final_dataset_scaled.drop(columns=id_cols).select_dtypes(include='number')






# %% [markdown]
#     ### PCA

# %%
pca_model = PCA(n_components=2, random_state=80085)
pca_result = pca_model.fit_transform(X)
X_pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
X_pca_df['season'] = final_dataset_scaled['season'].values


plt.figure(figsize=(10, 7))
lims = [
    min(np.floor(X_pca_df['PC1'].min()), np.floor(X_pca_df['PC2'].min())) - 1,
    max(np.ceil(X_pca_df['PC1'].max()), np.ceil(X_pca_df['PC2'].max())) + 1
]

plt.xlim(lims)
plt.ylim(lims)
plt.gca().set_aspect('equal', adjustable='box')

sns.scatterplot(
    data=X_pca_df,
    x='PC1',
    y='PC2',
    hue='season',
    palette='husl',
    s=75,
    hue_order=['Winter', 'Spring', 'Summer', 'Autumn']
)

plt.title('PCA projekcija dienos energijos profiliams')
plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.2f}%)')
plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print('Explained variance ratio:', np.round(pca_model.explained_variance_ratio_, 4))








# %%
def normalized_stress(X, X_emb):
	D_orig = pairwise_distances(X)
	D_emb = pairwise_distances(X_emb)
	return np.sum((D_orig - D_emb) ** 2) / np.sum(D_orig ** 2)





# %%
X_emb_pca = X_pca_df[['PC1', 'PC2']].values
t = trustworthiness(X.values, X_emb_pca, n_neighbors=10)
c = trustworthiness(X_emb_pca, X.values, n_neighbors=10)

D_orig = pairwise_distances(X.values)
D_emb = pairwise_distances(X_emb_pca)
stress = normalized_stress(X.values, X_emb_pca)

print(f"Trustworthiness: {t:.4f}")
print(f"Continuity: {c:.4f}")
print(f"Stress: {stress:.4f}")






# %% [markdown]
#      ### T-SNE
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# %%
def tsne_grid_search(X, n=2, param_grid=None, n_neighbors=10, random_state=42):
    grid = ParameterGrid(param_grid)
    results = []
    for params in grid:
        tsne = TSNE(n_components=n, random_state=random_state, max_iter=1000, **params)
        try:
            X_emb_pca = tsne.fit_transform(X)
            t = trustworthiness(X, X_emb_pca, n_neighbors=n_neighbors)
            c = trustworthiness(X_emb_pca, X, n_neighbors=n_neighbors)
            stress = normalized_stress(X.values, X_emb_pca)
            results.append({
                'params': params,
                'trustworthiness': t,
                'continuity': c,
                'stress': stress,
            })
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue
    return results
param_grid = {'perplexity': [5, 15, 20, 30, 50], 'learning_rate': [10, 20, 30, 50, 100]}
results = tsne_grid_search(X, n=2, param_grid=param_grid)
for res in results:
    print(res)










# %%
tsne = TSNE(n_components=2, random_state=42, perplexity=10, max_iter=1000)
tsne_result = tsne.fit_transform(X)
tsne_df = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'])
tsne_df['season'] = final_dataset_scaled['season'].values






# %%
plt.figure(figsize=(10, 7))
lims = [
    min(np.floor(tsne_df['TSNE1'].min()), np.floor(tsne_df['TSNE2'].min())) - 1,
    max(np.ceil(tsne_df['TSNE1'].max()), np.ceil(tsne_df['TSNE2'].max())) + 1
]

plt.xlim(lims)
plt.ylim(lims)
plt.gca().set_aspect('equal', adjustable='box')

sns.scatterplot(
    data=tsne_df,
    x='TSNE1',
    y='TSNE2',
    hue='season',
    palette='husl',
    s=75,
    hue_order=['Winter', 'Spring', 'Summer', 'Autumn']
)

plt.title('t-SNE projekcija dienos energijos profiliams')
plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()






# %%

X_emb_tsne = tsne_df[['TSNE1', 'TSNE2']].values

t = trustworthiness(X, X_emb_tsne, n_neighbors=10)
c = trustworthiness(X_emb_tsne, X, n_neighbors=10)

D_orig = pairwise_distances(X)
D_emb = pairwise_distances(X_emb_tsne)
stress = normalized_stress(X.values, X_emb_tsne)

print(f"Trustworthiness: {t:.4f}")
print(f"Continuity: {c:.4f}")
print(f"Stress: {stress:.4f}")








# %% [markdown]
#     ## MDS

# %%

def mds_grid_search(X, param_grid, n_neighbors=10, random_state=42):
    grid = ParameterGrid(param_grid)
    results = []
    for params in grid:
        mds = MDS(n_components=2, normalized_stress=True, n_jobs=-1, n_init=10, **params)
        try:
            X_emb_MDS = mds.fit_transform(X)
            t = trustworthiness(X, X_emb_MDS, n_neighbors=n_neighbors)
            c = trustworthiness(X_emb_MDS, X, n_neighbors=n_neighbors)
            stress = mds.stress_
            results.append({
                'params': params,
                'trustworthiness': t,
                'continuity': c,
                'stress': stress
            })
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue

    return results

param_grid = {'max_iter': [150, 200, 300, 500, 1000]}
results = mds_grid_search(X, param_grid)
for res in results:
    print(res)





# %%
mds = MDS(n_components=2, max_iter=1000, normalized_stress=True)
mds_result = mds.fit_transform(X)
mds_df = pd.DataFrame(mds_result, columns=['MDS1', 'MDS2'])
mds_df['season'] = final_dataset_scaled['season'].values





# %%

plt.figure(figsize=(10, 7))
lims = [
    min(np.floor(mds_df['MDS1'].min()), np.floor(mds_df['MDS2'].min())) - 1,
    max(np.ceil(mds_df['MDS1'].max()), np.ceil(mds_df['MDS2'].max())) + 1
]

plt.xlim(lims)
plt.ylim(lims)
plt.gca().set_aspect('equal', adjustable='box')

sns.scatterplot(
    data=mds_df,
    x='MDS1',
    y='MDS2',
    hue='season',
    palette='husl',
    s=75,
    hue_order=['Winter', 'Spring', 'Summer', 'Autumn']
)

plt.title(f'MDS projekcija dienos energijos profiliams. Stress: {mds.stress_:.4f}')
plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()








# %%
X_emb_MDS = mds_df[['MDS1', 'MDS2']].values

t = trustworthiness(X, X_emb_MDS, n_neighbors=10)
c = trustworthiness(X_emb_MDS, X, n_neighbors=10)

print(f"Trustworthiness: {t:.4f}")
print(f"Continuity: {c:.4f}")
print(f"Stress: {mds.stress_:.4f}")








# %% [markdown]
#     # Klasterizavimas

# %%
def draw_clusters(X_emb, labels, title):
	plt.figure(figsize=(10, 7))
	lims = [
		min(np.floor(X_emb[:, 0].min()), np.floor(X_emb[:, 1].min())) - 1,
		max(np.ceil(X_emb[:, 0].max()), np.ceil(X_emb[:, 1].max())) + 1
	]

	plt.xlim(lims)
	plt.ylim(lims)
	plt.gca().set_aspect('equal', adjustable='box')

	sns.scatterplot(
		x=X_emb[:, 0],
		y=X_emb[:, 1],
		hue=labels,
		palette='tab10',
		s=75,
		legend='full'
	)

	plt.title(title)
	plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.tight_layout()
	plt.show()





# %%
inertias = []
k_values = range(1, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=80085, n_init="auto")
    km.fit(X_emb_pca)
    inertias.append(km.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Klusterių skaičius (k)")
plt.ylabel("Inercija")
plt.title("Alkūnės metodas")
plt.grid(True)
plt.show()






# %%
K_means_model = KMeans(n_clusters=3, random_state=80085, n_init="auto")
clusters = K_means_model.fit_predict(X_emb_pca)





# %%
score = silhouette_score(X_emb_pca, clusters)
print(f"Silhouette Score for k=3: {score:.4f}")




# %%
emb_kmean = X_emb_pca.copy()
emb_kmean = pd.DataFrame(emb_kmean, columns=['PC1', 'PC2'])
emb_kmean["k3_cluster"] = clusters





# %%
draw_clusters(emb_kmean[['PC1', 'PC2']].values, emb_kmean['k3_cluster'], "K-means klasterizacija PCA erdvėje")





# %%
K_means_model_2 = KMeans(n_clusters=4, random_state=80085, n_init="auto")
clusters_2 = K_means_model_2.fit_predict(X_emb_pca)





# %%
score_2 = silhouette_score(X_emb_pca, clusters_2)
print(f"Silhouette Score for k=4: {score_2:.4f}")





# %%
emb_kmean["k4_cluster"] = clusters_2





# %%
draw_clusters(emb_kmean[['PC1', 'PC2']].values, emb_kmean['k4_cluster'], "K-means klasterizacija PCA erdvėje")





# %%
K_means_model_3 = KMeans(n_clusters=2, random_state=80085, n_init="auto")
clusters_3 = K_means_model_3.fit_predict(X_emb_pca)





# %%
score_3 = silhouette_score(X_emb_pca, clusters_3)
print(f"Silhouette Score for k=2: {score_3:.4f}")





# %%
emb_kmean["k2_cluster"] = clusters_3





# %%
draw_clusters(emb_kmean[['PC1', 'PC2']].values, emb_kmean['k2_cluster'], "K-means klasterizacija PCA erdvėje")





# %%
results=final_dataset[["Day", "season"]].copy()
results["cluster"] = clusters
results






# %%
# # Kryžminė lentelė: cluster x season
# if 'results' in globals() and {'cluster', 'season'}.issubset(results.columns):
#     base_df = results[['cluster', 'season']].copy()
# elif {'cluster', 'season'}.issubset(final_dataset_scaled.columns):
#     base_df = final_dataset_scaled[['cluster', 'season']].copy()
# else:
#     raise ValueError("Nerandu stulpelių 'cluster' ir 'season'. Pirma paleisk klasterizacijos celes.")

# # Kiekiai
# ct_cluster_season = pd.crosstab(base_df['cluster'], base_df['season'], margins=True)
# print('Kryžminė lentelė (kiekiai): cluster x season')

# # Eilučių procentai (kiekvieno klasterio sezoniškumo pasiskirstymas)
# ct_cluster_season_row_pct = pd.crosstab(
#     base_df['cluster'],
#     base_df['season'],
#     normalize='index'
# ).round(4) * 100
# print('Kryžminė lentelė (eilutės %, cluster -> season):')

# # Stulpelių procentai (kiekvieno sezono klasterių pasiskirstymas)
# ct_cluster_season_col_pct = pd.crosstab(
#     base_df['cluster'],
#     base_df['season'],
#     normalize='columns'
# ).round(4) * 100
# print('Kryžminė lentelė (stulpelai %, season -> cluster):')






# %%
import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
)


def stratified_sample_indices(strata_labels, sample_fraction, rng, min_total_size=None):
    strata_values = np.asarray(strata_labels)
    n_samples = len(strata_values)

    target_size = int(round(sample_fraction * n_samples))
    if min_total_size is not None:
        target_size = max(target_size, int(min_total_size))
    target_size = min(target_size, n_samples)

    sampled_parts = []
    for stratum in np.unique(strata_values):
        stratum_idx = np.where(strata_values == stratum)[0]
        n_take = int(round(sample_fraction * len(stratum_idx)))
        n_take = max(1, n_take)
        n_take = min(n_take, len(stratum_idx))
        sampled_parts.append(rng.choice(stratum_idx, size=n_take, replace=False))

    idx = np.sort(np.concatenate(sampled_parts))

    if len(idx) < target_size:
        remaining = np.setdiff1d(np.arange(n_samples), idx, assume_unique=False)
        add_n = min(target_size - len(idx), len(remaining))
        if add_n > 0:
            idx = np.sort(
                np.concatenate([idx, rng.choice(remaining, size=add_n, replace=False)])
            )

    if len(idx) > target_size:
        idx = np.sort(rng.choice(idx, size=target_size, replace=False))

    return idx


def stratified_sample_indices(strata_labels, sample_fraction, rng, min_total_size=None):
    strata_values = np.asarray(strata_labels)
    n_samples = len(strata_values)

    target_size = int(round(sample_fraction * n_samples))
    if min_total_size is not None:
        target_size = max(target_size, int(min_total_size))
    target_size = min(target_size, n_samples)

    sampled_parts = []
    for stratum in np.unique(strata_values):
        stratum_idx = np.where(strata_values == stratum)[0]
        n_take = int(round(sample_fraction * len(stratum_idx)))
        n_take = max(1, n_take)
        n_take = min(n_take, len(stratum_idx))
        sampled_parts.append(rng.choice(stratum_idx, size=n_take, replace=False))

    idx = np.sort(np.concatenate(sampled_parts))

    if len(idx) < target_size:
        remaining = np.setdiff1d(np.arange(n_samples), idx, assume_unique=False)
        add_n = min(target_size - len(idx), len(remaining))
        if add_n > 0:
            idx = np.sort(np.concatenate([idx, rng.choice(remaining, size=add_n, replace=False)]))

    if len(idx) > target_size:
        idx = np.sort(rng.choice(idx, size=target_size, replace=False))

    return idx


def run_clustering_stability(
    X_data,
    strata_labels,
    method="kmeans",
    param_values=None,
    n_runs=30,
    sample_fraction=0.8,
    base_seed=80085,
    linkage_method="ward",
    min_samples=5,
    max_noise_fraction=0.5,
):
    X_np = np.asarray(X_data)
    n_samples = X_np.shape[0]
    strata_values = np.asarray(strata_labels)
    row_ids_all = np.arange(n_samples)

    if param_values is None:
        param_values = np.round(np.arange(0.1, 3.05, 0.1), 2) if method == "dbscan" else range(2, 7)

    param_key = "eps" if method == "dbscan" else "k"
    all_assignments, all_run_metrics, all_ari_pairs, summary_rows = [], [], [], []

    for param in param_values:
        run_label_maps, run_row_sets = [], []

        for run in range(1, n_runs + 1):
            run_seed = base_seed + int(param * 1000) + run
            rng = np.random.default_rng(run_seed)

            idx = stratified_sample_indices(strata_values, sample_fraction, rng, min_total_size=int(param) + 1)
            X_run, row_ids_run = X_np[idx], row_ids_all[idx]

            if method == "kmeans":
                labels = KMeans(n_clusters=int(param), random_state=run_seed, n_init="auto").fit_predict(X_run)
            elif method == "hierarchical":
                labels = AgglomerativeClustering(n_clusters=int(param), linkage=linkage_method).fit_predict(X_run)
            elif method == "dbscan":
                labels = DBSCAN(eps=param, min_samples=min_samples).fit_predict(X_run)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise    = (labels == -1).sum()
            mask       = labels != -1

            valid = n_clusters >= 2 and (method != "dbscan" or n_noise < len(labels) * max_noise_fraction)
            X_score, l_score = (X_run[mask], labels[mask]) if method == "dbscan" else (X_run, labels)

            try:    sil = silhouette_score(X_score, l_score) if valid else np.nan
            except: sil = np.nan
            try:    db  = davies_bouldin_score(X_score, l_score) if valid else np.nan
            except: db  = np.nan

            all_assignments.append(pd.DataFrame({param_key: param, "seed": run_seed, "row_id": row_ids_run, "cluster": labels}))
            all_run_metrics.append({param_key: param, "seed": run_seed, "silhouette": sil, "davies_bouldin": db, "n_clusters": n_clusters, "n_noise": n_noise})
            run_label_maps.append(dict(zip(row_ids_run, labels)))
            run_row_sets.append(set(row_ids_run))

        k_ari_values = []
        for a, b in combinations(range(n_runs), 2):
            common = sorted(run_row_sets[a] & run_row_sets[b])
            if len(common) < 2:
                continue
            ari = adjusted_rand_score([run_label_maps[a][r] for r in common],
                                      [run_label_maps[b][r] for r in common])
            k_ari_values.append(ari)
            all_ari_pairs.append({param_key: param, "run_a": a+1, "run_b": b+1, "common_rows": len(common), "ari": ari})

        m = pd.DataFrame([r for r in all_run_metrics if r[param_key] == param])
        ari_arr = np.asarray(k_ari_values, dtype=float)
        summary_rows.append({
            param_key:             param,
            "mean_silhouette":     np.nanmean(m["silhouette"]),
            "std_silhouette":      np.nanstd(m["silhouette"], ddof=1),
            "mean_davies_bouldin": np.nanmean(m["davies_bouldin"]),
            "std_davies_bouldin":  np.nanstd(m["davies_bouldin"], ddof=1),
            "mean_ari":            np.nanmean(ari_arr) if len(ari_arr) else np.nan,
            "std_ari":             np.nanstd(ari_arr, ddof=1) if len(ari_arr) > 1 else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows)
    ranking = summary_df.dropna(subset=["mean_ari", "mean_silhouette"]).sort_values(
        by=["mean_ari", "mean_silhouette", "mean_davies_bouldin", "std_ari", "std_silhouette", param_key],
        ascending=[False, False, True, True, True, True],
    ).reset_index(drop=True)

    best = ranking[param_key].iloc[0] if not ranking.empty else None

    return {
        "assignments": pd.concat(all_assignments, ignore_index=True),
        "run_metrics": pd.DataFrame(all_run_metrics),
        "ari_pairs":   pd.DataFrame(all_ari_pairs),
        "summary":     summary_df,
        "ranking":     ranking,
        f"best_{param_key}": best,
    }



# %%
stability_results = run_clustering_stability(
    X_data=X_emb_pca,
    method="kmeans",
    strata_labels=final_dataset["season"],
    param_values=range(2, 7),
    n_runs=50,
    sample_fraction=0.8,
    base_seed=80085
)

cluster_assignments_runs = stability_results["assignments"]
k_run_metrics = stability_results["run_metrics"]
ari_pairs = stability_results["ari_pairs"]
stability_summary = stability_results["summary"]
best_k = stability_results["best_k"]

print("Stabilumo suvestinė pagal k:")
stability_summary


# %%
# Vizualus k palyginimas: vidurkiai ir sklaida (std) su pažymėtu rekomenduotu best_k.
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

axes[0].errorbar(
    stability_summary["k"],
    stability_summary["mean_silhouette"],
    yerr=stability_summary["std_silhouette"],
    fmt="o-",
    capsize=4,
)
axes[0].axvline(best_k, color="red", linestyle="--", alpha=0.7)
axes[0].set_title("Silhouette")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Reikšmė")
axes[0].grid(True, alpha=0.25)

axes[1].errorbar(
    stability_summary["k"],
    stability_summary["mean_davies_bouldin"],
    yerr=stability_summary["std_davies_bouldin"],
    fmt="o-",
    capsize=4,
)
axes[1].axvline(best_k, color="red", linestyle="--", alpha=0.7)
axes[1].set_title("Davies-Bouldin")
axes[1].set_xlabel("k")
axes[1].grid(True, alpha=0.25)

axes[2].errorbar(
    stability_summary["k"],
    stability_summary["mean_ari"],
    yerr=stability_summary["std_ari"],
    fmt="o-",
    capsize=4,
)
axes[2].axvline(best_k, color="red", linestyle="--", alpha=0.7)
axes[2].set_title("ARI")
axes[2].set_xlabel("k")
axes[2].grid(True, alpha=0.25)

plt.suptitle(f"KMeans stabilumo metrikos pagal k")
plt.tight_layout()
plt.show()










# %% [markdown]
#    <h1> HIERARCHINIS </h1>

# %%
Z = linkage(X_emb_pca, method='ward')
last = Z[-10:, 2]          
acceleration = np.diff(last, 2)  
k = acceleration[::-1].argmax() + 2 

print(f"Suggested k: {k}")
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("")
plt.xticks([])
plt.show()




# %%
#Applying the model here now
hierarchical_model = AgglomerativeClustering(n_clusters=2, linkage='ward')
hierarchical_clusters = hierarchical_model.fit_predict(X_emb_pca)



# %%
hierarchical_results = final_dataset[["Day", "season"]].copy()
hierarchical_results["hierarchical_cluster"] = hierarchical_clusters
hierarchical_score_silhouette = silhouette_score(X_emb_pca, hierarchical_clusters)
hierarchical_score_davies_bouldin = davies_bouldin_score(X_emb_pca, hierarchical_clusters)
print(f"Silhouette Score Hierarchical Model for k=2: {hierarchical_score_silhouette:.4f}")
print(f"Davies-Bouldin Score Hierarchical Model for k=2: {hierarchical_score_davies_bouldin:.4f}")




# %%
# # Kryžminė lentelė: cluster x season
# if 'hierarchical_results' in globals() and {'hierarchical_cluster', 'season'}.issubset(hierarchical_results.columns):
#     base_df = hierarchical_results[['hierarchical_cluster', 'season']].copy()
# elif {'hierarchical_cluster', 'season'}.issubset(final_dataset_scaled.columns):
#     base_df = final_dataset_scaled[['hierarchical_cluster', 'season']].copy()
# else:
#     raise ValueError("Nerandu stulpelių 'hierarchical_cluster' ir 'season'. Pirma paleisk klasterizacijos celes.")

# # Kiekiai
# ct_cluster_season = pd.crosstab(base_df['hierarchical_cluster'], base_df['season'], margins=True)
# print('Kryžminė lentelė (kiekiai): hierarchical_cluster x season')
# print(ct_cluster_season)

# # Eilučių procentai
# ct_cluster_season_row_pct = pd.crosstab(
#     base_df['hierarchical_cluster'],
#     base_df['season'],
#     normalize='index'
# ).round(4) * 100
# print('Kryžminė lentelė (eilutės %, cluster -> season):')
# print(ct_cluster_season_row_pct)

# # Stulpelių procentai
# ct_cluster_season_col_pct = pd.crosstab(
#     base_df['hierarchical_cluster'],
#     base_df['season'],
#     normalize='columns'
# ).round(4) * 100
# print('Kryžminė lentelė (stulpelai %, season -> cluster):')
# print(ct_cluster_season_col_pct)




# %%
plt.figure(figsize=(8, 8))
plt.scatter(X_emb_pca[:, 0], X_emb_pca[:, 1], c=hierarchical_clusters, cmap='Set1', s=50, alpha=0.7)
plt.title("Hierarchinio klasterizavimo rezultatai (k=2)")

# Same interval on both axes
lim_min = min(X_emb_pca[:, 0].min(), X_emb_pca[:, 1].min()) - 1
lim_max = max(X_emb_pca[:, 0].max(), X_emb_pca[:, 1].max()) + 1
plt.xlim(lim_min, lim_max)
plt.ylim(lim_min, lim_max)

plt.show()




# %% [markdown]
#    <h1> DBSCAN </h1>

# %%
for eps in np.arange(0.1, 5.1, 0.1):
    db = DBSCAN(eps=eps, min_samples=5).fit(X_emb_pca)
    labels = db.labels_

    mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    if n_clusters > 1 and mask.sum() > 1:
        sil = silhouette_score(X_emb_pca[mask], labels[mask])
        dbi = davies_bouldin_score(X_emb_pca[mask], labels[mask])
    else:
        sil, dbi = np.nan, np.nan

    print(f"eps={eps:.1f} → clusters={n_clusters}, noise={n_noise}, sil={sil:.3f}, db={dbi:.3f}")


# %% [markdown]
#  Geriausia silueta gavome:eps=2.1 → clusters=4, noise=52, sil=0.924, db=0.293
# 
# 

# %%
dbscan_model = DBSCAN(eps=2.1, min_samples=5)
dbscan_clusters = dbscan_model.fit_predict(X_emb_pca)


# %%
dbscan_results = final_dataset[["Day", "season"]].copy()
dbscan_results["dbscan_cluster"] = dbscan_clusters

mask = dbscan_clusters != -1

if len(set(dbscan_clusters[mask])) > 1:
    silhouette = silhouette_score(X_emb_pca[mask], dbscan_clusters[mask])
    db_index = davies_bouldin_score(X_emb_pca[mask], dbscan_clusters[mask])
else:
    silhouette = np.nan
    db_index = np.nan

print(f"Silhouette Score DBSCAN: {silhouette:.4f}")
print(f"Davies-Bouldin Score DBSCAN: {db_index:.4f}")
print(f"Number of clusters: {len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)}")
print(f"Noise points: {(dbscan_clusters == -1).sum()}")


# %%
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 8))

plt.scatter(
    X_emb_pca[:, 0],
    X_emb_pca[:, 1],
    c=dbscan_clusters,
    cmap='Set1',
    s=50,
    alpha=0.7
)

plt.title("DBSCAN klasterizavimo rezultatai (eps=2.1)")

lim_min = min(X_emb_pca[:, 0].min(), X_emb_pca[:, 1].min()) - 1
lim_max = max(X_emb_pca[:, 0].max(), X_emb_pca[:, 1].max()) + 1
plt.xlim(lim_min, lim_max)
plt.ylim(lim_min, lim_max)

plt.show() #cia noise yra pilkai pavaizduoti


# %% [markdown]
#  <h1> Dimensijos mažinimas iki 8 dimensijų </h1>

# %%
pca_model = PCA(n_components=8, random_state=80085)
pca_result = pca_model.fit_transform(X)
t = trustworthiness(X.values, pca_result, n_neighbors=8)
c = trustworthiness(pca_result, X.values, n_neighbors=8)

D_orig = pairwise_distances(X.values)
D_emb = pairwise_distances(X_emb_pca)
stress = normalized_stress(X.values, X_emb_pca)

print(f"Trustworthiness: {t:.4f}")
print(f"Continuity: {c:.4f}")
print(f"Stress: {stress:.4f}") # cia gauname labai geri rezultatai, todel kitu algoritmu netikriname, darome klasterizavima su PCA rezultatais


# %%
plt.figure(figsize=(10, 7))
lims = [
    min(np.floor(X_pca_df['PC1'].min()), np.floor(X_pca_df['PC2'].min())) - 1,
    max(np.ceil(X_pca_df['PC1'].max()), np.ceil(X_pca_df['PC2'].max())) + 1
]

plt.xlim(lims)
plt.ylim(lims)
plt.gca().set_aspect('equal', adjustable='box')

sns.scatterplot(
    data=X_pca_df,
    x='PC1',
    y='PC2',
    hue='season',
    palette='husl',
    s=75,
    hue_order=['Winter', 'Spring', 'Summer', 'Autumn']
)

plt.title('PCA projekcija dienos energijos profiliams')
plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.2f}%)')
plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print('Explained variance ratio:', np.round(pca_model.explained_variance_ratio_, 4))


# %%
X_pca_df_8 = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8'])
X_pca_df_8['season'] = final_dataset_scaled['season'].values

X_emb_pca_8 = X_pca_df_8[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8']].values



# %% [markdown]
#  <h2> K-means </h2>

# %%
inertias = []
k_values = range(1, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=80085, n_init="auto")
    km.fit(X_emb_pca_8)
    inertias.append(km.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Klusterių skaičius (k)")
plt.ylabel("Inercija")
plt.title("Alkūnės metodas (PCA su 8 komponentėmis)")
plt.grid(True)
plt.show()


# %%
K_means_model_8 = KMeans(n_clusters=2, random_state=80085, n_init="auto")
clusters_8 = K_means_model_8.fit_predict(X_emb_pca_8)

silhouette_score_8 = silhouette_score(X_emb_pca_8, clusters_8)
print(f"Silhouette Score for k=2 (PCA 8 components): {silhouette_score_8:.4f}")
davies_bouldin_score_8 = davies_bouldin_score(X_emb_pca_8, clusters_8)
print(f"Davies-Bouldin Score for k=2 (PCA 8 components): {davies_bouldin_score_8:.4f}")


# %%
stability_results_8 = run_clustering_stability(
    X_data=X_emb_pca_8,
    method="kmeans",
    strata_labels=final_dataset["season"],
    param_values=range(2, 7),
    n_runs=50,
    sample_fraction=0.8,
    base_seed=80085
)

cluster_assignments_runs = stability_results_8["assignments"]
k_run_metrics = stability_results_8["run_metrics"]
ari_pairs = stability_results_8["ari_pairs"]
stability_summary = stability_results_8["summary"]
best_k = stability_results_8["best_k"]

print("Stabilumo suvestinė pagal k:")
stability_summary


# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

axes[0].errorbar(
    stability_summary["k"],
    stability_summary["mean_silhouette"],
    yerr=stability_summary["std_silhouette"],
    fmt="o-",
    capsize=4,
)
axes[0].axvline(best_k, color="red", linestyle="--", alpha=0.7)
axes[0].set_title("Silhouette")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Reikšmė")
axes[0].grid(True, alpha=0.25)

axes[1].errorbar(
    stability_summary["k"],
    stability_summary["mean_davies_bouldin"],
    yerr=stability_summary["std_davies_bouldin"],
    fmt="o-",
    capsize=4,
)
axes[1].axvline(best_k, color="red", linestyle="--", alpha=0.7)
axes[1].set_title("Davies-Bouldin")
axes[1].set_xlabel("k")
axes[1].grid(True, alpha=0.25)

axes[2].errorbar(
    stability_summary["k"],
    stability_summary["mean_ari"],
    yerr=stability_summary["std_ari"],
    fmt="o-",
    capsize=4,
)
axes[2].axvline(best_k, color="red", linestyle="--", alpha=0.7)
axes[2].set_title("ARI")
axes[2].set_xlabel("k")
axes[2].grid(True, alpha=0.25)

plt.suptitle(f"KMeans stabilumo metrikos pagal k")
plt.tight_layout()
plt.show()



# %% [markdown]
#  <h2> Hierarchinis </h2>

# %%
Z = linkage(X_emb_pca_8, method='ward')
last = Z[-10:, 2]          
acceleration = np.diff(last, 2)  
k = acceleration[::-1].argmax() + 2 

print(f"Suggested k: {k}")
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("")
plt.xticks([])
plt.show()


# %%
hierarchical_model_8 = AgglomerativeClustering(n_clusters=2, linkage='ward')
hierarchical_clusters_8 = hierarchical_model.fit_predict(X_emb_pca_8)

hierarchical_results_8 = final_dataset[["Day", "season"]].copy()
hierarchical_results_8["hierarchical_cluster"] = hierarchical_clusters_8
hierarchical_score_silhouette_8 = silhouette_score(X_emb_pca_8, hierarchical_clusters_8)
hierarchical_score_davies_bouldin_8 = davies_bouldin_score(X_emb_pca_8, hierarchical_clusters_8)
print(f"Silhouette Score Hierarchical Model for k=2: {hierarchical_score_silhouette_8:.4f}")
print(f"Davies-Bouldin Score Hierarchical Model for k=2: {hierarchical_score_davies_bouldin_8:.4f}")


# %%
stability_results_8 = run_clustering_stability(
    X_data=X_emb_pca_8,
    method="hierarchical",         
    strata_labels=final_dataset["season"],
    param_values=range(2, 7),      
    n_runs=50,
    sample_fraction=0.8,
    base_seed=80085,
    linkage_method="ward"        
)

cluster_assignments_runs = stability_results_8["assignments"]
k_run_metrics = stability_results_8["run_metrics"]
ari_pairs = stability_results_8["ari_pairs"]
stability_summary= stability_results_8["summary"]
best_k = stability_results_8["best_k"]

print("Stabilumo suvestinė pagal k (hierarchical):")
stability_summary


# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

axes[0].errorbar(
    stability_summary["k"],
    stability_summary["mean_silhouette"],
    yerr=stability_summary["std_silhouette"],
    fmt="o-",
    capsize=4,
)
axes[0].axvline(best_k, color="red", linestyle="--", alpha=0.7)
axes[0].set_title("Silhouette")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Reikšmė")
axes[0].grid(True, alpha=0.25)

axes[1].errorbar(
    stability_summary["k"],
    stability_summary["mean_davies_bouldin"],
    yerr=stability_summary["std_davies_bouldin"],
    fmt="o-",
    capsize=4,
)
axes[1].axvline(best_k, color="red", linestyle="--", alpha=0.7)
axes[1].set_title("Davies-Bouldin")
axes[1].set_xlabel("k")
axes[1].grid(True, alpha=0.25)

axes[2].errorbar(
    stability_summary["k"],
    stability_summary["mean_ari"],
    yerr=stability_summary["std_ari"],
    fmt="o-",
    capsize=4,
)
axes[2].axvline(best_k, color="red", linestyle="--", alpha=0.7)
axes[2].set_title("ARI")
axes[2].set_xlabel("k")
axes[2].grid(True, alpha=0.25)

plt.suptitle(f"Hierarchical clustering stabilumo metrikos pagal k")  # <-- changed
plt.tight_layout()
plt.show()


# %% [markdown]
#  <h2> BDSCAN </h2>

# %%
stability_results_8 = run_clustering_stability(
    X_data=X_emb_pca_8,
    method="dbscan",
    strata_labels=final_dataset["season"], 
    param_values=np.round(np.linspace(1.8, 2.2, 5), 2),
    n_runs=50,
    sample_fraction=0.8,
    base_seed=80085,
    min_samples=5,
    max_noise_fraction=0.5
)

cluster_assignments_runs = stability_results_8["assignments"]
dbscan_run_metrics = stability_results_8["run_metrics"]
ari_pairs = stability_results_8["ari_pairs"]
stability_summary = stability_results_8["summary"]
best_eps = stability_results_8["best_eps"]

print("Stabilumo suvestinė pagal eps (DBSCAN):")
stability_summary


# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

axes[0].errorbar(
    stability_summary["eps"],
    stability_summary["mean_silhouette"],
    yerr=stability_summary["std_silhouette"],
    fmt="o-",
    capsize=4,
)
axes[0].axvline(best_eps, color="red", linestyle="--", alpha=0.7)
axes[0].set_title("Silhouette")
axes[0].set_xlabel("eps")
axes[0].set_ylabel("Reikšmė")
axes[0].grid(True, alpha=0.25)

axes[1].errorbar(
    stability_summary["eps"],
    stability_summary["mean_davies_bouldin"],
    yerr=stability_summary["std_davies_bouldin"],
    fmt="o-",
    capsize=4,
)
axes[1].axvline(best_eps, color="red", linestyle="--", alpha=0.7)
axes[1].set_title("Davies-Bouldin")
axes[1].set_xlabel("eps")
axes[1].grid(True, alpha=0.25)

axes[2].errorbar(
    stability_summary["eps"],
    stability_summary["mean_ari"],
    yerr=stability_summary["std_ari"],
    fmt="o-",
    capsize=4,
)
axes[2].axvline(best_eps, color="red", linestyle="--", alpha=0.7)
axes[2].set_title("ARI")
axes[2].set_xlabel("eps")
axes[2].grid(True, alpha=0.25)

plt.suptitle("DBSCAN stabilumo metrikos pagal eps")
plt.tight_layout()
plt.show()


# %%
dbscan_model_8 = DBSCAN(eps=2.1, min_samples=5)
dbscan_clusters_8 = dbscan_model_8.fit_predict(X_emb_pca_8)


# %%
dbscan_results_8 = final_dataset[["Day", "season"]].copy()
dbscan_results_8["dbscan_cluster"] = dbscan_clusters_8

mask = dbscan_clusters_8 != -1

if len(set(dbscan_clusters_8[mask])) > 1:
    silhouette = silhouette_score(X_emb_pca_8[mask], dbscan_clusters_8[mask])
    db_index = davies_bouldin_score(X_emb_pca_8[mask], dbscan_clusters_8[mask])
else:
    silhouette = np.nan
    db_index = np.nan

print(f"Silhouette Score DBSCAN: {silhouette:.4f}")
print(f"Davies-Bouldin Score DBSCAN: {db_index:.4f}")
print(f"Number of clusters: {len(set(dbscan_clusters_8)) - (1 if -1 in dbscan_clusters_8 else 0)}")
print(f"Noise points: {(dbscan_clusters_8 == -1).sum()}")



# %% [markdown]
#  <h1> Klasterizavimas originalioje dimensijoje </h1>

# %% [markdown]
#  <h2> K-means </h2>

# %%
inertias = []
k_values = range(1, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=80085, n_init="auto")
    km.fit(X)
    inertias.append(km.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Klusterių skaičius (k)")
plt.ylabel("Inercija")
plt.title("Alkūnės metodas (Originalioje dimensijoje)")
plt.grid(True)
plt.show()


# %%
K_means_model_orig = KMeans(n_clusters=2, random_state=80085, n_init="auto")
clusters_orig = K_means_model_orig.fit_predict(X)

silhouette_score_orig = silhouette_score(X, clusters_orig)
print(f"Silhouette Score for k=2 (Originalioje dimensijoje): {silhouette_score_orig:.4f}")
davies_bouldin_score_orig = davies_bouldin_score(X, clusters_orig)
print(f"Davies-Bouldin Score for k=2 (Originalioje dimensijoje): {davies_bouldin_score_orig:.4f}")


# %%
stability_results_orig = run_clustering_stability(
    X_data=X,
    method="kmeans",
    strata_labels=final_dataset["season"],
    param_values=range(2, 7),
    n_runs=50,
    sample_fraction=0.8,
    base_seed=80085
)

cluster_assignments_runs = stability_results_orig["assignments"]
k_run_metrics = stability_results_orig["run_metrics"]
ari_pairs = stability_results_orig["ari_pairs"]
stability_summary_orig = stability_results_orig["summary"]
best_k = stability_results_orig["best_k"]

print("Stabilumo suvestinė pagal k:")
stability_summary_orig


# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

axes[0].errorbar(
    stability_summary_orig["k"],
    stability_summary_orig["mean_silhouette"],
    yerr=stability_summary_orig["std_silhouette"],
    fmt="o-",
    capsize=4,
)
axes[0].axvline(best_k, color="red", linestyle="--", alpha=0.7)
axes[0].set_title("Silhouette")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Reikšmė")
axes[0].grid(True, alpha=0.25)

axes[1].errorbar(
    stability_summary_orig["k"],
    stability_summary_orig["mean_davies_bouldin"],
    yerr=stability_summary_orig["std_davies_bouldin"],
    fmt="o-",
    capsize=4,
)
axes[1].axvline(best_k, color="red", linestyle="--", alpha=0.7)
axes[1].set_title("Davies-Bouldin")
axes[1].set_xlabel("k")
axes[1].grid(True, alpha=0.25)

axes[2].errorbar(
    stability_summary_orig["k"],
    stability_summary_orig["mean_ari"],
    yerr=stability_summary_orig["std_ari"],
    fmt="o-",
    capsize=4,
)
axes[2].axvline(best_k, color="red", linestyle="--", alpha=0.7)
axes[2].set_title("ARI")
axes[2].set_xlabel("k")
axes[2].grid(True, alpha=0.25)

plt.suptitle(f"KMeans stabilumo metrikos pagal k")
plt.tight_layout()
plt.show()



# %% [markdown]
#  <h2> Hierarchinis </h2>

# %%
Z = linkage(X, method='ward')
last = Z[-10:, 2]          
acceleration = np.diff(last, 2)  
k = acceleration[::-1].argmax() + 2 

print(f"Suggested k: {k}")
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("")
plt.xticks([])
plt.show()


# %%
hierarchical_model_orig = AgglomerativeClustering(n_clusters=2, linkage='ward')
hierarchical_clusters_orig = hierarchical_model.fit_predict(X)

hierarchical_results_orig = final_dataset[["Day", "season"]].copy()
hierarchical_results_orig["hierarchical_cluster"] = hierarchical_clusters_orig
hierarchical_score_silhouette_orig = silhouette_score(X, hierarchical_clusters_orig)
hierarchical_score_davies_bouldin_orig = davies_bouldin_score(X, hierarchical_clusters_orig)
print(f"Silhouette Score Hierarchical Model for k=2: {hierarchical_score_silhouette_orig:.4f}")
print(f"Davies-Bouldin Score Hierarchical Model for k=2: {hierarchical_score_davies_bouldin_orig:.4f}")


# %%
stability_results_orig = run_clustering_stability(
    X_data=X,
    method="hierarchical",         
    strata_labels=final_dataset["season"],
    param_values=range(2, 7),      
    n_runs=50,
    sample_fraction=0.8,
    base_seed=80085,
    linkage_method="ward"        
)

cluster_assignments_runs = stability_results_orig["assignments"]
k_run_metrics = stability_results_orig["run_metrics"]
ari_pairs = stability_results_orig["ari_pairs"]
stability_summary_orig = stability_results_orig["summary"]
best_k = stability_results_orig["best_k"]

print("Stabilumo suvestinė pagal k (hierarchical):")
stability_summary_orig


# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

axes[0].errorbar(
    stability_summary_orig["k"],
    stability_summary_orig["mean_silhouette"],
    yerr=stability_summary_orig["std_silhouette"],
    fmt="o-",
    capsize=4,
)
axes[0].axvline(best_k, color="red", linestyle="--", alpha=0.7)
axes[0].set_title("Silhouette")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Reikšmė")
axes[0].grid(True, alpha=0.25)

axes[1].errorbar(
    stability_summary_orig["k"],
    stability_summary_orig["mean_davies_bouldin"],
    yerr=stability_summary_orig["std_davies_bouldin"],
    fmt="o-",
    capsize=4,
)
axes[1].axvline(best_k, color="red", linestyle="--", alpha=0.7)
axes[1].set_title("Davies-Bouldin")
axes[1].set_xlabel("k")
axes[1].grid(True, alpha=0.25)

axes[2].errorbar(
    stability_summary_orig["k"],
    stability_summary_orig["mean_ari"],
    yerr=stability_summary_orig["std_ari"],
    fmt="o-",
    capsize=4,
)
axes[2].axvline(best_k, color="red", linestyle="--", alpha=0.7)
axes[2].set_title("ARI")
axes[2].set_xlabel("k")
axes[2].grid(True, alpha=0.25)

plt.suptitle(f"Hierarchical clustering stabilumo metrikos pagal k")  # <-- changed
plt.tight_layout()
plt.show()


# %% [markdown]
#  <h2> BDSCAN </h2>

# %%
stability_results_orig = run_clustering_stability(
    X_data=X,
    method="dbscan",
    strata_labels=final_dataset["season"], 
    param_values=np.round(np.linspace(1.8, 2.2, 5), 2),
    n_runs=50,
    sample_fraction=0.8,
    base_seed=80085,
    min_samples=5,
    max_noise_fraction=0.5
)

cluster_assignments_runs = stability_results_orig["assignments"]
dbscan_run_metrics = stability_results_orig["run_metrics"]
ari_pairs = stability_results_orig["ari_pairs"]
stability_summary_orig = stability_results_orig["summary"]
best_eps = stability_results_orig["best_eps"]

print("Stabilumo suvestinė pagal eps (DBSCAN):")
stability_summary_orig


# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)

axes[0].errorbar(
    stability_summary_orig["eps"],
    stability_summary_orig["mean_silhouette"],
    yerr=stability_summary_orig["std_silhouette"],
    fmt="o-",
    capsize=4,
)
axes[0].axvline(best_eps, color="red", linestyle="--", alpha=0.7)
axes[0].set_title("Silhouette")
axes[0].set_xlabel("eps")
axes[0].set_ylabel("Reikšmė")
axes[0].grid(True, alpha=0.25)

axes[1].errorbar(
    stability_summary_orig["eps"],
    stability_summary_orig["mean_davies_bouldin"],
    yerr=stability_summary_orig["std_davies_bouldin"],
    fmt="o-",
    capsize=4,
)
axes[1].axvline(best_eps, color="red", linestyle="--", alpha=0.7)
axes[1].set_title("Davies-Bouldin")
axes[1].set_xlabel("eps")
axes[1].grid(True, alpha=0.25)

axes[2].errorbar(
    stability_summary_orig["eps"],
    stability_summary_orig["mean_ari"],
    yerr=stability_summary_orig["std_ari"],
    fmt="o-",
    capsize=4,
)
axes[2].axvline(best_eps, color="red", linestyle="--", alpha=0.7)
axes[2].set_title("ARI")
axes[2].set_xlabel("eps")
axes[2].grid(True, alpha=0.25)

plt.suptitle("DBSCAN stabilumo metrikos pagal eps")
plt.tight_layout()
plt.show()


# %%
dbscan_model_orig = DBSCAN(eps=2.1, min_samples=5)
dbscan_clusters_orig = dbscan_model_orig.fit_predict(X)


# %%
dbscan_results_orig= final_dataset[["Day", "season"]].copy()
dbscan_results_orig["dbscan_cluster"] = dbscan_clusters_orig

mask = dbscan_clusters_orig != -1

if len(set(dbscan_clusters_orig[mask])) > 1:
    silhouette = silhouette_score(X[mask], dbscan_clusters_orig[mask])
    db_index = davies_bouldin_score(X[mask], dbscan_clusters_orig[mask])
else:
    silhouette = np.nan
    db_index = np.nan

print(f"Silhouette Score DBSCAN: {silhouette:.4f}")
print(f"Davies-Bouldin Score DBSCAN: {db_index:.4f}")
print(f"Number of clusters: {len(set(dbscan_clusters_orig)) - (1 if -1 in dbscan_clusters_orig else 0)}")
print(f"Noise points: {(dbscan_clusters_orig == -1).sum()}")





