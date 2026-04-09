# %%
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, trustworthiness
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


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
#  Inverteris 2024 metais išsijungia random nuo 19:00 iki 02:00, todėl stebėjome keistumus, bet čia problemų yra ir kitų - skaityk duomenų kiekis atitinkantis 17 dienų yra tušti;

# %%
data_selected_features=data_selected_features[data_selected_features["timestamp"].dt.year == 2023]


# %% [markdown]
#  <span style="color: rgb(244, 12, 105);"> Daug geriau yra su praleistomis reikšmėmis -- čia yra tik viena diena kur visi inverteriai, jei imame tik 2023 metus, čia problema yra tik su 3 inverteriu, NA reikšmes čia užpildydami vidurkiu visai gerą aproksimacija gaunasi mano galva;

# %%
mask_all_na_2023 = data_selected_features[inv_cols].isna().all(axis=1)
all_empty_2023=data_selected_features[mask_all_na_2023]
all_empty_2023


# %%
data_raw = data_selected_features[~mask_all_na_2023]
data_raw


# %% [markdown]
#  Turime su 3 inverteriu daug praleistų reikšmelių (56 dienas) siūlau trinti, kol kas užpildau vidurkiu pagal eilutes

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
#  Patriminau laiką;

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
# ### PCA

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
# ### T-SNE
# 

# %%
def tsne_grid_search(X, param_grid, n_neighbors=10, random_state=42):
    grid = ParameterGrid(param_grid)
    results = []
    for params in grid:
        tsne = TSNE(n_components=2, random_state=random_state, max_iter=1000, **params)
        try:
            X_emb = tsne.fit_transform(X)

            # Compute metrics
            t = trustworthiness(X, X_emb, n_neighbors=n_neighbors)
            c = trustworthiness(X_emb, X, n_neighbors=n_neighbors)
            stress = normalized_stress(X.values, X_emb)
            results.append({
                'params': params,
                'trustworthiness': t,
                'continuity': c,
                'stress': stress,
                'score': score
            })
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue

    return results

param_grid = {'perplexity': [5, 15, 20, 30, 50], 'learning_rate': [10, 20, 30, 50, 100]}
results = tsne_grid_search(X, param_grid)
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
# ## MDS

# %%

def mds_grid_search(X, param_grid, n_neighbors=10, random_state=42):
    grid = ParameterGrid(param_grid)
    results = []
    for params in grid:
        mds = MDS(n_components=2, normalized_stress=True, n_jobs=-1, n_init=10, **params)
        try:
            X_emb = mds.fit_transform(X)

            t = trustworthiness(X, X_emb, n_neighbors=n_neighbors)
            c = trustworthiness(X_emb, X, n_neighbors=n_neighbors)
            stress = mds.stress_
            results.append({
                'params': params,
                'trustworthiness': t,
                'continuity': c,
                'stress': stress,
                'score': score
            })
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue

    return results

# Example usage:
param_grid = {'max_iter': [150, 200, 300, 500, 1000]}
results = mds_grid_search(X, param_grid)
for res in results:
    print(res)




# %%
#  MDS
mds = MDS(n_components=2, max_iter=1000, normalized_stress=True)
mds_result = mds.fit_transform(X)

mds_df = pd.DataFrame(mds_result, columns=['MDS1', 'MDS2'])
mds_df['season'] = data_clean['season'].values

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

# Compute trustworthiness and continuity
X_emb = mds_df[['MDS1', 'MDS2']].values

t = trustworthiness(X, X_emb, n_neighbors=10)
c = trustworthiness(X_emb, X, n_neighbors=10)

print(f"Trustworthiness: {t:.4f}")
print(f"Continuity: {c:.4f}")
print(f"Stress: {mds.stress_:.4f}")




# %%


# %%
X = data_clean.drop(columns=["Day","season","month"], errors="ignore").copy()

# scale first
X_scaled = StandardScaler().fit_transform(X)

inertias = []
k_values = range(1, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)
plt.show()



# %%
#k = 3

K_means_model = KMeans(n_clusters=3, random_state=42, n_init="auto")
clusters = K_means_model.fit_predict(X_scaled)

data_clean["cluster"] = clusters
#sil
score = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score for k=3: {score:.4f}")



# %%
data_clean



# %%

clusters = K_means_model.fit_predict(X_emb)
score = silhouette_score(X_emb, clusters)
print(f"Silhouette Score for k=3: {score:.4f}")








