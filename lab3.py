# %%
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns




# %%
data_raw = pd.read_csv('Elektrines_duomenys_2023-2024m.csv', sep=';', decimal=',')
data_selected_features= data_raw[["timestamp"] + [f"Total_active_power_INV-{i}" for i in range(1, 9)]]


inv_cols = [f"Total_active_power_INV-{i}" for i in range(1, 9)]
mask_all_empty = data_selected_features[inv_cols].fillna(0).eq(0).all(axis=1)
data_selected_features = data_selected_features.loc[~mask_all_empty]
data_selected_features
data_raw= data_selected_features


# %%
#sum across rows
data_raw["Total_active_power"] = data_raw[[f"Total_active_power_INV-{i}" for i in range(1, 9)]].sum(axis=1)
data_final = data_raw[["timestamp", "Total_active_power"]].copy()
data_final["timestamp"] = pd.to_datetime(data_final["timestamp"], errors="coerce")
data_final = data_final.dropna(subset=["timestamp"])







# %%
data_final.isna().sum()




# %%
data_final




# %%
# Convert data_final to wide format: Day + one column per 5-minute timestamp
data_final["Day"] = data_final["timestamp"].dt.date
data_final["Time"] = data_final["timestamp"].dt.strftime("%H:%M")

# If there are duplicate timestamps for a day, keep the summed value
sum_of_inv = (
    data_final.groupby(["Day", "Time"], as_index=False)["Total_active_power"]
    .sum()
)
sum_of_inv_wide = sum_of_inv.pivot(index="Day", columns="Time", values="Total_active_power")
sum_of_inv_wide.columns.name = None  # Remove index name
sum_of_inv_wide = sum_of_inv_wide.reset_index()

final_dataset = sum_of_inv_wide[["Day"] + sorted(sum_of_inv_wide.columns[1:])]
final_dataset




# %%
final_dataset["Day"] = pd.to_datetime(final_dataset["Day"], errors="coerce").dt.date
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
id_cols = ["Day", "month", "season"]
value_cols = [c for c in final_dataset.columns if c not in id_cols]

melted = final_dataset.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="time",
    value_name="power"
).dropna(subset=["power", "season"])

# Collect values per season for boxplot
data_by_season = [
    melted.loc[melted["season"] == s, "power"].values
    for s in season_order
]

plt.figure(figsize=(10, 6))
plt.boxplot(
    data_by_season,
    labels=season_order
)
plt.title("Power Distribution by Season")
plt.xlabel("Season")
plt.ylabel("Power")
plt.tight_layout()
plt.show()



# %%
threshold = 60000

final_dataset = final_dataset.fillna(0)

num_cols = final_dataset.select_dtypes(include="number").columns
final_data = final_dataset[(final_dataset[num_cols] <= threshold).all(axis=1)]

final_data


# %%
print(len(final_dataset), len(final_data)) #istrintos keturios eilutes


# %%
# Melt the dataset so all timestamp columns become rows
timestamp_cols = [col for col in final_data.columns if col not in ['Day', 'season']]

melted = final_data.melt(id_vars=['Day', 'season'], 
                         value_vars=timestamp_cols, 
                         var_name='timestamp', 
                         value_name='value')


# %%
# choose source table (final_data if you filtered out extreme rows, otherwise final_dataset)
df = final_data.copy()  # or: final_dataset.copy()

id_cols = ["Day", "month", "season"]
value_cols = [c for c in df.columns if c not in id_cols]

# melt wide -> long
melted = df.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="timestamp",
    value_name="value"
).dropna(subset=["season", "value"])

# make time sortable
melted["time_dt"] = pd.to_datetime(melted["timestamp"], format="%H:%M", errors="coerce")

# average profile per season across all days
line_df = (
    melted.groupby(["season", "timestamp", "time_dt"], as_index=False)["value"]
    .sum()
    .sort_values("time_dt")
)

season_order = ["Winter", "Spring", "Summer", "Autumn"]

plt.figure(figsize=(12, 6))
for s in season_order:
    part = line_df[line_df["season"] == s]
    plt.plot(part["timestamp"], part["value"], label=s, linewidth=1.8)

plt.title("Daily Power Curve by Season")
plt.xlabel("Time of day")
plt.ylabel("Power")
plt.xticks(ticks=range(0, len(part["timestamp"]), 12), rotation=45, ha="right")  # every 1 hour if 5-min data
plt.legend()
plt.tight_layout()
plt.show()


# %%
id_cols = ["Day", "month", "season"]

# pick whichever table you want to trim
df = final_data.copy()   # or final_dataset.copy()

time_cols = [c for c in df.columns if c not in id_cols]
time_dt = pd.to_datetime(time_cols, format="%H:%M", errors="coerce")

keep_time_cols = [
    c for c, t in zip(time_cols, time_dt)
    if pd.notna(t) and (t.hour * 60 + t.minute >= 2 * 60) and (t.hour * 60 + t.minute <= 19 * 60)
]

data_clean = df[id_cols + keep_time_cols]
data_clean


# %%
season_order = ["Winter", "Spring", "Summer", "Autumn"]
id_cols = ["Day", "month", "season"]
value_cols = [c for c in data_clean.columns if c not in id_cols]

season_labels_lt = {
    "Winter": "Žiema",
    "Spring": "Pavasaris",
    "Summer": "Vasara",
    "Autumn": "Ruduo"
}

melted = data_clean.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="time",
    value_name="power"
).dropna(subset=["power", "season"])

# Collect values per season for boxplot
data_by_season = [
    melted.loc[melted["season"] == s, "power"].values
    for s in season_order
]

plt.figure(figsize=(10, 6))
plt.boxplot(
    data_by_season,
    labels=[season_labels_lt.get(s, s) for s in season_order]
)
plt.title("Metų laikų stačiakampės diagramos")
plt.xlabel("Metų laikas")
plt.ylabel("Elektros energijos kiekis")
plt.tight_layout()
plt.show()



# %%
timestamp_cols = [col for col in data_clean.columns if col not in ['Day', 'season']]

melted = data_clean.melt(id_vars=['Day', 'season'], 
                         value_vars=timestamp_cols, 
                         var_name='timestamp', 
                         value_name='value')

print(melted.groupby("season").describe())


# %%
season_stats = melted.groupby('season')['value'].describe().round(4)

print(season_stats)


# %%
winter_data = data_clean[data_clean["season"] == "Winter"].copy()

num = winter_data.select_dtypes(include="number")
row_idx, col_name = num.stack().idxmax()   # location of absolute max
max_val = num.loc[row_idx, col_name]
max_day = winter_data.loc[row_idx, "Day"]

max_day, col_name, max_val


# %%
season_order = ["Winter", "Spring", "Summer", "Autumn"]
id_cols = ["Day", "month", "season"]
value_cols = [c for c in data_clean.columns if c not in id_cols]

melted = data_clean.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="time",
    value_name="power"
).dropna(subset=["power", "season"])

heatmap_by_season = (
    melted.groupby(["season", "time"], as_index=False)["power"]
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

xtick_positions = range(0, len(heatmap_by_season.columns), 12)
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


# %%
#PCA

from sklearn.decomposition import PCA
import numpy as np

id_cols = ['Day', 'month', 'season']
X = data_clean.drop(columns=id_cols).select_dtypes(include='number')

pca_model = PCA(n_components=2, random_state=80085)
pca_result = pca_model.fit_transform(X)

X_pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
X_pca_df['season'] = data_clean['season'].values


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


# %%
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
import numpy as np

X_emb = X_pca_df[['PC1', 'PC2']].values

# Trustworthiness and continuity
t = trustworthiness(X.values, X_emb, n_neighbors=10)
c = trustworthiness(X_emb, X.values, n_neighbors=10)

# Stress-like measure for the 2D embedding
D_orig = pairwise_distances(X.values)
D_emb = pairwise_distances(X_emb)
stress = np.sum((D_orig - D_emb) ** 2) / np.sum(D_orig ** 2)

print(f"Trustworthiness: {t:.4f}")
print(f"Continuity: {c:.4f}")
print(f"Stress: {stress:.4f}")

# %%


# %%
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances
import numpy as np

# Fit t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=10, max_iter=1000)
tsne_result = tsne.fit_transform(X)

tsne_df = pd.DataFrame(tsne_result, columns=['TSNE1', 'TSNE2'])
tsne_df['season'] = data_clean['season'].values

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

# Compute trustworthiness, continuity, and stress
X_emb = tsne_df[['TSNE1', 'TSNE2']].values

t = trustworthiness(X, X_emb, n_neighbors=10)
c = trustworthiness(X_emb, X, n_neighbors=10)

D_orig = pairwise_distances(X)
D_emb = pairwise_distances(X_emb)
stress = np.sum((D_orig - D_emb) ** 2) / np.sum(D_orig ** 2)

print(f"Trustworthiness: {t:.4f}")
print(f"Continuity: {c:.4f}")
print(f"Stress: {stress:.4f}")

# %%


# %%
from sklearn.model_selection import ParameterGrid
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances
import numpy as np

def tsne_grid_search(X, param_grid, n_neighbors=10, random_state=42):
    """
    Perform grid search for t-SNE hyperparameters and return all combinations with metrics.

    Parameters:
    - X: Input data (scaled)
    - param_grid: Dict of parameters to search, e.g., {'perplexity': [5, 30, 50], 'learning_rate': [10, 200]}
    - n_neighbors: Number of neighbors for trustworthiness/continuity
    - random_state: Random state for reproducibility

    Returns:
    - results: List of dicts with params, trustworthiness, continuity, stress, and score
    """
    grid = ParameterGrid(param_grid)
    results = []

    for params in grid:
        tsne = TSNE(n_components=2, random_state=random_state, max_iter=1000, **params)
        try:
            X_emb = tsne.fit_transform(X)

            # Compute metrics
            t = trustworthiness(X, X_emb, n_neighbors=n_neighbors)
            c = trustworthiness(X_emb, X, n_neighbors=n_neighbors)
            D_orig = pairwise_distances(X)
            D_emb = pairwise_distances(X_emb)
            stress = np.sum((D_orig - D_emb) ** 2) / np.sum(D_orig ** 2)

            score = t  # Use trustworthiness as the score

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
param_grid = {'perplexity': [5, 15, 20, 30, 50], 'learning_rate': [10, 20, 30, 50, 100]}
results = tsne_grid_search(X, param_grid)
for res in results:
    print(res)

# %%


# %%
from sklearn.manifold import MDS, trustworthiness
from sklearn.metrics import pairwise_distances
import numpy as np

id_cols = ['Day', 'month', 'season']
X = data_clean.drop(columns=id_cols).select_dtypes(include='number')


# Fit MDS
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
from sklearn.model_selection import ParameterGrid
from sklearn.manifold import MDS, trustworthiness
from sklearn.metrics import pairwise_distances
import numpy as np

def mds_grid_search(X, param_grid, n_neighbors=10, random_state=42):
    """
    Perform grid search for MDS hyperparameters and return all combinations with metrics.

    Parameters:
    - X: Input data (scaled)
    - param_grid: Dict of parameters to search, e.g., {'max_iter': [300, 1000], 'n_init': [4, 10]}
    - n_neighbors: Number of neighbors for trustworthiness/continuity
    - random_state: Random state for reproducibility

    Returns:
    - results: List of dicts with params, trustworthiness, continuity, stress, and score
    """
    grid = ParameterGrid(param_grid)
    results = []

    for params in grid:
        mds = MDS(n_components=2, normalized_stress=True, n_jobs=-1, n_init=10, **params)
        try:
            X_emb = mds.fit_transform(X)

            # Compute metrics
            t = trustworthiness(X, X_emb, n_neighbors=n_neighbors)
            c = trustworthiness(X_emb, X, n_neighbors=n_neighbors)
            stress = mds.stress_

            score = t  # Use trustworthiness as the score

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



