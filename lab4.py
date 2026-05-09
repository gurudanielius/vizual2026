# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import warnings

# %%
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# %%
final_dataset=pd.read_csv("final_dataset.csv")

# %%
final_dataset.head()


# %%
label_counts = final_dataset["season"].value_counts()
label_counts # klases subalansuotos


# %%
# final_dataset.drop(columns=["month"], inplace=True)


# %%
sums=final_dataset.copy()
sums=sums.select_dtypes(include="number").mean(axis=1).to_frame(name="mean")

# %%
sums

# %%
sums["Day"]=final_dataset["Day"]
sums["season"]=final_dataset["season"]

# %%
sums

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

plt.figure(figsize=(6, 3))
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
plot_means=sums.groupby("season")["mean"].mean().to_frame(name="mean")
plot_means[ "std" ] = sums.groupby("season")["mean"].std().to_frame(name="std")

# %%
plot_means

# %%
# point plot of mean with std by season (no connecting line)
plot_means_ordered = plot_means.reindex(season_order)
x = range(len(plot_means_ordered.index))
plt.figure(figsize=(6, 3))
plt.errorbar(
    x,
    plot_means_ordered["mean"],
    yerr=plot_means_ordered["std"],
    fmt="o",
    color="black",
    ecolor="black",
    capsize=5,
    linewidth=1.5
 )
plt.xticks(ticks=x, labels=["Žiema", "Pavasaris", "Vasara", "Ruduo"])
plt.title("Vidutinė galia pagal sezoną")
plt.xlabel("Sezonas")
plt.ylabel("Vidutinė galia")
plt.tight_layout()
plt.show()


# %%
#kodel grafikas toks gaunasi:DD edit:viskas ok dabar
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
plt.figure(figsize=(6, 3))
#color lines according to season
for i,s in enumerate(season_order):
    part = line_df[line_df["season"] == s]
    plt.plot(
        part["time"],
        part["power"],
        label=season_labels_lt.get(s, s),
        linewidth=1.8,
        color=season_colors[i]
    )

plt.title("Galios generavimas pagal laiką ir sezoną")
plt.xlabel("Laikas")
plt.ylabel("Galia")
plt.xticks(ticks=range(0, len(part["time"])), rotation=45, ha="right") 
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
#  # Duomenų padalinimas

# %%
X = final_dataset.drop(columns=["season", "Day", "month"])
y = final_dataset["season"]

# %%
#Originalios aibes padalinimas
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=80085)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=80085)

# %% [markdown]
#  # Dimensijos mažinimas

# %%
def normalized_stress(X, X_emb):
	D_orig = pairwise_distances(X)
	D_emb = pairwise_distances(X_emb)
	return np.sum((D_orig - D_emb) ** 2) / np.sum(D_orig ** 2)

def emb_metrics(X_orig, X_emb, n_neighbors=10):
    X_orig = np.asarray(X_orig)
    X_emb = np.asarray(X_emb)

    t = trustworthiness(X_orig, X_emb, n_neighbors=n_neighbors)
    c = trustworthiness(X_emb, X_orig, n_neighbors=n_neighbors)
    stress = normalized_stress(X_orig, X_emb)

    print(f"Trustworthiness: {t:.4f}")
    print(f"Continuity:      {c:.4f}")
    print(f"Stress:          {stress:.4f}")

# %% [markdown]
#  ### PCA
# 
#  Praeitame laboratoriniame darbe naudotas tas pats duomenų rinkinys ir ten gauta, kad geriausias dimensijos mažinimo algoritmas yra PCA. Šiame laboratorinyje taip pat naudosime PCA.

# %%
pca_model = PCA(n_components=2, random_state=80085)

# %%
#PCA aibes padalinimas (padariau kad musu orignalios aibes padalinimas butu tas pats kaip ir PCA del consistency)
scaler = RobustScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_val   = scaler.transform(X_val)
X_scaled_test  = scaler.transform(X_test)

X_train_pca = pca_model.fit_transform(X_scaled_train)
X_val_pca   = pca_model.transform(X_scaled_val)
X_test_pca  = pca_model.transform(X_scaled_test)

# %% [markdown]
#  # Atsitiktinių miškų klasifikatorius
# 
# 
# 
#  **Pagrindiniai hiperparametrai:**
# 
#  1. n_estimators
# 
#  2. max_depth
# 
#  3. min_samples_split
# 
#  4. min_samples_leaf
# 
#  5. max_features

# %% [markdown]
#  ## Originali duomenų aibė

# %%
rf_param_grid = {
    'n_estimators': [50, 200, 500],
    'max_depth': [5, 10, None],
    'max_features': ['sqrt', 'log2', 0.5],
}

rf_results = []
for rf_params in ParameterGrid(rf_param_grid):
    rf_model = RandomForestClassifier(**rf_params, random_state=80085, n_jobs=-1).fit(X_train, y_train)
    rf_results.append({
        **rf_params,
        'train_acc': rf_model.score(X_train, y_train),
        'val_acc': rf_model.score(X_val, y_val),
    })

rf_results_df = pd.DataFrame(rf_results).sort_values('val_acc', ascending=False).reset_index(drop=True)
rf_results_df

# %%
best_rf = RandomForestClassifier(
    max_depth=10, max_features=0.5, n_estimators=50,
    random_state=80085
).fit(X_train, y_train)

test_acc_rf = best_rf.score(X_test, y_test)
print(f"Test accuracy: {test_acc_rf :.4f}")

# %%
y_test_pred_rf = best_rf.predict(X_test)

print(f"Test accuracy: {best_rf.score(X_test, y_test):.4f}\n")
print(classification_report(y_test, y_test_pred_rf, digits=3))

# %%
cm_test_rf = confusion_matrix(y_test, y_test_pred_rf, labels=["Winter", "Spring", "Summer", "Autumn"])
disp_rf = ConfusionMatrixDisplay(
    cm_test_rf,
    display_labels=["Žiema", "Pavasaris", "Vasara", "Ruduo"]
)
disp_rf.plot(cmap="Blues")
disp_rf.ax_.set_xlabel("Prognozuota klasė")
disp_rf.ax_.set_ylabel("Tikroji klasė")
plt.title("Atsitiktinių miškų sumaišymo matrica - testinė aibė")
plt.tight_layout()
plt.show()

# %% [markdown]
#  ## Dviejų dimensijų aibė

# %%
rf_results_pca = []
for rf_params in ParameterGrid(rf_param_grid):
    rf_model = RandomForestClassifier(**rf_params, random_state=80085, n_jobs=-1).fit(X_train_pca, y_train)
    rf_results_pca.append({
        **rf_params,
        'train_acc': rf_model.score(X_train_pca, y_train),
        'val_acc': rf_model.score(X_val_pca, y_val),
    })

rf_results_pca_df = pd.DataFrame(rf_results_pca).sort_values('val_acc', ascending=False).reset_index(drop=True)
rf_results_pca_df

# %% [markdown]
#  Prasti popieriai, PCA duomenų aibė labai pablogina rezultatus - validacijos aibė realiai spėlioja duomenis, o klasifikacvimo tikslumas (ten kur validacija geriausia) sieki tik 0,767...

# %%
best_rf_pca = RandomForestClassifier(
    max_depth=5, max_features="sqrt", n_estimators=200,
    random_state=80085, n_jobs=1
).fit(X_train_pca, y_train)


# %%
y_test_pred_rf_pca = best_rf_pca.predict(X_test_pca)
test_acc_rf_pca = best_rf_pca.score(X_test_pca, y_test)

print(f"Test accuracy: {test_acc_rf_pca :.4f}\n")
print(classification_report(y_test, y_test_pred_rf_pca, digits=3))

# %%
cm_test_rf_pca = confusion_matrix(y_test, y_test_pred_rf_pca, labels=["Winter", "Spring", "Summer", "Autumn"])
disp_rf_pca = ConfusionMatrixDisplay(
    cm_test_rf_pca,
    display_labels=["Žiema", "Pavasaris", "Vasara", "Ruduo"]
)
disp_rf_pca.plot(cmap="Blues")
disp_rf_pca.ax_.set_xlabel("Prognozuota klasė")
disp_rf_pca.ax_.set_ylabel("Tikroji klasė")
plt.title("Atsitiktinių miškų sumaišymo matrica - testinė aibė (PCA)")
plt.tight_layout()
plt.show()

# %%
def plot_classification_pca(X_test_pca, y_test, y_pred, title):
    season_colors_map = {
        "Winter": "#4C78A8", "Spring": "#59A14F",
        "Summer": "#F28E2B", "Autumn": "#9C755F",
    }
    season_lt = {"Winter": "Žiema", "Spring": "Pavasaris",
                 "Summer": "Vasara", "Autumn": "Ruduo"}

    y_test_arr = np.array(y_test)
    correct = y_test_arr == np.array(y_pred)

    lim_min = np.floor(min(X_test_pca[:, 0].min(), X_test_pca[:, 1].min())) - 1
    lim_max = np.ceil(max(X_test_pca[:, 0].max(), X_test_pca[:, 1].max())) + 1

    plt.figure(figsize=(9, 9))

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        mask = y_test_arr == season
        plt.scatter(
            X_test_pca[mask, 0], X_test_pca[mask, 1],
            c=season_colors_map[season],
            label=season_lt[season],
            s=70, alpha=0.8,
            edgecolor="white", linewidth=0.5,
        )

    plt.scatter(
        X_test_pca[~correct, 0], X_test_pca[~correct, 1],
        facecolors="none", edgecolors="red",
        s=200, linewidths=2,
        label=f"Klaidos ({(~correct).sum()})",
    )

    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(title="Sezonas", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%
plot_classification_pca(X_test_pca, y_test, y_test_pred_rf,
                        title="Atsitiktinių miškų klasifikavimas PCA erdvėje")

# %% [markdown]
#  #Todo: labiau pasižiūėti kuo klaidos išsiskiria (tiketina, kad bus tie taskai, kurie yra perainamajame laikotarpyje ruduo -> ziema, ziema -> pavasaris, pavasaris -> vasara, vasara -> ruduo)

# %% [markdown]
# # k-NN klasifikatorius
# 
# **Pagrindiniai hiperparametrai:**
# 1. n_neighbors
# 2. weights
# 3. metric

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=80085)

knn_param_grid = {
    "knn__n_neighbors": [1, 3, 5, 7, 9, 10, 15, 20, 30],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan", "cosine"]
}

# %% [markdown]
# ## Originali duomenų aibė

# %%
knn_pipe = Pipeline([
    ("scaler", RobustScaler()),
    ("knn", KNeighborsClassifier())
])

grid_knn = GridSearchCV(
    estimator=knn_pipe,
    param_grid=knn_param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid_knn.fit(X_train, y_train)

print("Best CV accuracy:", grid_knn.best_score_)
print("Best parameters:", grid_knn.best_params_)

# %%
knn_results_df = pd.DataFrame(grid_knn.cv_results_)

knn_results_df = knn_results_df[
    [
        "param_knn__n_neighbors",
        "param_knn__weights",
        "param_knn__metric",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]
].sort_values("rank_test_score").reset_index(drop=True)

knn_results_df

# %%
best_knn = grid_knn.best_estimator_

y_test_pred_knn = best_knn.predict(X_test)

test_acc_knn = best_knn.score(X_test, y_test)
print(f"Test accuracy: {test_acc_knn:.4f}\n")
print(classification_report(y_test, y_test_pred_knn, digits=3))

# %%
cm_test_knn = confusion_matrix(
    y_test,
    y_test_pred_knn,
    labels=["Winter", "Spring", "Summer", "Autumn"]
)

disp_knn = ConfusionMatrixDisplay(
    cm_test_knn,
    display_labels=["Žiema", "Pavasaris", "Vasara", "Ruduo"]
)

disp_knn.plot(cmap="Blues")
disp_knn.ax_.set_xlabel("Prognozuota klasė")
disp_knn.ax_.set_ylabel("Tikroji klasė")
plt.title("k-NN sumaišymo matrica - testinė aibė")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Dviejų dimensijų aibė

# %%
knn_param_grid_pca = {
    "n_neighbors": [1, 3, 5, 7, 9, 10, 15, 20, 30],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "cosine"]
}

grid_knn_pca = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=knn_param_grid_pca,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid_knn_pca.fit(X_train_pca, y_train)

print("Best PCA CV accuracy:", grid_knn_pca.best_score_)
print("Best PCA parameters:", grid_knn_pca.best_params_)

# %%
knn_results_pca_df = pd.DataFrame(grid_knn_pca.cv_results_)

knn_results_pca_df = knn_results_pca_df[
    [
        "param_n_neighbors",
        "param_weights",
        "param_metric",
        "mean_test_score",
        "std_test_score",
        "rank_test_score"
    ]
].sort_values("rank_test_score").reset_index(drop=True)

knn_results_pca_df

# %%
best_knn_pca = grid_knn_pca.best_estimator_

y_test_pred_knn_pca = best_knn_pca.predict(X_test_pca)

test_acc_knn_pca = best_knn_pca.score(X_test_pca, y_test)

print(f"Test accuracy: {test_acc_knn_pca:.4f}\n")
print(classification_report(y_test, y_test_pred_knn_pca, digits=3))

# %%
cm_test_knn_pca = confusion_matrix(
    y_test,
    y_test_pred_knn_pca,
    labels=["Winter", "Spring", "Summer", "Autumn"]
)

disp_knn_pca = ConfusionMatrixDisplay(
    cm_test_knn_pca,
    display_labels=["Žiema", "Pavasaris", "Vasara", "Ruduo"]
)

disp_knn_pca.plot(cmap="Blues")
disp_knn_pca.ax_.set_xlabel("Prognozuota klasė")
disp_knn_pca.ax_.set_ylabel("Tikroji klasė")
plt.title("k-NN sumaišymo matrica - testinė aibė (PCA)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Roc originalioj

# %%
classes = ["Winter", "Spring", "Summer", "Autumn"]

# tikimybės
y_test_proba_knn = best_knn.predict_proba(X_test)

# binarinės klasės
y_test_bin = label_binarize(y_test, classes=classes)

# macro AUC
auc_macro_knn = roc_auc_score(
    y_test_bin,
    y_test_proba_knn,
    average="macro",
    multi_class="ovr"
)

print(f"Macro AUC: {auc_macro_knn:.4f}")

# %%
plt.figure(figsize=(6, 4))

for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_test_proba_knn[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{cls} AUC = {roc_auc:.2f}"
    )

plt.plot([0, 1], [0, 1], "k--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("k-NN ROC kreivės")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Klaidos originalioj Knn

# %%
errors_knn = X_test[y_test != y_test_pred_knn].copy()

errors_knn["True"] = y_test[y_test != y_test_pred_knn]
errors_knn["Predicted"] = y_test_pred_knn[y_test != y_test_pred_knn]

errors_knn.head()

# %%
print(f"Klaidų skaičius: {len(errors_knn)}")

# %%
pd.crosstab(
    errors_knn["True"],
    errors_knn["Predicted"]
)

# %% [markdown]
# ## Klaidos PCA knn

# %%
plot_classification_pca(
    X_test_pca,
    y_test,
    y_test_pred_knn_pca,
    title="k-NN klasifikavimas PCA erdvėje"
)


