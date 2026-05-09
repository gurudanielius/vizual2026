# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid, train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, trustworthiness
from sklearn.metrics import (pairwise_distances, classification_report, confusion_matrix, 
                           accuracy_score, ConfusionMatrixDisplay, balanced_accuracy_score, 
                           recall_score, f1_score, precision_score, roc_curve, auc, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize



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



# %%
#Skaitines charakteristikos pagal sezona
print(final_dataset_melted[["power","season"]].groupby("season").describe())



# %%



# %% [markdown]
#   # Duomenų padalinimas

# %%
X = final_dataset.drop(columns=["season", "Day", "month"])
y = final_dataset["season"]



# %%
#Originalios aibes padalinimas
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=80085)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=80085)



# %% [markdown]
#   # Dimensijos mažinimas

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
#   ### PCA
# 
# 
# 
#   Praeitame laboratoriniame darbe naudotas tas pats duomenų rinkinys ir ten gauta, kad geriausias dimensijos mažinimo algoritmas yra PCA. Šiame laboratorinyje taip pat naudosime PCA.

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
#   # Atsitiktinių miškų klasifikatorius
# 
# 
# 
# 
# 
# 
# 
#   **Pagrindiniai hiperparametrai:**
# 
# 
# 
#   1. n_estimators
# 
# 
# 
#   2. max_depth
# 
# 
# 
#   3. min_samples_split
# 
# 
# 
#   4. min_samples_leaf
# 
# 
# 
#   5. max_features

# %% [markdown]
#   ## Originali duomenų aibė

# %% [markdown]
#  ### Holdout

# %%
rf_param_grid = {
    'n_estimators': [50, 200, 500],
    'max_depth': [5, 10, None],
    'max_features': ['sqrt', 'log2', 0.5],
}

def rf_holdout(X_train, y_train, X_val, y_val, param_grid, random_state=80085):
    results = []
    for params in ParameterGrid(param_grid):
        rf = RandomForestClassifier(**params, random_state=random_state, n_jobs=-1).fit(X_train, y_train)
        results.append({
            **params,
            'train_acc': rf.score(X_train, y_train),
            'val_acc': rf.score(X_val, y_val),
        })
    return pd.DataFrame(results).sort_values('val_acc', ascending=False).reset_index(drop=True)

holdout_results = rf_holdout(X_train, y_train, X_val, y_val, rf_param_grid)



# %%
holdout_results


# %%
best_holdout_params = holdout_results.iloc[0][["n_estimators", "max_depth", "max_features"]].to_dict()
best_holdout_params["n_estimators"] = int(best_holdout_params["n_estimators"])
best_holdout_params["max_depth"] = None if best_holdout_params["max_depth"] == "None" else int(best_holdout_params["max_depth"])

best_rf_holdout = RandomForestClassifier(
    **best_holdout_params,
    random_state=80085
).fit(X_train, y_train)

test_acc_rf_holdout = best_rf_holdout.score(X_test, y_test)
print(f"Test accuracy HOLDOUT: {test_acc_rf_holdout:.4f}")



# %%
y_test_pred_rf_holdout = best_rf_holdout.predict(X_test)

print(f"Test accuracy HOLDOUT: {best_rf_holdout.score(X_test, y_test):.4f}\n")
print(classification_report(y_test, y_test_pred_rf_holdout, digits=3))


# %% [markdown]
#  ### Kryzmine validacija

# %%
def rf_cv(X, y, param_grid, cv=5, random_state=80085):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []
    for params in ParameterGrid(param_grid):
        rf = RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)
        scores = cross_val_score(rf, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        results.append({
            **params,
            'mean_acc': scores.mean(),
            'std_acc': scores.std(),
        })
    return pd.DataFrame(results).sort_values('mean_acc', ascending=False).reset_index(drop=True)

# sujungiam train ir val nes cv nereikia validaciijos aibes
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

cv_results = rf_cv(X_trainval, y_trainval, rf_param_grid, cv=5)
cv_results


# %%
best_cv_params = cv_results.iloc[0][["n_estimators", "max_depth", "max_features"]].to_dict()
best_cv_params["n_estimators"] = int(best_cv_params["n_estimators"])
best_cv_params["max_depth"] = None if pd.isna(best_cv_params["max_depth"]) else int(best_cv_params["max_depth"])

best_rf_cv = RandomForestClassifier(
    **best_cv_params,
    random_state=80085
).fit(X_trainval, y_trainval)

test_acc_rf_cv = best_rf_cv.score(X_test, y_test)
print(f"Test accuracy CV: {test_acc_rf_cv:.4f}")


# %%
y_test_pred_rf_CV = best_rf_cv.predict(X_test)

print(f"Test accuracy CV: {best_rf_cv.score(X_test, y_test):.4f}\n")
print(classification_report(y_test, y_test_pred_rf_CV, digits=3))


# %% [markdown]
#  ### Bootstrap

# %%
# #KODAS TRUNKA 4 MINUTES RUN AT YOU OWN RISK
# def rf_bootstrap(X, y, param_grid, n_iter=30, random_state=80085):
#     rng = np.random.default_rng(random_state)
#     results = []
    
#     for params in ParameterGrid(param_grid):
#         scores = []
#         for i in range(n_iter):
#             # Atsitiktinis sampling su pakartojimais
#             idx = rng.choice(len(X), size=len(X), replace=True)
#             oob_idx = np.setdiff1d(np.arange(len(X)), idx)
            
#             if len(oob_idx) < 5:
#                 continue
            
#             X_boot = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
#             y_boot = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
#             X_oob = X.iloc[oob_idx] if hasattr(X, "iloc") else X[oob_idx]
#             y_oob = y.iloc[oob_idx] if hasattr(y, "iloc") else y[oob_idx]
            
#             rf = RandomForestClassifier(**params, random_state=random_state, n_jobs=-1).fit(X_boot, y_boot)
#             scores.append(rf.score(X_oob, y_oob))
        
#         results.append({
#             **params,
#             'mean_acc': np.mean(scores),
#             'std_acc': np.std(scores),
#         })
#     return pd.DataFrame(results).sort_values('mean_acc', ascending=False).reset_index(drop=True)

# bootstrap_results = rf_bootstrap(X_trainval, y_trainval, rf_param_grid, n_iter=10)



# %%
# bootstrap_results


# %%
# best_boot_params = bootstrap_results.iloc[0][["n_estimators", "max_depth", "max_features"]].to_dict()
# best_boot_params["n_estimators"] = int(best_boot_params["n_estimators"])
# best_boot_params["max_depth"] = None if pd.isna(best_boot_params["max_depth"]) else int(best_boot_params["max_depth"])

# best_rf_boot = RandomForestClassifier(
#     **best_boot_params,
#     random_state=80085
# ).fit(X_trainval, y_trainval)

# test_acc_rf_boot = best_rf_boot.score(X_test, y_test)
# print(f"Test accuracy BOOTSTRAP: {test_acc_rf_boot:.4f}")


# %%
# y_test_pred_rf_boot = best_rf_boot.predict(X_test)

# print(f"Test accuracy BOOTSTRAP: {best_rf_boot.score(X_test, y_test):.4f}\n")
# print(classification_report(y_test, y_test_pred_rf_boot, digits=3))


# %%
#LYGINAM testines
print("=== HOLDOUT ===")
print(test_acc_rf_holdout)
print("\n=== 5-FOLD CV ===")
print(test_acc_rf_cv)
print("\n=== BOOTSTRAP ===")
# print(test_acc_rf_boot) #=> geriausias modelis holdout ir cv


# %% [markdown]
#  ### ROC

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_roc_curves_combined(models_dict, X_test, y_test, suptitle="ROC kreivių palyginimas"):
    season_lt = {"Winter": "Žiema", "Spring": "Pavasaris",
                 "Summer": "Vasara", "Autumn": "Ruduo"}
    season_colors = {"Winter": "#4C78A8", "Spring": "#59A14F",
                     "Summer": "#F28E2B", "Autumn": "#9C755F"}
    classes = ["Winter", "Spring", "Summer", "Autumn"]
    
    y_test_bin = label_binarize(y_test, classes=classes)
    
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(9 * n_models, 9))
    if n_models == 1:
        axes = [axes]
    
    auc_scores_all = {}
    
    for ax, (name, model) in zip(axes, models_dict.items()):
        model_classes = list(model.classes_)
        class_order_idx = [model_classes.index(c) for c in classes]
        y_score = model.predict_proba(X_test)[:, class_order_idx]
        
        auc_scores = {}
        for i, season in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores[season] = roc_auc
            
            ax.plot(fpr, tpr, color=season_colors[season], linewidth=3,
                    label=f"{season_lt[season]} (AUC = {roc_auc:.3f})")
        
        auc_scores_all[name] = auc_scores
        macro_auc = np.mean(list(auc_scores.values()))
        
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, label="Atsitiktinis")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("FPR", fontsize=16)
        ax.set_ylabel("TPR", fontsize=16)
        ax.set_title(f"{name}\nMacro AUC = {macro_auc:.3f}", fontsize=18)
        ax.legend(loc="lower right", fontsize=16)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=16)
        ax.set_aspect("equal")
    
    plt.suptitle(suptitle, fontsize=20)
    plt.tight_layout()
    plt.show()
    
    return auc_scores_all


# %%
auc_all = plot_roc_curves_combined(
    {
        "Holdout": best_rf_holdout,
        "CV": best_rf_cv,
        "Bootstrap": best_rf_boot,
    },
    X_test, y_test,
    suptitle=""
)


# %% [markdown]
#  #TODO: pasitarti kuri strategija geriausia
# 
#  Pagal ROC kreives geriausias yra bootstrap modelis - didziausias macro AUC

# %%
cm_test_rf_holdout = confusion_matrix(y_test, y_test_pred_rf_holdout, labels=["Winter", "Spring", "Summer", "Autumn"])
disp_rf_holdout = ConfusionMatrixDisplay(
    cm_test_rf_holdout,
    display_labels=["Žiema", "Pavasaris", "Vasara", "Ruduo"]
)
disp_rf_holdout.plot(cmap="Blues")
disp_rf_holdout.ax_.set_xlabel("Prognozuota klasė")
disp_rf_holdout.ax_.set_ylabel("Tikroji klasė")
plt.title("Atsitiktinių miškų sumaišymo matrica - holdout")
plt.tight_layout()
plt.show()


# %%
cm_test_rf_CV = confusion_matrix(y_test, y_test_pred_rf_CV, labels=["Winter", "Spring", "Summer", "Autumn"])
disp_rf_CV = ConfusionMatrixDisplay(
    cm_test_rf_CV,
    display_labels=["Žiema", "Pavasaris", "Vasara", "Ruduo"]
)
disp_rf_CV.plot(cmap="Blues")
disp_rf_CV.ax_.set_xlabel("Prognozuota klasė")
disp_rf_CV.ax_.set_ylabel("Tikroji klasė")
plt.title("Atsitiktinių miškų sumaišymo matrica - kryžminė validacija")
plt.tight_layout()
plt.show()


# %%
cm_test_rf_boot = confusion_matrix(y_test, y_test_pred_rf_boot, labels=["Winter", "Spring", "Summer", "Autumn"])
disp_rf_boot = ConfusionMatrixDisplay(
    cm_test_rf_boot,
    display_labels=["Žiema", "Pavasaris", "Vasara", "Ruduo"]
)
disp_rf_boot.plot(cmap="Blues")
disp_rf_boot.ax_.set_xlabel("Prognozuota klasė")
disp_rf_boot.ax_.set_ylabel("Tikroji klasė")
plt.title("Atsitiktinių miškų sumaišymo matrica - bootstrap")
plt.tight_layout()
plt.show()



# %% [markdown]
#  ### Klaidu tyrimas
# 
#  Dabar fiksuotas bootstrap'as

# %%
#Neteisingai suklasifikuoti tašku tyrimas
mistakes_rf = X_test.copy()
mistakes_rf.insert(0, "True", np.array(y_test))
mistakes_rf.insert(1, "Predicted", y_test_pred_rf_boot)
mistakes_rf = mistakes_rf[mistakes_rf["True"] != mistakes_rf["Predicted"]]
mistakes_rf


# %%
def plot_misclassified_profiles(mistakes_df, final_dataset, time_cols, suptitle):
    season_lt = {"Winter": "Žiema", "Spring": "Pavasaris", 
                 "Summer": "Vasara", "Autumn": "Ruduo"}
    
    # Sezonų vidurkiai iš VISO dataset'o
    season_means = {}
    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        mask = final_dataset["season"] == season
        season_means[season] = final_dataset.loc[mask, time_cols].mean(axis=0).values
    
    n_show = len(mistakes_df)
    
    # Renkam optimalų grid - kuo "kvadratiškesnis", tuo geriau
    n_cols = int(np.ceil(np.sqrt(n_show)))
    n_rows = int(np.ceil(n_show / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), sharey=True)
    axes = np.array(axes).ravel() if n_show > 1 else [axes]
    
    for ax, (idx, row) in zip(axes, mistakes_df.iterrows()):
        true_lbl = row["True"]
        pred_lbl = row["Predicted"]
        profile = row[time_cols].values.astype(float)
        
        ax.plot(time_cols, profile, "k-", linewidth=2.5, label="Prognozuota diena")
        ax.plot(time_cols, season_means[true_lbl], "g--", linewidth=1.8,
                label=f"Vid. {season_lt[true_lbl]} (T)")
        ax.plot(time_cols, season_means[pred_lbl], "r:", linewidth=1.8,
                label=f"Vid. {season_lt[pred_lbl]} (P)")
        
        ax.set_title(f"Tikra: {season_lt[true_lbl]} → Prognozuota: {season_lt[pred_lbl]}", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_aspect("auto")
        ax.set_box_aspect(1)
        
        tick_step = 3
        tick_positions = list(range(0, len(time_cols), tick_step))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([time_cols[i] for i in tick_positions], rotation=45, fontsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_xlabel("Valanda", fontsize=12)
    
    for ax in axes[n_show:]:
        ax.set_visible(False)
    
    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    plt.show()


# %%
time_cols = [c for c in mistakes_rf.columns if c not in ["True", "Predicted"]]
plot_misclassified_profiles(
    mistakes_rf, final_dataset, time_cols,
    suptitle=""
)


# %%
mistakes_summary_rf = pd.DataFrame({
    "Data": final_dataset.loc[mistakes_rf.index, "Day"].values,
    "Tikra": mistakes_rf["True"].values,
    "Prognozuota": mistakes_rf["Predicted"].values,
})
mistakes_summary_rf



# %% [markdown]
#  ## Dviejų dimensijų aibė

# %% [markdown]
#  ### Holdout

# %%
holdout_results_pca = rf_holdout(X_train_pca, y_train, X_val_pca, y_val, rf_param_grid)


# %%
holdout_results_pca


# %% [markdown]
#   Prasti popieriai, PCA duomenų aibė labai pablogina rezultatus - validacijos aibė realiai spėlioja duomenis, o klasifikacvimo tikslumas (ten kur validacija geriausia) sieki tik 0,767...

# %%
best_holdout_params_pca = holdout_results_pca.iloc[0][["n_estimators", "max_depth", "max_features"]].to_dict()
best_holdout_params_pca["n_estimators"] = int(best_holdout_params_pca["n_estimators"])
best_holdout_params_pca["max_depth"] = None if best_holdout_params_pca["max_depth"] == "None" else int(best_holdout_params_pca["max_depth"])

best_rf_holdout_pca = RandomForestClassifier(
    **best_holdout_params_pca,
    random_state=80085
).fit(X_train_pca, y_train)

test_acc_rf_holdout = best_rf_holdout_pca.score(X_test_pca, y_test)
print(f"Test accuracy HOLDOUT PCA: {test_acc_rf_holdout:.4f}")



# %%
y_test_pred_rf_pca_holdout = best_rf_holdout_pca.predict(X_test_pca)
test_acc_rf_pca = best_rf_holdout_pca.score(X_test_pca, y_test)

print(f"Test accuracy HOLDOUT PCA: {test_acc_rf_pca :.4f}\n")
print(classification_report(y_test, y_test_pred_rf_pca_holdout, digits=3))


# %% [markdown]
#  ### Kryzmine validavija

# %%
X_trainval_pca = np.concatenate([X_train_pca, X_val_pca])
y_trainval = pd.concat([y_train, y_val])
cv_results_pca = rf_cv(X_trainval_pca, y_trainval, rf_param_grid, cv=5)


# %%
cv_results_pca


# %%
best_cv_params_pca = cv_results_pca.iloc[0][["n_estimators", "max_depth", "max_features"]].to_dict()
best_cv_params_pca["n_estimators"] = int(best_cv_params_pca["n_estimators"])
best_cv_params_pca["max_depth"] = None if best_cv_params_pca["max_depth"] == "None" else int(best_cv_params_pca["max_depth"])

best_rf_cv_pca = RandomForestClassifier(
    **best_cv_params_pca,
    random_state=80085
).fit(X_trainval_pca, y_trainval)

test_acc_rf_cv = best_rf_cv_pca.score(X_test_pca, y_test)
print(f"Test accuracy CV PCA: {test_acc_rf_cv:.4f}")


# %%
y_test_pred_rf_pca_cv = best_rf_cv_pca.predict(X_test_pca)
test_acc_rf_pca = best_rf_cv_pca.score(X_test_pca, y_test)

print(f"Test accuracy CV PCA: {test_acc_rf_pca :.4f}\n")
print(classification_report(y_test, y_test_pred_rf_pca_cv, digits=3))


# %% [markdown]
#  ### Bootstraping

# %%
# boot_results_pca = rf_bootstrap(X_trainval_pca, y_trainval, rf_param_grid, n_iter=10)


# %%
# boot_results_pca


# %%
# best_boot_params_pca = boot_results_pca.iloc[0][["n_estimators", "max_depth", "max_features"]].to_dict()
# best_boot_params_pca["n_estimators"] = int(best_boot_params_pca["n_estimators"])
# best_boot_params_pca["max_depth"] = None if pd.isna(best_boot_params_pca["max_depth"]) else int(best_boot_params_pca["max_depth"])

# best_rf_boot_pca = RandomForestClassifier(
#     **best_boot_params_pca,
#     random_state=80085
# ).fit(X_trainval_pca, y_trainval)

# test_acc_rf_boot_pca = best_rf_boot_pca.score(X_test_pca, y_test)
# print(f"Test accuracy BOOTSTRAP PCA: {test_acc_rf_boot_pca:.4f}")


# %%
# y_test_pred_rf_pca_boot = best_rf_boot_pca.predict(X_test_pca)
# test_acc_rf_pca = best_rf_boot_pca.score(X_test_pca, y_test)

# print(f"Test accuracy BOOTSTRAP PCA: {test_acc_rf_pca :.4f}\n")
# print(classification_report(y_test, y_test_pred_rf_pca_boot, digits=3))


# %% [markdown]
#  ## ROC kreives

# %%
auc_all = plot_roc_curves_combined(
    {
        "Holdout": best_rf_holdout_pca,
        "CV": best_rf_cv_pca,
        # "Bootstrap": best_rf_boot_pca,
    },
    X_test_pca, y_test,
    suptitle=""
)


# %% [markdown]
#  Pagal ROC kreive PCA geriausias yra holdout'as

# %%
#holdout
cm_test_rf_pca_holdout = confusion_matrix(y_test, y_test_pred_rf_pca_holdout, labels=["Winter", "Spring", "Summer", "Autumn"])
disp_rf_pca_holdout = ConfusionMatrixDisplay(
    cm_test_rf_pca_holdout,
    display_labels=["Žiema", "Pavasaris", "Vasara", "Ruduo"]
)
disp_rf_pca_holdout.plot(cmap="Blues")
disp_rf_pca_holdout.ax_.set_xlabel("Prognozuota klasė")
disp_rf_pca_holdout.ax_.set_ylabel("Tikroji klasė")
plt.title("Atsitiktinių miškų sumaišymo matrica - holdout PCA")
plt.tight_layout()
plt.show()


# %%
#cv
cm_test_rf_pca_cv = confusion_matrix(y_test, y_test_pred_rf_pca_cv, labels=["Winter", "Spring", "Summer", "Autumn"])
disp_rf_pca_cv = ConfusionMatrixDisplay(
    cm_test_rf_pca_cv,
    display_labels=["Žiema", "Pavasaris", "Vasara", "Ruduo"]
)
disp_rf_pca_cv.plot(cmap="Blues")
disp_rf_pca_cv.ax_.set_xlabel("Prognozuota klasė")
disp_rf_pca_cv.ax_.set_ylabel("Tikroji klasė")
plt.title("Atsitiktinių miškų sumaišymo matrica - kryžminė validacija PCA")
plt.tight_layout()
plt.show()


# %%
#bootstrap
# cm_test_rf_pca_bootstrap = confusion_matrix(y_test, y_test_pred_rf_pca_boot, labels=["Winter", "Spring", "Summer", "Autumn"])
# disp_rf_pca_bootstrap = ConfusionMatrixDisplay(
#     cm_test_rf_pca_bootstrap,
#     display_labels=["Žiema", "Pavasaris", "Vasara", "Ruduo"]
# )
# disp_rf_pca_bootstrap.plot(cmap="Blues")
# disp_rf_pca_bootstrap.ax_.set_xlabel("Prognozuota klasė")
# disp_rf_pca_bootstrap.ax_.set_ylabel("Tikroji klasė")
# plt.title("Atsitiktinių miškų sumaišymo matrica - bootstrap PCA")
# plt.tight_layout()
# plt.show()



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
#holdout PCA
plot_classification_pca(X_test_pca, y_test, y_test_pred_rf_pca_holdout, title="Atsitiktinių miškų klasifikavimas PCA erdvėje - holdout")


# %%
#cv pca
plot_classification_pca(X_test_pca, y_test, y_test_pred_rf_pca_cv, title="Atsitiktinių miškų klasifikavimas PCA erdvėje - kryžminė validacija")


# %%
#bootstrap pca
# plot_classification_pca(X_test_pca, y_test, y_test_pred_rf_pca_boot, title="Atsitiktinių miškų klasifikavimas PCA erdvėje - bootstrap")



# %% [markdown]
#  ### Klaidu analize

# %%
#Neteisingai suklasifikuoti tašku tyrimas
X_test_pca_df = pd.DataFrame(X_test_pca, columns=["PC1", "PC2"], index=X_test.index)
mistakes_rf_pca = X_test_pca_df.join(X_test)
mistakes_rf_pca.insert(0, "True", np.array(y_test))
mistakes_rf_pca.insert(1, "Predicted", y_test_pred_rf_pca_holdout)
mistakes_rf_pca = mistakes_rf_pca[mistakes_rf_pca["True"] != mistakes_rf_pca["Predicted"]]
mistakes_rf_pca.drop(columns=["PC1", "PC2"], inplace=True)
mistakes_rf_pca


# %%
plot_misclassified_profiles(
    mistakes_rf_pca, final_dataset, time_cols,
    suptitle=""
)


# %%
mistakes_summary_rf_pca = pd.DataFrame({
    "Data": final_dataset.loc[mistakes_rf_pca.index, "Day"].values,
    "Tikra": mistakes_rf_pca["True"].values,
    "Prognozuota": mistakes_rf_pca["Predicted"].values,
})
mistakes_summary_rf_pca


# %% [markdown]
# # KNN klasifikatorius
#
# **Pagrindiniai hiperparametrai:**
#
# 1. n_neighbors
# 2. weights
# 3. metric

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import RobustScaler, label_binarize
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
knn_param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 15],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

# %% [markdown]
# ## Originali duomenų aibė

# %% [markdown]
# ### Duomenų standartizavimas KNN modeliui

# %%
scaler_knn = RobustScaler()

X_train_knn = scaler_knn.fit_transform(X_train)
X_val_knn   = scaler_knn.transform(X_val)
X_test_knn  = scaler_knn.transform(X_test)

# %% [markdown]
# ### Holdout

# %%
def knn_holdout(X_train, y_train, X_val, y_val, param_grid):
    results = []

    for params in ParameterGrid(param_grid):
        knn = KNeighborsClassifier(**params)
        knn.fit(X_train, y_train)

        results.append({
            **params,
            "train_acc": knn.score(X_train, y_train),
            "val_acc": knn.score(X_val, y_val),
        })

    return pd.DataFrame(results).sort_values(
        "val_acc", ascending=False
    ).reset_index(drop=True)

# %%
holdout_results_knn = knn_holdout(
    X_train_knn, y_train,
    X_val_knn, y_val,
    knn_param_grid
)

holdout_results_knn

# %%
best_holdout_params_knn = holdout_results_knn.iloc[0][
    ["n_neighbors", "weights", "metric"]
].to_dict()

best_holdout_params_knn["n_neighbors"] = int(best_holdout_params_knn["n_neighbors"])

best_knn_holdout = KNeighborsClassifier(
    **best_holdout_params_knn
).fit(X_train_knn, y_train)

test_acc_knn_holdout = best_knn_holdout.score(X_test_knn, y_test)

print(f"Test accuracy HOLDOUT: {test_acc_knn_holdout:.4f}")

# %%
y_test_pred_knn_holdout = best_knn_holdout.predict(X_test_knn)

print(f"Test accuracy HOLDOUT: {best_knn_holdout.score(X_test_knn, y_test):.4f}\n")
print(classification_report(y_test, y_test_pred_knn_holdout, digits=3))

# %% [markdown]
# ### Kryžminė validacija

# %%
def knn_cv(X, y, param_grid, cv=5, random_state=80085):
    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state
    )

    results = []

    for params in ParameterGrid(param_grid):
        knn = KNeighborsClassifier(**params)

        scores = cross_val_score(
            knn, X, y,
            cv=skf,
            scoring="accuracy",
            n_jobs=-1
        )

        results.append({
            **params,
            "mean_acc": scores.mean(),
            "std_acc": scores.std(),
        })

    return pd.DataFrame(results).sort_values(
        "mean_acc", ascending=False
    ).reset_index(drop=True)

# %%
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

scaler_knn_cv = RobustScaler()
X_trainval_knn = scaler_knn_cv.fit_transform(X_trainval)
X_test_knn_cv  = scaler_knn_cv.transform(X_test)

# %%
cv_results_knn = knn_cv(
    X_trainval_knn,
    y_trainval,
    knn_param_grid,
    cv=5
)

cv_results_knn

# %%
best_cv_params_knn = cv_results_knn.iloc[0][
    ["n_neighbors", "weights", "metric"]
].to_dict()

best_cv_params_knn["n_neighbors"] = int(best_cv_params_knn["n_neighbors"])

best_knn_cv = KNeighborsClassifier(
    **best_cv_params_knn
).fit(X_trainval_knn, y_trainval)

test_acc_knn_cv = best_knn_cv.score(X_test_knn_cv, y_test)

print(f"Test accuracy CV: {test_acc_knn_cv:.4f}")

# %%
y_test_pred_knn_cv = best_knn_cv.predict(X_test_knn_cv)

print(f"Test accuracy CV: {best_knn_cv.score(X_test_knn_cv, y_test):.4f}\n")
print(classification_report(y_test, y_test_pred_knn_cv, digits=3))

# %% [markdown]
# ### Bootstrap

# %%
# # KODAS GALI TRUKTI ILGIAU
# def knn_bootstrap(X, y, param_grid, n_iter=30, random_state=80085):
#     rng = np.random.default_rng(random_state)
#     results = []
#
#     for params in ParameterGrid(param_grid):
#         scores = []
#
#         for i in range(n_iter):
#             idx = rng.choice(len(X), size=len(X), replace=True)
#             oob_idx = np.setdiff1d(np.arange(len(X)), idx)
#
#             if len(oob_idx) < 5:
#                 continue
#
#             X_boot = X[idx]
#             y_boot = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
#             X_oob = X[oob_idx]
#             y_oob = y.iloc[oob_idx] if hasattr(y, "iloc") else y[oob_idx]
#
#             knn = KNeighborsClassifier(**params)
#             knn.fit(X_boot, y_boot)
#             scores.append(knn.score(X_oob, y_oob))
#
#         results.append({
#             **params,
#             "mean_acc": np.mean(scores),
#             "std_acc": np.std(scores),
#         })
#
#     return pd.DataFrame(results).sort_values(
#         "mean_acc", ascending=False
#     ).reset_index(drop=True)

# %%
# bootstrap_results_knn = knn_bootstrap(
#     X_trainval_knn,
#     y_trainval,
#     knn_param_grid,
#     n_iter=10
# )
#
# bootstrap_results_knn

# %%
# best_boot_params_knn = bootstrap_results_knn.iloc[0][
#     ["n_neighbors", "weights", "metric"]
# ].to_dict()
#
# best_boot_params_knn["n_neighbors"] = int(best_boot_params_knn["n_neighbors"])
#
# best_knn_boot = KNeighborsClassifier(
#     **best_boot_params_knn
# ).fit(X_trainval_knn, y_trainval)
#
# test_acc_knn_boot = best_knn_boot.score(X_test_knn_cv, y_test)
#
# print(f"Test accuracy BOOTSTRAP: {test_acc_knn_boot:.4f}")

# %%
# y_test_pred_knn_boot = best_knn_boot.predict(X_test_knn_cv)
#
# print(f"Test accuracy BOOTSTRAP: {best_knn_boot.score(X_test_knn_cv, y_test):.4f}\n")
# print(classification_report(y_test, y_test_pred_knn_boot, digits=3))

# %% [markdown]
# ### Modelių palyginimas

# %%
print("=== HOLDOUT ===")
print(test_acc_knn_holdout)

print("\n=== 5-FOLD CV ===")
print(test_acc_knn_cv)

print("\n=== BOOTSTRAP ===")
# print(test_acc_knn_boot)

# %% [markdown]
# ### ROC kreivės

# %%
def plot_roc_curves_combined(models_dict, X_test, y_test, suptitle="ROC kreivių palyginimas"):
    season_lt = {
        "Winter": "Žiema",
        "Spring": "Pavasaris",
        "Summer": "Vasara",
        "Autumn": "Ruduo"
    }

    season_colors = {
        "Winter": "#4C78A8",
        "Spring": "#59A14F",
        "Summer": "#F28E2B",
        "Autumn": "#9C755F"
    }

    classes = ["Winter", "Spring", "Summer", "Autumn"]

    y_test_bin = label_binarize(y_test, classes=classes)

    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(9 * n_models, 9))

    if n_models == 1:
        axes = [axes]

    auc_scores_all = {}

    for ax, (name, model) in zip(axes, models_dict.items()):
        model_classes = list(model.classes_)
        class_order_idx = [model_classes.index(c) for c in classes]

        y_score = model.predict_proba(X_test)[:, class_order_idx]

        auc_scores = {}

        for i, season in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores[season] = roc_auc

            ax.plot(
                fpr, tpr,
                color=season_colors[season],
                linewidth=3,
                label=f"{season_lt[season]} (AUC = {roc_auc:.3f})"
            )

        auc_scores_all[name] = auc_scores
        macro_auc = np.mean(list(auc_scores.values()))

        ax.plot(
            [0, 1], [0, 1],
            "k--",
            linewidth=1.5,
            alpha=0.5,
            label="Atsitiktinis"
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("FPR", fontsize=16)
        ax.set_ylabel("TPR", fontsize=16)
        ax.set_title(f"{name}\nMacro AUC = {macro_auc:.3f}", fontsize=18)
        ax.legend(loc="lower right", fontsize=16)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="both", labelsize=16)
        ax.set_aspect("equal")

    plt.suptitle(suptitle, fontsize=20)
    plt.tight_layout()
    plt.show()

    return auc_scores_all

# %%
auc_all_knn = plot_roc_curves_combined(
    {
        "Holdout": best_knn_holdout,
        "CV": best_knn_cv,
        # "Bootstrap": best_knn_boot,
    },
    X_test_knn,
    y_test,
    suptitle=""
)

# %% [markdown]
# ### Sumaišymo matricos

# %%
cm_test_knn_holdout = confusion_matrix(
    y_test,
    y_test_pred_knn_holdout,
    labels=["Winter", "Spring", "Summer", "Autumn"]
)

disp_knn_holdout = ConfusionMatrixDisplay(
    cm_test_knn_holdout,
    display_labels=["Žiema", "Pavasaris", "Vasara", "Ruduo"]
)

disp_knn_holdout.plot(cmap="Blues")
disp_knn_holdout.ax_.set_xlabel("Prognozuota klasė")
disp_knn_holdout.ax_.set_ylabel("Tikroji klasė")
plt.title("KNN sumaišymo matrica - holdout")
plt.tight_layout()
plt.show()

# %%
cm_test_knn_cv = confusion_matrix(
    y_test,
    y_test_pred_knn_cv,
    labels=["Winter", "Spring", "Summer", "Autumn"]
)

disp_knn_cv = ConfusionMatrixDisplay(
    cm_test_knn_cv,
    display_labels=["Žiema", "Pavasaris", "Vasara", "Ruduo"]
)

disp_knn_cv.plot(cmap="Blues")
disp_knn_cv.ax_.set_xlabel("Prognozuota klasė")
disp_knn_cv.ax_.set_ylabel("Tikroji klasė")
plt.title("KNN sumaišymo matrica - kryžminė validacija")
plt.tight_layout()
plt.show()

# %%
# cm_test_knn_boot = confusion_matrix(
#     y_test,
#     y_test_pred_knn_boot,
#     labels=["Winter", "Spring", "Summer", "Autumn"]
# )
#
# disp_knn_boot = ConfusionMatrixDisplay(
#     cm_test_knn_boot,
#     display_labels=["Žiema", "Pavasaris", "Vasara", "Ruduo"]
# )
#
# disp_knn_boot.plot(cmap="Blues")
# disp_knn_boot.ax_.set_xlabel("Prognozuota klasė")
# disp_knn_boot.ax_.set_ylabel("Tikroji klasė")
# plt.title("KNN sumaišymo matrica - bootstrap")
# plt.tight_layout()
# plt.show()

# %% [markdown]
# ### Klaidų analizė
#
# Toliau fiksuojamas CV modelis.

# %%
mistakes_knn = X_test.copy()
mistakes_knn.insert(0, "True", np.array(y_test))
mistakes_knn.insert(1, "Predicted", y_test_pred_knn_cv)

mistakes_knn = mistakes_knn[
    mistakes_knn["True"] != mistakes_knn["Predicted"]
]

mistakes_knn

# %%
def plot_misclassified_profiles(mistakes_df, final_dataset, time_cols, suptitle):
    season_lt = {
        "Winter": "Žiema",
        "Spring": "Pavasaris",
        "Summer": "Vasara",
        "Autumn": "Ruduo"
    }

    season_means = {}

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        mask = final_dataset["season"] == season
        season_means[season] = final_dataset.loc[
            mask, time_cols
        ].mean(axis=0).values

    n_show = len(mistakes_df)

    n_cols = int(np.ceil(np.sqrt(n_show)))
    n_rows = int(np.ceil(n_show / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        sharey=True
    )

    axes = np.array(axes).ravel() if n_show > 1 else [axes]

    for ax, (idx, row) in zip(axes, mistakes_df.iterrows()):
        true_lbl = row["True"]
        pred_lbl = row["Predicted"]
        profile = row[time_cols].values.astype(float)

        ax.plot(
            time_cols,
            profile,
            "k-",
            linewidth=2.5,
            label="Prognozuota diena"
        )

        ax.plot(
            time_cols,
            season_means[true_lbl],
            "g--",
            linewidth=1.8,
            label=f"Vid. {season_lt[true_lbl]} (T)"
        )

        ax.plot(
            time_cols,
            season_means[pred_lbl],
            "r:",
            linewidth=1.8,
            label=f"Vid. {season_lt[pred_lbl]} (P)"
        )

        ax.set_title(
            f"Tikra: {season_lt[true_lbl]} → Prognozuota: {season_lt[pred_lbl]}",
            fontsize=13
        )

        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_aspect("auto")
        ax.set_box_aspect(1)

        tick_step = 3
        tick_positions = list(range(0, len(time_cols), tick_step))

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [time_cols[i] for i in tick_positions],
            rotation=45,
            fontsize=11
        )

        ax.tick_params(axis="y", labelsize=11)
        ax.set_xlabel("Valanda", fontsize=12)

    for ax in axes[n_show:]:
        ax.set_visible(False)

    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()
    plt.show()

# %%
time_cols = [c for c in mistakes_knn.columns if c not in ["True", "Predicted"]]

plot_misclassified_profiles(
    mistakes_knn,
    final_dataset,
    time_cols,
    suptitle=""
)

# %%
mistakes_summary_knn = pd.DataFrame({
    "Data": final_dataset.loc[mistakes_knn.index, "Day"].values,
    "Tikra": mistakes_knn["True"].values,
    "Prognozuota": mistakes_knn["Predicted"].values,
})

mistakes_summary_knn

# %% [markdown]
# ## Dviejų dimensijų aibė

# %% [markdown]
# ### Holdout

# %%
holdout_results_knn_pca = knn_holdout(
    X_train_pca,
    y_train,
    X_val_pca,
    y_val,
    knn_param_grid
)

holdout_results_knn_pca

# %%
best_holdout_params_knn_pca = holdout_results_knn_pca.iloc[0][
    ["n_neighbors", "weights", "metric"]
].to_dict()

best_holdout_params_knn_pca["n_neighbors"] = int(
    best_holdout_params_knn_pca["n_neighbors"]
)

best_knn_holdout_pca = KNeighborsClassifier(
    **best_holdout_params_knn_pca
).fit(X_train_pca, y_train)

test_acc_knn_holdout_pca = best_knn_holdout_pca.score(X_test_pca, y_test)

print(f"Test accuracy HOLDOUT PCA: {test_acc_knn_holdout_pca:.4f}")

# %%
y_test_pred_knn_pca_holdout = best_knn_holdout_pca.predict(X_test_pca)

print(f"Test accuracy HOLDOUT PCA: {best_knn_holdout_pca.score(X_test_pca, y_test):.4f}\n")
print(classification_report(y_test, y_test_pred_knn_pca_holdout, digits=3))

# %% [markdown]
# ### Kryžminė validacija

# %%
X_trainval_pca = np.concatenate([X_train_pca, X_val_pca])
y_trainval = pd.concat([y_train, y_val])

# %%
cv_results_knn_pca = knn_cv(
    X_trainval_pca,
    y_trainval,
    knn_param_grid,
    cv=5
)

cv_results_knn_pca

# %%
best_cv_params_knn_pca = cv_results_knn_pca.iloc[0][
    ["n_neighbors", "weights", "metric"]
].to_dict()

best_cv_params_knn_pca["n_neighbors"] = int(
    best_cv_params_knn_pca["n_neighbors"]
)

best_knn_cv_pca = KNeighborsClassifier(
    **best_cv_params_knn_pca
).fit(X_trainval_pca, y_trainval)

test_acc_knn_cv_pca = best_knn_cv_pca.score(X_test_pca, y_test)

print(f"Test accuracy CV PCA: {test_acc_knn_cv_pca:.4f}")

# %%
y_test_pred_knn_pca_cv = best_knn_cv_pca.predict(X_test_pca)

print(f"Test accuracy CV PCA: {best_knn_cv_pca.score(X_test_pca, y_test):.4f}\n")
print(classification_report(y_test, y_test_pred_knn_pca_cv, digits=3))

# %% [markdown]
# ### Bootstrap

# %%
# bootstrap_results_knn_pca = knn_bootstrap(
#     X_trainval_pca,
#     y_trainval,
#     knn_param_grid,
#     n_iter=10
# )
#
# bootstrap_results_knn_pca

# %%
# best_boot_params_knn_pca = bootstrap_results_knn_pca.iloc[0][
#     ["n_neighbors", "weights", "metric"]
# ].to_dict()
#
# best_boot_params_knn_pca["n_neighbors"] = int(
#     best_boot_params_knn_pca["n_neighbors"]
# )
#
# best_knn_boot_pca = KNeighborsClassifier(
#     **best_boot_params_knn_pca
# ).fit(X_trainval_pca, y_trainval)
#
# test_acc_knn_boot_pca = best_knn_boot_pca.score(X_test_pca, y_test)
#
# print(f"Test accuracy BOOTSTRAP PCA: {test_acc_knn_boot_pca:.4f}")

# %%
# y_test_pred_knn_pca_boot = best_knn_boot_pca.predict(X_test_pca)
#
# print(f"Test accuracy BOOTSTRAP PCA: {best_knn_boot_pca.score(X_test_pca, y_test):.4f}\n")
# print(classification_report(y_test, y_test_pred_knn_pca_boot, digits=3))

# %% [markdown]
# ### ROC kreivės PCA aibei

# %%
auc_all_knn_pca = plot_roc_curves_combined(
    {
        "Holdout": best_knn_holdout_pca,
        "CV": best_knn_cv_pca,
        # "Bootstrap": best_knn_boot_pca,
    },
    X_test_pca,
    y_test,
    suptitle=""
)

# %% [markdown]
# ### Sumaišymo matricos PCA aibei

# %%
cm_test_knn_pca_holdout = confusion_matrix(
    y_test,
    y_test_pred_knn_pca_holdout,
    labels=["Winter", "Spring", "Summer", "Autumn"]
)

disp_knn_pca_holdout = ConfusionMatrixDisplay(
    cm_test_knn_pca_holdout,
    display_labels=["Žiema", "Pavasaris", "Vasara", "Ruduo"]
)

disp_knn_pca_holdout.plot(cmap="Blues")
disp_knn_pca_holdout.ax_.set_xlabel("Prognozuota klasė")
disp_knn_pca_holdout.ax_.set_ylabel("Tikroji klasė")
plt.title("KNN sumaišymo matrica - holdout PCA")
plt.tight_layout()
plt.show()

# %%
cm_test_knn_pca_cv = confusion_matrix(
    y_test,
    y_test_pred_knn_pca_cv,
    labels=["Winter", "Spring", "Summer", "Autumn"]
)

disp_knn_pca_cv = ConfusionMatrixDisplay(
    cm_test_knn_pca_cv,
    display_labels=["Žiema", "Pavasaris", "Vasara", "Ruduo"]
)

disp_knn_pca_cv.plot(cmap="Blues")
disp_knn_pca_cv.ax_.set_xlabel("Prognozuota klasė")
disp_knn_pca_cv.ax_.set_ylabel("Tikroji klasė")
plt.title("KNN sumaišymo matrica - kryžminė validacija PCA")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Klasifikavimo rezultatai PCA erdvėje

# %%
def plot_classification_pca(X_test_pca, y_test, y_pred, title):
    season_colors_map = {
        "Winter": "#4C78A8",
        "Spring": "#59A14F",
        "Summer": "#F28E2B",
        "Autumn": "#9C755F",
    }

    season_lt = {
        "Winter": "Žiema",
        "Spring": "Pavasaris",
        "Summer": "Vasara",
        "Autumn": "Ruduo"
    }

    y_test_arr = np.array(y_test)
    correct = y_test_arr == np.array(y_pred)

    lim_min = np.floor(
        min(X_test_pca[:, 0].min(), X_test_pca[:, 1].min())
    ) - 1

    lim_max = np.ceil(
        max(X_test_pca[:, 0].max(), X_test_pca[:, 1].max())
    ) + 1

    plt.figure(figsize=(9, 9))

    for season in ["Winter", "Spring", "Summer", "Autumn"]:
        mask = y_test_arr == season

        plt.scatter(
            X_test_pca[mask, 0],
            X_test_pca[mask, 1],
            c=season_colors_map[season],
            label=season_lt[season],
            s=70,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

    plt.scatter(
        X_test_pca[~correct, 0],
        X_test_pca[~correct, 1],
        facecolors="none",
        edgecolors="red",
        s=200,
        linewidths=2,
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
plot_classification_pca(
    X_test_pca,
    y_test,
    y_test_pred_knn_pca_holdout,
    title="KNN klasifikavimas PCA erdvėje - holdout"
)

# %%
plot_classification_pca(
    X_test_pca,
    y_test,
    y_test_pred_knn_pca_cv,
    title="KNN klasifikavimas PCA erdvėje - kryžminė validacija"
)

# %%
# plot_classification_pca(
#     X_test_pca,
#     y_test,
#     y_test_pred_knn_pca_boot,
#     title="KNN klasifikavimas PCA erdvėje - bootstrap"
# )

# %% [markdown]
# ### Klaidų analizė PCA aibei
#
# Toliau fiksuojamas holdout PCA modelis.

# %%
X_test_pca_df = pd.DataFrame(
    X_test_pca,
    columns=["PC1", "PC2"],
    index=X_test.index
)

mistakes_knn_pca = X_test_pca_df.join(X_test)

mistakes_knn_pca.insert(0, "True", np.array(y_test))
mistakes_knn_pca.insert(1, "Predicted", y_test_pred_knn_pca_holdout)

mistakes_knn_pca = mistakes_knn_pca[
    mistakes_knn_pca["True"] != mistakes_knn_pca["Predicted"]
]

mistakes_knn_pca.drop(columns=["PC1", "PC2"], inplace=True)

mistakes_knn_pca

# %%
plot_misclassified_profiles(
    mistakes_knn_pca,
    final_dataset,
    time_cols,
    suptitle=""
)

# %%
mistakes_summary_knn_pca = pd.DataFrame({
    "Data": final_dataset.loc[mistakes_knn_pca.index, "Day"].values,
    "Tikra": mistakes_knn_pca["True"].values,
    "Prognozuota": mistakes_knn_pca["Predicted"].values,
})

mistakes_summary_knn_pca

# %% [markdown]
#  # Support vector classifier

# %% [markdown]
#  ## Tinklelio paieška

# %%

pipe = Pipeline([
    ("scaler", RobustScaler()),
    ("svm", SVC())
])

param_grid = {	
    "svm__C": [0.1, 1, 10, 50, 100],
    "svm__kernel": ["linear", "rbf", "poly"],
    "svm__gamma": ["scale", 0.01, 0.1, 1]
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)


# %%
display(pd.DataFrame(grid.cv_results_).sort_values("mean_test_score", ascending=False))



# %%
param_grid_2 = {
    "svm__C": np.linspace(50, 100, 10),
    "svm__kernel": ["rbf"],
    "svm__gamma": np.linspace(0.1, 1, 10) 
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid_2,
    cv=5,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)


# %%
display(pd.DataFrame(grid.cv_results_).sort_values("mean_test_score", ascending=False))


# %%
svm_final = grid.best_estimator_
test_score = svm_final.score(X_temp, y_temp)
print(f"SVM test accuracy: {test_score:.4f}")


# %%
y_pred = svm_final.predict(X_temp)
labels_lt = ["Ruduo", "Pavasaris", "Vasara", "Žiema"]  
labels_en = ["Autumn", "Spring", "Summer", "Winter"]

print(classification_report(y_temp, y_pred))

cm = confusion_matrix(y_temp, y_pred, labels=labels_en)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_lt)
disp.plot(cmap="Blues")
disp.ax_.set_xlabel("Prognozuota klasė")
disp.ax_.set_ylabel("Tikroji klasė")
plt.title("SVM sumaišymo matrica - testinė aibė")
plt.tight_layout()
plt.show()


# %%
print("Balanced accuracy:", balanced_accuracy_score(y_temp, y_pred))
print("Macro precision:", precision_score(y_temp, y_pred, average="macro"))
print("Macro recall:", recall_score(y_temp, y_pred, average="macro"))
print("Macro F1:", f1_score(y_temp, y_pred, average="macro"))


# %%
y_score = svm_final.decision_function(X_temp)

classes = svm_final.classes_
y_temp_bin = label_binarize(y_temp, classes=classes)

plt.figure(figsize=(8, 6))

for i, class_name in enumerate(classes):
    class_name_lt = {
		"Winter": "Žiema",
		"Spring": "Pavasaris",
		"Summer": "Vasara",
		"Autumn": "Ruduo"
	}.get(class_name, class_name)
    fpr, tpr, _ = roc_curve(y_temp_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"Klasė {class_name_lt} AUC = {roc_auc:.3f}")

plt.plot([0, 1], [0, 1], linestyle="--", label="Atsitiktinis klasifikatorius")

plt.xlabel("1-Specifiškumas (FPR)")
plt.ylabel("Jautrumas (TPR)")
plt.title("SVM ROC kreivės - testinė aibė")
plt.legend()
plt.grid(True)
plt.show()


