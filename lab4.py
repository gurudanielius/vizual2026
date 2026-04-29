# %%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

# %%
final_dataset=pd.read_csv("final_dataset.csv")

# %%
final_dataset.head()

# %%
label_counts = final_dataset["season"].value_counts()
label_counts # klases subalansuotos

# %%
final_dataset.drop(columns=["month"], inplace=True)

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

X=final_dataset.drop(columns=["season"])
y=final_dataset["season"]

# %%
X_temp, X_test, y_temp, y_test = train_test_split(X, y, train_size=0.8, stratify=y, random_state=80085)
X_train, y_train, X_val, y_val = train_test_split(X_temp, y_temp, train_size=0.8, stratify=y_temp, random_state=80085)

# %%



