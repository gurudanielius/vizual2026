# %%
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns



# %%
data_raw = pd.read_csv('Elektrines_duomenys_2023-2024m.csv', sep=';', decimal=',')
data_raw



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

#%%
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


