# %%
import pandas as pd


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
final_dataset.to_csv("final_dataset.csv", index=False)

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


