# %%
import pandas as pd
import matplotlib.pyplot as plt	



# %%
data_raw=pd.read_csv('Elektrines_duomenys_2023-2024m.csv',sep=';')
data_raw


# %%
data=data_raw[["timestamp"]+[f"Daily_energy_INV-{i}" for i in range(1, 9)]]
data


# %%
# Daily totals table with timestamp column
cols = [f"Daily_energy_INV-{i}" for i in range(1, 9)]

data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
data[cols] = data[cols].apply(pd.to_numeric, errors="coerce")

daily_totals = (
    data.groupby(data["timestamp"].dt.date)[cols]
    .max()
    .reset_index()
)

daily_totals.columns = ["timestamp"] + [f"Total_energy_INV-{i}" for i in range(1, 9)]
daily_totals


# %%
daily_totals["timestamp"] 


# %%



# %%
summer_data





