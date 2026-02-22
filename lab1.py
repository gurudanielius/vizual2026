#%%  
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib.dates as mdates


data = pd.read_csv('INV12.csv')
data.describe()
data[2730:2750]
# Praleistų stebėjimų nėra.
data['Timestamp']= pd.to_datetime(data['Timestamp']) #Pakeiciam i datetime agregavimo logikai veliau
data["month_day_hour"]=data['Timestamp'].dt.strftime('%m-%d-%H')

data.groupby(data["month_day_hour"]).sum(numeric_only=True)
data.columns = ['Timestamp'] + [f'string_{i}' for i in range(1, 11)] + ['month_day_hour']

## NUSPRENDEME NA REIKSMES TIESIOG PASALINTI
data=data.dropna(subset=[f'string_{i}' for i in range(1, 11)])

# %%

filtered_data = data[data['Timestamp'].dt.hour.between(10, 17)]
data_hourly=filtered_data.groupby(filtered_data["month_day_hour"]).sum(numeric_only=True)
data_hourly

# %%
#Aprašomoji statistika 
data_hourly.describe() 

# %%
#Taškai atsiskyrėliai
numeric = data_hourly.select_dtypes(include="number")

outliers = {}

for col in numeric.columns:
    Q1 = numeric[col].quantile(0.25)
    Q3 = numeric[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    mask = (numeric[col] < lower) | (numeric[col] > upper)
    outliers[col] = numeric.loc[mask, col]

for col, vals in outliers.items():

    print(f"{col}: {len(vals)} outliers")

# %%
#Skaičiuojamos koreliacijos
correlation_matrix = data_hourly.select_dtypes(include=['number']).corr()
print(correlation_matrix)

# %%
correlation_matrix.index = [f"Grandinė {i+1}" for i in range(len(correlation_matrix.index))]
correlation_matrix.columns = [f"Grandinė {i+1}" for i in range(len(correlation_matrix.columns))]

plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8}
)

plt.title("Koreliacijos šilumos žemėlapis", fontsize=18)
plt.tight_layout()
plt.savefig("grafikai/koreliacijos_heatmap.png")
plt.show()
# %%

# min max
numeric_cols = data.select_dtypes(include=['number']).columns
normalized_minmax = (data_hourly[numeric_cols] - data_hourly[numeric_cols].min()) / (data_hourly[numeric_cols].max() - data_hourly[numeric_cols].min())
# %% 
#standartizacija
standartizuotas = (data_hourly[numeric_cols] - data_hourly[numeric_cols].mean()) / data_hourly[numeric_cols].std()
# %%
colors = ["#D00000", "#ffba08", "#cbff8c", "#8fe388", "#1b998b", "#3185fc",
          "#5d2e8c", "#46237a", "#ff7b9c", "#ff9b85"]

# Taškiniai grafikai
columns = [f"string_{i}" for i in range(1, 11)]
display_names = [f"Grandinė {i}" for i in range(1, 11)]

temp_df = data_hourly[columns].copy()
temp_df.columns = display_names  

sm = scatter_matrix(
    temp_df,
    alpha=0.6,
    figsize=(20, 20),
    diagonal='kde',
    color=colors[5]  
)

plt.suptitle("Grandinėse pagamintos elektros sklaidos diagrama", fontsize=24)
plt.savefig("grafikai/sklaidos.png")

# %%
# Dažnio diagramos - histogramos
fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
axes = axes.flatten()
columns = [f"string_{i}" for i in range(1, 11)]

for i, col in enumerate(columns):
    axes[i].hist(data_hourly[col], bins=15, color = colors[i], 
                 alpha = 0.7, edgecolor=colors[i])
    axes[i].set_title(col)
    axes[i-1].set_title(f"Grandinė {i}")

fig.supxlabel("Elektra, pagaminta grandinėje (kWh)")
fig.supylabel("Dažnis")
fig.suptitle("Elektros, pagamintos grandinėse per valandą, histograma", fontsize=18)

plt.tight_layout()
plt.savefig("grafikai/histogramos.png")
plt.show()

# %%
# Stačiakampės diagramos
columns = [f"string_{i}" for i in range(1, 11)]
data = [data_hourly[col] for col in columns]

plt.figure(figsize=(12, 6))
box = plt.boxplot(data, patch_artist=True)

for patch, c in zip(box["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)

plt.xticks(
    range(1, 11),
    [f"Grandinė {i}" for i in range(1, 11)]
)

plt.ylabel("Elektra, pagaminta grandinėje (kWh)")
plt.title("Per valandą pagaminta elektra skirtingose grandinėse") 

plt.grid(axis="y", alpha=0.3)
plt.savefig("grafikai/boxplotai.png")
plt.show()

#%%
#laikute skirtinguose plot'uose
# fig, axes = plt.subplots(5, 2, figsize=(20, 16),sharey=True)
# axes = axes.flatten()

# for i, col in enumerate(columns):
#     axes[i].plot(data_hourly.index, data_hourly[col], 
#                  color=colors[i], linewidth=2)
#     axes[i].set_title(f"Grandinė {i+1}")
#     axes[i].xaxis.set_major_locator(mdates.DayLocator(interval=10))
#     axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
#     axes[i].grid(alpha=0.3)

# plt.tight_layout()
# plt.savefig("grafikai/laikute_subplots.png")
# plt.show()
# %%
plt.figure(figsize=(20, 8))
sns.heatmap(data_hourly[columns].T, cmap='YlOrRd', cbar_kws={'label': 'kWh'})
ax = plt.gca()
new_labels = [f"Grandinė {i}" for i in range(1, 11)]
ax.set_yticklabels(new_labels, rotation=0)
tick_positions = range(0, len(data_hourly.index), 16)
ax.set_xticks(tick_positions)
ax.set_xticklabels([data_hourly.index[i].strftime('%m-%d') for i in tick_positions])
plt.xlabel("Data")
plt.title("Pagamintos elektros grandinėse šilumos žemėlapis")
plt.tight_layout()
plt.savefig("grafikai/laikute_heatmap.png")
plt.show()
print(tick_positions)
## DG: As manau sitas geras irgi bet cia demokratija, tai subalsuokime
# %%
#Standartizuotos aibes histogramos

fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharey=True)
axes = axes.flatten()
columns = [f"string_{i}" for i in range(1, 11)]

for i, col in enumerate(columns):
    axes[i].hist(standartizuotas[col], bins=15, color = colors[i], 
                 alpha = 0.7, edgecolor=colors[i])
    axes[i].set_title(col)
    axes[i-1].set_title(f"Grandinė {i}")

fig.supxlabel("Elektra, pagaminta grandinėje (kWh)")
fig.supylabel("Dažnis")
fig.suptitle("Elektros, pagamintos grandinėse per valandą, histograma (standartizuotos reikšmės)", fontsize=18)

plt.tight_layout()
plt.savefig("grafikai/histogramos_standartizuotos.png")
plt.show()

# %%
#Staciakampes diagramos standartizuotos
columns = [f"string_{i}" for i in range(1, 11)]
data = [standartizuotas[col] for col in columns]

plt.figure(figsize=(12, 6))
box = plt.boxplot(data, patch_artist=True)

for patch, c in zip(box["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.7)

plt.xticks(
    range(1, 11),
    [f"Grandinė {i}" for i in range(1, 11)]
)

plt.ylabel("Elektra, pagaminta grandinėje (kWh)")
plt.title("Per valandą pagaminta elektra skirtingose grandinėse (standartizuotos reikšmės)") 

plt.grid(axis="y", alpha=0.3)
plt.savefig("grafikai/boxplotai_standartizuoti.png")
plt.show()

#%%
#heatmapas laikutei standartizuotos reikšmės

plt.figure(figsize=(20, 8))
sns.heatmap(standartizuotas[columns].T, cmap='YlOrRd', cbar_kws={'label': 'kWh'})
ax = plt.gca()
new_labels = [f"Grandinė {i}" for i in range(1, 11)]
ax.set_yticklabels(new_labels, rotation=0)
tick_positions = range(0, len(standartizuotas.index), 16)
ax.set_xticks(tick_positions)
ax.set_xticklabels([standartizuotas.index[i].strftime('%m-%d') for i in tick_positions])
plt.xlabel("Data")
plt.title("Pagamintos elektros grandinėse šilumos žemėlapis (standartizuotos reikšmės)")
plt.tight_layout()
plt.savefig("grafikai/laikute_heatmap_standartizuotas.png")
plt.show()
print(tick_positions)

# %%

columns = [f"string_{i}" for i in range(1, 11)]
display_names = [f"Grandinė {i}" for i in range(1, 11)]

temp_df = standartizuotas[columns].copy()
temp_df.columns = display_names  

sm = scatter_matrix(
    temp_df,
    alpha=0.6,
    figsize=(20, 20),
    diagonal='kde',
    color=colors[5]  
)

plt.suptitle("Grandinėse pagamintos elektros sklaidos diagrama (standartizuotos reikšmės)", fontsize=24)
plt.savefig("grafikai/sklaidos_standartizuoti.png")