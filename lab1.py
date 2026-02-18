#%%  
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import matplotlib.dates as mdates

data = pd.read_csv('INV12.csv')
data.describe()
data[2730:2750]
# Praleistų stebėjimų nėra.
data['Timestamp']= pd.to_datetime(data['Timestamp']) #Pakeiciam i datetime agregavimo logikai veliau
data["month_day"]=data['Timestamp'].dt.strftime('%m-%d')
data.groupby(data["month_day"]).sum(numeric_only=True)
data.columns = ['Timestamp'] + [f'string_{i}' for i in range(1, 11)] + ['month_day']

#%%
### NaN reikšmių užpildymas naudojant slenkamąjį vidurkį
data_ma=data.copy()
data_ma['moving_avg_string_1'] = data_ma['string_1'].rolling(window=5).mean()
data_ma['moving_avg_string_2'] = data_ma['string_2'].rolling(window=5).mean()
data_ma['moving_avg_string_3'] = data_ma['string_3'].rolling(window=5).mean()
data_ma['moving_avg_string_4'] = data_ma['string_4'].rolling(window=5).mean()
data_ma['moving_avg_string_5'] = data_ma['string_5'].rolling(window=5).mean()
data_ma['moving_avg_string_6'] = data_ma['string_6'].rolling(window=5).mean()
data_ma['moving_avg_string_7'] = data_ma['string_7'].rolling(window=5).mean()
data_ma['moving_avg_string_8'] = data_ma['string_8'].rolling(window=5).mean()
data_ma['moving_avg_string_9'] = data_ma['string_9'].rolling(window=5).mean()
data_ma['moving_avg_string_10'] = data_ma['string_10'].rolling(window=5).mean()

data_ma['string_1']= data_ma['string_1'].fillna(data_ma['moving_avg_string_1'].ffill())
data_ma['string_2']= data_ma['string_2'].fillna(data_ma['moving_avg_string_2'].ffill())
data_ma['string_3']= data_ma['string_3'].fillna(data_ma['moving_avg_string_3'].ffill())
data_ma['string_4']= data_ma['string_4'].fillna(data_ma['moving_avg_string_4'].ffill())
data_ma['string_5']= data_ma['string_5'].fillna(data_ma['moving_avg_string_5'].ffill())
data_ma['string_6']= data_ma['string_6'].fillna(data_ma['moving_avg_string_6'].ffill())
data_ma['string_7']= data_ma['string_7'].fillna(data_ma['moving_avg_string_7'].ffill())
data_ma['string_8']= data_ma['string_8'].fillna(data_ma['moving_avg_string_8'].ffill())
data_ma['string_9']= data_ma['string_9'].fillna(data_ma['moving_avg_string_9'].ffill())
data_ma['string_10'] = data_ma['string_10'].fillna(data_ma['moving_avg_string_10'].ffill())

# Drop moving average columns

data_ma = data_ma.drop(columns=[f'moving_avg_string_{i}' for i in range(1, 11)]+['Timestamp'])

data_ma.groupby(data_ma["month_day"]).sum(numeric_only=True)

# %%
data_ma_days = data.copy()
intervals_per_day = 96  

for i in range(1, 11):
    col_name = f'string_{i}'
    shifted_values = [data_ma_days[col_name].shift(intervals_per_day * day) for day in range(1, 4)]
    data_ma_days[f'moving_avg_string_{i}'] = pd.concat(shifted_values, axis=1).mean(axis=1)

for i in range(1, 11):
    col_name = f'string_{i}'
    data_ma_days[col_name] = data_ma_days[col_name].fillna(data_ma_days[f'moving_avg_string_{i}'].ffill())

#^
##        0      1      2      3      4
# 0    NaN    NaN    NaN    NaN    NaN
# 1    NaN    NaN    NaN    NaN    NaN
# ...
# 96   10.0   NaN    NaN    NaN    NaN      
# 97   11.0   NaN    NaN    NaN    NaN
# ...
# 192  100.0  10.0   NaN    NaN    NaN      
# 193  101.0  11.0   NaN    NaN    NaN

#Minusas sio metodo yra tai kad reikia daugiau stebejimu, lieka daugiau neuzpildytu atveju, taip pat
#praeitu penkiu dienu oras gali skirtis labai skirtis nuo kitos dienos, todel nebutina kad tai bus geras budas uzpildyti trukstamus duomenis;

#pvz 06-29 12:15
# data_ma_days.iloc[[2737,2737-96,2737-192,2737-288,2737-384]] #paimsim 5 dienas atgal ir paziuresim ar oras buvo panasus


vidurkio_uzpildymas=data.fillna(data.mean(numeric_only=True))

print(vidurkio_uzpildymas[2730:2750])
# Per vasara oro pokyciai (o tuo tarpu ir saules skleidziamos sviesos) gali buti labai smarkus, tad visos imties vidurkis nera labai 
# puiki reiksme uzpildyti praleistus stebejimus;

# %%
data_day = data_ma.groupby(data_ma["month_day"]).sum(numeric_only=True)
# %%
#Aprašomoji statistika
data_day.describe()

# %%
#Taškai atsiskyrėliai
numeric = data_day.select_dtypes(include="number")

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

outliers["string_2"]

# %%
#Skaičiuojamos koreliacijos
correlation_matrix = data_day.select_dtypes(include=['number']).corr()
print(correlation_matrix)
# %%
#Sklaida svarbi todel, kad dauguma ML metodų sprendžia pagal atstumus, o sklaida čia daug reiškia.
#Jei vieno požymio sd didelė, jis natūraliai varijuoja smarkiai ir jo indėlis į atstumą būna didesnis. Jei kito sd maža jis mazai varijuoa ir jo indelis į atstuma buna mazas. Tada modelis netycia pradeda laikyti viena požymi svarbesniu.
# %%
numeric_cols = data.select_dtypes(include=['number']).columns
normalized_minmax = (data[numeric_cols] - data[numeric_cols].min()) / (data[numeric_cols].max() - data[numeric_cols].min())
# %% 
standartizuotas = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()


# %%
#Spalvų paletė
colors = ["#d00000", "#ffba08", "#cbff8c", "#8fe388", "#1b998b", "#3185fc",
          "#5d2e8c", "#46237a", "#ff7b9c", "#ff9b85"]

# Taškiniai grafikai
columns = [f"string_{i}" for i in range(1, 11)]
display_names = [f"Grandinė {i}" for i in range(1, 11)]

temp_df = data_day[columns].copy()
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
plt.show()

# %%
# Dažnio diagramos - histogramos
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
columns = [f"string_{i}" for i in range(1, 11)]

for i, col in enumerate(columns):
    axes[i].hist(data_day[col], bins=15, color = colors[i], 
                 alpha = 0.7, edgecolor=colors[i])
    axes[i].set_title(col)
    axes[i-1].set_title(f"Grandinė {i}")

fig.supxlabel("Elektra, pagaminta grandinėje (kWh)")
fig.supylabel("Dažnis")
fig.suptitle("Elektros, pagamintos grandinėse per dieną, histograma", fontsize=18)

plt.tight_layout()
plt.savefig("grafikai/histogramos.png")
plt.show()

# %%
# Stačiakampės diagramos
columns = [f"string_{i}" for i in range(1, 11)]
data = [data_day[col] for col in columns]

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
plt.title("Pagamintos elektros pasiskirstymas pagal grandines")

plt.grid(axis="y", alpha=0.3)
plt.savefig("grafikai/boxplotai.png")
plt.show()

#%%
#Laiko eilučių grafikai

data_day.index = pd.to_datetime(data_day.index, format="%m-%d")

columns = [f"string_{i}" for i in range(1, 11)]
display_names = [f"Grandinė {i}" for i in range(1, 11)]

plt.figure(figsize=(20, 8))

for i, col in enumerate(columns):
    plt.plot(
        data_day.index,
        data_day[col],
        label=display_names[i],
        linewidth=2,
        color=colors[i]
    )

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

plt.xticks(rotation=45)

plt.xlabel("Data")
plt.ylabel("Pagaminta elektra (kWh)")
plt.title("Grandinių elektros gamybos kitimas laike")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("grafikai/laikute.png")
plt.show()

# %%
