#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Duomenų įkėlimas
data = pd.read_csv('Pradine_analize/INV12.csv')
data.columns = ['Timestamp'] + [f'string_{i}' for i in range(1, 11)]
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Dienos identifikatorius
data["month_day"] = data['Timestamp'].dt.strftime('%m-%d')

# Dienų skirstymas į High / Medium / Low pagal bendrą dienos generaciją
mapping = data.groupby("month_day").sum(numeric_only=True)
mapping["sum"] = mapping.sum(axis=1)
mapping["category"] = np.where(
    mapping["sum"] > mapping["sum"].quantile(0.75), "High",
    np.where(mapping["sum"] < mapping["sum"].quantile(0.25), "Low", "Medium")
)

data["category"] = data["month_day"].map(mapping["category"])

# Laiko stulpelis
data["hour"] = data["Timestamp"].dt.strftime('%H:%M')

# Pašalinam trūkstamas reikšmes
string_cols = [f'string_{i}' for i in range(1, 11)]
data = data.dropna(subset=string_cols)

# Long formatas
melted = data.melt(
    id_vars=['month_day', 'hour', 'category'],
    value_vars=string_cols,
    var_name='string',
    value_name='value'
)

# Wide formatas: eilutės = diena + string + kategorija, stulpeliai = laikas
result = melted.pivot_table(
    index=['month_day', 'string', 'category'],
    columns='hour',
    values='value',
    aggfunc='sum'
)

result["month_day"] = result.index.get_level_values('month_day')
result["string"] = result.index.get_level_values('string')
result["category"] = result.index.get_level_values('category')
result.reset_index(drop=True, inplace=True)

cols = result.columns.tolist()
cols = ['month_day', 'string', 'category'] + [col for col in cols if col not in ['month_day', 'string', 'category']]
result = result[cols]
result = result.sort_values(by=['month_day', 'string']).reset_index(drop=True)
result.index.name = 'id'

# Padalinimas pagal kategorijas

data_high = result[result['category'] == 'High'].drop(columns=['category', 'month_day'])
data_medium = result[result['category'] == 'Medium'].drop(columns=['category', 'month_day'])
data_low = result[result['category'] == 'Low'].drop(columns=['category', 'month_day'])
#%%
# Aprašomoji statistika
string_order = [f'string_{i}' for i in range(1, 11)]
data_high_long = data_high.melt(
    id_vars='string',
    var_name='hour',
    value_name='value'
)
data_high_final = data_high_long.pivot_table(
    index='hour',
    columns='string',
    values='value',
    aggfunc='mean'
).reset_index()
data_high_final = data_high_final[string_order]
data_high_final.describe()
#%%
data_medium_long = data_medium.melt(
    id_vars='string',
    var_name='hour',
    value_name='value'
)
data_medium_final = data_medium_long.pivot_table(
    index='hour',
    columns='string',
    values='value',
    aggfunc='mean'
).reset_index()
data_medium_final = data_medium_final[string_order]
data_medium_final.describe()
#%%
data_low_long = data_low.melt(
    id_vars='string',
    var_name='hour',
    value_name='value'
)
data_low_final = data_low_long.pivot_table(
    index='hour',
    columns='string',
    values='value',
    aggfunc='mean'
).reset_index()
data_low_final = data_low_final[string_order]
data_low_final.describe()
#%%
# Heatmap
rename_strings = {f'string_{i}': f'Grandinė {i}' for i in range(1, 11)}
string_order = [f'string_{i}' for i in range(1, 11)]

heatmap_high = data_high.groupby('string').mean().reindex(string_order).rename(index=rename_strings)
heatmap_medium = data_medium.groupby('string').mean().reindex(string_order).rename(index=rename_strings)
heatmap_low = data_low.groupby('string').mean().reindex(string_order).rename(index=rename_strings)

vmin = min(
    heatmap_high.min().min(),
    heatmap_medium.min().min(),
    heatmap_low.min().min()
)
vmax = max(
    heatmap_high.max().max(),
    heatmap_medium.max().max(),
    heatmap_low.max().max()
)

def plot_heatmap(data, title):
    plt.figure(figsize=(14, 5))
    sns.heatmap(data, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel('Laikas')
    plt.ylabel('Grandinės')
    plt.xticks(
        ticks=range(0, len(data.columns), 8),
        labels=data.columns[::8],
        rotation=45,
        ha='right'
    )
    plt.tight_layout()
    plt.show()

plot_heatmap(heatmap_high, 'Didelės generacijos dienos')
plot_heatmap(heatmap_medium, 'Vidutinės generacijos dienos')
plot_heatmap(heatmap_low, 'Mažos generacijos dienos')
# %%
