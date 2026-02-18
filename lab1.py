#%%  
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('INV12.csv')
data.describe()
data[2730:2750]
# Praleistų stebėjimų nėra.
data['Timestamp']= pd.to_datetime(data['Timestamp']) #Pakeiciam i datetime agregavimo logikai veliau
data["month_day"]=data['Timestamp'].dt.strftime('%m-%d')
data[90:110:4]["month_day"] #paimsim dienas ir sugrupuosim ir sudesim i viena diena
data.groupby(data["month_day"]).sum(numeric_only=True)#%% 
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

data_ma[2730:2750].head(30)

data_ma["month_day"]=data['Timestamp'].dt.strftime('%m-%d')
data_ma = data_ma.drop(columns=[f'moving_avg_string_{i}' for i in range(1, 11)]+['Timestamp'])

data_ma.groupby(data_ma["month_day"]).sum(numeric_only=True)

# %%
data_ma_days = data.copy()
intervals_per_day = 96  

for i in range(1, 11):
    col_name = f'string_{i}'
    # Get values from 1 day ago, 2 days ago, ..., 5 days ago
    shifted_values = [data_ma_days[col_name].shift(intervals_per_day * day) for day in range(1, 6)]
    data_ma_days[f'moving_avg_string_{i}'] = pd.concat(shifted_values, axis=1).mean(axis=1)



for i in range(1, 11):
    col_name = f'string_{i}'
    data_ma_days[col_name] = data_ma_days[col_name].fillna(data_ma_days[f'moving_avg_string_{i}'].ffill())


##TODO: Check if its alright:

# Test: verify calculation is correct
# Pick a row far enough to have 5 days of history
test_row = 500
test_col = 'string_1'

print(f"Test row: {test_row}")
print(f"Timestamp: {data_ma_days.loc[test_row, 'Timestamp']}")
print(f"Current value: {data_ma_days.loc[test_row, test_col]}")
print()

# Show values from 1-5 days ago at same time
values_prev_days = []
for day in range(1, 6):
    prev_row = test_row - (intervals_per_day * day)
    if prev_row >= 0:
        val = data_ma_days.loc[prev_row, test_col]
        print(f"{day} day(s) ago (row {prev_row}): {val}")
        values_prev_days.append(val)

manual_average = sum(values_prev_days) / len(values_prev_days)
calculated_average = data_ma_days.loc[test_row, f'moving_avg_string_1']

print(f"Manual average of previous 5 days: {manual_average}")
print(f"Calculated moving average: {calculated_average}")

data_ma_days[2730:2750]
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

