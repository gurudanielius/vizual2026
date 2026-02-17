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

# When string_1 is NaN, fill with moving_avg_5 (propagates through consecutive NaNs)
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

data[2730:2750].head(30)
# %%
