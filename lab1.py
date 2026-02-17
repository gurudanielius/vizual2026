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
data.groupby(data["month_day"]).sum(numeric_only=True)
data_2 = pd.DataFrame({
	'values': [1, 2, 2, 3, 3,
	4, 4, 4, 5, 5, 5, 5]
})
plt.hist(data_2['values'], bins=5, edgecolor='black')
plt.title('Histogram of Values')
plt.show()
#%% 
