import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv("housing.csv")
print(data.head())
print(data.info())
print(data.describe().transpose())
data["households"].hist()
plt.show()
data.hist(figsize=(20,20))
plt.show()
plt.figure(figsize=(20,20))
plt.plot(data["latitude"].values,data["longitude"].values,"rX")
plt.show()
# plt.figure()
data.plot(kind="scatter",x="latitude",y="longitude",c="median_house_value",cmap=plt.get_cmap("jet"))
plt.show()
print(data.head())
# remove latitude and longitude
data = data.iloc[:,2:]
print(data)
print(data["ocean_proximity"].unique())
# arrange the uniqe values in Deending order
l = ['ISLAND','NEAR OCEAN','NEAR BAY','<1H OCEAN','INLAND' ]

print(l.index("NEAR BAY"))
def convert(x):
    return l.index(x)+1
convert("NEAR BAY")
data["ocean_proximity"] = data["ocean_proximity"].apply(convert)
data[data["total_bedrooms"].isna()]
data = data.dropna()
print(data.shape)
print(data.head())
from sklearn.preprocessing import MinMaxScaler
# train test split
X = data[["housing_median_age","total_rooms","total_bedrooms","population","households","median_income","ocean_proximity"]]
Y = data["median_house_value"]
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)
model = LinearRegression()
model.fit(x_train,y_train)
predicted = model.predict(x_test)

MSE = ((y_test-predicted)**2).mean()
print(MSE)
RMSE = MSE**0.5
print(RMSE)
MAE = (np.abs(y_test-predicted)).mean()
print(MAE)
import pickle
f = open("LR_without_scaling.sav","wb")
pickle.dump(model,f)
f.close()
