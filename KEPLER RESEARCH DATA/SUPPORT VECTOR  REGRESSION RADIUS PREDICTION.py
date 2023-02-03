import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

dataset = pd.read_csv("C:/Users/microsoft/Desktop/#/python programs/ML/OM/KEPLER RESEARCH DATA/dataset.csv")
x = dataset.iloc[:,[29]].values
y = dataset.iloc[:,[44]].values

from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean' , fill_value=None, verbose="deprecated", copy=True, add_indicator=False)# imputer objecet
imputer = imputer.fit(y[:, :]) 
y[:, :] = imputer.transform(y[:, :]) 

from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean' , fill_value=None, verbose="deprecated", copy=True, add_indicator=False)# imputer objecet
imputer = imputer.fit(x[:, :]) 
x[:, :] = imputer.transform(x[:, :]) 

# Feature Scaling cause it is not preavailable in SVR
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)


# Fitting regressor to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(x,y,train_size=0.25) 
# arr = np.array([[6.5],[0]])
# # arr.reshape(-1,1)
# arr2 = regressor.predict(sc_x.transform(arr))
# arr2 = [arr2]
# y_prd = sc_y.inverse_transform(np.array(arr2))

plt.scatter(X_train, y_train, color ='blue')
plt.plot(X_train, regressor.predict(X_train), color ='red')
plt.title('RADIUS PREDICTION TRAIN SET')
plt.xlabel("PLANET RADIUS")
plt.ylabel("SUN RADIUS")


plt.scatter(X_test, y_test, color ='blue')
plt.plot(X_train, regressor.predict(X_train), color ='red')
plt.title('RADIUS PREDICTION TEST SET ')
plt.xlabel("PLANET RADIUS")
plt.ylabel("SUN RADIUS")
plt.show()

prediction = regressor.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print(mean_absolute_error(y_test,prediction))
print(mean_squared_error(y_test,prediction))
print(np.sqrt(mean_squared_error(y_test,prediction)))
print(r2_score(y_test,prediction))