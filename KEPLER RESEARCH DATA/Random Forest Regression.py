### non continues non linear model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("C:/Users/microsoft/Desktop/#/python programs/ML/OM/KEPLER RESEARCH DATA/dataset.csv")
x = dataset.iloc[:, [26,24,19,11,41,17]].values
y = dataset.iloc[:, [44]].values


from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean' , fill_value=None, verbose="deprecated", copy=True, add_indicator=False)# imputer objecet
imputer = imputer.fit(y[:, :]) 
y[:, :] = imputer.transform(y[:, :]) 

from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(missing_values=np.nan, strategy='mean' , fill_value=None, verbose="deprecated", copy=True, add_indicator=False)# imputer objecet
imputer = imputer.fit(x[:, :]) 
x[:, :] = imputer.transform(x[:, :]) 


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(x,y,train_size=0.25)  

   
# Fitting the Random Forrest regressor to dataset 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10)
regressor.fit(X_train, y_train)

# visualising tree regressor
# x_grid = np.arange(min(x), max(x), 0.01)
# x_grid = x_grid.reshape((len(x_grid),1))

plt.scatter(X_train, y_train, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Random forest Prediction (train)')
plt.ylabel("PLAENTS RADIUS")
plt.xlabel("SUNS RADIUS")
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Random forest Prediction (test)')
plt.ylabel("PLANETS RADIUS")
plt.xlabel("SUNS RADIUS")
plt.show()


prediction = regressor.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print(mean_absolute_error(y_test,prediction))
print(mean_squared_error(y_test,prediction))
print(np.sqrt(mean_squared_error(y_test,prediction)))
print(r2_score(y_test,prediction))