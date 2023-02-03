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

from sklearn.preprocessing  import PolynomialFeatures
plr = PolynomialFeatures(degree= 2)
x_poly_train = plr.fit_transform(x)

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(x,y,train_size=0.25) 

from sklearn.preprocessing  import PolynomialFeatures
plr = PolynomialFeatures(degree= 2)
x_poly = plr.fit_transform(X_train)

# Linear Regression model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_poly,y_train)

from sklearn.linear_model import LinearRegression
linr =LinearRegression()
linr.fit(X_train , y_train)



# visual linear regression 
plt.scatter(X_train, y_train, color = 'red')
plt.plot(x_poly, lr.predict(x_poly), color='blue')
plt.plot(X_train, linr.predict(X_train), color='green')
plt.title('Polynomial Regression Prediction (train)')
plt.ylabel("suns radius ")
plt.xlabel("Planets radius")
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(x_poly, lr.predict(x_poly), color='blue')
plt.plot(X_train, linr.predict(X_train), color='green')
plt.title('Polynomial regression  Prediction (test)')
plt.ylabel("suns rasdius")
plt.xlabel("planets radius")
plt.show()

prediction = linr.predict(X_test)


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print(mean_absolute_error(y_test,prediction))
print(mean_squared_error(y_test,prediction))
print(np.sqrt(mean_squared_error(y_test,prediction)))
print(r2_score(y_test,prediction))