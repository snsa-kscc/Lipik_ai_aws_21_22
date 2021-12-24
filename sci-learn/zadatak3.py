from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

data = datasets.fetch_california_housing()

cali_X, cali_Y = datasets.fetch_california_housing(return_X_y=True)

nrows, ncols = 2, 5
fig = plt.figure()

cali_X = cali_X[:, 1]
cali_X = cali_X[:, np.newaxis]

cali_X_train, cali_X_test, cali_Y_train, cali_Y_test = train_test_split(
    cali_X, cali_Y, test_size=0.3)

linear_model = linear_model.LinearRegression()

linear_model.fit(cali_X_train, cali_Y_train)
cali_Y_pred = linear_model.predict(cali_X_test)

mse = mean_squared_error(cali_Y_test, cali_Y_pred)
r2 = r2_score(cali_Y_test, cali_Y_pred)
print(mse, r2)

plt.scatter(cali_X_test, cali_Y_test, color='black')
plt.scatter(cali_X_test, cali_Y_pred,
            color='red')
plt.show()
