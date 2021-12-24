from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

diabetes_X, diabetes_Y = datasets.load_diabetes(return_X_y=True)

nrows, ncols = 2, 5
fig = plt.figure()

# for i in range(1, 11):
#     ax = fig.add_subplot(nrows, ncols, i)
#     ax.scatter(diabetes_X[:, i - 1], diabetes_Y)

# plt.show()

diabetes_X = diabetes_X[:, 2]
diabetes_X = diabetes_X[:, np.newaxis]

poly = PolynomialFeatures(degree=2)
# krivulja višeg reda (ax+b -> ax^2+bx+c)
diabetes_X = poly.fit_transform(diabetes_X)

diabetes_X_train, diabetes_X_test, diabetes_Y_train, diabetes_Y_test = train_test_split(
    diabetes_X, diabetes_Y, test_size=0.2)

linear_model = linear_model.LinearRegression()

linear_model.fit(diabetes_X_train, diabetes_Y_train)
diabetes_Y_pred = linear_model.predict(diabetes_X_test)

mse = mean_squared_error(diabetes_Y_test, diabetes_Y_pred)
r2 = r2_score(diabetes_Y_test, diabetes_Y_pred)
print(mse, r2)
# plt.scatter(diabetes_X_test, diabetes_Y_test, color='black')
# plt.plot(diabetes_X_test, diabetes_Y_pred,
#          color='red', linewidth=3, marker='o')
# plt.show()
plt.scatter(diabetes_X_test[:, 1], diabetes_Y_test, color='black')
plt.scatter(diabetes_X_test[:, 1], diabetes_Y_pred,
            color='red')  # blago zakrivljen
plt.show()
