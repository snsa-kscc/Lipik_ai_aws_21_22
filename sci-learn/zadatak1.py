import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm

xtrain = np.array([[0], [1], [2]])
ytrain = np.array([0, 1, 2])

plt.scatter(xtrain, ytrain)
plt.show()

linear_model = lm.LinearRegression()
linear_model.fit(xtrain, ytrain)

# print(linear_model.coef_)
# print(linear_model.intercept_)

xtest = np.array([[0.5], [3]])
ypred = linear_model.predict(xtest)

print(ypred)
