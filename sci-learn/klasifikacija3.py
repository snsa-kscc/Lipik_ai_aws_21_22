import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

iris_x, iris_y = load_iris(return_X_y=True)

iris_x = StandardScaler().fit_transform(iris_x)

x_train, x_test, y_train, y_test = train_test_split(
    iris_x, iris_y, test_size=0.15)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

class_report = classification_report(y_test, y_pred)
print(class_report)
