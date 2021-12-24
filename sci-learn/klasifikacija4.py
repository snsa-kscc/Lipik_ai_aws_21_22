import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


df = pd.read_csv('sci-learn/heart.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15)

classifier = LogisticRegression()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

class_report = classification_report(y_test, y_pred)
cfsn = confusion_matrix(y_test, y_pred)
print(cfsn)
print(class_report)
