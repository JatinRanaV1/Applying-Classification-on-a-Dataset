import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('us.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                   test_size = 0.25, random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_test, y_pred, color = 'blue')
plt.title('K Neighbors Classifier')
plt.xlabel('Cases')
plt.ylabel('Deaths')
plt.show()

