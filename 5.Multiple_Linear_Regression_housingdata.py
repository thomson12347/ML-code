# Multiple Logistic Regression using Housing Data

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split

boston = datasets.load_boston(return_X_y=False)

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

reg = linear_model.LinearRegression()

reg.fit(X_train, y_train)

print('Coefficient :\n', reg.coef_)
print("Variance score:",reg.score(X_test, y_test))
plt.style.use("fivethirtyeight")
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, color='green', s=10, label='Train data')
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, color='blue', s=10, label='Test data')

plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)
plt.legend(loc='upper right')
plt.title("Residual errors")
plt.show()
