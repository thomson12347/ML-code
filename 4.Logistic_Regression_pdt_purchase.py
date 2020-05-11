# Plotting Logistic Regression using New Product Purchase Datasheet

from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_set():
    datset = pd.read_csv('LogisticsRegDataSet.csv')
    x = datset.iloc[:, [2, 3]].values
    y = datset.iloc[:, 4].values
    return x, y


def split(x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)
    sc_x = StandardScaler()
    xtrain = sc_x.fit_transform(xtrain)
    xtest = sc_x.transform(xtest)
    print("Top 10 training Data : ",xtrain[0:10, :])
    return xtrain, ytrain, xtest, ytest


def train_test_plot(xtrain, ytrain, xtest, ytest):
    classifier = LogisticRegression(random_state=0)
    classifier.fit(xtrain, ytrain)
    y_pred = classifier.predict(xtest)
    cm = confusion_matrix(ytest, y_pred)
    print("Confusion Matrix : \n", cm)
    print("Accuracy : ", accuracy_score(ytest, y_pred))
    X_set, y_set = xtest, ytest
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1,
                                   stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1,
                                   stop=X_set[:, 1].max() + 1, step=0.01))
    print(X1,X2)
    plt.contourf(X1, X2, classifier.predict(
        np.array([X1.ravel(), X2.ravel()]).T).reshape(
        X1.shape), alpha=0.75, cmap=ListedColormap(('blue', 'yellow')))

    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        l = lambda j: 'Yes' if (j == 1) else 'No'
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(('black', 'red')) (i), label=l(j))

        plt.title("Classifier(Test set)")
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend(loc='best')
    plt.show()


# driver
x, y = read_set()
xtrain, ytrain, xtest, ytest = split(x, y)
train_test_plot(xtrain, ytrain, xtest, ytest)
