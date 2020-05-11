# program to implement Naive Bayes Classification using Iris data

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
print("X training values :", X_train[:10])
print("y training values : ", y_train[:10])
print("X test values : ", X_test[:10])
print("Y test values : ", y_test)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("y predict values : ", y_pred)
print("Count of test values : ", len(y_test))
print("Gaussian Naive Bayes model accuracy(in %) : ", metrics.accuracy_score(y_test, y_pred) * 100)
