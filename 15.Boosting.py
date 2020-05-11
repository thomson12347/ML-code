# program to implement Boosting using Mushroom Dataset and AdaBoost Classifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

dataset = pd.read_csv('mushroom.csv', header=None)
dataset = dataset.sample(frac=1)
dataset.columns = ['target','cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing',
                   'gill-size', 'gill-color','gill-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                   'stalk-color-above-ring', 'stalk-color-below-ring','veil-type', 'veil-color','ring-number', 'ring-type', 'spore-print-color',
                   'population', 'habitat']

# Encode the feature values from strings to integers since the sklearn Decision Tree
# Classifier only takes numerical values

for label in dataset.columns:
    dataset[label] = LabelEncoder().fit(dataset[label]).transform(dataset[label])

X = dataset.drop(['target'], axis=1)
Y = dataset['target']

model = DecisionTreeClassifier(criterion='entropy', max_depth=11)
AdaBoost = AdaBoostClassifier(base_estimator=model, n_estimators=400, learning_rate=1)
AdaBoost.fit(X, Y)
prediction = AdaBoost.score(X, Y)
print("The accuracy for Decision Tree Model is : ", prediction * 100, '%')

AdaBoost = AdaBoostClassifier(n_estimators=400, learning_rate=1, algorithm='SAMME')
AdaBoost.fit(X, Y)
prediction = AdaBoost.score(X, Y)
print("The accuracy for SAMME algorithm is : ", prediction * 100, '%')
