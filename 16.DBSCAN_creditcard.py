# program using DBSCAN with credit card Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

X=pd.read_csv('CC_GENERAL.csv')
X=X.drop('CUST_ID',axis=1)
#print(X)
#Sometimes csv file has null values, which are later displayed as NaN in Data Frame. Just like pandas dropna() method manage and
# remove Null values from a data frame, fillna() manages  and let the user replace NaN values with some value of their own.
X.fillna(method='ffill',inplace=True)
#Pandas head() method is used to return top n (5 by default) rows of a data frame or series.
print(X.head())

#StandardScaler() function(): # This function Standardize features by removing the mean and scaling to unit variance.
#The standard score of a sample x is calculated as: z = (x – u) / s, where x = variable, u = mean and s = standard deviation

scaler=StandardScaler()
#fit_transform means to do some calculation and then do transformation (say calculating the means of columns from some data and
# then replacing the missing values). So for training set, you need to both calculate and do transformation.
X_scaled=scaler.fit_transform(X)

#Normalization refers to rescaling real valued numeric attributes into the range 0 and 1.
X_normalized=normalize(X_scaled)

#print(X_normalized)
X_normalized=pd.DataFrame(X_normalized)

pca=PCA(n_components=2)
X_principal=pca.fit_transform(X_normalized)
X_principal=pd.DataFrame(X_principal)
X_principal.columns=['P1','P2']
print(X_principal.head())

#Numpy array of all the cluster lABELS ASSIGNED TO EACH DATA POINT

db_default=DBSCAN(eps=0.0375,min_samples=3).fit(X_principal)
labels=db_default.labels_

colours={}
colours[0]='r'
colours[1]='y'
colours[2]='b'
colours[-1]='k'

cvec=[colours[label] for label in labels]

r=plt.scatter(X_principal['P1'],X_principal['P2'],color='r');
ye=plt.scatter(X_principal['P1'],X_principal['P2'],color='y');
b=plt.scatter(X_principal['P1'],X_principal['P2'],color='b');
k=plt.scatter(X_principal['P1'],X_principal['P2'],color='k');

plt.figure(figsize=(9,9))
plt.scatter(X_principal['P1'],X_principal['P2'],c=cvec)
plt.legend((r,ye,b,k),('Label 0','Label 1','Label 2','Label -1'))

plt.show()

db=DBSCAN(eps=0.0375,min_samples=50).fit(X_principal)
labels1=db.labels_

colours1={}
colours1[0]='r'
colours1[1]='y'
colours1[2]='b'
colours1[3]='c'
colours1[4]='y'
colours1[5]='m'
colours1[-1]='k'

cvec=[colours1[label] for label in labels]
colors=['r','y','b','c','y','m','k']

r=plt.scatter(X_principal['P1'],X_principal['P2'],marker='o',color=colors[0])
ye=plt.scatter(X_principal['P1'],X_principal['P2'],marker='x',color=colors[1])
b=plt.scatter(X_principal['P1'],X_principal['P2'],marker='o',color=colors[2])
c=plt.scatter(X_principal['P1'],X_principal['P2'],marker='x',color=colors[3])
y=plt.scatter(X_principal['P1'],X_principal['P2'],marker='o',color=colors[4])
m=plt.scatter(X_principal['P1'],X_principal['P2'],marker='x',color=colors[5])
k=plt.scatter(X_principal['P1'],X_principal['P2'],marker='o',color=colors[-1])

plt.figure(figsize=(9,9))
plt.scatter(X_principal['P1'],X_principal['P2'],c=cvec)
plt.legend((r,ye,b,c,y,m,k),('Label 0','Label 1','Label 2','Label 3','Label 4','Label 5','Label -1'),scatterpoints=1,loc='upper left',ncol=3,fontsize=8)
plt.show()