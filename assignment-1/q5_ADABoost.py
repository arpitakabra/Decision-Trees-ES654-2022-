"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
# from linearRegression.linearRegression import LinearRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def array_to_dataframe(x):
    if not isinstance(X, pd.DataFrame):
        if x.dtype == 'float64':
            x = pd.DataFrame(x)
        else:
            x = pd.DataFrame(x, dtype = 'category')
    x = x.reset_index(drop=True)
    return x

def array_to_series(x):
    if not isinstance(X, pd.Series):
        if x.dtype == 'float64':
            x = pd.Series(x)
        else:
            x = pd.Series(x, dtype = 'category')
    x = x.reset_index(drop= True)
    return x

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################

N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
tree = DecisionTree(criterion=criteria)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))



##### AdaBoostClassifier on Classification data set using the entire data set

# Read dataset
# ...
# 
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)
# print(X.dtype)

#shuffling the dataset
X, y = array_to_dataframe(X), array_to_series(y)
X['y'] = y
X = X.sample(frac=1).reset_index(drop=True)
y = X['y']
X = X.drop('y', axis=1)

#splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=False)

#converting back to dataframe and series
X_train, X_test = array_to_dataframe(X_train), array_to_dataframe(X_test)
y_train, y_test = array_to_series(y_train), array_to_series(y_test)


#implementing Decision Stump using sklearn to compare to the accuracies
sk_tree = DecisionTreeClassifier(max_depth=1)
sk_tree.fit(X_train, y_train)
y_predict = sk_tree.predict(X_test)
y_predict = array_to_series(y_predict)
print()
print('Accuracy on Decision Stump using sklearn')
print()
print('Accuracy: ', accuracy(y_predict, y_test))
for cls in y.unique():
    print('Precision: ', precision(y_predict, y_test, cls))
    print('Recall: ', recall(y_predict, y_test, cls))



#using ADABoost
print()
print('Accuracy on Decision Stump with ADABoost')
print()
Classifier_AB = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=3)
Classifier_AB.fit(X_train, y_train)
y_hat = Classifier_AB.predict(X_test)
[fig1, fig2] = Classifier_AB.plot()
# print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))
# print(X_train)