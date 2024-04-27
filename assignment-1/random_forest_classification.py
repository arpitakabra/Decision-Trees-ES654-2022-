import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

###Write code here

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

criteria = 'information_gain'
Classifier_RF = RandomForestClassifier(10, criterion = criteria)
Classifier_RF.fit(X_train, y_train)
y_hat = Classifier_RF.predict(X_test)
Classifier_RF.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))