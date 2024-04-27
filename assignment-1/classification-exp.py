import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

np.random.seed(42)

#function to convert dataset to dataframe if not
def array_to_dataframe(x):
    if not isinstance(X, pd.DataFrame):
        if x.dtype == 'float64':
            x = pd.DataFrame(x)
        else:
            x = pd.DataFrame(x, dtype = 'category')
    x = x.reset_index(drop=True)
    return x

#function to convert labels to series if not already
def array_to_series(x):
    if not isinstance(X, pd.Series):
        if x.dtype == 'float64':
            x = pd.Series(x)
        else:
            x = pd.Series(x, dtype = 'category')
    x = x.reset_index(drop= True)
    return x


# Read dataset
# ...
# 
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

#splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



################### Part (a) ###################
print('-------------- Part (a) --------------')
print()

X_train, X_test = array_to_dataframe(X_train), array_to_dataframe(X_test)
y_train, y_test = array_to_series(y_train), array_to_series(y_test)

criteria = 'information_gain'

tree = DecisionTree(criterion=criteria, max_depth=6) #Split based on Inf. Gain
tree.fit(X_train, y_train)
y_hat = tree.predict(X_test)
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y_test.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


################### Part (b) ###################

print()
print('-------------- Part (b) --------------')
print()

kf = KFold(n_splits=5, shuffle=False, random_state=None)

kf.get_n_splits(X_train)

accuracy_matrix = pd.DataFrame([])
criteria = 'information_gain'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test, y_test = array_to_dataframe(X_test), array_to_series(y_test)

for depth in range(1,10):

    print('Depth = ', depth)

    acc_val = []
    acc_test = []

    for train_index, test_index in kf.split(X_train):

        #converting train and validation data to dataframes and series
        x_train, x_test = array_to_dataframe(X_train[train_index]), array_to_dataframe(X_train[test_index])
        yy_train, yy_test = array_to_series(y_train[train_index]), array_to_series(y_train[test_index])

        
        tree = DecisionTree(criterion=criteria, max_depth=depth) #Split based on Inf. Gain
        tree.fit(x_train, yy_train)
        # tree.plot()

        #prediction on validation set
        y_hat = tree.predict(x_test)
        acc_val.append(accuracy(y_hat, yy_test))

        #prediction on test (30%) data
        y_predict = tree.predict(X_test)
        acc_test.append(accuracy(y_predict, y_test))

    print('Average Accuracy for depth ', depth, 'Validation Data: ',np.mean(acc_val), ' | Test Data: ', np.mean(acc_test) )
    print('===========================================')






            
