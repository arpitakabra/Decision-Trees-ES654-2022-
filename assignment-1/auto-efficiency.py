
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor
np.random.seed(42)

# Read real-estate data set
# ...
# 
#reading and prerocessing data: removing car names and empty rows
X = pd.read_csv(r'assignment-1\auto-mpg.data', sep = '\s+', header=None)
X = X.drop(8, axis=1)
X = X.dropna().reset_index(drop=True)

for i in range(len(X.columns)):
    X = X[pd.to_numeric(X[i], errors='coerce').notnull()]

y = X[0]
X = X.drop(0, axis = 1).reset_index(drop=True)
# print(X.iloc[31,:])
y = y.astype(np.float64)
y = y.reset_index(drop=True)

for i in range(len(X.columns)):
    X = X.rename(columns = {i+1:i})
    X[i] = pd.to_numeric(X[i], downcast = 'float')

X[X.select_dtypes(np.float32).columns] = X.select_dtypes(np.float32).astype(np.float64)

tree = DecisionTree(criterion='information_gain', max_depth=15) #Split based on Inf. Gain
tree.fit(X, y)
y_hat = tree.predict(X)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))


#Checking from sklearn tree
tree = DecisionTreeRegressor(max_depth=15)
tree.fit(X,y)
y_predict = tree.predict(X)
print('RMSE: ', rmse(y_predict, y))
print('MAE: ', mae(y_predict, y))
# print(X)
