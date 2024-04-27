from pyexpat import features
import sklearn
import math
import numpy as np
import pandas as pd
from .base import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt 
from sklearn import tree

class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators = n_estimators  
        if criterion == 'gini_index':
            criterion = 'gini'
        else:
            criterion = 'entropy'  
        self.base_estimator = DecisionTreeClassifier(criterion=criterion)   #decision tree from sklearn is used
        self.features = []
        self.classifiers = []  #to store each weak classifier (decision stump)
        self.X = None
        self.y = None

        pass



    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X
        self.y = y

        no_features = len(X.columns)

        select_features = math.ceil(np.sqrt(no_features))
        tr = self.base_estimator

        for i in range(self.n_estimators):

            X_train = X.sample(n = select_features, axis='columns')

            
            tr.fit(X_train, y)

            self.classifiers.append(tr)
            feat = X_train.columns
            self.features.append(feat)

        pass



    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        predictions = pd.DataFrame([])

        tr = self.base_estimator

        for i in range(self.n_estimators):

            X_train = self.X[self.features[i]]

            tr.fit(X_train, self.y)
            X_test = X[self.features[i]]
            y_hat = tr.predict(X_test)
            predictions[i] = y_hat
            
        
        final = predictions.mode(axis=1)[0]

        return final

        pass



    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """

        tr = self.base_estimator

        for i in range(self.n_estimators):

            X_train = self.X[self.features[i]]

            
            tr.fit(X_train, self.y)
            # t = self.classifiers[i]
            tree.plot_tree(tr)
            plt.title('Tree {}'.format(i+1))

            plt.savefig('assignment-1\q7\q7b_plot_info_gain{}.png'.format(i+1))
            plt.show()


        # fig1 = plt.plot()
        dx = 0.05
        
        plot_predictions = pd.DataFrame([])
        for i in range(self.n_estimators):

            plt.subplot(2,int(self.n_estimators)/2,i+1)

            #for plotting decision surface we will have to select only 2 features
            #from each sample of features, first two features are selected
            X_train = self.X[[self.features[i][0],self.features[i][1]]]
            
            tr.fit(X_train, self.y)
            
            #getting min and max axis quantitites
            x1_min, x1_max = self.X.iloc[:, 0].min() - 1, self.X.iloc[:,0].max() +1
            x2_min, x2_max = self.X.iloc[:, 1].min() - 1, self.X.iloc[:,1].max() +1

            X1, X2 = np.meshgrid(np.arange(x1_min, x1_max, dx), np.arange(x2_min, x2_max, dx))

            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

            #decision stump from each iteration

            #generating predictions over entire meshgrid
            Z = tr.predict(np.c_[X1.ravel(), X2.ravel()])
            plot_predictions[i] = Z
            Z = Z.reshape(X1.shape)
            c = plt.contourf(X1, X2, Z, cmap=plt.cm.coolwarm)     #decision surface plot

            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Tree {}'.format(i))

            X0, X1 = self.X.iloc[:, 0], self.X.iloc[:, 1] 
            #scatter plot for train data in each iteration
            plt.scatter(X0, X1, c=self.y, cmap=plt.cm.coolwarm, s = 15)
            

        plt.suptitle('Decision surface for each decision tree')
        plt.savefig('assignment-1\q7\q7b_rfs_classifier_info_gain.png')
        plt.show()

        # final classifier plotted based on final predicitons

        fig2 = plt.figure()

        x1_min, x1_max = self.X.iloc[:, 0].min() - 1, self.X.iloc[:,0].max() +1
        x2_min, x2_max = self.X.iloc[:, 1].min() - 1, self.X.iloc[:,1].max() +1

        X1, X2 = np.meshgrid(np.arange(x1_min, x1_max, dx), np.arange(x2_min, x2_max, dx))

        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        Z_predict = plot_predictions.mode(axis=1)[0]

        Z_predict = np.array(Z_predict.to_list()).reshape(X1.shape)     #predictions from  Bagging 

        p = plt.contourf(X1, X2, Z_predict, cmap = plt.cm.coolwarm)       #decision surface plot

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        #train data scattered plot
        X0, X1 = self.X.iloc[:, 0], self.X.iloc[:, 1] 
            #scatter plot for train data in each iteration
        plt.scatter(X0, X1, c=self.y, cmap=plt.cm.coolwarm, s = 15)

        plt.suptitle('Random Forest Classifier')
        plt.savefig('assignment-1\q7\q7b_rfs_combined_classifier_info_gain.png')
        plt.show()
            

        pass










class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''

        self.n_estimators = n_estimators   
        self.base_estimator = DecisionTreeRegressor(criterion='squared_error')   #decision tree from sklearn is used
        self.features = []
        self.classifiers = []  #to store each weak classifier (decision stump)
        self.X = None
        self.y = None

        pass

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """

        self.X = X
        self.y = y

        no_features = len(X.columns)

        select_features = math.ceil(np.sqrt(no_features))
        tr = self.base_estimator

        for i in range(self.n_estimators):

            X_train = X.sample(n = select_features, axis='columns')

            
            tr.fit(X_train, y)

            self.classifiers.append(tr)
            feat = X_train.columns
            self.features.append(feat)

        pass

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """

        predictions = pd.DataFrame([])

        tr = self.base_estimator

        for i in range(self.n_estimators):

            X_train = self.X[self.features[i]]

            tr.fit(X_train, self.y)
            X_test = X[self.features[i]]
            y_hat = tr.predict(X_test)
            predictions[i] = y_hat
            
        
        final = predictions.mean(axis=1)

        return final

        pass

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """

        tr = self.base_estimator

        for i in range(self.n_estimators):

            X_train = self.X[self.features[i]]

            
            tr.fit(X_train, self.y)
            # t = self.classifiers[i]
            tree.plot_tree(tr)
            plt.title('Tree {}'.format(i+1))

            # plt.savefig('assignment-1\q7\q7_regressor{}.png'.format(i+1))
            plt.show()


        # fig1 = plt.plot()
        dx = 0.05
        
        plot_predictions = pd.DataFrame([])
        for i in range(self.n_estimators):

            plt.subplot(2,int(self.n_estimators)/2,i+1)

            #for plotting decision surface we will have to select only 2 features
            #from each sample of features, first two features are selected
            X_train = self.X[[self.features[i][0],self.features[i][1]]]
            
            tr.fit(X_train, self.y)
            
            #getting min and max axis quantitites
            x1_min, x1_max = self.X.iloc[:, 0].min() - 1, self.X.iloc[:,0].max() +1
            x2_min, x2_max = self.X.iloc[:, 1].min() - 1, self.X.iloc[:,1].max() +1

            X1, X2 = np.meshgrid(np.arange(x1_min, x1_max, dx), np.arange(x2_min, x2_max, dx))

            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

            #decision stump from each iteration

            #generating predictions over entire meshgrid
            Z = tr.predict(np.c_[X1.ravel(), X2.ravel()])
            plot_predictions[i] = Z
            Z = Z.reshape(X1.shape)
            c = plt.contourf(X1, X2, Z, cmap=plt.cm.coolwarm)     #decision surface plot

            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Tree {}'.format(i))

            X0, X1 = self.X.iloc[:, 0], self.X.iloc[:, 1] 
            #scatter plot for train data in each iteration
            plt.scatter(X0, X1, c=self.y, cmap=plt.cm.coolwarm, s = 15)

        plt.suptitle('Decision surface for each decision tree')
        # plt.savefig('assignment-1\q7\q7_rfs_regressor.png')
        plt.show()

        # final classifier plotted based on final predicitons

        fig2 = plt.figure()

        x1_min, x1_max = self.X.iloc[:, 0].min() - 1, self.X.iloc[:,0].max() +1
        x2_min, x2_max = self.X.iloc[:, 1].min() - 1, self.X.iloc[:,1].max() +1

        X1, X2 = np.meshgrid(np.arange(x1_min, x1_max, dx), np.arange(x2_min, x2_max, dx))

        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        Z_predict = plot_predictions.mean(axis=1)

        Z_predict = np.array(Z_predict.to_list()).reshape(X1.shape)     #predictions from  Bagging 

        p = plt.contourf(X1, X2, Z_predict, cmap = plt.cm.coolwarm)       #decision surface plot

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        #train data scattered plot
        X0, X1 = self.X.iloc[:, 0], self.X.iloc[:, 1] 
            #scatter plot for train data in each iteration
        plt.scatter(X0, X1, c=self.y, cmap=plt.cm.coolwarm, s = 15)

        plt.suptitle('Random Forest Classifier')
        # plt.savefig('assignment-1\q7\q7_rfs_combined_regressor.png')
        plt.show()

        pass
