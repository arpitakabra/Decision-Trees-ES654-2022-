from asyncore import compact_traceback
from zlib import Z_RLE
import matplotlib
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import jaccard_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
# %matplotlib inline 

class AdaBoostClassifier():

    def __init__(self, base_estimator, n_estimators): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.n_estimators = n_estimators    
        self.base_estimator = DecisionTreeClassifier(max_depth=1)   #decision tree from sklearn is used
        self.alpha = [0]*self.n_estimators  #array to store alpha with each iteration
        self.weak_classifiers = []  #to store each weak classifier (decision stump)
        self.X = None
        self.y = None
        self.weight_list = []

        pass

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """

        self.X = X
        self.y = y
        weights = [1/len(y)]*len(y)     #initialising equal weights for all elements

        for i in range(self.n_estimators):

            tree = self.base_estimator
            t = tree.fit(X,y,sample_weight=weights) #fitting decision stump
            self.weak_classifiers.append(t)  #saving decision stump for this iteration
            self.weight_list.append(weights)    #saving weights of thi iteration

            y_hat = tree.predict(X)             #predicted output

            weight_error = self.weighted_error(y, y_hat, weights)   #normalised summation of error 
            self.alpha[i] = 0.5*np.log((1-weight_error)/weight_error)   #updating alpha

            #updating weights
            for j in range(len(y)):
                if y_hat[j] == y[j]:
                    weights[j] = weights[j]*np.exp(-self.alpha[i])
                else:
                    weights[j] = weights[j]*np.exp(self.alpha[i])
            
            weights = weights/sum(weights)  #normalising weights such that sum(weights) = 1

        pass


    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """

        predictions = pd.DataFrame([])  #predictions from each iteration is added as a column to the dataframe
        final = [0]*len(X)

        for i in range(self.n_estimators):
            tree = self.weak_classifiers[i]
            y_hat = tree.predict(X) #predictions
            predictions[i] = y_hat
            predictions[i] = predictions[i].replace(to_replace=0, value=-1)*self.alpha[i]   #final prediction = sign(alpha_1*predictions of stump 1 + ...)
            final += predictions[i]
        
        final = pd.Series(final)

        #predicting 0 or 1 based on sign of the values weighted by alpha
        final[final<=0] = 0
        final[final>0] = 1

        return final
        pass
        

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        dx = 0.05
        classes = self.y.nunique()
        

        fig1 = plt.figure()
        for i in range(self.n_estimators):

            plt.subplot(1,self.n_estimators,i+1)

            weights = self.weight_list[i]

            #getting min and max axis quantitites
            x1_min, x1_max = self.X.iloc[:, 0].min() - 1, self.X.iloc[:,0].max() +1
            x2_min, x2_max = self.X.iloc[:, 1].min() - 1, self.X.iloc[:,1].max() +1

            X1, X2 = np.meshgrid(np.arange(x1_min, x1_max, dx), np.arange(x2_min, x2_max, dx))

            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

            #decision stump from each iteration
            tree = self.weak_classifiers[i]

            #generating predictions over entire meshgrid
            Z = tree.predict(np.c_[X1.ravel(), X2.ravel()])
            Z = Z.reshape(X1.shape)
            c = plt.contourf(X1, X2, Z, cmap=plt.cm.RdYlBu)     #decision surface plot

            alpha = self.alpha[i]
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('alpha = {:.3f}'.format(alpha))
            
            #scatter plot for train data with size corresponding to weight in each iteration
            for j in range(len(self.y)):
                if self.y[j] == 0:
                    plt.scatter(self.X.iloc[j,0], self.X.iloc[j,1], c = 'r', s = 150*weights[j])
                else:
                    plt.scatter(self.X.iloc[j,0], self.X.iloc[j,1], c = 'b', s = 150*weights[j])
            # for j, color in zip(range(classes), 'ry'):

            #     idex = np.where(self.y==j)
            #     plt.scatter(self.X.iloc[idex, 0], self.X.iloc[idex, 1], c=color, s=weights[idex],)

        plt.suptitle('Decision surface for each weak classifier')
        plt.show()

        #final classifier plotted based on final predicitons

        fig2 = plt.figure()

        x1_min, x1_max = self.X.iloc[:, 0].min() - 1, self.X.iloc[:,0].max() +1
        x2_min, x2_max = self.X.iloc[:, 1].min() - 1, self.X.iloc[:,1].max() +1

        X1, X2 = np.meshgrid(np.arange(x1_min, x1_max, dx), np.arange(x2_min, x2_max, dx))

        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        Z_predict = self.predict(np.c_[X1.ravel(), X2.ravel()])
        Z_predict = np.array(Z_predict.tolist())

        Z_predict = Z_predict.reshape(X1.shape)     #predictions from  ADABoost 

        p = plt.contourf(X1, X2, Z_predict, cmap = plt.cm.RdYlBu)       #decision surface plot

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        #train data scattered plot
        for j in range(len(self.y)):
            if self.y[j] == 0:
                plt.scatter(self.X.iloc[j,0], self.X.iloc[j,1], c = 'r', s = 25)
            else:
                plt.scatter(self.X.iloc[j,0], self.X.iloc[j,1], c = 'b', s = 25)

        plt.suptitle('ADABoost Classifier')
        plt.show()

        return fig1, fig2

        pass



    def weighted_error(self, y, y_hat, weights):

        #the function returns the normalised sum of error in the weights after prediction
        if isinstance(y, pd.Series):

            y = np.array(y.tolist())
        
        if isinstance(y_hat, pd.Series):
            y_hat = np.array(y_hat.tolist())

        error = sum(weights*np.not_equal(y,y_hat))

        error = error/sum(weights)

        return error


#References:
# ADABoost implementation: https://towardsdatascience.com/adaboost-from-scratch-37a936da3d50
# Plotting Decision Surface: https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html#sphx-glr-auto-examples-tree-plot-iris-dtc-py