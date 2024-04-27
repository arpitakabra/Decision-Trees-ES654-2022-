from random import seed
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import jaccard_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.n_estimators = n_estimators    
        self.base_estimator = DecisionTreeClassifier()   #decision tree from sklearn is used
        self.strong_classifiers = []  #to store each weak classifier (decision stump)
        self.X = None
        self.y = None
        self.X_samples = []
        self.y_samples = []
        self.no_samples = None

        pass

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.X = X
        self.y = y
        X['y'] = y
        

        for i in range(self.n_estimators):

            no_samples = int(0.7*len(X))    #the number of samples considered in each iteration is 70% of the dataset
            self.no_samples = no_samples

            X_train = X.sample(frac = no_samples/len(X)).reset_index(drop=True)     #randomly selecting n data points
            y_train = X_train['y']
            X_train = X_train.drop('y', axis =1)
            self.X_samples.append(X_train)
            self.y_samples.append(y_train)

            tree = self.base_estimator
            t = tree.fit(X_train,y_train)   #training sklearn decision tree
            self.strong_classifiers.append(t)

        X = X.drop('y', axis = 1)

        pass

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """

        predictions = pd.DataFrame([])

        if 'y' in X:
            X = X.drop('y', axis = 1)
        

        for i in range(self.n_estimators):
            tree = self.strong_classifiers[i]
            y_predict = tree.predict(X)
            predictions[i] = np.array(y_predict)
        
        final = predictions.mode(axis=1)[0]

        return final


        pass

    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """

        dx = 0.05
        
        fig1 = plt.figure()
        for i in range(self.n_estimators):

            plt.subplot(1,self.n_estimators,i+1)
            
            X = self.X_samples[i]

            #getting min and max axis quantitites
            x1_min, x1_max = X.iloc[:, 0].min() - 1, X.iloc[:,0].max() +1
            x2_min, x2_max = X.iloc[:, 1].min() - 1, X.iloc[:,1].max() +1

            X1, X2 = np.meshgrid(np.arange(x1_min, x1_max, dx), np.arange(x2_min, x2_max, dx))

            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

            #decision stump from each iteration
            tree = self.strong_classifiers[i]

            #generating predictions over entire meshgrid
            Z = tree.predict(np.c_[X1.ravel(), X2.ravel()])
            Z = Z.reshape(X1.shape)
            c = plt.contourf(X1, X2, Z, cmap=plt.cm.RdYlBu)     #decision surface plot

            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title('Round {}'.format(i))
            
            #scatter plot for train data in each iteration
            for j in range(self.no_samples):
                if self.y_samples[i][j] == 0:
                    plt.scatter(X.iloc[j,0], X.iloc[j,1], c = 'r', s = 15)
                else:
                    plt.scatter(X.iloc[j,0], X.iloc[j,1], c = 'b', s = 15)
            

        plt.suptitle('Decision surface for each strong classifier')
        plt.show()

        #final classifier plotted based on final predicitons

        fig2 = plt.figure()

        x1_min, x1_max = self.X.iloc[:, 0].min() - 1, self.X.iloc[:,0].max() +1
        x2_min, x2_max = self.X.iloc[:, 1].min() - 1, self.X.iloc[:,1].max() +1

        X1, X2 = np.meshgrid(np.arange(x1_min, x1_max, dx), np.arange(x2_min, x2_max, dx))

        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

        Z_predict = self.predict(np.c_[X1.ravel(), X2.ravel()])
        Z_predict = np.array(Z_predict.tolist())

        Z_predict = Z_predict.reshape(X1.shape)     #predictions from  Bagging 

        p = plt.contourf(X1, X2, Z_predict, cmap = plt.cm.RdYlBu)       #decision surface plot

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        
        #train data scattered plot
        for j in range(len(self.y)):
            if self.y[j] == 0:
                plt.scatter(self.X.iloc[j,0], self.X.iloc[j,1], c = 'r', s = 25)
            else:
                plt.scatter(self.X.iloc[j,0], self.X.iloc[j,1], c = 'b', s = 25)

        plt.suptitle('Bagging Classification')
        plt.show()

        return fig1, fig2

        pass
