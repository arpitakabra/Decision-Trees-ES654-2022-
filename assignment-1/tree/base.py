"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

from os import X_OK
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index, variance, loss_function, information_gain_rido

np.random.seed(42)

class DecisionTree():

    def __init__(self, criterion, max_depth=None):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """

        self.max_depth = max_depth if max_depth else 20
        self.criterion = criterion
        self.yhat = None
        self.root_node = None
        self.input_type = None
        self.output_type = None
        
        pass



    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """

        self.input_type = X[0].dtype
        self.output_type = y.dtype

        if self.input_type != 'float64':    #discrete input

            no_features = len(X.columns)
            if no_features < self.max_depth:    #if number of features is less then specified depth, set it as the depth
                self.max_depth = no_features
            
            if self.max_depth > 0:
                
                self.root_node = self.dd_split(X, y)
                self.dd_grow(self.root_node, X, y, 1)

            else:   #if no feature is present then set the predicted value as mean/mode of the outcome
                 self.yhat = self.best_value(y)
        
        else:
            if self.max_depth > 0:

                self.root_node = self.rr_split(X,y)
                self.rr_grow(self.root_node, X, y, 1)

            else:
                self.yhat = self.best_value(y)
        




    def best_value(self, y):    #function to return the most likely/mean outcome based on output is dicrete or real

        best_outcome = 0

        if y.dtype != 'float':

            best_outcome = y.mode()
        
        else:

            best_outcome = y.mean()

        return best_outcome




    def dd_split(self, X, y):   #function to split tree for discrete input

        #splitting dataset
        X_copy = X.copy()   
        X_copy['y'] = y     

        best_feature = None #best feature at each level of the decision tree

        features = list(X.columns)

        if y.dtype != 'float64':   #output is also discrete

            if self.criterion == 'information_gain':    #best split is found using entropy-information gain

                info_gain_max = 0
                for feature in features:

                    feature_sorted = X_copy.sort_values(feature)    #information gain corresponding to each feature
                    info_gain = information_gain(y, X_copy[feature])

                    if info_gain > info_gain_max:
                        best_feature = feature
                        info_gain_max = info_gain   #selecting the best feature based on the maximum information gain
            

            if self.criterion == 'gini_index':   #best split is found using gini index

                sample_space = len(y)   #total number of samples considered for the split
                overall_gini_index = 0
                gini_base = gini_index(y)   #gini index of the overall sample

                for feature in features:    #finding gini index for each attribute in a given feature

                    feature_sorted = X_copy.sort_values(feature)    #sorting the given feature  
                    feature_unique = feature_sorted[feature].unique()   #no. of unique quantitites in a single feature

                    gini_i = 0
                    for i in range(len(feature_unique)):

                        gini_sample = feature_sorted[feature_sorted[feature]==feature_unique[i]]['y']   #sample of output corresponding to each quantity in a given feature
                        sample_count = len(feature_sorted[feature_sorted[feature] == feature_unique[i]])
                        gini_i = gini_i + sample_count/sample_space*gini_index(gini_sample)   #weighted gini index
                    
                    gini_g = gini_base - gini_i #reduction of impurity as compared to overall gini index

                    if gini_g > overall_gini_index:
                        best_feature = feature  #selecting the best feature based on maximum reduction in gini index
                        overall_gini_index = gini_g


        else:   # for real output

            info_gain_max = -999

            for feature in features:    #best feature found using reduction in variance 

                feature_sorted = X_copy.sort_values(feature)
                info_gain = variance(y, X_copy[feature])
                #negative info gain refers to increased variance of the output for a given feature as compared to overall variance
                #positive info gain refers to decreased variance of the output for a given feature as compared to overall variance of the sample space

                if info_gain >= info_gain_max:   #the max info_gain is chosen for the best feature to split
                    best_feature = feature
                    info_gain_max = info_gain

        if best_feature == None:
            best_feature = feature
        if best_feature is not None:

            children = {}
            children['best_feature'] = best_feature

            feature_unique = X_copy[best_feature].unique()
            for i in range(len(feature_unique)):
                temp = X_copy[X_copy[best_feature] == feature_unique[i]]
                temp = temp.drop(best_feature, axis=1)
                children[feature_unique[i]] = temp

        

        return children

        

    def dd_grow(self, node, X, y, depth):    #recursively growing the tree for discrete input

        best_feature = node['best_feature']
        no_children = X[best_feature].nunique()

        if(depth < self.max_depth):

            for i in range(no_children):

                if i in node:

                    X_temp = node[i]
                    y_temp = X_temp['y']
                    X_temp = X_temp.drop('y', axis = 1)

                    if y.dtype != 'float64':

                        if y_temp.nunique() > 1:

                            node[i] = self.dd_split(X_temp, y_temp)
                            self.dd_grow(node[i], X, y, depth+1)
                        
                        else:
                            node[i] = self.best_value(y_temp)
                    
                    else:
                        node[i] = self.dd_split(X_temp, y_temp)
                        self.dd_grow(node[i], X, y, depth+1)        
        else:

            for i in range(no_children):

                if i in node:
                    X_temp = node[i]
                    y_temp = X_temp['y']
                    node[i] = self.best_value(y_temp)
            
            return



    def rr_split(self, X, y):

        #splitting dataset
        X_copy = X.copy()
        features = list(X.columns)
        unique_features = len(features)

        X_copy['y'] = y     

        best_feature = None #best feature at each level of the decision tree
        best_value = None

        if self.output_type == 'float64':

            min_loss = 999
            count = 0

            for feature in features:

                feature_sorted = X_copy.sort_values(feature)    #information gain corresponding to each feature
                local_loss = 998
                local_value = 0
                for i in range(len(y)-1):
                    value = (feature_sorted.iloc[i,count] + feature_sorted.iloc[i+1, count])/2
                    loss = loss_function(feature_sorted['y'], feature_sorted[feature], value)
                    if loss < local_loss:
                        local_loss = loss
                        local_value = value
                
                if local_loss < min_loss:

                    min_loss = local_loss
                    best_feature = feature
                    best_value = local_value
                
                count += 1
        
        else:

            max_info_gain = 0
            best_feature = None
            best_value = None
            count = 0
            for feature in features:

                feature_sorted = X_copy.sort_values(feature)
                
                local_gain = 0
                local_value = 0

                for i in range(len(y)-1):

                    if feature_sorted.iloc[i,unique_features] != feature_sorted.iloc[i+1,unique_features]:

                        value = (feature_sorted.iloc[i, count] + feature_sorted.iloc[i+1, count])/2
                        info_gain = information_gain_rido(feature_sorted['y'], feature_sorted[feature], value)

                        if info_gain > local_gain:

                            local_gain = info_gain
                            local_value = value
                
                if local_gain >= max_info_gain:

                    max_info_gain = local_gain
                    best_feature = feature
                    best_value = local_value
                
                count += 1



        if best_feature is not None:

            children = {}
            children['best_feature'] = best_feature
            children['best_value'] = best_value

            children[0] = X_copy[X_copy[best_feature] < best_value]
            children[1] = X_copy[X_copy[best_feature] >= best_value]

        return children


    
    def rr_grow(self, node, X, y, depth):

        best_feature = node['best_feature']
        best_value = node['best_value']

        if depth < self.max_depth:

            if isinstance(node[0], pd.DataFrame):

                x_temp1 = node[0]
                y_temp1 = x_temp1['y']
                x_temp1 = x_temp1.drop('y', axis = 1)

                if len(y_temp1) == 1:

                    node[0] = self.best_value(y_temp1)
                
                else:

                    if y_temp1.nunique() == 1:
                        node[0] = self.best_value(y_temp1)
                    else:
                        node[0] = self.rr_split(x_temp1, y_temp1)
                        self.rr_grow(node[0], X, y, depth+1)

            if isinstance(node[1], pd.DataFrame):

                x_temp2 = node[1]
                y_temp2 = x_temp2['y']
                x_temp2 = x_temp2.drop('y', axis = 1)

                if len(y_temp2) == 1:

                    node[1] = self.best_value(y_temp2)
                
                else:

                    if y_temp2.nunique() == 1:
                        node[1] = self.best_value(y_temp2)
                    else:
                        node[1] = self.rr_split(x_temp2, y_temp2)
                        self.rr_grow(node[1], X, y, depth+1)
        
        else:

            if isinstance(node[0], pd.DataFrame):

                x_temp1 = node[0]
                y_temp1 = x_temp1['y']
                node[0] = self.best_value(y_temp1)
            
            if isinstance(node[1], pd.DataFrame):

                x_temp2 = node[1]
                y_temp2 = x_temp2['y']
                node[1] = self.best_value(y_temp2)
            
        return
        



    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        predictions = []
        if self.input_type != 'float64':

            for _, x in X.iterrows():
                predict = self.predict_value_dd(self.root_node, x)
                if self.output_type != 'float64':
                    predictions.append(predict[0])
                else:
                    predictions.append(predict)
            
            predictions = pd.Series(np.array(predictions))
        
        else:
            for _, x in X.iterrows():
                predict = self.predict_value_rr(self.root_node, x)
                predictions.append(predict)
            predictions = pd.Series(np.array(predictions))
        return predictions
        pass


    def predict_value_dd(self, node, x):

        if self.max_depth > 0:

            best_feature = node['best_feature']
            if isinstance(node[x[best_feature]], dict):
                return self.predict_value_dd(node[x[best_feature]], x)
            else:
                return node[x[best_feature]]
        else:

            return self.yhat


    def predict_value_rr(self, node, x):

        if self.max_depth > 0:

            best_feature = node['best_feature']
            best_value = node['best_value']
            input_value = x[best_feature]

            if input_value < best_value:
                if isinstance(node[0], dict):
                    return self.predict_value_rr(node[0], x)
                else:
                    if self.output_type == 'float64':
                        return node[0]
                    else:
                        return node[0][0]
            else:
                if isinstance(node[1], dict):
                    return self.predict_value_rr(node[1], x)
                else:
                    if self.output_type == 'float64':
                        return node[1]
                    else:
                        return node[1][0]




    def plot(self):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        if self.input_type != 'float64':
            self.print_tree_dd(self.root_node, 0)
        else:
            self.print_tree_rr(self.root_node, 0)

        pass


    def print_tree_dd(self, node, depth):
        
        print('      '*depth,'Feature: ', node['best_feature'])
        for i in node:

            if i != 'best_feature':

                if isinstance(node[i], dict):
                    print('      '*depth,'Value: ', i)
                    self.print_tree_dd(node[i], depth+1)
                else:
                    print('      '*depth, 'Value: ', i)
                    if self.output_type != 'float64':
                        print('      '*(depth+1), 'Class: ', node[i][0])
                    else:
                        print('      '*(depth+1), 'Predict: ', node[i])  

        return


    def print_tree_rr(self, node, depth):

        print('     '*depth, 'Feature ', node['best_feature'], ' < ', node['best_value'])

        if isinstance(node[0], dict):
            print('     '*depth,'Yes')
            self.print_tree_rr(node[0], depth+1)
        else:
            if self.output_type != 'float64':
                print('     '*depth,'Yes | Class: ', node[0][0])
            else:
                print('     '*depth,'Yes | Predict: ', node[0])

        if isinstance(node[1], dict):
            print('     '*depth,'No')
            self.print_tree_rr(node[1], depth+1)
        else:
            if self.output_type != 'float64':
                print('     '*depth,'No | Class: ', node[1][0])
            else:
                print('     '*depth,'No | Predict: ', node[1])
        
        return

