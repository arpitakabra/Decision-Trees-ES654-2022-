from cmath import nan
from codecs import ignore_errors
import math
import pandas as pd
import numpy as np

# np.random.seed(42)

# N = 10
# P = 3
# X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(P)})
# y = pd.Series(np.random.randint(P, size = N), dtype="category")

def entropy(Y):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    dict = {}   #stores the number of times each calss occur
    for i in Y:
        if i in dict:
            dict[i] += 1
        else:
            dict[i] = 1
    
    total = sum(dict.values())  #total sample space
    e = 0   #entropy
    for i in dict:
        e = e - dict[i]/total*math.log2(dict[i]/total)  #summation of product of probability and log2 of probability of each class for entropy
    
    return e
    pass


def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    dict = {}    #stores the number of times each calss occur
    for i in Y:
        if i in dict:
            dict[i] += 1
        else:
            dict[i] = 1
    
    total = sum(dict.values())  #total sample space

    g = 0
    for i in dict:
        g = g + (dict[i]/total)**2 #gini index is 1 - summation of square of porbability of each class

    g = 1 - g

    return g
    pass



def information_gain(Y, attr):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    # print(attr)
    # print(Y)
    entropy_y = entropy(Y)  #entropy of the values
    total_samples = len(Y)
    unique_attr = attr.unique() #unique entries in attribute
   
    attr_entropy = 0
    Y = Y.rename('y')
    attr = pd.DataFrame(attr)
    attr = attr.join(Y)

    info_gain = 0
    # print(attr.sort_values(0))
    for i in unique_attr:

        temp = entropy(attr[attr.iloc[:, 0]==i]['y'])
        attr_entropy += len(attr[attr.iloc[:, 0]==i]['y'])/total_samples*temp
        # print('i=', i, '|  temp = ', temp,  '  | probabiility: ', len(attr[attr[0]==i]['y'])/total_samples, '  |  attr_e = ', attr_entropy)

    info_gain = entropy_y - attr_entropy
    return info_gain

    pass


def variance(Y, attr):   #for discrete input real output, information gain is found using variance

    #returns reduction of variance for the given feature as compared to overall variance
    attr = attr.sort_values(0)
    attr_unique = attr.unique()
    variance_y = Y.var()    #overall variance of the output

    Y = Y.rename('y')
    attr = pd.DataFrame(attr)
    attr = attr.join(Y)

    var_attr = 0

    for i in range(len(attr_unique)):   #weighted variance for each of the attribute quantitites based on probability of the attribute

        temp1 = attr[attr.iloc[:,0] == attr_unique[i]]['y']

        temp = temp1.var()
        var_attr += len(temp1)/len(Y)*temp

    info_gain = variance_y - var_attr   #information gain is measure of reduction in variance
    if math.isnan(info_gain):
        info_gain = 0

    return info_gain
        
def loss_function(Y, attr, value):

    Y = Y.rename('y')
    attr = pd.DataFrame(attr)
    attr = attr.join(Y)

    y1 = attr[attr.iloc[:,0]<value]['y']
    y1_loss = (abs(y1 - y1.mean())).sum()

    y2 = attr[attr.iloc[:,0]>=value]['y']
    y2_loss = (abs(y2 - y2.mean())).sum()

    return y1_loss+y2_loss


def information_gain_rido(Y, attr, value):

    entropy_y = entropy(Y)
    total_samples = len(Y)
    attr = pd.DataFrame(attr)
    Y = Y.rename('y')
    attr = attr.join(Y)

    entropy_1 = entropy(attr[attr.iloc[:,0] < value]['y'])
    len1 = len(attr[attr.iloc[:,0] < value]['y'])

    entropy_2 = entropy(attr[attr.iloc[:,0] >= value]['y'])
    len2 = len(attr[attr.iloc[:,0] >= value]['y'])

    info_gain = entropy_y - len1/total_samples*entropy_1 - len2/total_samples*entropy_2

    return info_gain

    
# print(y.sort_values())
# print(X[0])
# print(entropy(y))
# print(information_gain(y,X[0]))

