import numpy as np
import pandas as pd

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    correct = 0
    assert(y_hat.size == y.size)
    # TODO: Write here
    for i in range(y.size):
        if y[i] == y_hat[i]:
            correct += 1
    
    accuracy = correct/y.size*100
    return accuracy

    pass

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    precise = 0
    cls_count = 0
    assert(y_hat.size == y.size)
    for i in range(y_hat.size):
        if y_hat[i] == cls:
            cls_count += 1
            if y[i] == y_hat[i]:
                precise += 1
    if cls_count != 0:
        precise = precise/cls_count
    return precise


    pass

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    recal = 0
    cls_count = 0
    assert(y_hat.size == y.size)
    for i in range(y.size):
        if y[i] == cls:
            cls_count += 1
            if y[i] == y_hat[i]:
                recal += 1

    if cls_count != 0:
        recal = recal/cls_count

    return recal

    pass

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    error_sqr = 0
    assert(y_hat.size == y.size)
    for i in range(y.size):
        error_sqr += np.square(y[i]-y_hat[i])
    
    error_sqr = error_sqr/y.size
    
    return np.sqrt(error_sqr)

    pass

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    err = 0
    assert(y_hat.size == y.size)
    for i in range(y.size):
        err += abs(y[i]-y_hat[i])
    
    mean_error = err/y.size
    return mean_error

    pass
