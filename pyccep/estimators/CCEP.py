# Importing required packages
import numpy as np
from numpy.linalg import inv

def reshape_to_matrix(x):
    """
    Function to reshape a 1D array to a column matrix.

    Parameters:
        x (numpy.ndarray): The 1D array to be reshaped.

    Returns:
        numpy.ndarray: A column matrix containing the elements of the input array.
    """
    return np.reshape(x, (x.size, 1))

def CCEP(model):
    """
    Applies the Common Correlated Effects Pooled estimation in dynamic panels.

    Args:
    - model: A data object that contains the following attributes:
        - T: An integer that represents the number of time periods in the data.
        - N: An integer that represents the number of  cross-sectional units in the data.
        - X: A 2D numpy array with shape (T, N) that contains individual-specific kx x 1 column vector of strictly exogenous regressors for the period t to T.
        - Y: A 2D numpy array with shape (T, N) that contains observations on the dependent variable for the period t to T.

    Returns:
    - delta_hat: A list that contains the estimates of the CCEP.
    """
    # Get the number of time periods and cross-sectional units
    T = model.T
    N = model.N

    # Calculate the cross-sectional averages of X and Y
    cross_sectional_averages_y = np.nansum(model.y, axis=1, dtype='float')
    cross_sectional_averages_X = [np.nansum(cross_sec_avg, axis=1, dtype='float') for cross_sec_avg in model.X]

    # Create a vector of ones and a matrix Q for the CCEP estimator
    Q = np.vstack((np.ones(T), cross_sectional_averages_y, *cross_sectional_averages_X)).T
    
    # Potentially remove certain CSA from the specification
    if (model.CSA != []):
        Q = np.delete(Q, model.CSA, 1)

    # Calculate the projection matrix H and the residual matrix M
    H = np.matmul(np.matmul(Q,inv(np.matmul(Q.transpose(),Q))), Q.transpose())
    M = np.identity(T) - H

    # Calculate the CCEP estimator delta_hat
    first_sum = 0
    second_sum = 0
    for n in range(0,N):
        # Create the augmented data matrix for cross-sectional unit i
        w_i_temp = np.matrix(model.X[0][:,n])  
        for i in range(1,len(model.X)):
            w_i_temp = np.append(w_i_temp, np.matrix(model.X[i][:,n]),axis=0)
        w_i = np.matrix(w_i_temp).transpose()
        y_i = model.y[:,n]
        indices = np.unique(np.append(np.argwhere(np.isnan(w_i))[:,0], np.argwhere(np.isnan(model.y[:,n]))[:,0]  ))
        
        # Check whether the data is unbalanced
        if len(indices)>0:
            model.unbalanced = True
            w_i = np.delete(w_i, indices, axis=0)
            y_i = np.delete(model.y[:,n], indices, axis=0)
            M_adjusted = compute_M_missing_values(Q, indices, T)
             # Calculate the two sums needed for the CCEP estimates
            first_sum += np.matmul(np.matmul(w_i.transpose(),M_adjusted),w_i)
            second_sum += np.matmul(np.matmul(w_i.transpose(),M_adjusted),reshape_to_matrix(y_i))
        else:
            # Calculate the two sums needed for the CCEP estimates
            first_sum += np.matmul(np.matmul(w_i.transpose(),M),w_i)
            second_sum += np.matmul(np.matmul(w_i.transpose(),M),reshape_to_matrix(y_i))
    delta_hat = np.matmul(inv(first_sum),second_sum)

    delta_hat = np.array(delta_hat.transpose()).flatten()
    return delta_hat.tolist()


def compute_M_missing_values(Q, indices, T):
    """
    Function to compute matrix M for handling unbalanced data.

    Parameters:
        Q (numpy.ndarray): The matrix Q.
        indices (list): A list of row indices corresponding to missing values.
        T (int): The total number of time periods.

    Returns:
        numpy.ndarray: The adjusted matrix M.
    """
    # Remove rows corresponding to missing values from the matrix Q
    Q = np.delete(Q, indices, axis=0)

    # Compute the H matrix for valid observations in Q
    H = np.matmul(np.matmul(Q, inv(np.matmul(Q.transpose(), Q))), Q.transpose())

    # Compute the matrix M to handle missing values based on the valid observations
    M = np.identity(T - len(indices)) - H
    return M


