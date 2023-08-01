# Importing required packages
import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize
from pyccep.estimators.CCEP import CCEP
from copy import deepcopy

def CCEPbc(model):
    """
    Applies a Bias-Corrected Common Correlated Effects Pooled estimation in dynamic panels.

    Args:
    - model: A data object that contains the following attributes:
        - T: An integer that represents the number of time periods in the data.
        - N: An integer that represents the number of  cross-sectional units in the data.
        - X: A 2D numpy array with shape (T, N) that contains individual-specific kx x 1 column vector of strictly exogenous regressors for the period t to T.
        - Y: A 2D numpy array with shape (T, N) that contains observations on the dependent variable for the period t to T.

    Returns:
    - delta_hat_bc: A list that contains the estimates of the CCEPbc.
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
        indices = np.unique(np.append(np.argwhere(np.isnan(w_i))[:,0], np.argwhere(np.isnan(model.y[:,n]))[:,0]))
        if len(indices)>0:
            raise Exception('Bias-corrections are not supported for unbalanced panels.') 
        else:
            # Calculate the two sums needed for the CCEP estimates
            first_sum += np.matmul(np.matmul(w_i.transpose(),M),w_i)
            second_sum += np.matmul(np.matmul(w_i.transpose(),M),reshape_to_matrix(y_i))
    delta_hat = np.matmul(inv(first_sum),second_sum)


    if model.dynamic == True:
        # Calculate the dynamic CCEPbc estimator delta_hat_bc using a nummerical solver.
        SIGMA_hat = 1/(model.N * (model.T)) *first_sum
        delta_hat_bc = minimize(
            solver,
            np.copy(delta_hat),
            args=(delta_hat, model,H,SIGMA_hat,M, Q.shape[1], delta_hat.shape,Q),
            tol=0
        ).x.reshape(delta_hat.shape)

    elif model.dynamic == False:
        # Calculate the static CCEPbc estimator delta_hat_bc using a cross-sectional bootstrap.
        delta_hat_bc = boostrap_bias_correction(model,delta_hat,2000)

    delta_hat_bc = delta_hat_bc.flatten().tolist()
    return delta_hat_bc



def solver(d_0, delta_hat, model, H, SIGMA_hat, M, c, shape,Q):
    """
    Solves for the bias correction in the CCEP method.
    
    Args:
        d_0 (numpy array): Initial estimate of the delta vector.
        delta_hat (numpy array): Estimate of the delta vector.
        model (object): Object containing data model for the CCEP method.
        H (numpy matrix): Projection matrix.
        SIGMA_hat (numpy array): Estimate of the covariance matrix.
        M (numpy matrix): Orthogonal projection matrix.
        c (int): Number of constraints.
        shape (tuple): Shape of the delta vector.
        
    Returns:
        float: Value of the objective function.
    """
    
    # Compute the sum of the squared norm of the residuals.
    sum = 0
    d_0 = d_0.reshape(shape)
    for n in range(0, model.N):
        w_i_temp = np.matrix(model.X[0][:,n])  
        for i in range(1,len(model.X)):
            w_i_temp = np.append(w_i_temp, np.matrix(model.X[i][:,n]),axis=0)
        w_i = np.matrix(w_i_temp).transpose()
        y_i = model.y[:,n]
        indices = np.unique(np.append(np.argwhere(np.isnan(w_i))[:,0], np.argwhere(np.isnan(model.y[:,n]))[:,0]  ))

        if len(indices)>0:
            raise Exception('Bias-corrections are not supported for unbalanced panels.') 
        else:
           sigma_sum = np.matmul(M,(reshape_to_matrix(y_i)- np.matmul(w_i,d_0)))
        sum += np.power(np.linalg.norm(sigma_sum,'fro'),2) 
        
    # Compute the estimate of the unkown variance.
    sigma_eta = 1/(model.N*(model.T-c)) * sum.flat[0]
    
    # Compute the estimate of the linear term.
    rho_sum = 0
    for t in range(1, model.T):
        h_sum = 0
        for s in range(t+1,model.T):
            h_sum += H.item((s-1,s-t-1))
        rho_sum += np.power(d_0[0], t-1) *h_sum
    q_1 = np.matrix([[1.0]])
    q_1 = np.append(q_1, np.zeros((len(model.X)-1,1)), axis=0).transpose()
    v = (rho_sum * q_1).transpose()

    # Compute the feasible version of the asymptotic bias expression
    m_hat = d_0 - (1/(model.T)) * np.matmul(sigma_eta*inv(SIGMA_hat),v)
    
    # Compute the objective function to minimize.
    return 0.5 * np.power(np.linalg.norm((delta_hat - m_hat), 'fro'),2)

def boostrap_bias_correction(model, delta_hat, iterations=2000):
    """
    Function to compute bootstrap bias-corrected estimates of a parameter.

    Parameters:
        model (HomogenousPanelModel): The panel data model to perform bias correction on.
        delta_hat (numpy.ndarray): The estimated parameter that requires bias correction.
        iterations (int): The number of bootstrap samples to generate for bias correction.

    Returns:
        numpy.ndarray: An array containing the bias-corrected estimates.
    """
    delta_hat_bc = []
    for i in range(iterations):
        sample = bootstrap_sample(deepcopy(model))
        delta_hat_bc.append(CCEP(sample))

    # Calculate the mean of the bootstrap estimates
    delta_mean = np.mean(np.matrix(delta_hat_bc), axis=0)

    # Calculate the bias-corrected estimates using the delta_hat and delta_mean
    bc = 2 * delta_hat - delta_mean.transpose()
    return np.array(bc)

def bootstrap_sample(model):
    """
    Function to generate a bootstrap sample from the panel data model.

    Parameters:
        model (HomogenousPanelModel): The panel data model to bootstrap.

    Returns:
        HomogenousPanelModel: A new HomogenousPanelModel instance with the bootstrap sample.
    """
    # Randomly choose indices of the data with replacement
    indices = np.random.choice(model.N, model.N)

    # Create row indices for the bootstrap sample
    rows = np.indices((model.T,)).reshape(-1, 1)

    # Update dependent variable y by selecting rows based on rows and columns based on indices
    model.y = model.y[rows, indices]

    # Select rows and columns for each element in model.X and append to the bootstrapped sample of X
    X = []
    for x in range(len(model.X)):
        X.append(model.X[x][rows, indices])

    # Update the exogenous regressors (model.X) with the modified bootstrapped sample of X
    model.X = X
    return model

def reshape_to_matrix(x):
    """
    Function to reshape a 1D array to a column matrix.

    Parameters:
        x (numpy.ndarray): The 1D array to be reshaped.

    Returns:
        numpy.ndarray: A column matrix containing the elements of the input array.
    """
    return np.reshape(x, (x.size, 1))
