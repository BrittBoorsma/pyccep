# Importing required packages
from pyccep.estimators.CCEP import CCEP
from pyccep.estimators.CCEPbc import CCEPbc
import numpy as np
from tqdm.auto import tqdm
from copy import deepcopy
from scipy.stats import norm


def get_bootstrap_sdt_error(model, iterations):
    """
    Calculates bootstrap standard errors and related statistics for a given model using the specified number of bootstrap iterations.

    Args:
    - model: The input model object. It should have attributes `coef` (coefficients of the model) and `estimator` (the type of estimator used).
    - iterations: The number of bootstrap iterations to perform.

    Returns:
    A tuple containing the following elements:
    - std_errors: An array of standard errors calculated based on the bootstrap estimates. Each element corresponds to a specific coefficient or parameter.
    - lower_bound: An array of lower bounds of the confidence intervals. Each element corresponds to a specific coefficient or parameter.
    - upper_bound: An array of upper bounds of the confidence intervals. Each element corresponds to a specific coefficient or parameter.
    """

    # Coefficients of the model
    coef = model.coef  

    # Estimator used for calculations
    estimator = model.estimator  

    # Empty list to store bootstrap estimates
    bootstrap_estimates = []  
    
    # Iterate for the specified number of bootstrap iterations
    print('\n-Collecting bootstrap standard errors-')
    for b in tqdm(range(iterations)): 
        # Create a bootstrap sample of the model
        sample_model = bootstrap_sample(deepcopy(model))  
        
        # Depending on the estimator, perform calculations using CCEP or CCEPbc and append results
        if estimator == 'CCEP':
            bootstrap_estimates.append(np.array(CCEP(sample_model)))
        elif estimator == 'CCEPbc':
            bootstrap_estimates.append(np.array(CCEPbc(sample_model)))
    
    # Convert bootstrap estimates to NumPy array
    bootstrap_estimates = np.array(bootstrap_estimates)  
    
    # Calculate lower and upper bounds using 5th and 95th percentiles of bootstrap estimates
    lower_bound = np.percentile(bootstrap_estimates, 2.5, axis=0)
    upper_bound = np.percentile(bootstrap_estimates, 97.5, axis=0)
    
    # Compute standard errors by taking the standard deviation of bootstrap estimates
    std_errors = np.std(bootstrap_estimates,ddof=1, axis=0)

    for c in range(0,len(coef)):
        z = coef[c]/std_errors[c]
        print('p-value van')
        print(coef[c])
        print(norm.sf(abs(z))*2)
        
    return std_errors, lower_bound, upper_bound




def bootstrap_sample(model):
    """
    Generates a bootstrap sample of the given model by resampling the data with replacement.

    Args:
        model: An instance of the model class containing the data and variables.

    Returns:
        model: The updated model instance with the bootstrap sample applied.
    """
    # Randomly choose indices of the data with replacement
    indices = np.random.choice(model.N, model.N)  

    # Create row indices for the bootstrap sample
    rows = np.indices((model.T,)).reshape(-1, 1)  
    
    # Update dependent variable y by selecting rows based on rows and columns based on indices
    model.y = model.y[rows, indices]  
    
    # Select rows and columns for each element in model.X and append to bootstraped sample of X
    X = []
    for x in range(len(model.X)):
        X.append(model.X[x][rows, indices])  
    
    # Update the exogenous regressors (model.X) with the modified bootstraped sample of X
    model.X = X  
    return model