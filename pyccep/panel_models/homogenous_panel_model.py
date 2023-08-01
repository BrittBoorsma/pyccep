# Importing required packages
from pyccep.bootstrap import get_bootstrap_sdt_error
from pyccep.estimators.CCEPbc import CCEPbc
from pyccep.estimators.CCEP import CCEP
from prettytable import PrettyTable
import pandas as pd
import numpy as np 
import re 

# Creating a class for a Homogenous Panel Model
# The HomogenousPanelModel class represents a model for estimating panel data using homogeneous specifications.
# This class provides functionality for preprocessing data, estimating the model, and printing model regression summaries.
class HomogenousPanelModel:
    def __init__(self, formula, data, group, time='No time unit', CSA=[]):
        # Constructor to initialize the model with the given parameters
        self.formula = formula
        self.data = data
        self.group = group
        self.time = time
        self.CSA = CSA

        # Variables to store processed data and results
        self.y = None
        self.X = None
        self.T = None
        self.N = None
        self.y_name = None
        self.y_names_formula = []
        self.X_names = []
        self.X_names_formula = []
        self.dynamic = False

        self.estimator = None
        self.coef = None
        self.std_err = None
        self.lower_bound = None 
        self.upper_bound = None
        self.unbalanced = None
        
        # Preprocess the data upon object creation
        self.preprocess_data()
    
    def preprocess_data(self):
        """
        Function to preprocess the data and prepare it for model estimation.

        Parameters:
            None (Uses class attributes: formula, data, group, time, CSA)

        Returns:
            None (Updates class attributes: y, X, T, N, y_name, y_names_formula, X_names, X_names_formula, dynamic, CSA)
        """
         
        # Helper function to convert DataFrame into numpy matrix format for panel data
        def to_numpy_matrix( df, group):
            g = df.groupby(group).cumcount()
            L = (df.set_index([group,g])
                .unstack(fill_value= np.nan)
                .to_numpy())
            return(L.transpose())

        # Unicode subscripts used for variable naming
        subcript_unicodes =['','\u2081','\u2082','\u2083','\u2084','\u2085','\u2086','\u2087','\u2088','\u2089']
        
        # Extracting the dependent variable name (y_name) and adding unicode subscript
        self.y_name  = self.formula.split("~")[0].strip()
        self.y_names_formula = self.y_name  +'\u1D62\u209C'
        
        # Extracting the dependent variable (y) and converting it into a numpy matrix
        self.y  = to_numpy_matrix(self.data[[self.y_name,self.group]],self.group)
        
        # Extracting and processing explanatory variables (X) from the formula
        explainitory_vars = self.formula.split("~")[1]
        vars = explainitory_vars.split("+")
        vars = [var.strip() for var in vars]
        
        # Checking for duplicates in the formula
        if (any(vars.count(x) > 1 for x in vars)):
            raise Exception('Duplicates found in formula. Rewrite the formula: ' + self.formula)
        
        # Lists to store time-lagged variables, variable names, and dynamic flag
        X = []
        max_lags = 0
        num_vars = 1
        self.dynamic = False

        csa = []
        if self.y_name not in self.CSA:
            csa.append(1)
        count = 2

        # Loop through each variable in the formula
        for x in vars:
            x = x.strip()
            if x not in self.CSA:
                csa.append(count)

            # Handling time-lagged variables
            if '_{t-' in x:
                t = int(re.search('_{t-(.*)}', x).group(1))
                x_stripped = x.replace('_{t-'+ str(t)+'}', '')
                X.append(to_numpy_matrix(self.data[[x_stripped,self.group]],self.group)[:-t])
                
                # Updating max_lags if a higher lag is found
                if t > max_lags:
                    max_lags = t

                # Handling dynamic panel data variables
                if x_stripped == self.y_name and self.dynamic ==False:
                    if (t > 1):
                        raise Exception('Currently this package only supports first-order dynamic panel data models. Change the model to a first-order dynamic panel data model.')
                    self.X_names_formula.append('\u03C1'+x_stripped +'\u1D62\u209C\u208B'+subcript_unicodes[t])
                    self.dynamic = True
                elif (x_stripped == self.y_name ):
                     raise Exception('Currently this package only supports first-order dynamic panel data models. Change the model to a first-order dynamic panel data model.')       
                else:
                    self.X_names_formula.append("\u03B2"+ subcript_unicodes[num_vars]+ x_stripped +'\u1D62\u209C\u208B'+subcript_unicodes[t])
                    num_vars += 1
            else:
                # Handling regular variables
                if (self.y_name in explainitory_vars and len(vars) < 3) or (self.y_name not in explainitory_vars and len(vars) < 2):
                    X.append(to_numpy_matrix(self.data[[x,self.group]],self.group))
                    self.X_names_formula.append("\u03B2"+ x + "\u1D62\u209C")
                    num_vars += 1
                else:
                    X.append(to_numpy_matrix(self.data[[x,self.group]],self.group))
                    self.X_names_formula.append("\u03B2"+ subcript_unicodes[num_vars]+ x + "\u1D62\u209C")
                    num_vars += 1
            self.X_names.append(x)
            count +=1
            
        # Finalizing processed data
        self.y = self.y[max_lags:]
        self.T = self.y.shape[0]
        self.N = self.y.shape[1]

        for i in range(0,len(X)):
            if len(X[i]) > self.T:
                diff = len(X[i]) - self.T
                X[i] = X[i][diff:]
        self.X = X

        if (self.CSA != []):
            self.CSA = csa


    def info(self):
        """
        Function to display information about the panel data model.

        Returns:
            None (Prints model information)
        """
        # Creating the model structure formula string
        s = ' + '.join(self.X_names_formula)
        model_structure = self.y_names_formula + " = \u03B1\u1D62 + " + s + " + e\u1D62\u209C"
        error_structure = "e\u1D62\u209C = \033[1m\u03B3\033[0m'\u1D62\033[1mf\033[0m\u209C + \u03B5\u1D62\u209C"

        # Displaying model information
        print('Model structure:')
        print(model_structure)
        print(error_structure)
        print('\nNumber of groups: ' + str(self.N))
        print('Obs per group (T):  ' + str(self.T))
        print('Panel Variable (i): '+ self.group)
        print('Time Variable (t):  '+ self.time) 


    def fit(self, estimator='CCEP',  iterations = 2000, get_std_error=True):
        """
        Function to fit the panel model using the chosen estimator.

        Args:
            estimator (str): The estimator to use for model fitting. Options are 'CCEP' or 'CCEPbc'.
            iterations (int): Number of iterations for bootstrapping (if applicable).
            get_std_error (bool): Whether to calculate standard errors using bootstrapping.

        Returns:
            None (Updates the model coefficients and standard errors)
        """
        if (self.coef != None):
            raise Exception('Model has already been fitted. Create a new model object.')

        self.estimator = estimator
        if estimator == 'CCEP':  
            self.coef = CCEP(self)
        elif estimator == 'CCEPbc': 
            self.coef = CCEPbc(self)
        
        if get_std_error:
            self.std_err, self.lower_bound, self.upper_bound = get_bootstrap_sdt_error(self, iterations)
        else:
            self.std_err = ['-'] * len(self.X)
            self.lower_bound = ['-'] * len(self.X)
            self.upper_bound = ['-'] * len(self.X)

        self.print_regression_summary()
         

    def print_regression_summary(self):
        """
        Function to print a summary of the model regression results.

        Returns:
            None (Prints the regression summary)
        """
        to_print = []
        if (self.dynamic == True):
            to_print.append(' Dynamic panel-data estimation, with the '+str(self.estimator)+' estimator')
        else:
            to_print.append(' Static panel-data estimation, with the '+str(self.estimator)+' estimator')
        to_print.append(self.basic_information())
        to_print.append(self.regression_table())
        for line in to_print:
            print(line)

    def regression_table(self):
        """
        Function to create a table with model regression results.

        Returns:
            str: A string representing the regression results table.
        """
        # Getting model coefficients and standard errors
        coeff = self.coef
        std_err = self.std_err
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound

        # Creating a DataFrame to store regression results
        self.regression_table = pd.DataFrame(list(zip(self.X_names, coeff, std_err,  lower_bound, upper_bound)),
                                             columns=['variable', 'coefficient', 'std_err',  'lower_bound',
                                                      'upper_bound'])

        dep_name = self.y_name
        r_table = PrettyTable()
        r_table.field_names = [dep_name, "Coef.", "Bootstrap Std. Err.", "[95% Conf. Interval] "]
        r_table.float_format = '.7'
        regression_table = self.regression_table
        num_indep = len(regression_table.index)

        # Populating the PrettyTable with regression results
        for i in range(num_indep):
            var_name = regression_table['variable'][i]
            coeff = regression_table['coefficient'][i]
            std_err = regression_table['std_err'][i]
            lower_bound = regression_table['lower_bound'][i]
            upper_bound = regression_table['upper_bound'][i]
            try:
                r_table.add_row([var_name, format(coeff, '.4f'), format(std_err, '.4f'), [np.round(lower_bound, 4), np.round(upper_bound, 4)]])
            except:
                r_table.add_row([var_name, format(coeff, '.4f'), std_err, [lower_bound, upper_bound]])

        return r_table.get_string()

    

    def basic_information(self):
        """
        Function to create a basic information table for the panel model.

        Returns:
            str: A string representing the basic information table.
        """
        basic_table = PrettyTable()
        middle_space = '         '
        basic_table.field_names = ["    ", "   ", "  "]
        basic_table.border = False
        basic_table.header = False
        basic_table.align = 'l'

        s = ' + '.join(self.X_names_formula)

        # Populating the basic information table with model details
        basic_table.add_row([middle_space, middle_space, middle_space])
        basic_table.add_row(['Model structure: ', middle_space, middle_space])
        basic_table.add_row([self.y_names_formula + " = \u03B1\u1D62 + " + s + " + e\u1D62\u209C", middle_space, middle_space])
        basic_table.add_row(["e\u1D62\u209C = \033[1m\u03B3\033[0m'\u1D62\033[1mf\033[0m\u209C + \u03B5\u1D62\u209C", middle_space, middle_space])
        basic_table.add_row([middle_space, middle_space, middle_space])
        basic_table.add_row(['Panel Variable (i):   '+ self.group, middle_space, middle_space])
        basic_table.add_row(['Time Variable (t):    '+ self.time, middle_space, middle_space])
        basic_table.add_row(['Number of groups (N): ' + str(self.N), middle_space, middle_space])
        basic_table.add_row(['Obs per group (T):    ' + str(self.T), middle_space, middle_space])

        return basic_table.get_string()



