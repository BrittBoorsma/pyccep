from bootstrap import get_bootstrap_sdt_error
from estimators.CCEPbc import CCEPbc
from estimators.CCEP import CCEP
from prettytable import PrettyTable
import pandas as pd
import numpy as np 
import re 


class HomogenousPanelModel:
    def __init__(self, formula, data, group, time='No time unit found'):
        self.formula = formula
        self.data = data
        self.group = group
        self.time = time
        self.y = None
        self.X = None
        self.T = None
        self.N = None
        self.y_name = None
        self.y_names_formula = []
        self.X_names = []
        self.X_names_formula = []

        self.estimator = None
        self.coef = None
        self.std_err = None
        self.p_value = None 
        self.lower_bound = None 
        self.upper_bound = None
        self.dynamic = False

        self.preprocess_data()
    
    def preprocess_data(self):
        subcript_unicodes =['','\u2081','\u2082','\u2083','\u2084','\u2085','\u2086','\u2087','\u2088','\u2089']
        self.y_name  = self.formula.split("~")[0].strip()
        self.y_names_formula = self.y_name  +'\u1D62\u209C'
        self.y  = self.to_numpy_matrix(self.data[[self.y_name,self.group]],self.group)
        explainitory_vars = self.formula.split("~")[1]
        vars = explainitory_vars.split("+")
        X = []
        max_lags = 0
        num_vars = 1
        lim_rho = False
        for x in vars:
            x = x.strip()
            if '_{t-' in x:
                t = int(re.search('_{t-(.*)}', x).group(1))
                x_stripped = x.replace('_{t-'+ str(t)+'}', '')
                X.append(self.to_numpy_matrix(self.data[[x_stripped,self.group]],self.group)[:-t])
                if t > max_lags:
                    max_lags = t

                if x_stripped == self.y_name and lim_rho ==False:
                    self.X_names_formula.append('\u03C1'+x_stripped +'\u1D62\u209C\u208B'+subcript_unicodes[t])
                    lim_rho = True
                else:
                    self.X_names_formula.append("\u03B2"+ subcript_unicodes[num_vars]+ x_stripped +'\u1D62\u209C\u208B'+subcript_unicodes[t])
                    num_vars += 1
            else:
                if (self.y_name in explainitory_vars and len(vars) < 3) or (self.y_name not in explainitory_vars and len(vars) < 2):
                    X.append(self.to_numpy_matrix(self.data[[x,self.group]],self.group))
                    self.X_names_formula.append("\u03B2"+ x + "\u1D62\u209C")
                    num_vars += 1
                else:
                    X.append(self.to_numpy_matrix(self.data[[x,self.group]],self.group))
                    self.X_names_formula.append("\u03B2"+ subcript_unicodes[num_vars]+ x + "\u1D62\u209C")
                    num_vars += 1
            self.X_names.append(x)
            
        self.y = self.y[max_lags:]
        self.T = self.y.shape[0]
        self.N = self.y.shape[1]

        for i in range(0,len(X)):
            if len(X[i]) > self.T:
                diff = len(X[i]) - self.T
                X[i] = X[i][diff:]
        self.X = X

        if  lim_rho == True:
            raise Exception("The current model is considered a dynamic model. Please remove the dynamic component. For a dynamic model use: HomogenousDynamicPanelModel")


    def to_numpy_matrix(self, df, group):
        g = df.groupby(group).cumcount()
        L = (df.set_index([group,g])
            .unstack(fill_value= np.nan)
            .to_numpy())
        return(L.transpose())


    def info(self):
        s = ' + '.join(self.X_names_formula)
        print('Model structure:')
        print(self.y_names_formula +" = \u03B1\u1D62 + "+ s+ " + e\u1D62\u209C")
        print("e\u1D62\u209C = \033[1m\u03B3\033[0m'\u1D62\033[1mf\033[0m\u209C + \u03B5\u1D62\u209C")

        print('\nNumber of groups: ' + str(self.N))
        print('Obs per group (T):  ' + str(self.T))
        print('Panel Variable (i): '+ self.group)
        print('Time Variable (t):  '+ self.time) 


    def fit(self, estimator='CCEP',  itterations = 2000, get_std_error=True):
        self.estimator = estimator
        if estimator == 'CCEP':  
            self.coef = CCEP(self)
        elif estimator == 'CCEPbc': 
            self.coef= CCEPbc(self)
        
        if get_std_error == True:
            self.std_err, self.p_value, self.lower_bound, self.upper_bound  = get_bootstrap_sdt_error(self,itterations)
        else:
            self.std_err  = ['-'] * len(self.X)
            self.p_value  = ['-'] * len(self.X) 
            self.lower_bound  = ['-'] * len(self.X)
            self.upper_bound  = ['-'] * len(self.X)

        self.print_regression_summary()
         

    def print_regression_summary(self):
        to_print = []
        to_print.append(' Dynamic panel-data estimation, with the '+str(self.estimator)+' estimator')
        to_print.append(self.basic_information())
        to_print.append(self.regression_table())
        # to_print.append(self.test_results(model))
        for line in to_print:
            print(line)

    def regression_table(self):
        coeff = self.coef
        std_err = self.std_err
        p_value = self.p_value
        lower_bound = self.lower_bound
        upper_bound = self.upper_bound
        self.regression_table = pd.DataFrame(list(zip(self.X_names, coeff, std_err,  p_value, lower_bound, upper_bound)),
                                             columns=['variable', 'coefficient', 'std_err', 'p_value', 'lower_bound',
                                                      'upper_bound'])
        dep_name = self.y_name

        r_table = PrettyTable()

        r_table.field_names = [dep_name, "Coef.", "Bootstrap Std. Err.", "p-value", "[95% Conf. Interval] "]

        r_table.float_format = '.7'
        regression_table = self.regression_table
        num_indep = len(regression_table.index)

        for i in range(num_indep):
            var_name = regression_table['variable'][i]
            coeff = regression_table['coefficient'][i]
            std_err = regression_table['std_err'][i]

            p = regression_table['p_value'][i]
            lower_bound = regression_table['lower_bound'][i]
            upper_bound = regression_table['upper_bound'][i]
            try :
                r_table.add_row([var_name, format(coeff, '.4f'), format(std_err, '.4f'), format(p, '.4f'), [np.round(lower_bound, 4), np.round(upper_bound, 4)]])
            except:
                r_table.add_row([var_name, format(coeff, '.4f'), std_err, p, [lower_bound, upper_bound]])
        return r_table.get_string()

    

    def basic_information(self):

        basic_table = PrettyTable()
        middle_space = '         '
        basic_table.field_names = ["    ", "   ", "  "]
        basic_table.border = False
        basic_table.header = False
        basic_table.align = 'l'

        s = ' + '.join(self.X_names_formula)
        basic_table.add_row(
            [ middle_space, middle_space,middle_space
             ])
        basic_table.add_row(
            ['Model structure: ', middle_space,middle_space
             ])
        basic_table.add_row(
            [ self.y_names_formula +" = \u03B1\u1D62 + "+ s+ " + e\u1D62\u209C", middle_space,middle_space
             ])
        basic_table.add_row(
            ["e\u1D62\u209C = \033[1m\u03B3\033[0m'\u1D62\033[1mf\033[0m\u209C + \u03B5\u1D62\u209C",middle_space,middle_space])
        basic_table.add_row(
            [ middle_space, middle_space,middle_space
             ])
        basic_table.add_row(
            ['Panel Variable (i):   '+ self.group, middle_space,
             middle_space])
        basic_table.add_row(
            ['Time Variable (t):    '+ self.time, middle_space,
             middle_space])
        basic_table.add_row(
            ['Number of groups (N): ' + str(self.N), middle_space,
             middle_space])
        basic_table.add_row(
            ['Obs per group (T):    ' + str(self.T), middle_space,
             middle_space])


        return (basic_table.get_string())



