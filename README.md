# pyccep: Common Correlated Effects Pooled Estimator with Bias Corrections

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**pyccep** is a Python package that implements the Common Correlated Effects Pooled (CCEP) estimator with various bias corrections for static and dynamic homogenous panel data models. It provides tools to estimate and analyze panel data with consistent specifications across groups while addressing common correlated effects.

## Installation

To install the **pyccep** package, you can use `pip`:

```
pip install pyccep
```

## Getting Started

### Importing Required Libraries

```python

from prettytable import PrettyTable
import pandas as pd
import numpy as np 
import re
from tqdm.auto import tqdm
from copy import deepcopy
```

### HomogenousPanelModel Class

```python
class HomogenousPanelModel:
    def __init__(self, formula, data, group, time='No time unit', CSA=[]):
        # Constructor and initialization of model attributes
        # ...

    def preprocess_data(self):
        """
        Function to preprocess the data and prepare it for model estimation.

        Parameters:
            None (Uses class attributes: formula, data, group, time, CSA)

        Returns:
            None (Updates class attributes: y, X, T, N, y_name, y_names_formula, X_names, X_names_formula, dynamic, CSA)
        """
        # ...

    def info(self):
        """
        Function to display information about the panel data model.

        Returns:
            None (Prints model information)
        """
        # ...

    def fit(self, estimator='CCEP', iterations=2000, get_std_error=True):
        """
        Function to fit the panel model using the chosen estimator.

        Parameters:
            estimator (str): The estimator to use for model fitting. Options are 'CCEP' or 'CCEPbc'.
            iterations (int): Number of iterations for bootstrapping (if applicable).
            get_std_error (bool): Whether to calculate standard errors using bootstrapping.

        Returns:
            None (Updates the model coefficients and standard errors)
        """
        # ...

    def print_regression_summary(self):
        """
        Function to print a summary of the model regression results.

        Returns:
            None (Prints the regression summary)
        """
        # ...

    def regression_table(self):
        """
        Function to create a table with model regression results.

        Returns:
            str: A string representing the regression results table.
        """
        # ...

    def basic_information(self):
        """
        Function to create a basic information table for the panel model.

        Returns:
            str: A string representing the basic information table.
        """
        # ...
```



## Examples

```python

>>> from pyccep.panel_models import HomogenousPanelModel
>>> import pandas as pd

>>> df = pd.read_excel("terrorism.xlsx")  
>>> df["Terrorist Events"] = df["Terrorist Events"]/1000

>>> model = HomogenousPanelModel("Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}", df, "Country Codes", time="Year")
>>> model.fit(estimator="CCEPbc",get_std_error=False)

Dynamic panel-data estimation, with the CCEPbc estimator
                                                                                     
Model structure:   
 Growth rateᵢₜ = αᵢ + ρGrowth rateᵢₜ₋₁ + βTerrorist Eventsᵢₜ₋₁ + eᵢₜ                       
 eᵢₜ = γ'ᵢfₜ + εᵢₜ                                                                     
                                                                                     
Panel Variable (i):   Country Codes                                                        
Time Variable (t):    Year                                                                 
Number of groups (N): 96                                                                   
Obs per group (T):    38   

+----------------------+-------+-------------------+---------------------+
|    Growth rate       | Coef. |Bootstrap Std. Err.|[95% Conf. Interval] |
+----------------------+-------+-------------------+---------------------+
|  Growth rate_{t-1}   | 0.0433|       0.0438      |  [-0.0119, 0.1616]  |
|Terrorist Events_{t-1}|-0.0513|       0.1421      |  [-0.3587, 0.2068]  |
+----------------------+-------+-------------------+---------------------+
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

The **pyccep** package is based on the Common Correlated Effects Pooled (CCEP) estimator developed in the academic literature. We acknowledge the contributions of the original authors and researchers in the field of econometrics and panel data analysis.

Please refer to the official documentation and academic literature for more detailed explanations and references on the CCEP estimator and bias corrections.