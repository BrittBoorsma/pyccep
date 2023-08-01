# pyccep: Common Correlated Effects Pooled Estimator with Bias Corrections

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**pyccep** is a Python package that implements the Common Correlated Effects Pooled (CCEP) estimator with various bias corrections for static and dynamic homogenous panel data models. 

## Installation

To install the **pyccep** package, you can use clone the repository:

```
git clone https://github.com/BrittBoorsma/pyccep.git
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


## Examples

```python

from pyccep.panel_models import HomogenousPanelModel
import pandas as pd

df = pd.read_excel("terrorism.xlsx")  
df["Terrorist Events"] = df["Terrorist Events"]/1000

model = HomogenousPanelModel("Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}", df, "Country Codes", time="Year")
model.fit(estimator="CCEPbc")

"""
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
"""
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

The **pyccep** package is based on the Common Correlated Effects Pooled (CCEP) estimator developed in the academic literature. We acknowledge the contributions of the original authors and researchers in the field of econometrics and panel data analysis.

Please refer to the official documentation and academic literature for more detailed explanations and references on the CCEP estimator and bias corrections.

## References

If you use the **pyccep** package or find it helpful, consider citing the original papers that introduced the CCEP estimator and any relevant bias corrections:

1. De Vos, I. and Everaert, G. (2021). Bias-corrected common correlated effects pooled estimation in dynamic panels. Journal of Business & Economic Statistics, 39(1):294–306. [Link](https://doi.org/10.1080/07350015.2019.1654879)

2. De Vos, I. and Stauskas, O. (2022). Bootstrap-improved inference for cce regressions. Technical report, Working Papers 2021: 16, Lund University, Department of Economics [Link](https://www.researchgate.net/profile/Ignace-De-Vos/publication/362932059_Bootstrap-Improved_Inference_for_CCE_Regressions/links/6307cdc81ddd4470210aaf0a/Bootstrap-Improved-Inference-for-CCE-Regressions.pdf)

3. Pesaran, M. H. (2006). Estimation and inference in large heterogeneous panels with a multifactor error structure. Econometrica, 74(4):967–1012. [Link]( https://doi.org/10.1111/j.1468-0262.2006.00692.x)