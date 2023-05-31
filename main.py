from pyccep.monte_carlo_data import generate_data
from pyccep.panel_models.homogenous_dynamic_panel_model import HomogenousDynamicPanelModel
from pyccep.panel_models.homogenous_panel_model import HomogenousPanelModel
import pandas as pd
import numpy as np


rho = 0.8
beta = 1 - rho
RI = 1
m = 1
lambda_ = [0]
theta = 0.6
f_j_init = 0
z_i_init = 0
y_i_init = 0
T = [10, 15, 20,30,50, 100]
N = [25, 50, 100, 500, 1000, 5000]
gamma_u = RI
discard_observations = 50
lags= 1

# experiment 1 Monte Carlo 

df = generate_data(y_i_init,z_i_init, f_j_init, discard_observations, 10,25, beta, rho,lambda_, lags, m, gamma_u,theta )        
# model = HomogenousDynamicPanelModel('y ~ y_{t-1} + X', df, 'N', 'T')
# model.fit(estimator='CCEP', itterations=500)

# model = HomogenousDynamicPanelModel('y ~ y_{t-1} + X', df, 'N', 'T')
# model.fit(estimator='CCEPbc', itterations=500)


model = HomogenousPanelModel('y ~ X + X_{t-1} ', df, 'N', 'T')
model.fit(estimator='CCEPbc')



df = pd.read_csv('data/PanelData.csv')  
# model = HomogenousDynamicPanelModel('Q ~ Q_{t-1} + LF + LF_{t-1} + LF_{t-2}', df, 'I')
# model.fit(estimator='CCEPbc')

# df2 = df.mask(np.random.random(df.shape) < .1)
# df['Q'] = df2['Q']
# df['PF'] = df2['PF']
# df['LF'] = df2['LF']
# df['C'] = df2['C']

# model = HomogenousDynamicPanelModel('Q ~ Q_{t-1} + PF + PF_{t-1} + C + LF_{t-2}', df, 'I')
# model.fit(estimator='CCEPbc',get_std_error=False)

# model = HomogenousDynamicPanelModel('Q ~ Q_{t-1} + PF + PF_{t-1} + C + LF_{t-2}', df, 'I')
# model.fit(estimator='CCEP',get_std_error=False)

# df = pd.read_excel('data/Data Dell et al.xlsx')  
# model = HomogenousDynamicPanelModel('g ~ g_{t-1} + temperature (deg. C) + temperature (deg. C)_{t-1}', df, 'FIPS country code', time='year')
# # model = HomogenousDynamicPanelModel('Q ~ Q_{t-1} + PF + PF_{t-1} + C + LF_{t-2}', df, 'I')
# model.info()
# print(model.X)
# # model.fit()


