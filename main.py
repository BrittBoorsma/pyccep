from pyccep.panel_models.homogenous_panel_model import HomogenousPanelModel
import pandas as pd
import numpy as np



df =  pd.read_excel('pyccep/data/terrorism.xlsx')  
df['Delta Terrorist Events'] = df['Delta Terrorist Events']/1000
print('_______________START________________')

df_america = df[df['Region'] == 'america']
df_asia = df[df['Region'] == 'asia']
df_me_na = df[df['Region'] == 'middle east']
df_ss_afrika = df[df['Region'] == 'africa sub sahara']
df_w_europe = df[df['Region'] == 'europe']



model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df, 'Country Codes', time='Year', CSA=['Delta Terrorist Events_{t-1}', 'Growth rate_{t-1}'])
model.info()
print(model.X)
print(model.X_names)
print(model.X_names_formula)
print(model.y_names_formula)
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df, 'Country Codes', 'Year')
# model.fit(estimator='CCEPbc')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df, 'Country Codes', 'Year')
model.fit(estimator='CCEP', get_std_error= False)

# print('______________________________________________________________America')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_america, 'Country Codes', 'Year')
# model.fit(estimator='CCEPbc')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_america, 'Country Codes', 'Year')
# model.fit(estimator='CCEP')

# print('_____________________________________________________________Asia')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_asia, 'Country Codes', 'Year')
# model.fit(estimator='CCEPbc')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_asia, 'Country Codes', 'Year')
# model.fit(estimator='CCEP')

# print('_____________________________________________________________Middle East and North Afrika')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_me_na, 'Country Codes', 'Year')
# model.fit(estimator='CCEPbc')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_me_na, 'Country Codes', 'Year')
# model.fit(estimator='CCEP')

# print('_____________________________________________________________Sub Saharah Afrika')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_ss_afrika, 'Country Codes', 'Year')
# model.fit(estimator='CCEPbc')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_ss_afrika, 'Country Codes', 'Year')
# model.fit(estimator='CCEP')

# print('_____________________________________________________________Western Europe')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_w_europe, 'Country Codes', 'Year')
# model.fit(estimator='CCEPbc')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_w_europe, 'Country Codes', 'Year')
# model.fit(estimator='CCEP')


# print('_____________________________________________________________Devided by population')
# df =  pd.read_excel('pyccep/data/terrorism.xlsx')  
# df['Delta Terrorist Events'] = df['Delta Terrorist Events']/(df['POP']*1000)
# print(df.head())

# df_america = df[df['Region'] == 'america']
# df_asia = df[df['Region'] == 'asia']
# df_me_na = df[df['Region'] == 'middle east']
# df_ss_afrika = df[df['Region'] == 'africa sub sahara']
# df_w_europe = df[df['Region'] == 'europe']

# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df, 'Country Codes', 'Year')
# model.fit(estimator='CCEPbc')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df, 'Country Codes', 'Year')
# model.fit(estimator='CCEP')

# print('_____________________________________________________________America')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_america, 'Country Codes', 'Year')
# model.fit(estimator='CCEPbc')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_america, 'Country Codes', 'Year')
# model.fit(estimator='CCEP')

# print('_____________________________________________________________Asia')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_asia, 'Country Codes', 'Year')
# model.fit(estimator='CCEPbc')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_asia, 'Country Codes', 'Year')
# model.fit(estimator='CCEP')

# print('_____________________________________________________________Middle East and North Afrika')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_me_na, 'Country Codes', 'Year')
# model.fit(estimator='CCEPbc')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_me_na, 'Country Codes', 'Year')
# model.fit(estimator='CCEP')

# print('_____________________________________________________________Sub Saharah Afrika')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_ss_afrika, 'Country Codes', 'Year')
# model.fit(estimator='CCEPbc')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_ss_afrika, 'Country Codes', 'Year')
# model.fit(estimator='CCEP')

# print('_____________________________________________________________Western Europe')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_w_europe, 'Country Codes', 'Year')
# model.fit(estimator='CCEPbc')
# model = HomogenousDynamicPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_w_europe, 'Country Codes', 'Year')
# model.fit(estimator='CCEP')


# warning taht dataframe needs to be sorted ascendingl
