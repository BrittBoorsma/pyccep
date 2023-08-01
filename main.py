# Importing required packages
from pyccep.panel_models.homogenous_panel_model import HomogenousPanelModel
import pandas as pd

df =  pd.read_excel('pyccep/data/terrorism.xlsx')  
df['Terrorist Events'] = df['Terrorist Events']/1000
print('_______________START________________')

df_america = df[df['Region'] == 'america']
df_asia = df[df['Region'] == 'asia']
df_me_na = df[df['Region'] == 'middle east']
df_ss_afrika = df[df['Region'] == 'africa sub sahara']
df_w_europe = df[df['Region'] == 'europe']

model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('______________________________________________________________America')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_america, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_america, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________Asia')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_asia, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_asia, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________Middle East and North Afrika')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_me_na, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_me_na, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________Sub Saharah Afrika')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_ss_afrika, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_ss_afrika, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________Western Europe')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_w_europe, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_w_europe, 'Country Codes', 'Year')
model.fit(estimator='CCEP')


print('_____________________________________________________________Devided by population')
df =  pd.read_excel('pyccep/data/terrorism.xlsx')  
df['Terrorist Events'] = df['Terrorist Events']/(df['POP']*1000)
print(df.head())

df_america = df[df['Region'] == 'america']
df_asia = df[df['Region'] == 'asia']
df_me_na = df[df['Region'] == 'middle east']
df_ss_afrika = df[df['Region'] == 'africa sub sahara']
df_w_europe = df[df['Region'] == 'europe']

model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________America')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_america, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_america, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________Asia')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_asia, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_asia, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________Middle East and North Afrika')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_me_na, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_me_na, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________Sub Saharah Afrika')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_ss_afrika, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_ss_afrika, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________Western Europe')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_w_europe, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Terrorist Events_{t-1}', df_w_europe, 'Country Codes', 'Year')
model.fit(estimator='CCEP')


# calculations with Delta Terrorist Events


df =  pd.read_excel('pyccep/data/terrorism.xlsx')  
df['Delta Terrorist Events'] = df['Delta Terrorist Events']/1000
print('_______________START__WITH_____Delta_Terrorist_Events_________')

df_america = df[df['Region'] == 'america']
df_asia = df[df['Region'] == 'asia']
df_me_na = df[df['Region'] == 'middle east']
df_ss_afrika = df[df['Region'] == 'africa sub sahara']
df_w_europe = df[df['Region'] == 'europe']

model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('______________________________________________________________America')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_america, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_america, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________Asia')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_asia, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_asia, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________Middle East and North Afrika')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_me_na, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_me_na, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________Sub Saharah Afrika')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_ss_afrika, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_ss_afrika, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________Western Europe')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_w_europe, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_w_europe, 'Country Codes', 'Year')
model.fit(estimator='CCEP')


print('_____________________________________________________________Devided by population')
df =  pd.read_excel('pyccep/data/terrorism.xlsx')  
df['Delta Terrorist Events'] = df['Delta Terrorist Events']/(df['POP']*1000)
print(df.head())

df_america = df[df['Region'] == 'america']
df_asia = df[df['Region'] == 'asia']
df_me_na = df[df['Region'] == 'middle east']
df_ss_afrika = df[df['Region'] == 'africa sub sahara']
df_w_europe = df[df['Region'] == 'europe']

model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________America')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_america, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_america, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________Asia')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_asia, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_asia, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________Middle East and North Afrika')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_me_na, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_me_na, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________Sub Saharah Afrika')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_ss_afrika, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_ss_afrika, 'Country Codes', 'Year')
model.fit(estimator='CCEP')

print('_____________________________________________________________Western Europe')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_w_europe, 'Country Codes', 'Year')
model.fit(estimator='CCEPbc')
model = HomogenousPanelModel('Growth rate ~ Growth rate_{t-1} + Delta Terrorist Events_{t-1}', df_w_europe, 'Country Codes', 'Year')
model.fit(estimator='CCEP')
