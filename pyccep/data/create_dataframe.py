import pandas as pd


df_gpd = pd.read_csv('pwt70_w_country_names.csv', index_col=False)
df_gpd= df_gpd[['country','year','POP', 'rgdpch']]  
df_gpd = df_gpd.rename(columns={"country": "Country Name"})
df_gpd = df_gpd.rename(columns={"year": "Year"})
df_gpd = df_gpd.rename(columns={"rgdpch": "GDP"})
df_gpd = df_gpd[df_gpd['Year'] >= 1970]
df_gpd = df_gpd[df_gpd['Year'] <= 2009]

df_terrorists = pd.read_excel('ITERATE_COMMON_FILE_1968_2018.xls')  
df_terrorists['Terrorist Events'] = 1
df_terrorists = df_terrorists.rename(columns={"Location start": "Country Codes"})


df_terrorists.loc[df_terrorists['Country Codes'] == 265, 'Country Codes'] = 255
df_terrorists.loc[df_terrorists['Country Codes'] == 817, 'Country Codes'] = 815
df_terrorists.loc[df_terrorists['Country Codes'] == 816, 'Country Codes'] = 815

df_terrorists.loc[df_terrorists['Country Codes'] == 206, 'Country Codes'] = 200
df_terrorists.loc[df_terrorists['Country Codes'] == 204, 'Country Codes'] = 200
df_terrorists.loc[df_terrorists['Country Codes'] == 202, 'Country Codes'] = 200

df_terrorists = df_terrorists[['Year','Country Codes','Terrorist Events']].groupby(['Year','Country Codes']).count().reset_index(drop=False)

all_codes= [['002', 'United States', 'america'],
['006', 'Puerto Rico', 'america'],
['020', 'Canada', 'america'],
['040', 'Cuba', 'america'],
['041', 'Haiti', 'america'],
['042', 'Dominican Republic', 'america'],
["070", "Mexico", 'america'],
["090", "Guatemala", 'america'],
["091", "Honduras", 'america'],
["092", "El Salvador", 'america'],
["093", "Nicaragua", 'america'],
["094", "Costa Rica", 'america'],
["095", "Panama", 'america'],
["100", "Colombia", 'america'],
["101", "Venezuela", 'america'],
['130', 'Ecuador', 'america'],
['135', 'Peru', 'america'],
['140', 'Brazil', 'america'],
['145', 'Bolivia', 'america'],
['155', 'Chile', 'america'],
['160', 'Argentina', 'america'],
['165', 'Uruguay', 'america'],
['200', 'United Kingdom', 'europe'],
# ['202', 'Guernsey and dependencies', 'europe'],
# ['204', 'Northern Ireland', 'europe'],
['205', 'Ireland', 'europe'],
# ['206', 'Scotland', 'europe'],
['210', 'Netherlands', 'europe'],
['211', 'Belgium', 'europe'],
['212', 'Luxembourg', 'europe'],
['220', 'France', 'europe'],
['225', 'Switzerland', 'europe'],
['230', 'Spain', 'europe'],
['235', 'Portugal', 'europe'],
['255', 'Germany', 'europe'],
# ['265', ' German Democratic Republic (East Germany)',],
['305', 'Austria', 'europe'],
['325', 'Italy', 'europe'],
['338', 'Malta', 'europe'],
['350', 'Greece', 'europe'],
# ['375', 'Finland', 'europe'],
['380', 'Sweden',  'europe'],
['385', 'Norway', 'europe'],
['390', 'Denmark', 'europe'],
['395', 'Iceland',  'europe'],
['540', 'Angola', 'africa sub sahara'],
# ['625', 'Former Sudan','africa sub sahara'],
['625', 'Sudan','africa sub sahara'],
['432', 'Mali','africa sub sahara'],
['433', 'Senegal','africa sub sahara'],
['436', 'Niger','africa sub sahara'],
['437', "Cote d`Ivoire",'africa sub sahara'],
['461', 'Togo','africa sub sahara'],
['475', 'Nigeria','africa sub sahara'],
['483', 'Chad','africa sub sahara'],
['484', 'Congo, Republic of','africa sub sahara'],
['485', 'Congo, Dem. Rep.','africa sub sahara'],
['500', 'Uganda','africa sub sahara'],
['451', 'Sierra Leone','africa sub sahara'],
['501', 'Kenya','africa sub sahara'],
['516', 'Burundi','africa sub sahara'],
['517', 'Rwanda','africa sub sahara'],
['520', 'Somalia','africa sub sahara'],
['530', 'Ethiopia','africa sub sahara'],
# ['530', 'Former Ethiopia','africa sub sahara'],
['541', 'Mozambique','africa sub sahara'],
['551', 'Zambia','africa sub sahara'],
['552', 'Zimbabwe','africa sub sahara'],
['560', 'South Africa','africa sub sahara'],
['565', 'Namibia','africa sub sahara'],
['570', 'Lesotho','africa sub sahara'],
['615', 'Algeria', 'middle east'],
['616', 'Tunisia','middle east'],
['630', 'Iran','middle east'],
['640', 'Turkey','middle east'],
['645', 'Iraq','middle east'],
['352', 'Cyprus','middle east'],
['651', 'Egypt','middle east'],
['652', 'Syria','middle east'],
['660', 'Lebanon','middle east'],
['663', 'Jordan','middle east'],
['666', 'Israel','middle east'],
# ['600', 'Morocco','middle east'],
['692', 'Bahrain','middle east'],
['710','China Version 1','asia'],
['713','Taiwan','asia'],
['720','Hong Kong','asia'],
# ['721','Macao','asia'],
['732','Korea, Republic of','asia'],
['740','Japan','asia'],
['750','India','asia'],
['765','Bangladesh','asia'],
['700','Afghanistan','asia'],
['770','Pakistan','asia'],
['780','Sri Lanka','asia'],
['781','Maldives','asia'],
['790','Nepal','asia'],
['800','Thailand','asia'],
['811','Cambodia','asia'],
['812',"Laos",'asia'],
['815','Vietnam','asia'],
# ['816','North Vietnam','asia'],
# ['817','South Vietnam','asia'],
['820','Malaysia','asia'],
['830','Singapore','asia'],
['840','Philippines','asia'],
['850','Indonesia', 'asia']]

df_country_codes = pd.DataFrame(all_codes,columns=['Country Codes', 'Country Name', 'Region'])
df_country_codes['Country Codes'] = df_country_codes['Country Codes'].astype(int)

print(df_country_codes[['Region']].value_counts())
df_country_codes['Year'] = 1970
df_country_codes = df_country_codes.loc[df_country_codes.index.repeat(40)]
df_country_codes.loc[:, 'Year'] += df_country_codes.groupby(level=0).cumcount()


df_terrorists['Country Codes'] = df_terrorists['Country Codes'].astype(int)


data = df_country_codes.merge(df_terrorists, on=['Country Codes', 'Year'], how='left')
data = data.fillna(0)
df_gpd['Year'] = df_gpd['Year'].astype(int)


df_gpd = df_country_codes.merge(df_gpd, on=['Country Name', 'Year' ], how='left')
df_gpd = df_gpd[df_gpd['Year'] >= 1970]
df_gpd = df_gpd[df_gpd['Year'] <= 2009]

print(df_gpd[df_gpd['GDP'].isna()]['Country Name'].value_counts())
df_gpd['Growth rate'] = df_gpd['GDP'].shift(1)
df_gpd['Growth rate'] = (df_gpd['GDP'] -df_gpd['Growth rate'])/df_gpd['Growth rate']

data = data.merge(df_gpd, on=['Country Codes', 'Year', 'Country Name', 'Region'], how='left')
data['Delta Terrorist Events'] =  data['Terrorist Events'] -data['Terrorist Events'].shift(1) 
data = data[data['Year'] > 1970]
data = data[data['Year'] <= 2009]



print(df_gpd[df_gpd['GDP'].isna()]['Country Name'].value_counts())

data = data[data['GDP'].notna()]

data.to_excel("terrorism.xlsx") 

print(data.head(60))




