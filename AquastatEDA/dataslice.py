import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os,sys
import warnings
warnings.filterwarnings('ignore')
sns.set_context("poster", font_scale=1.3)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

path = 'data' + os.sep + 'aquastat.csv.gzip'
data = pd.read_csv(path, compression='gzip')
print(data.head(5))
print(data.info())

print(data[['variable','variable_full']].drop_duplicates()) #输出这两个特征值，但是去除掉描述一样的，实际就是想看有多少种描绘
print(data['country'].nunique())

countries = data.country.unique()

time_period = data['time_period'].unique()  #查看时间间隔
print(time_period)

var_null = data[data['variable'] == 'total_area'].value.isnull().sum()  #看下variable列有多少重复
print(var_null)

print(data.describe())

'''
查看数据的角度
* 横截面：一个时期内所有国家
* 时间序列：一个国家随着时间的推移
* 面板数据：所有国家随着时间的推移（作为数据给出）
* 地理空间：所有地理上相互联系的国家
'''

#时间切片
#三维切片，纵坐标：country，横坐标：variable，填入值：vlaue
def time_slice(df, time_period):
    df = df[df['time_period'] ==time_period]
    df = df.pivot(index='country', columns='variable', values='value')
    df.columns.name = time_period
    return df

print(time_slice(data, time_period[0]).head())

#国家切片
#纵轴为variable， 横轴为time——period，填充值：value
def country_slice(df, country):
    df = df[df.country == country]
    df = df.pivot(index='variable', columns='time_period', values='value')
    df.index.name = country
    return df

print(country_slice(data, countries[40]).head())

#变量切片
def variable_slice(df, variable):
    df = df[df.variable == variable]
    df = df.pivot(index='country', columns='time_period', values='value')
    df.columns.name = 'time_period'
    return df

print(variable_slice(data, 'total_pop').head())


#给定国家和variable，显示此国家的此项variable随时间变化
def time_series(df, country, variable):
    series = df[(df.country == country) & (df.variable == variable)]    #符合条件的样本所有列都提取来了

    series = series.dropna()[['year_measured', 'value']]

    series.year_measured = series.year_measured.astype(int)
    series.set_index('year_measured', inplace=True)
    series.columns = [variable]
    return series

print(time_series(data, 'Belarus', 'total_pop'))


'''区域显示'''
#国家数太多，用字典将国家全放入大洲里
simple_regions ={
    'World | Asia':'Asia',
    'Americas | Central America and Caribbean | Central America': 'North America',
    'Americas | Central America and Caribbean | Greater Antilles': 'North America',
    'Americas | Central America and Caribbean | Lesser Antilles and Bahamas': 'North America',
    'Americas | Northern America | Northern America': 'North America',
    'Americas | Northern America | Mexico': 'North America',
    'Americas | Southern America | Guyana':'South America',
    'Americas | Southern America | Andean':'South America',
    'Americas | Southern America | Brazil':'South America',
    'Americas | Southern America | Southern America':'South America',
    'World | Africa':'Africa',
    'World | Europe':'Europe',
    'World | Oceania':'Oceania'
}

################################################################
data['region'] = data['region'].apply(lambda x: simple_regions[x])
################################################################

print(data['region'].unique())

#提取对应地区数据
def subregion(df, region):
    return df[df['region'] == 'region']
