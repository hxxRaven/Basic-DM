import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("poster", font_scale=1.3)
import folium

# system packages
import os, sys
import warnings
warnings.filterwarnings('ignore')

# basic wrangling
import numpy as np
import pandas as pd

# eda tools
import pivottablejs
import missingno as msno
import pandas_profiling

# interactive
import ipywidgets as widgets

# more technical eda
import sklearn
import scipy
from aqua_helper import time_slice, country_slice, time_series, simple_regions, subregion, variable_slice

data = pd.read_csv('data/aquastat.csv.gzip', compression='gzip')

# simplify regions
data.region = data.region.apply(lambda x: simple_regions[x])

# remove exploitable fields and national rainfall index
data = data.loc[~data.variable.str.contains('exploitable'),:]
data = data.loc[~(data.variable=='national_rainfall_index')]
recent = time_slice(data, '2013-2017')

'''
# Exploring *population*
## Cross-section
For numerical data, look at: 

* *Location*: 均值，中位数，模式，四分位
* *Spread*: 标准差、方差、范围、间距范围
* *Shape*: 偏度、峰度
'''

#查看大致是数据
print(recent)
print(recent[['total_pop', 'urban_pop', 'rural_pop']].describe().astype(int))
#发现有个负数，看下是哪个国家
print(recent.sort_values('rural_pop')[['total_pop','urban_pop','rural_pop']].head())
#得知是Qatar，再看看他的其他相关（'total_pop','urban_pop','rural_pop' ）
print(time_series(data, 'Qatar', 'total_pop').join(time_series(data, 'Qatar', 'urban_pop')).join(time_series(data, 'Qatar', 'rural_pop')))

'''
一般情况中位数和均值相近最好，更接近正态分布。如果相差过大就要考虑了，会影响训练模型
### Shape of the data
* 数据分布是倾斜的吗？
* 有异常值吗？它们可行吗？ 
* 有不连续的吗? 
'''
print(recent[['total_pop', 'urban_pop', 'rural_pop']].describe().astype(int))
print(recent[['total_pop', 'urban_pop', 'rural_pop']].apply(scipy.stats.skew))
#三个偏度均大于0，右偏

'''
峰度值
'''
print(recent[['total_pop', 'urban_pop', 'rural_pop']])
print(recent[['total_pop', 'urban_pop', 'rural_pop']].apply(scipy.stats.kurtosis))

#绘制原图形
# fig1, ax1 = plt.subplots(figsize=(12, 8))
# ax1.hist(recent.total_pop.values, bins=50)
# ax1.set_xlabel('Total population')
# ax1.set_ylabel('Number of countries')
# ax1.set_title('Distribution of population of countries 2013-2017')

'''数据变换使数据正常分布——log变换'''
print(recent[['total_pop']].apply(np.log).apply(scipy.stats.skew))
print(recent[['total_pop']].apply(np.log).apply(scipy.stats.kurtosis))

#画图脚本
def plot_hist(df, variable, bins=20, xlabel=None, by=None,
              ylabel=None, title=None, logx=False, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(12, 8))
    if logx:
        if df[variable].min() <= 0:
            df[variable] = df[variable] - df[variable].min() + 1
            print('Warning: data <=0 exists, data transformed by %0.2g before plotting' % (- df[variable].min() + 1))

        bins = np.logspace(np.log10(df[variable].min()),
                           np.log10(df[variable].max()), bins)
        ax.set_xscale("log")

    ax.hist(df[variable].dropna().values, bins=bins);

    if xlabel:
        ax.set_xlabel(xlabel);
    if ylabel:
        ax.set_ylabel(ylabel);
    if title:
        ax.set_title(title);

    return ax
'''画图时，plot_hist(logx=True)'''
#测试log化后的峰度
# plot_hist(recent, 'total_pop', bins=25, logx=True,
#           xlabel='Log of total population', ylabel='Number of countries',
#           title='Distribution of total population of countries 2013-2017')



'''数据的观察角度'''

#先添加一个平均人口
recent['population_density'] = recent.total_pop.divide(recent.total_area)
# fig2, ax2 = plt.figure()
# time_series(data, 'United States of America', 'total_pop').plot(ax=ax2)
#
# ax2.xlabel('Year')
# ax2.ylabel('Population')
# ax2.title('United States population over time')

#绘制每国人口变化情况
with sns.color_palette(sns.diverging_palette(220, 280, s=85, l=25, n=23)):
    north_america = time_slice(subregion(data, 'North America'), '1958-1962').sort_values('total_pop').index.tolist()
    for country in north_america:
        plt.plot(time_series(data, country, 'total_pop'), label=country)
        plt.xlabel('Year')
        plt.ylabel('Population')
        plt.title('North American populations over time')
    plt.legend(loc=2,prop={'size':10})

#很多国家很难观察，搜索一用相对min的比值来表示
with sns.color_palette(sns.diverging_palette(220, 280, s=85, l=25, n=23)):
    for country in north_america:
        ts = time_series(data, country, 'total_pop')
        ts['norm_pop'] = ts.total_pop/ts.total_pop.min()*100
        plt.plot(ts['norm_pop'], label=country)
        plt.xlabel('Year')
        plt.ylabel('Percent increase in population')
        plt.title('Percent increase in population from 1960 in North American countries')
    plt.legend(loc=2,prop={'size':10})

#再用热度图
north_america_pop = variable_slice(subregion(data, 'North America'), 'total_pop')
north_america_norm_pop = north_america_pop.div(north_america_pop.min(axis=1), axis=0)*100
north_america_norm_pop = north_america_norm_pop.loc[north_america]
fig, ax = plt.subplots(figsize=(16, 12));
sns.heatmap(north_america_norm_pop, ax=ax, cmap=sns.light_palette((214, 90, 60), input="husl", as_cmap=True));
plt.xticks(rotation=45);
plt.xlabel('Time period');
plt.ylabel('Country, ordered by population in 1960 (<- greatest to least ->)');
plt.title('Percent increase in population from 1960');
