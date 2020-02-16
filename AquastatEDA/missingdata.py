
# plotting
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context("poster", font_scale=1.3)
import folium
#pip install folium

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

# File with functions from prior notebook(s)

from AquastatEDA.aqua_helper import time_slice, country_slice, time_series, simple_regions, subregion, variable_slice

'''
# 数据质量评估和分析
在试图了解数据中哪些信息之前，请确保您理解了数据代表什么和丢失了什么。

## Overview 
###  Basic things to do 
* 分类：计数，区分计数，评估唯一值
* 数值：计数，最小，最大
* 抽查你熟悉的随机样品 
* 切片和切块 

### Main questions
* 那里没有什么数据？
* 那里的数据对吗？
* 数据是按照你想象的方式生成的吗？

### Helpful packages
* [`missingno`](https://github.com/ResidentMario/missingno)
* [`pivottablejs`](https://github.com/nicolaskruchten/jupyter_pivottablejs)
* [`pandas_profiling`](https://github.com/JosPolfliet/pandas-profiling)

### Example backlog
* 评估缺失数据在所有数据字段中的普遍性，评估其丢失是随机的还是系统的，并在缺少数据时确定模式
* 标识包含给定字段丢失数据的默认值。
* 确定质量评估抽样策略和初始EDA
* datetime数据类型，保证格式的一致性和粒度的数据，并执行对数据的所有日期的检查.
* 在多个字段捕获相同或相似信息的情况下，了解它们之间的关系并评估最有效的字段使用。
* 评估每个字段数据类型
* 对于离散值类型，确保数据格式一致。
* 对于离散值类型，评估不同值和唯一百分比的数目，并对答案的类型进行正确检查。
* 对于连续数据类型，评估描述性统计，并对值进行检查。
* 了解时间戳和评估使用的分析之间的关系
* 按设备类型、操作系统、软件版本对数据进行切片，保证跨切片数据的一致性
* 对于设备或应用程序数据，确定版本发布日期，并评估这些日期前后格式或值的任何更改数据。
'''


path = 'data' + os.sep + 'aquastat.csv.gzip'
data = pd.read_csv(path, compression='gzip')

data.region = data.region.apply(lambda x: simple_regions[x])
recent = time_slice(data, '2013-2017')

#查看所有属性的缺失情况
msno.matrix(recent, labels=True)

#查看水资源的总量缺失值
msno.matrix(variable_slice(data, 'exploitable_total'), inline=False, sort='descending')
plt.xlabel('Time period');
plt.ylabel('Country');
plt.title('Missing total exploitable water resources data across countries and time periods \n \n \n \n')
print(variable_slice(data, 'exploitable_total'))
#缺失值太多，直接去掉
data = data.loc[~data.variable.str.contains('exploitable'),:]   #保存variable值不为exploitable的样本
# print(data[data['variable']=='exploitable_total'])

#查看全国降水指数
msno.matrix(variable_slice(data, 'national_rainfall_index'),
            inline=False, sort='descending')
plt.xlabel('Time period');
plt.ylabel('Country');
plt.title('Missing national rainfall index data across countries and time periods \n \n \n \n')

#2002之后基本不再统计，所以缺失值也很多，去掉
data = data.loc[~(data.variable=='national_rainfall_index')]
print(data.shape)

##仪式上处理过后看下美洲的数据情况
# north_america = subregion(data, 'North America')
# msno.matrix(msno.nullity_sort(time_slice(north_america, '2013-2017'), sort='descending').T, inline=False)

#查看巴哈马的情况
# print(msno.nullity_filter(country_slice(data, 'Bahamas').T, filter='bottom', p=0.1))
#空值巨多


'''
地图显示某值的缺失分布情况
'''
geo = 'data' + os.sep + 'world.json'

null_data = recent['agg_to_gdp'].notnull()*1
map = folium.Map(location=[48, -102], zoom_start=2)
map.choropleth(geo_data=geo,
               data=null_data,
               columns=['country', 'agg_to_gdp'],
               key_on='feature.properties.name', reset=True,
               fill_color='GnBu', fill_opacity=1, line_opacity=0.2,
               legend_name='Missing agricultural contribution to GDP data 2013-2017')
map.save('data\map.html')

#脚本
def plot_null_map(df, time_period, variable,
                  legend_name=None):
    geo = 'data' + os.sep + 'world.json'

    ts = time_slice(df, time_period).reset_index().copy()
    ts[variable] = ts[variable].notnull() * 1
    map = folium.Map(location=[48, -102], zoom_start=2)
    map.choropleth(geo_data=geo,
                   data=ts,
                   columns=['country', variable],
                   key_on='feature.properties.name', reset=True,
                   fill_color='GnBu', fill_opacity=1, line_opacity=0.2,
                   legend_name=legend_name if legend_name else variable)
    return map
#测试营养不良缺失值分布
plot_null_map(data, '2013-2017', 'number_undernourished', 'Number undernourished is missing')

#不同国家，随着时间某些值提取的数量
fig, ax = plt.subplots(figsize=(16, 16))
sns.heatmap(data.groupby(['time_period','variable']).value.count().unstack().T , ax=ax)
plt.xticks(rotation=45)
plt.xlabel('Time period')
plt.ylabel('Variable')
plt.title('Number of countries with data reported for each variable over time')


'''
profiling
'''
print(data.head())
aly = pandas_profiling.ProfileReport(time_slice(data, '2003-2017'))
aly.to_html('data\profiliing.html')