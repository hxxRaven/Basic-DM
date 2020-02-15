from __future__ import absolute_import, division, print_function
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
import os,sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
sns.set_context("poster", font_scale=1.3)

import missingno as msno
import pandas_profiling

from sklearn.datasets import make_blobs
import time

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

def load_subgroup(filename, index_col=[0]):
    path = 'data' + os.sep + filename
    return pd.read_csv(path, compression='gzip', index_col=index_col)

players = load_subgroup("raw_players.csv.gz")
print(players.head())

#缺失值的处理，太多的可以考虑drop，少的进行下一步处理
'''missingno模块_查看缺失值的分布情况
https://github.com/ResidentMario/missingno
'''
# msno.matrix(players.sample(500), figsize=(16, 7), width_ratios=(15, 1))
# msno.bar(players.sample(500), figsize=(16, 7))
msno.heatmap(players.sample(500), figsize=(16, 7))  #两个特征之间，一方缺失另一方缺失的可能性

print("All player：", len(players))
print('rater1 nulls：', len(players[players["rater1"].isnull()]))
print("rater2 nulls:", len(players[players.rater2.isnull()]))
print("Both nulls:", len(players[(players.rater1.isnull()) & (players.rater2.isnull())]))

#移除掉缺失过多的
players = players[players['rater1'].notnull()]
# print(players.shape)



#两个rater的处理，看一下两个rater的有什么关系(是适用于离散的)
print(pd.crosstab(players['rater1'], players['rater2']))
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(pd.crosstab(players.rater1, players.rater2), cmap='Blues', annot=True, fmt='d', ax=ax)
ax.set_title("Correlation between Rater 1 and Rater 2\n")
fig.tight_layout()
#均值吧
players['skintone'] = players[['rater1', 'rater2']].mean(axis=1)
print(players.head())
fig3 = plt.figure()
sns.distplot(players['skintone'], kde=False)


#position的分析
# fig4, ax4 = plt.subplots(figsize=(12, 8))
# players.position.value_counts(dropna=False, ascending=True).plot(kind='barh', ax=ax4)
# ax4.set_ylabel("Position")
# ax4.set_xlabel("Counts")
# fig4.tight_layout()

#发现分的太细，所以重新划分大区域
position_types = players.position.unique()
print(position_types)
defense = ['Center Back','Defensive Midfielder', 'Left Fullback', 'Right Fullback', ]
midfield = ['Right Midfielder', 'Center Midfielder', 'Left Midfielder',]
forward = ['Attacking Midfielder', 'Left Winger', 'Right Winger', 'Center Forward']
keeper = 'Goalkeeper'

#players中定位position再指定集或等于指定集的，为其添加新的特征并写入数据
players.loc[players['position'].isin(defense), 'position_agg'] = "Defense"
players.loc[players['position'].isin(midfield), 'position_agg'] = "Midfield"
players.loc[players['position'].isin(forward), 'position_agg'] = "Forward"
players.loc[players['position'].eq(keeper), 'position_agg'] = "Keeper"

print(players.head(6))
fig5, ax5 = plt.subplots(figsize=(12, 9))
players['position_agg'].value_counts(dropna=False, ascending=True).plot(kind='barh', ax=ax5)
ax5.set_ylabel("position_agg")
ax5.set_xlabel("Counts")
fig5.tight_layout()


#多变量之间关系
#发现仅身高体重正相关
from pandas.plotting import scatter_matrix
fig6, ax6 = plt.subplots(figsize=(10, 10))
scatter_matrix(players[['height', 'weight', 'skintone']], alpha=0.2, diagonal='hist', ax=ax6)

#画一条回归图看看，regplot
fig7, ax7 = plt.subplots(figsize=(10, 10))
sns.regplot('weight', 'height', data=players, ax=ax7)
ax7.set_ylabel('Height [cm]')
ax7.set_xlabel('Weight [kg]')
fig7.tight_layout()

#把身高的值分bin切分(qcut)
height_categories = ["vlow_weight", "low_weight", "mid_weight", "high_weight", "vhigh_weight"]
#qcut() 三个参数，切分那个特质，切分几份，切分的参考
players['heightclass'] = pd.qcut(players['height'], len(height_categories), height_categories)

weight_categories = ["vlow_weight", "low_weight", "mid_weight", "high_weight", "vhigh_weight",]

players['weightclass'] = pd.qcut(players['weight'], len(weight_categories), weight_categories)

print(players.head(5))

players['skintoneclass'] = pd.qcut(players['skintone'], 3)
print(players.head(4))

'''存储csv脚本'''
def save_subgroup(dataframe, subgroup_name, prefix='raw_'):
    save_subgroup_filename = "".join([prefix, subgroup_name, ".csv.gz"])
    path = 'data' + os.sep + save_subgroup_filename
    dataframe.to_csv(path, compression='gzip', encoding='UTF-8')


save_subgroup(players, "stepplayers")