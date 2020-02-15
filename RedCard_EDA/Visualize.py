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

players = load_subgroup('raw_stepplayers.csv.gz')
print(players.head())

'''
pandas_profiling处理数据的关系,直接显示网页
'''
aly = pandas_profiling.ProfileReport(players)
aly.to_file("example.html")

#处理下生日和年龄的问题
players['birth_date'] = pd.to_datetime(players.birthday, format='%d.%m.%Y')
players['age_years'] = ((pd.to_datetime("2013-01-01") - players['birth_date']).dt.days)/365.25  #dt.days日期直接变成天数


#选取和红黄牌有关系的特征，之前用过的已经被去除或者处理了
players_cleaned_variables = ['height', 'weight', 'skintone', 'position_agg', 'weightclass', 'heightclass', 'skintoneclass', 'age_years']
aly2 = pandas_profiling.ProfileReport(players[players_cleaned_variables])
aly2.to_file("example2.html")

players[players_cleaned_variables].to_csv("data\cleaned_players.csv.gz", compression='gzip')