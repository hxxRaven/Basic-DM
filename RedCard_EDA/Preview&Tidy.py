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

path = 'data' + os.sep + 'redcard.csv.gz'
df = pd.read_csv(path, compression='gzip')
print(df.shape)
# print(df.head(5))

#导入后先描述，统计下
print(df.describe().T)

#再查看每个特征的类型
print(df.dtypes)

all_columns = df.columns.tolist()
print(all_columns)

#直接求均值，可能被异常值重复值空值影响

print(df['height'].mean())  #此方法不用

##先按人名分组，求其均值，在对每个group相加求均值（groupby的用法）
a1 = np.mean(df.groupby('playerShort').height.mean())

#按人名分组，同一个人名下，设定的特征重复出现次数（为1正常，为2则需要drop）
player_index = 'playerShort'
player_cols = ['birthday', 'height', 'weight', 'position', 'photoID', 'rater1', 'rater2']

all_cols_unique_players = df.groupby('playerShort').agg({col:'nunique' for col in player_cols})

print(all_cols_unique_players.head())

#把上面分好的矩阵，判断有值大于1的，则去除掉空值再显示
print(all_cols_unique_players[all_cols_unique_players > 1].dropna().head(5))

'''脚本'''
def get_subgroup(dataFrame, g_index, g_columns):
    g = dataFrame.groupby(g_index).agg({col:'nunique' for col in g_columns})
    if g[g > 1].dropna().shape[0] !=0:
        print("数据有重复")
    else:
        print("无重复数据")
    return dataFrame.groupby(g_index).agg({col:'max' for col in g_columns})

'''存储csv脚本'''
def save_subgroup(dataframe, g_index, subgroup_name, prefix='raw_'):
    save_subgroup_filename = "".join([prefix, subgroup_name, ".csv.gz"])
    path = 'data' + os.sep + save_subgroup_filename
    dataframe.to_csv(path, compression='gzip', encoding='UTF-8')

    test_df = pd.read_csv(path, compression='gzip', index_col=g_index, encoding='UTF-8')
    # Test that we recover what we send in
    if dataframe.equals(test_df):
        print("Test-passed: we recover the equivalent subgroup dataframe.")
    else:
        print("Warning -- equivalence test!!! Double-check.")

players = get_subgroup(df, player_index, player_cols)
save_subgroup(players, player_index, "players")

#切分俱乐部与国家数据
club_index = 'club'
club_cols = ['leagueCountry']
clubs = get_subgroup(df, club_index, club_cols)
print(clubs.head(10))

print(clubs['leagueCountry'].value_counts())
save_subgroup(clubs, club_index, "clubs", )

#切分裁判与所属国家的
referee_index = 'refNum'
referee_cols = ['refCountry']
referees = get_subgroup(df, referee_index, referee_cols)
print(referees.head(5))
print(referees.refCountry.nunique())    #nunique直接统计数量，unique则会全部列出
save_subgroup(referees, referee_index, "referees")

#把裁判和球员结合起来考虑
dyad_index = ['refNum', 'playerShort']
dyad_cols = ['games', 'victories', 'ties', 'defeats', 'goals', 'yellowCards', 'yellowReds', 'redCards']
dyads = get_subgroup(df, g_index=dyad_index, g_columns=dyad_cols)
print(dyads.head())
print(dyads[dyads["redCards"] > 1].head(5))
print(dyads['redCards'].max())
save_subgroup(dyads, dyad_index, "dyads")

country_index = 'refCountry'
country_cols = ['Alpha_3', # rename this name of country
                'meanIAT',
                'nIAT',
                'seIAT',
                'meanExp',
                'nExp',
                'seExp',
               ]
countries = get_subgroup(df, country_index, country_cols)
countries.head()
rename_columns = {'Alpha_3':'countryName', }
countries = countries.rename(columns=rename_columns)
countries.head()
save_subgroup(countries, country_index, "countries")