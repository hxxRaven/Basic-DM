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

def save_subgroup(dataframe, subgroup_name, prefix='raw_'):
    save_subgroup_filename = "".join([prefix, subgroup_name, ".csv.gz"])
    path = 'data' + os.sep + save_subgroup_filename
    dataframe.to_csv(path, compression='gzip', encoding='UTF-8')

def load_subgroup(filename, index_col=[0]):
    path = 'data' + os.sep + filename
    return pd.read_csv(path, compression='gzip', index_col=index_col)

#读取之前的数据

clean_players = load_subgroup("cleaned_players.csv.gz")
players = load_subgroup("raw_players.csv.gz", )
countries = load_subgroup("raw_countries.csv.gz")
referees = load_subgroup("raw_referees.csv.gz")
path = 'data' + os.sep + "raw_dyads.csv.gz"
agg_dyads = pd.read_csv(path, compression='gzip', index_col=[0, 1])

#判断总比赛局数，红黄牌次数整合
all(agg_dyads['games'] == agg_dyads['victories'] + agg_dyads['ties'] + agg_dyads['defeats'])
# agg_dyads = agg_dyads.reset_index().set_index('playerShort')
print(len(agg_dyads))
agg_dyads['totalRedCards'] = agg_dyads['yellowReds'] + agg_dyads['redCards']
agg_dyads.rename(columns={'redCards': 'strictRedCards'}, inplace=True)
print(agg_dyads.head())


#整合之前的数据
#球员和裁判合并，以球员为index
player_dyad = (clean_players.merge(agg_dyads.reset_index().set_index('playerShort'),
                                   left_index=True,
                                   right_index=True))

clean_dyads = (agg_dyads.reset_index()[agg_dyads.reset_index()
                                   .playerShort
                                   .isin(set(clean_players.index))
                                  ]).set_index(['refNum', 'playerShort'])

#整理比赛场数和红黄牌数关系
colnames = ['games', 'totalRedCards']
j = 0
out = [0 for _ in range(sum(clean_dyads['games']))]

for index, row in clean_dyads.reset_index().iterrows():
    n = row['games']
    d = row['totalRedCards']
    ref = row['refNum']
    player = row['playerShort']
    for _ in range(n):
        row['totalRedCards'] = 1 if (d-_) > 0 else 0
        rowlist=list([ref, player, row['totalRedCards']])
        out[j] = rowlist
        j += 1

tidy_dyads = pd.DataFrame(out, columns=['refNum', 'playerShort', 'redcard'],).set_index(['refNum', 'playerShort'])

clean_referees = (referees.reset_index()[referees.reset_index()
                                                 .refNum.isin(tidy_dyads.reset_index().refNum
                                                                                       .unique())
                                        ]).set_index('refNum')

clean_countries = (countries.reset_index()[countries.reset_index()
                                           .refCountry
                                           .isin(clean_referees.refCountry
                                                 .unique())
                                          ].set_index('refCountry'))
datapath = 'data' + os.sep + "cleaned_dyads.csv.gz"
tidy_dyads.to_csv(datapath, compression='gzip')