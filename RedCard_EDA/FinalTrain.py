import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import GridSpec
import seaborn as sns
import numpy as np
import pandas as pd
import os, sys
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

#取出数据
clean_players = load_subgroup("cleaned_players.csv.gz")
players = load_subgroup("raw_players.csv.gz", )
countries = load_subgroup("raw_countries.csv.gz")
referees = load_subgroup("raw_referees.csv.gz")
path1 = 'data' + os.sep + 'raw_dyads.csv.gz'
agg_dyads = pd.read_csv(path1, compression='gzip', index_col=[0, 1])
path2 = 'data' + os.sep + 'cleaned_dyads.csv.gz'
tidy_dyads = pd.read_csv(path2, compression='gzip', index_col=[0, 1])

clean_players = load_subgroup("cleaned_players.csv.gz")

#合并下裁判和球员，当做临时工作数据
temp = tidy_dyads.reset_index().set_index('playerShort').merge(clean_players, left_index=True, right_index=True)

print(tidy_dyads.head())

print((tidy_dyads.groupby(level=0)  #level 0以合并时基底为准（裁判给牌数）
           .sum()
           .sort_values('redcard', ascending=False)
           .rename(columns={'redcard':'total redcards given'})).head())

print((tidy_dyads.groupby(level=1)  #为1是后加入的球员为准
           .sum()
           .sort_values('redcard', ascending=False)
           .rename(columns={'redcard':'total redcards received'})).head())

total_ref_games = tidy_dyads.groupby(level=0).size().sort_values(ascending=False)
total_player_games = tidy_dyads.groupby(level=1).size().sort_values(ascending=False)

total_ref_given = tidy_dyads.groupby(level=0).sum().sort_values(ascending=False,by='redcard')
total_player_received = tidy_dyads.groupby(level=1).sum().sort_values(ascending=False, by='redcard')
# sns.distplot(total_player_received, kde=False)


#提一个球员信息
player_ref_game = (tidy_dyads.reset_index()
                               .set_index('playerShort')
                                       .merge(clean_players,
                                              left_index=True,
                                              right_index=True)
                  )
print(player_ref_game.sample(replace=True, n=10000).groupby('skintone').mean())

#看下肤色和红牌数的关系, 结果石锤了
bootstrap = pd.concat([player_ref_game.sample(replace=True,
                                              n=10000).groupby('skintone').mean()
                       for _ in range(100)])

ax1 = sns.regplot(bootstrap.index.values,
                 y='redcard',
                 data=bootstrap,
                 lowess=True,
                 scatter_kws={'alpha':0.4,},
                 x_jitter=(0.125 / 4.0))
ax1.set_xlabel("Skintone")