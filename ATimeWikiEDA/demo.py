'''
探索性数据发现
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

path = 'data' + os.sep + 'train_1.csv'
train = pd.read_csv(path).fillna(0)
print(train.info())

#查看数据发现值都是整数，却用了float型，为减小内存进行转换
for col in train.columns[1:]:
    train[col] = pd.to_numeric(train[col], downcast='integer')
print(train.info())

#统计下有几种语言，和其出现次数
def get_language(page):
    res =re.search('[a-z][a-z].wikipedia.org', page)
    if res:
        a = res.group()
        return res.group()[0:2]
    return 'na'

train['lang'] = train['Page'].map(get_language)

from collections import Counter
print(Counter(train['lang']))

#按国家，把数据分块存入一个大字典
lang_sets = {}
lang_sets['en'] = train[train.lang=='en'].iloc[:,0:-1]
lang_sets['ja'] = train[train.lang=='ja'].iloc[:,0:-1]
lang_sets['de'] = train[train.lang=='de'].iloc[:,0:-1]
lang_sets['na'] = train[train.lang=='na'].iloc[:,0:-1]
lang_sets['fr'] = train[train.lang=='fr'].iloc[:,0:-1]
lang_sets['zh'] = train[train.lang=='zh'].iloc[:,0:-1]
lang_sets['ru'] = train[train.lang=='ru'].iloc[:,0:-1]
lang_sets['es'] = train[train.lang=='es'].iloc[:,0:-1]
sums = {}

#sums list中再按国家分块，每块存储日期和对应日期（Page阅读量求和再除以Page的页均阅读量）
for key in lang_sets:
    sums[key] = lang_sets[key].iloc[:,1:].sum(axis=0) / lang_sets[key].shape[0]
#days存储日期的列表
days = [r for r in range(sums['en'].shape[0])]


fig = plt.figure(1, figsize=[10, 10])
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title('Pages in Different Languages')
labels = {'en': 'English', 'ja': 'Japanese', 'de': 'German',
          'na': 'Media', 'fr': 'French', 'zh': 'Chinese',
          'ru': 'Russian', 'es': 'Spanish'
          }

for key in sums:
    plt.plot(days, sums[key], label=labels[key])

plt.legend()
plt.show()

#再指定国家中，显示指定id词条的阅读变化情况
def plot_entry(key, idx):
    data = lang_sets[key].iloc[idx, 1:]
    fig = plt.figure(1, figsize=(10, 5))
    plt.plot(days, data)
    plt.xlabel('day')
    plt.ylabel('views')
    plt.title(train.iloc[lang_sets[key].index[idx], 0])

    plt.show()

idx = [1, 5, 10, 50, 100, 250,500, 750,1000,1500,2000,3000,4000,5000]
for i in idx:
    plot_entry('en',i)

#选出每组的top5词条
npages = 5
top_pages = {}
for key in lang_sets:
    print(key)
    sum_set = pd.DataFrame(lang_sets[key][['Page']])#以PAGE和index建立新的dataframe
    sum_set['total'] = lang_sets[key].sum(axis=1)
    sum_set = sum_set.sort_values('total',ascending=False)
    print(sum_set.head(10))
    top_pages[key] = sum_set.index[0]   #存储每区域top1的id
    print('\n\n')

#打印每个区域的top1词条的变化情况
for key in top_pages:
    fig = plt.figure(1,figsize=(10,5))
    cols = train.columns
    cols = cols[1:-1]
    data = train.loc[top_pages[key],cols]
    plt.plot(days,data)
    plt.xlabel('Days')
    plt.ylabel('Views')
    plt.title(train.loc[top_pages[key],'Page'])
    plt.show()