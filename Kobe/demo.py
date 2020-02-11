import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

path = 'data' + os.sep + 'data.csv'
raw = pd.read_csv(path)
print(raw.head())
print(raw.shape)

kobe = raw[pd.notnull(raw['shot_made_flag'])]   #把非空的取出
print(kobe.shape)

'''
对坐标和经纬度绘制投篮点分布
也可以用极坐标标志
'''
alpha = 0.02
fig1 = plt.figure(figsize=(1,5))

plt.subplot(1, 2, 1)
plt.scatter(kobe['loc_x'], kobe['loc_y'], color='blue', alpha=alpha)
plt.title('loc_x & loc_y')

plt.subplot(1, 2, 2)
plt.scatter(kobe['lon'], kobe['lat'], color='red', alpha=alpha)
plt.title('lon & lat')
# plt.show()

'''极坐标'''
raw['dist'] = np.sqrt(raw['loc_x']**2 + raw['loc_y']**2)

loc_x_zero = raw['loc_x'] == 0
#print (loc_x_zero)
raw['angle'] = np.array([0]*len(raw))
raw['angle'][~loc_x_zero] = np.arctan(raw['loc_y'][~loc_x_zero] / raw['loc_x'][~loc_x_zero])
raw['angle'][loc_x_zero] = np.pi / 2

#分钟和秒统一为秒
raw['remaining_time'] = raw['minutes_remaining'] * 60 + raw['seconds_remaining']

#打印出各特征的值得种类
print(kobe['action_type'].unique())
print(kobe['combined_shot_type'].unique())
print(kobe['shot_type'].unique())
print(kobe['shot_type'].value_counts()) #显示种类，还显示每种种类的数量

'''apply用法，把2000-1取后半部分的赛季'''
print(kobe['season'].unique())
raw['season'] = raw['season'].apply(lambda x : int(x.split('-')[1]))
print(raw['season'].unique())

'生成dataframe'
mid_step = pd.DataFrame({'matchup' : kobe['matchup'], 'opponent' : kobe['opponent']})
print(mid_step.head(10))

'''如果两个属性表现强烈的正相关，取其中之一即可'''
raw['dist'] = np.sqrt(raw['loc_x']**2 + raw['loc_y']**2)    #取出投篮点离篮筐的距离
fig2 = plt.figure()
plt.scatter(raw['dist'], raw['shot_distance'], color='red')
plt.title('dist & shot_distance')
plt.show()      #图形显示确实强烈正相关

gs = kobe.groupby('shot_zone_area') #以选定特征，为全部数据分块
print(kobe['shot_zone_area'].value_counts())
print(len(gs))

import matplotlib.cm as cm
fig3 = plt.figure(figsize=(20,10))

def scatter_plot_by_category(feat):
    alpha = 0.1
    gs = kobe.groupby(feat)
    cs = cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1].loc_x, g[1].loc_y, color=c, alpha=alpha)   #gs为按特征分块，g为了遍历几个分块，g[1]为值所在

# shot_zone_area
plt.subplot(131)
scatter_plot_by_category('shot_zone_area')
plt.title('shot_zone_area')

# shot_zone_basic
plt.subplot(132)
scatter_plot_by_category('shot_zone_basic')
plt.title('shot_zone_basic')

# shot_zone_range
plt.subplot(133)
scatter_plot_by_category('shot_zone_range')
plt.title('shot_zone_range')

'''去掉某些特征'''
drops = ['shot_id', 'team_id', 'team_name', 'shot_zone_area', 'shot_zone_range', 'shot_zone_basic', \
         'matchup', 'lon', 'lat', 'seconds_remaining', 'minutes_remaining', \
         'shot_distance', 'loc_x', 'loc_y', 'game_event_id', 'game_id', 'game_date']
for drop in drops:
    raw = raw.drop(drop, axis=1)

'''把一些非数值特征转化为数值型编码'''
categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'period', 'season']
for var in categorical_vars:
    raw = pd.concat([raw, pd.get_dummies(raw[var], prefix=var)], axis=1)
    raw = raw.drop(var, axis=1) #去掉原来的特征


'''训练模型,用处理后'shot_made_flag'非空的数据去预测为空的数据'''
train_kobe = raw[pd.notnull(raw['shot_made_flag'])]
train_label = train_kobe['shot_made_flag']
train_kobe = train_kobe.drop('shot_made_flag', axis=1)
test_kobe = raw[pd.isnull(raw['shot_made_flag'])]
test_kobe = test_kobe.drop('shot_made_flag', axis=1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import KFold
import time

'''树的个数，树的深度'''
print('Finding best n_estimators for RandomForestClassifier...')    #树的个数
min_score = 100000
best_n = 0
scores_n = []
range_n = np.logspace(0, 2, num=3).astype(int)
for n in range_n:
    print("the number of trees : {0}".format(n))
    t1 = time.time()

    rfc_score = 0.
    rfc = RandomForestClassifier(n_estimators=n)
    for train_k, test_k in KFold(n_splits=10, shuffle=True, random_state=7).split(train_kobe, train_label):
        rfc.fit(train_kobe.iloc[train_k], train_label.iloc[train_k])
        
        pred = rfc.predict(train_kobe.iloc[test_k])
        rfc_score += log_loss(train_label.iloc[test_k], pred) / 10
    scores_n.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_n = n

    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(n, t2 - t1))
print(best_n, min_score)


print('Finding best max_depth for RandomForestClassifier...')   #树的深度
min_score = 100000
best_m = 0
scores_m = []
range_m = np.logspace(0, 2, num=3).astype(int)
for m in range_m:
    print("the max depth : {0}".format(m))
    t1 = time.time()

    rfc_score = 0.
    rfc = RandomForestClassifier(max_depth=m, n_estimators=best_n)
    for train_k, test_k in KFold(n_splits=10, shuffle=True, random_state=7).split(train_kobe, train_label):
        rfc.fit(train_kobe.iloc[train_k], train_label.iloc[train_k])

        pred = rfc.predict(train_kobe.iloc[test_k])
        rfc_score += log_loss(train_label.iloc[test_k], pred) / 10
    scores_m.append(rfc_score)
    if rfc_score < min_score:
        min_score = rfc_score
        best_m = m

    t2 = time.time()
    print('Done processing {0} trees ({1:.3f}sec)'.format(m, t2 - t1))
print(best_m, min_score)

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.plot(range_n, scores_n)
plt.ylabel('score')
plt.xlabel('number of trees')

plt.subplot(122)
plt.plot(range_m, scores_m)
plt.ylabel('score')
plt.xlabel('max depth')

'''用选出的参数训练正式使用的模型'''
model = RandomForestClassifier(n_estimators=best_n, max_depth=best_m)
model.fit(train_kobe, train_label)
