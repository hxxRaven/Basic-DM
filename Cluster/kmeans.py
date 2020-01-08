import pandas as pd
import os

path = 'data' + os.sep + 'data.txt'
beer = pd.read_csv(path, sep=' ')

print(beer)

X = beer[['calories', 'sodium', 'alcohol', 'cost']]

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3).fit(X)
km2 = KMeans(n_clusters=2).fit(X)

print(km.labels_)

beer['cluster'] = km.labels_
beer['cluster2'] = km2.labels_

print(beer.groupby('cluster').mean())       #排个序，按cluster分组，显示均值差异

from pandas.plotting import scatter_matrix


cluster_centers = km.cluster_centers_
cluster_centers_2 = km2.cluster_centers_
centers = beer.groupby("cluster").mean().reset_index()  #与上一次groupby差不多，但将index改变了赋给了centers这个变量

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow'])

plt.scatter(beer["calories"], beer["alcohol"],c=colors[beer["cluster"]])

plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker='+', s=300, c='black')

plt.xlabel("Calories")
plt.ylabel("Alcohol")


'''
colors = np.array(['red', 'green', 'blue', 'yellow'])
c=colors[beer["cluster"]]
会将cluster的属性对应上colors里的颜色
'''

'''各个维度显示当前特征的分布情况——3个簇的情况下'''
scatter_matrix(beer[["calories","sodium","alcohol","cost"]],s=100, alpha=1, c=colors[beer["cluster"]], figsize=(10,10))
plt.suptitle("With 3 centroids initialized")



'''各个维度显示当前特征的分布情况——2个簇的情况下'''
scatter_matrix(beer[["calories","sodium","alcohol","cost"]],s=100, alpha=1, c=colors[beer["cluster2"]], figsize=(10,10))
plt.suptitle("With 2 centroids initialized")

'''标准化,其实是要衡量各个特征的权重'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)     #会被转换成ndarry，但保持维度

km3 = KMeans(n_clusters=3).fit(X_scaled)
beer["scaled_cluster"] = km3.labels_
print(beer.sort_values("scaled_cluster"))

print(beer.groupby('scaled_cluster').mean())

scatter_matrix(X, s=100, alpha=1, c=colors[beer["scaled_cluster"]], figsize=(10,10))
plt.suptitle("With Scaled initialized")

'''轮廓系数评估'''
from sklearn import metrics
score_scaled = metrics.silhouette_score(X, beer.scaled_cluster) #参数为原始数据集，和分类结果
score = metrics.silhouette_score(X, beer.cluster)

print(score_scaled, score)  #标准化不一定使结果更好

'''遍历k值，看评价情况，选最优K值'''
scores = []
for k in range(2,20):
    labels = KMeans(n_clusters=k).fit(X).labels_
    score_k = metrics.silhouette_score(X,labels)
    scores.append(score_k)

print(scores)

plt.plot(list(range(2,20)), scores)
plt.xlabel("Number of Clusters Initialized")
plt.ylabel("Sihouette Score")

