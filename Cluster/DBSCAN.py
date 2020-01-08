from sklearn.cluster import DBSCAN
import os
import pandas as pd
import numpy as np

colors = np.array(['red', 'green', 'blue', 'yellow'])
path = 'data' + os.sep + 'data.txt'
beer = pd.read_csv(path, sep=' ')

X = beer[['calories', 'sodium', 'alcohol', 'cost']]

'''不规则数据集适用
    eps和数据值分布有关，归一化之后要小些0.几
'''
db = DBSCAN(eps=10, min_samples=2).fit(X)
labels = db.labels_
beer['cluster_db'] = labels
print(beer.sort_values('cluster_db'))

print(beer.groupby('cluster_db').mean())

from pandas.plotting import scatter_matrix
scatter_matrix(X, c=colors[beer.cluster_db], figsize=(10,10), s=100)

from sklearn.metrics import silhouette_score
scores = []
for e, m in zip(range(5,15), range(1-6)):
    labels = DBSCAN(eps=e, min_samples=m).fit(X).labels_
    score_em = silhouette_score(X, labels)
    scores.append()
print(scores)