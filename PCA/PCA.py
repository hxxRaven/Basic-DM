import numpy as np
import pandas as pd
import os


path = 'data' + os.sep + 'iris.data'
df = pd.read_csv(path)
df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']    #重新规定列名
print(df.head())

X = df.iloc[ : , 0:4].values
y = df.iloc[ : , 4].values
# print(X)
# print(y)

from matplotlib import pyplot as plt
import math

label_dict = {1: 'Iris-Setosa',
              2: 'Iris-Versicolor',
              3: 'Iris-Virgnica'}

feature_dict = {0: 'sepal length [cm]',
                1: 'sepal width [cm]',
                2: 'petal length [cm]',
                3: 'petal width [cm]'}

#查看数据重合程度
plt.figure(figsize=(8, 6))
for cnt in range(4):
    plt.subplot(2, 2, cnt+1)
    for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
        plt.hist(X[y==lab, cnt],
                     label=lab,
                     bins=10,
                     alpha=0.3,)
    plt.xlabel(feature_dict[cnt])
    plt.legend(loc='upper right', fancybox=True, fontsize=8)

plt.tight_layout()
plt.show()

'''标准化'''
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)
print (X_std)

'''归一化并计算协方差矩阵'''
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std-mean_vec).T.dot((X_std-mean_vec)) / (X_std.shape[0] - 1)   #方差要除以自由度（样本数-1）

print('Covariance matrix \n%s' %cov_mat)

'''也可以直接使用numpy计算协方差矩阵'''
np_cov = np.cov(X_std.T)
print(np_cov)

'''计算特征值与特征向量'''
eig_vals, eig_vecs = np.linalg.eig(np_cov)
print(eig_vals)
print(eig_vecs)

'''将特征值和特征向量组成对应关系'''
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
print (eig_pairs)
print ('----------')

'''将特征值从数值转为比重关系'''
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
print (var_exp)
cum_var_exp = np.cumsum(var_exp)    #此变量为累加和，为后面展示趋势而坐
print(cum_var_exp)

plt.figure(figsize=(6, 4))

plt.bar(range(4), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
plt.step(range(4), cum_var_exp, where='mid',
             label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

'''可视化显示，选择最优特征向量'''
matrix = np.hstack((eig_pairs[0][1].reshape(4, 1),
                    eig_pairs[1][1].reshape(4, 1)))
print('Matrix W:\n', matrix)

Y = X_std.dot(matrix)
print(Y)

'''原有维度下分布情况'''
plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
     plt.scatter(X[y==lab, 0],
                X[y==lab, 1],
                label=lab,
                c=col)
plt.xlabel('sepal_len')
plt.ylabel('sepal_wid')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

'''PCA后分布情况'''
plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
     plt.scatter(Y[y==lab, 0],
                Y[y==lab, 1],
                label=lab,
                c=col)
     print(Y[y==lab, 0])
     print(Y[y==lab, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='lower center')
plt.tight_layout()
plt.show()