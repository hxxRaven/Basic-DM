'''
1.learning rate                  一般 0.1 以下
2.tree                           决策树的参数
max_depth
min_child_weight
subsample, colsample_bytree
gamma
3.正则化参数
lambda
alpha

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,             #树的个数
 max_depth=5,
 min_child_weight=1,            #叶子节点最新权重系数
 gamma=0,                       #惩罚项系数（T总叶子节点数前的值）
 subsample=0.8,                 #随机选择样本的比例，1为不随机
 colsample_bytree=0.8,          #每次选用特征建树的随机比例
 objective= 'binary:logistic',  #用什么损失函数（求解G和H的必须）
 nthread=4,
 scale_pos_weight=1,
 seed=27)
'''

import os
import xgboost
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

'''生成可用数据集，并切分训练测试集'''
path = 'data' + os.sep + 'pima-indians-diabetes.csv'
dataset = np.loadtxt(path,  delimiter=',')

X = dataset[ : , 0:8]
y = dataset[ : , 8]

seed = 7
test_size = 0.33    #切分比例

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

'''训练模型,并依次检测学习率对结果的影响'''
model = XGBClassifier()
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]    #待用学习率
param_grid = dict(learning_rate = learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring='neg_log_loss', n_jobs=-1, cv=kfold)  #n_jobs多线程全部跑
grid_result = grid_search.fit(X, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for means, params in zip(means, params):
    print("%f  with: %r" % (means, params))