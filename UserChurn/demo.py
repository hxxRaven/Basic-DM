import pandas as pd
import numpy as np
import os


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

path = 'data' + os.sep + 'churn.csv'
churn = pd.read_csv(path)

col_names = churn.columns.tolist()

print('Column names')
print(col_names)

to_show = col_names[:6] + col_names[-6:]    #只显示前6后6特征
print(churn[to_show].head(10))

churn_result = churn['Churn?']  #提取label

y = np.where(churn_result == 'True', 1,0)   #True用1和0替换

to_drop = ['State', 'Area Code', 'Phone', 'Churn?']
churn_feat_space = churn.drop(to_drop, axis=1)

#yes和no转换为True和False
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = (churn_feat_space[yes_no_cols] == 'yes')
print(churn_feat_space.head(10))

features = churn_feat_space.columns

# print(churn_feat_space.info())

X = churn_feat_space.values.astype(np.float)   #X表示只有数值的二维矩阵

# print(X)

#标准化数据
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
print("Feature space holds %d observations and %d features" % X.shape)
print ("Unique target labels:", np.unique(y))
print (X[0])
print (len(y[y == 0]))


#多分类器训练模块
from sklearn.model_selection import KFold

def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(n_splits=7, shuffle=True)
    y_pred = y.copy()


    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]

        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN

def accuracy(y_true,y_pred):
    # NumPy interprets True and False as 1. and 0.
    return np.mean(y_true == y_pred)

#输出准确率
print("Support vector machines:")
print("%.3f" % accuracy(y, run_cv(X,y,SVC)))
print("Random forest:")
print("%.3f" % accuracy(y, run_cv(X,y,RF)))
print("K-nearest-neighbors:")
print("%.3f" % accuracy(y, run_cv(X,y,KNN)))


#获取预测概率，人数，实际流失比例的表格
def run_prob_cv(X, y, clf_class, **kwargs):
    kf = KFold(n_splits=7, shuffle=True)
    y_prob = np.zeros((len(y),2))

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)

        y_prob[test_index] = clf.predict_proba(X_test)
    return y_prob

import warnings
warnings.filterwarnings('ignore')

pred_prob = run_prob_cv(X, y, RF, n_estimators=10)  #获得预测表

pred_churn = pred_prob[:,1]
is_churn = (y == 1) #把y中等于1的标记为True，否则为False

counts = pd.value_counts(pred_churn)    #统计各种结果的总数

true_prob = {}
for prob in counts.index:
    true_prob['prob'] = np.mean(is_churn[pred_churn == prob])   #pred_churn中选取概率等于prob的，再统计True和False数量，求比值
    true_prob = pd.Series(true_prob)

counts = pd.concat([counts,true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']