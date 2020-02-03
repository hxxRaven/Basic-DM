import os
import xgboost
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''生成可用数据集，并切分训练测试集'''
path = 'data' + os.sep + 'pima-indians-diabetes.csv'
dataset = np.loadtxt(path,  delimiter=',')

X = dataset[ : , 0:8]
y = dataset[ : , 8]

seed = 7
test_size = 0.33    #切分比例

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

'''训练模型'''
model = XGBClassifier()
# model .fit(X_train, y_train)
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10,
          eval_metric='logloss', eval_set=eval_set, verbose=True)   #early_stopping_rounds连续多少次添加模型，loss值没有下降则停止
                                                                    #指定eval_set（测试集）后，每添加一个模型救护进行一次loss值检测
                                                                    #eval_metric哪种loss进行评估，verbose=True每加一个模型是否进行loss检测
'''第一次预测'''
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]    #把预测值取出

accuary = accuracy_score(y_test, predictions)
print('Accuacy：%.2f%%' % (accuary * 100.0)) #accuracy_score后要*100.0

from matplotlib import pyplot as plt
from xgboost import plot_importance
plot_importance(model)      #展示特征的重要程度
plt.show