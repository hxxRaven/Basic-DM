'''
对于线性回归的分值，先进行e的分值次方，保证全为正值，然后的压缩到0-1的范围（t/1+t）,再设置阈值，进行分类
'''
import numpy as np
def logistic(x):
    return np.exp(x) / (1 + np.exp(x))


import pandas as pd
import matplotlib.pyplot as plt
import os

path = 'data' + os.sep + 'admissions.csv'
admission = pd.read_csv(path)
print(admission.head(10))

fig1 = plt.figure()
plt.scatter(admission['gpa'], admission['admit'])
plt.show()

X = np.linspace(-6, 6, dtype=float)
y = logistic(X)
fig2 = plt.figure()
plt.plot(X, y)
plt.ylabel('Probability')
plt.show()  #画一下sigmod函数

from sklearn.linear_model import LinearRegression
liner = LinearRegression()
liner.fit(admission[['gpa']], admission['admit'])

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(admission[['gpa']], admission['admit'])

pred_prob = logr.predict_proba(admission[['gpa']])  #两列，一列为0，一列为1，表示被拒绝和录取的概率
fig3 = plt.figure()
plt.scatter(admission['gpa'], pred_prob[ : , 1])
plt.show()

'''
直接逻辑回归取类别结果
'''
labels = logr.predict(admission[['gpa']])
admission['predicted_label'] = labels
print(admission['predicted_label'].value_counts())
print(admission.head(10))

matches = admission['predicted_label'] == admission['admit']
correct_prediction = admission[matches]
print(correct_prediction.head(10))
accuracy = len(correct_prediction) / float(len(admission))  #精度 正确数量/总的数量
print(accuracy)

true_positive_filter = (admission['predicted_label'] == 1) & (admission['admit'] == 1)
true_positive = len(admission[true_positive_filter])

true_negative_filter = (admission['predicted_label'] == 0) & (admission['admit'] == 0)
true_negative = len(admission[true_negative_filter])
print(true_positive)
print(true_negative)

false_negative_filter = (admission["predicted_label"] == 0) & (admission["admit"] == 1)
false_negatives = len(admission[false_negative_filter])

sensitivity = true_positive / float(true_positive + false_negatives)
print(sensitivity)

true_positive_filter = (admission["predicted_label"] == 1) & (admission["admit"] == 1)
true_positives = len(admission[true_positive_filter])
false_negative_filter = (admission["predicted_label"] == 0) & (admission["admit"] == 1)
false_negatives = len(admission[false_negative_filter])
true_negative_filter = (admission["predicted_label"] == 0) & (admission["admit"] == 0)
true_negatives = len(admission[true_negative_filter])
false_positive_filter = (admission["predicted_label"] == 1) & (admission["admit"] == 0)
false_positives = len(admission[false_positive_filter])
specificity = (true_negatives) / float((false_positives + true_negatives))
print(specificity)