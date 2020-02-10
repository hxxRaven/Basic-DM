import pandas as pd
import matplotlib.pyplot as plt
import os

path = 'data' + os.sep + 'admissions.csv'
admission = pd.read_csv(path)
print(admission.head(10))

import numpy as np

np.random.seed(8)
shuffled_index = np.random.permutation(admission.index)
# print(shuffled_index)
shuffled_admission = admission.loc[shuffled_index]  #按新的index排列
'''切分训练测试集'''
train = shuffled_admission[0:515]
test = shuffled_admission[515:len(shuffled_admission)]

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(train[['gpa']], train['admit'])
labels = model.predict(test[["gpa"]])
test["predicted_label"] = labels
matches = test["predicted_label"] == test["admit"]
correct_predictions = test[matches]
accuracy = len(correct_predictions) / len(test)
true_positive_filter = (test["predicted_label"] == 1) & (test["admit"] == 1)
true_positives = len(test[true_positive_filter])
false_negative_filter = (test["predicted_label"] == 0) & (test["admit"] == 1)
false_negatives = len(test[false_negative_filter])

sensitivity = true_positives / float((true_positives + false_negatives))
print(sensitivity)

false_positive_filter = (test["predicted_label"] == 1) & (test["admit"] == 0)
false_positives = len(test[false_positive_filter])
true_negative_filter = (test["predicted_label"] == 0) & (test["admit"] == 0)
true_negatives = len(test[true_negative_filter])

specificity = (true_negatives) / float((false_positives + true_negatives))
print(specificity)

'''绘制ROC曲线'''
from sklearn import metrics
probabiltities = model.predict_proba(test[['gpa']])
fpr, tpr, thresholds = metrics.roc_curve(test['admit'], probabiltities[:, 1])
print(thresholds)
plt.plot(fpr, tpr)
plt.show()

'''ROC打分'''
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(test['admit'], probabiltities[:, 1])
print(auc_score)

import pandas as pd
import numpy as np

admissions = pd.read_csv("admissions.csv")
admissions["actual_label"] = admissions["admit"]
admissions = admissions.drop("admit", axis=1)

shuffled_index = np.random.permutation(admissions.index)
shuffled_admissions = admissions.loc[shuffled_index]
admissions = shuffled_admissions.reset_index()
admissions.ix[0:128, "fold"] = 1    #给0-128的数据切块

