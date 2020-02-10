'''交叉验证'''
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import os

path = 'data' + os.sep + 'admissions.csv'
admission = pd.read_csv(path)

kf = KFold(n_splits=5, shuffle=True, random_state=7)
lr = LogisticRegression()

accuracies = cross_val_score(lr, admission[['gpa']], admission['admit'], scoring='roc_auc', cv=kf)
average_acc = sum(accuracies) / len(accuracies)
print(accuracies)
print(average_acc)