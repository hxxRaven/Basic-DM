import numpy as  np


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, GridSearchCV

import os
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

import re

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import GradientBoostingClassifier

path = 'data' + os.sep + 'train.csv'
titanic = pd.read_csv(path)

print(titanic.describe())

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())# 缺失值用均值填充

print(titanic.describe())

print(titanic["Sex"].unique())# 返回参数中所有不同的值，并且按照从小到大的顺序排列

# 将male和female用0和1代替

titanic.loc[titanic["Sex"] =="male", "Sex"] = 0
titanic.loc[titanic["Sex"] =="female", "Sex"] =1

print(titanic["Embarked"].unique())

titanic["Embarked"] = titanic["Embarked"].fillna("S")# 有缺失值，我们采用谁多用谁

titanic.loc[titanic["Embarked"] =="S", "Embarked"] =0
titanic.loc[titanic["Embarked"] =="C", "Embarked"] =1
titanic.loc[titanic["Embarked"] =="Q", "Embarked"] =2

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg = LinearRegression()

kf = KFold(n_splits=3, shuffle=False)

predictions = []

for train, test in kf.split(titanic[predictors]):
    train_predictors = (titanic[predictors].iloc[train, :])
    train_target = titanic["Survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic[predictors].iloc[test, :])
    print(test_predictions)
    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)

print(predictions)
predictions[predictions >.5] =1
predictions[predictions <=.5] =0

accuracy =sum(predictions == titanic["Survived"]) /len(predictions)
print(accuracy)
