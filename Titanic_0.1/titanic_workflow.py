import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)


path = 'data' + os.sep + 'train.csv'
titanic = pd.read_csv(path)

print(titanic.head())
print(titanic.describe())

'''重要的特征，存在缺失。需要补缺值，可以用均值'''

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())     #填充均值

'''字符量不易于分析，用数值分析'''
titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1

'''上船地点在这里也是，Embarked，用出现频率最大值填充缺失值，但还是要数值化'''
print(titanic['Embarked'].unique())     #按出现次数，统计显示值
titanic['Embarked'] = titanic['Embarked'].fillna('S')
titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2

'''..........................预处理简单完成，开始进行处理.............................'''
'''线性回归部分'''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold       #交叉验证

predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']    #提取特征
alg = LinearRegression()

print(titanic.shape[0])
kf = KFold(n_splits=3, random_state=1,shuffle=True)


predictions = []
for train_index,test_index in kf.split(titanic[predictors]):
    train_predictors = (titanic[predictors].iloc[train_index, :])
    train_target = titanic['Survived'].iloc[train_index]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(titanic[predictors].iloc[test_index, :])
    test_predictions = np.where(test_predictions <= 0.5, 0, 1)
    real_label = np.array(titanic['Survived'].iloc[test_index])
    score = (test_predictions == real_label)
    predictions = predictions + score.tolist()
# predictions = np.concatenate(predictions, axis=0)
# predictions[predictions > 0.5] =1
# predictions[predictions <= 0.5] = 0
# acc = sum(predictions[predictions == titanic['Survived']]) / len(predictions)
# print(acc)
from collections import Counter
import operator
from functools import reduce
print(predictions)
last_result = dict(Counter(predictions))
# acc = last_result.get('True') / len(predictions)
acc = int(last_result.get(True)) / len(predictions)
print(acc)

'''线性回归的精确度78%，不符合预期，使用逻辑回归尝试'''
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

alg = LogisticRegression(random_state=1)
scores = cross_val_score(alg, titanic[predictors], titanic['Survived'], cv=3)
print(scores.mean())
# print(acc)

'''引入test集'''

path = 'data' + os.sep + 'test.csv'
titanic_test = pd.read_csv(path)
titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].median())     #填充均值

'''train中的预处理同步骤'''
titanic_test.loc[titanic_test['Sex'] == 'male', 'Sex'] = 0
titanic_test.loc[titanic_test['Sex'] == 'female', 'Sex'] = 1

print(titanic['Embarked'].unique())
titanic_test['Embarked'] = titanic_test['Embarked'].fillna('S')
titanic_test.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic_test.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic_test.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2

'''回归太过简单，使用随机森林尝试'''
from sklearn.ensemble import RandomForestClassifier
predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
rfm1 = RandomForestClassifier(random_state=1, n_estimators=10,
                             min_samples_split=2, min_samples_leaf=1)
# '''n_estimators生成几棵树， min_samples_split 少于两个样本时不在切分，min_sample_leaf叶子节点最小个数
#     防止每棵树很高，防止过拟合'''
kf = KFold(n_splits=3, shuffle=False, random_state=1)
scores1 = cross_val_score(rfm1, titanic[predictors], titanic["Survived"],cv=kf)
print(scores.mean())

'''初始效果并不好，准备调参'''
rfm2 = RandomForestClassifier(random_state=1, n_estimators=50,
                              min_samples_split=4, min_samples_leaf=2)
'''增加森林规模，放宽划分数量'''
scores2 = cross_val_score(rfm2, titanic[predictors],titanic['Survived'], cv=kf)
print(scores.mean())


'''没什么太大变化,考虑是否是特征不够
    考虑家庭成员数，名字长度，名字中的身份，看是否有影响'''
titanic['FamilySize'] = titanic['SibSp'] +titanic['Parch']
titanic['NameLength'] = titanic['Name'].apply(lambda x:len(x))

'''提取称呼模块'''
import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)        #正则表达式匹配称呼模式
    if title_search:
        a = title_search.group()
        return title_search.group(1)                        #将所有能匹配模式中第一个括号内的项打包为元组
    return ""

titles = titanic['Name'].apply(get_title)                   #按样板进行，同时赋给新的ndarry
print(pd.value_counts(titles))

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4,
                 "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                 "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10,
                 "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():                                   #字典循环映射
    titles[titles == k] =v

print(pd.value_counts(titles))

titanic['Title'] = titles

'''预选完特征后，分析下这些特征的价值，把要分析特征线参与构建模型，再破坏掉数据构建模型，看准确率（随机森林）'''
from sklearn.feature_selection import SelectKBest, f_classif
predictors1 = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
              "Embarked", "FamilySize", "Title", "NameLength"]      #x新候选特征
selector = SelectKBest(f_classif, k=5)  #对于分类用f_classif。对于回归用f_regression。k值为显示几个优秀参数
selector.fit(titanic[predictors1], titanic['Survived'])

selector_scores = -np.log10(selector.pvalues_)

plt.bar(range(len(predictors1)), selector_scores)
plt.xticks(range(len(predictors1)), predictors1, rotation='vertical')
plt.show()

'''根据分析图，选择优化参数。选择三个不同分类器，集成解决'''
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

predictors_last = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]
# The algorithms we want to ensemble.
# We're using the more linear predictors for the logistic regression, and everything with the gradient boosting classifier.
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors_last],
    [LogisticRegression(random_state=1), predictors_last]
]

# Initialize the cross validation folds
kf = KFold(n_splits=3,shuffle=False, random_state=1)

predictions = []
for train, test in kf.split(titanic[predictors_last]):
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # Make predictions for each algorithm on each fold
    for alg, predictors_last in algorithms:
        # Fit the algorithm on the training data.
        alg.fit(titanic[predictors_last].iloc[train,:], train_target)
        # Select and predict on the test fold.
        # The .astype(float) is necessary to convert the dataframe to all floats and avoid an sklearn error.
        test_predictions = alg.predict_proba(titanic[predictors_last].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# Put all the predictions together into one array.

predictions = np.concatenate(predictions, axis=0)           #按行将array拼接起来,构成一个一行n列的ndarry

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions == titanic["Survived"]) / len(predictions)
print(accuracy)