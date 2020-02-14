import pandas as pd

loans = pd.read_csv('data\cleaned_loans.csv')
print(loans.info())

#逻辑回归
from sklearn.linear_model import LogisticRegression #逻辑回归一般都是分类
lr = LogisticRegression()
cols = loans.columns
train_cols = cols.drop('loan_status')   #去掉那个label列

features = loans[train_cols]
target = loans['loan_status']
lr.fit(features, target)
predictions = lr.predict(features)  #训练集和测试集相同

print(features.shape)

#交叉验证和逻辑回归
from sklearn.model_selection import cross_val_predict, KFold
lr1 = LogisticRegression()

kf = KFold(n_splits=7, random_state=1)

predictions1 = cross_val_predict(lr1, features, target, cv=kf)
predictions1 = pd.Series(predictions1)

fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / float((tp + fn))
fpr = fp / float((fp + tn))

print(tpr)
print(fpr)
print(predictions[:20])


'''
tpr和fpr都很大，这很异常，明显是将所有结果都判断为一种情况了
原因来自于样本里，两种结果的比例太过悬殊
可以通过重采样（补全或者计算）
也可以通过调整正负样本权重
'''


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
lr = LogisticRegression(class_weight="balanced")    #权重参数，也可以自己按照设置

penalty = {0:6, 1:1}    #前者为二分类结果，class_weight=penaly

kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)


fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])


tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])


fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])


tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])


tpr = tp / float((tp + fn))
fpr = fp / float((fp + tn))

print(tpr)
print(fpr)
print(predictions[:20])

#使用随机森林
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

rf = RandomForestClassifier(class_weight='balanced', random_state=1)
kf = KFold(n_splits=7, random_state=1)
predictions = cross_val_predict(rf, features, target, cv=kf)
predictions =pd.Series(predictions)

fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])


tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])


fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])


tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])


tpr = tp / float((tp + fn))
fpr = fp / float((fp + tn))

print(tpr)
print(fpr)
print(predictions[:20])