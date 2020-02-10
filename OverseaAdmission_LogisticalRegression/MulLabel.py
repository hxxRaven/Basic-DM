'''
one 2 all 思想
'''
import pandas as pd
import matplotlib.pyplot as plt
import os

pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)


path = 'data' + os.sep + 'auto-mpg.data'
columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
           'acceleration', 'model year', 'origin', 'car name']
cars = pd.read_table(path, delim_whitespace=True, names=columns)

'''将原有特征转化为多维形式（如年龄有4、6、8、10岁，将4,6,8,10作为特征名，为每行填写0和1表示此行实际数据）'''
dummy_cylinders = pd.get_dummies(cars["cylinders"], prefix="cyl")
cars = pd.concat([cars, dummy_cylinders], axis=1)
# print(dummy_cylinders)
dummy_years = pd.get_dummies(cars['model year'], prefix='year')
cars = pd.concat([cars,dummy_years], axis=1)

cars = cars.drop('model year', axis=1)
cars = cars.drop('cylinders', axis=1)

# print(cars.head(5))
import numpy as np
shuffled_rows =np.random.permutation(cars.index)
shuffled_cars = cars.loc[shuffled_rows]
train_rows = int(cars.shape[0] * 0.7)
train = shuffled_cars[0:train_rows]
test = shuffled_cars[train_rows:]

from sklearn.linear_model import LogisticRegression
unique_origins = np.unique(cars["origin"])
unique_origins.sort()

models = {}
features = [c for c in train.columns if c.startswith('cyl') or c.startswith('year')] #选取处理后的两特征

'''对特征，进行迭代训练'''
for origin in unique_origins:
    model = LogisticRegression()

    X_train = train[features]
    y_train = train['origin'] == origin

    model.fit(X_train, y_train)
    models[origin] = model

test_probs = pd.DataFrame(columns=unique_origins)
print(test_probs)
'''每个模型进行test，结果存在test——prob中'''
for origin in unique_origins:
    X_test = test[features]
    test_probs[origin] = models[origin].predict_proba(X_test)[:, 1]

# print(test_probs)

'''选出最大可能性，并以类别存储'''
convert = test_probs.values
print(convert)
test_probs['last_predict'] = np.argmax(convert, axis=1)
test_probs['last_predict'] = test_probs['last_predict'] .values + 1
print(test_probs)

