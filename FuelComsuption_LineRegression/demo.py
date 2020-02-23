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
# print(cars.head(10))

fig1 = plt.figure()
ax1 = fig1.add_subplot(2, 1, 1)
ax2 = fig1.add_subplot(2, 1, 2)
cars.plot('weight', 'mpg', kind='scatter', ax=ax1)
cars.plot('acceleration', 'mpg', kind='scatter', ax=ax2)
plt.show()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(cars[['weight']],cars['mpg']) #训练样本只选一个特征[[]]
predicitons = lr.predict(cars[['weight']])  #用训练数据来测试其实是不好的
print(predicitons[0:5])
print(cars['mpg'][0:5])

fig2 = plt.figure()
plt.scatter(cars['weight'], cars['mpg'], c='red')
plt.scatter(cars['weight'], predicitons, c='blue')
plt.show()

'''
模型训练和测试完，需要有个评判指标
均方误差（MSE）：1/n * 求1到n和（（Y预测-Y实际）的平方）
'''
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(cars['mpg'], predicitons)
print(mse)

rmse = mse ** (0.5) #对mse开根号（**幂级运算）
print(rmse)
