'''
数据重采样
时间数据由一个频率转换到另一个频率
降采样(多的的小的数据点，整合成大的)
升采样（大的数据，拆分成小的）
'''

import pandas as pd
import numpy as np

rang = pd.date_range('2011-01-01', periods=90, freq='D')
ts = pd.Series(np.random.rand(len(rang)), index=rang)
print(ts.head(10))

print(ts.resample('M').sum())   #以月，降采样
print(ts.resample('3D').sum())  #以三天降采样

#差值填充，让原来三天为单位的升采样为一天
day3Ts = ts.resample('3D').mean()
print(day3Ts)

# print(day3Ts.resample('D').asfreq())    #不填充直接为空
'''
插值方法：
ffill 空值取前面的值
bfill 空值取后面的值
interpolate 线性取值
'''
print(day3Ts.resample('D').ffill(1))    #用前面的值填充后面一个
print(day3Ts.resample('D').bfill(1))    #用后面面的值填充后面一个

print(day3Ts.resample('D').interpolate('linear'))   #根据数据变化拟合出一个线性函数来填充