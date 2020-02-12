'''
###  时间序列 ###
- 时间戳（timestamp）
- 固定周期（period）
- 时间间隔（interval）
'''

import pandas as pd
import numpy as np

'''
### date_range ###
- 可以指定开始时间与周期
- H：小时
- D：天
- M：月
'''
#起始时间戳，生成几个周期，每个周期多长
rng = pd.date_range('2016-07-01', periods = 10, freq = '3D')

#生成20个数据来填充20个时间戳
time = pd.Series(np.random.rand(20), index=pd.date_range('2020-01-01', periods=20))
print(time)
print(time['2020-01-06'])   #通过时间戳来取数据（也可以切片取）

data = pd.date_range('2009-12-31', '2011-01-01', freq='M')  #也可以通过指定首尾和间隔
print(data)

time.truncate(before='2020-01-10')  #truncate过滤-afterhe1before
time.truncate(after='2020-01-15')

#时间戳
print(pd.Timestamp('2016-07-10'))
print(pd.Timestamp('2016-07-10 10'))
print(pd.Timestamp('2016-07-10 10:15'))

# 时间区间
print(pd.Period('2016-01'))
print(pd.Period('2016-01-01'))

# TIME OFFSETS(时间进行加减)
print(pd.Period('2016-01-01 10:10') + pd.Timedelta('1 day'))    #加上一天
print(pd.Timestamp('2016-01-01 10:10') + pd.Timedelta('1 day'))

#data和period的使用形式差不多相同
p1 = pd.date_range('2016-01-01 10:10', freq = '1D1H', periods = 10)
p2 = pd.period_range('2016-01-01 10:10', freq = '1D1H', periods = 10)
print(p2)
print(p1)

rng = pd.date_range('2016 Jul 1', periods = 10, freq = 'D')

c1 = pd.Series(range(len(rng)), index = rng)
print(c1)

#时间戳切片时不会包含断点，时间段则会包含
ts = pd.Series(range(10), pd.date_range('07-10-16 8:00', periods = 10, freq = 'H'))
ts_period = ts.to_period()
print(ts_period['2016-07-10 08:30':'2016-07-10 11:45'])
print(ts['2016-07-10 08:30':'2016-07-10 11:45'])