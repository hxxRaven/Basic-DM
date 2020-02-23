import pandas as pd
import pandas_datareader.data as web
import os
import datetime
import matplotlib.pylab as plt
import seaborn as sns
from matplotlib.pylab import style
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

stockPath = 'data' + os.sep + 'T10yr.csv'
stock = pd.read_csv(stockPath, index_col=0, parse_dates=[0]) #index_col索引是第一列，parse_dates对日期格式处理

print(stock.head(10))

stock_week = stock['Close'].resample('W-MON').mean()    #W(以周为单位)-MON(以每周几为开始)
stock_train = stock_week['2000':'2015']
print(stock_train.tail(10))
stock_train.plot()
plt.legend(loc='best')
plt.title('Stock Close')
sns.despine()

#一阶差分
stock_diff = stock_train.diff()
stock_diff = stock_diff.dropna()
plt.figure()
plt.plot(stock_diff)
plt.title('一阶差分')
plt.show()

#绘制acf和pacf

acf = plot_acf(stock_diff, lags=20)
plt.title('ACF')
acf.show()

pacf = plot_pacf(stock_diff, lags=20)
plt.title('PACF')
pacf.show()

#选取p=1，q=1， d=1 训练模型
model = ARIMA(stock_train, order=(1, 1, 1), freq='W-MON')
result = model.fit()

print(stock.tail(10))
#用模型预测
pred = model.predict(params=stock_train, start='2014-06-09', end='2015-12-28')
print(pred)