import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.Series(np.random.rand(600), index=pd.date_range('2016-7-1', freq='D', periods=600))
print(df.head(10))

r = df.rolling(window=10)   #Rolling [window=10,center=False,axis=0] center为是否从中间滑动

print(r.mean().head(15))    #求每个窗口均值

fig1 = plt.figure(figsize=(15, 5))
df.plot(style='r--')
r.mean().plot(style='b')