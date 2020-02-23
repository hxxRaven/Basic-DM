import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (10.0, 8.0)  #初始化参数
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(0)   #保证每次运行随机结果一致
N = 100 # 没类有100个点
D = 2 # 2维片面
K = 3 # 3类
X = np.zeros((N*K,D))   #构建二维矩阵存放点坐标
y = np.zeros(N*K, dtype='uint8')  #构建label列表

'''建立螺旋的三条线
   分三次循环，先均等步长确定半径，再（（0-4取100个数）+（0-99取随机数*0.2）确定t值，然后半径*sint和半径*cost作为坐标，最后打上对应标签）
'''
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim([-1,1])
plt.ylim([-1,1])
plt.show()
    