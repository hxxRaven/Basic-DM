#Train a Linear Classifier
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D))
y = np.zeros(N*K, dtype='uint8')
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j



W = 0.01 * np.random.randn(D,K)
b = np.zeros((1,K))

# 步长和正则化惩罚项
step_size = 1e-0
reg = 1e-3

# gradient descent loop
num_examples = X.shape[0]

for i in range(1000):

  #以下为求Li交叉熵损失的步骤
  scores = np.dot(X, W) + b
  #print scores.shape 
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K] probs:300*3

  # print(probs.shape)

  corect_logprobs = -np.log(probs[range(num_examples),y]) #probs[range(num_examples),y]为在0-299索引中按（找与y的label标签对应的概率值取出）

  # print(corect_logprobs.shape)

  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss

#计算Li到此结束

  if i % 100 == 0:

    print("iteration %d: loss %f" % (i, loss))
  #反向传播
  #计算梯度（均一化后结果先把应当正确项-1，然后除以规模，再用X的转置相乘得到dW，）
  dscores = probs
  dscores[range(num_examples),y] -= 1
  dscores /= num_examples
  
  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(X.T, dscores)

  db = np.sum(dscores, axis=0, keepdims=True) #处理后的dscores每类累加得db

  dW += reg*W # regularization gradient
  
  # 梯度下降通过步长得到新的W和b
  W += -step_size * dW
  b += -step_size * db
  scores = np.dot(X, W) + b

predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))

'''计以算出的W和b，计算边界画出等高线划分'''
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

print(x_min,x_max)
print(y_min,y_max)

#根据每个维度的取值区间，以0.02为步长求得点坐标，再按图像reshape为对应尺寸网格点坐标矩阵
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))



#np.c化列想连接。np.r化行相连接（ravel把原nparry转为一行）



Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b  #将n*2和2*3点乘，加上b得到预测备用矩阵
Z = np.argmax(Z, axis=1)  #求最大概率（进行类别划分）
Z = Z.reshape(xx.shape)   #划分后的Z，reshape为网格点坐标，便于画出等高线图

fig = plt.figure()

plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

