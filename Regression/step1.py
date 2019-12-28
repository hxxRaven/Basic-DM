import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from sklearn import preprocessing as pre

'''
            逻辑回归 The Logistic Regression
        目标：建立分类器（求解出三个参数 𝜃0 𝜃1 𝜃2  
        确定阈值，根据输入成绩判断结果
                     要完成的模块
        *sigmoid : 映射到概率的函数
        •model : 返回预测结果值
        •cost : 根据参数计算损失
        •gradient : 计算每个参数的梯度方向
        •descent : 进行参数更新
        •accuracy: 计算精度
'''

'''比较三种梯度下降方法'''
STOP_ITER = 0                                   #根据迭代次数
STOP_COST = 1                                   #根据损失值
STOP_GRAD = 2                                   #根据梯度

def sigmod(z):
    return 1 / (1 + np.exp(-z))

# nums = np.arange(-10,10,step = 1)
# fig,ax = plt.subplots(figsize = (12,4))                   #查看sigmod
# ax.plot(nums,sigmod(nums),c = 'r')
# plt.show()

def model(X,theta):
    # print(theta.shape)
    # print(X.shape)
    return sigmod(np.dot(X,theta.T))            #返回预测值

def cost(X,y,theta):
    left = np.multiply(-y,np.log(model(X,theta)))
    right = np.multiply(1 - y,np.log(1 - model(X,theta)))
    return np.sum(left - right) / (len(X))

# print(cost(X,Y,theta))                          #测试损失函数是否好用

def gradient(X,y,theta):
    grad = np.zeros(theta.shape)                #求解三个参数
    error = (model(X,theta) - y).ravel()
    for j in range(len(theta.ravel())):         #三个参数，需求三次偏导
        term = np.multiply(error,X[ : ,j])
        grad[0,j] = np.sum(term) / len(X)
    return grad

def stopCriterion(type, value, threshlod):
    if type == STOP_ITER:   return value > threshlod
    elif type == STOP_COST: return abs(value[-1]-value[-2]) < threshlod             #最后的值-倒数第二数，看损失值变化大不大
    elif type == STOP_GRAD: return np.linalg.norm(value) < threshlod                #范数

def shuffleData(data):              #元素洗牌，消除收集数据时的规律性。提高泛化能力
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[ : , 0 : cols-1 ]
    y = data[ : , cols-1 : ]
    return X ,y

def descent(data, theta, batchSize, stopType, thresh, alpha):           #梯度下降
    init_time = time.time()
    i = 0
    k = 0
    X, y = shuffleData(data)
    grad = np.zeros(theta.shape)
    costs = [cost(X, y, theta)]

    while True:
        grad = gradient(X[k:k+batchSize], y[k:k+batchSize], theta)
        k += batchSize
        if k >= n:
            k =0
            X, y =shuffleData(data)
        theta = theta - alpha * grad
        costs.append(cost(X,y,theta))
        i += 1

        if stopType == STOP_ITER:       value = i
        elif stopType == STOP_COST:     value = costs
        elif stopType == STOP_GRAD:     value = grad
        print(stopCriterion(stopType, value, thresh))
        if stopCriterion(stopType, value, thresh):  break
    return theta, i-1, costs, grad, time.time() - init_time

def runExpe(data, theta, batchSize, stopType, thresh, alpha):
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    print((data[:,1]>2).sum() > 1)
    print(data)
    print(data[:,1])
    name += " data —— learning rate: {} - ".format(alpha)
    if batchSize == n: strDescType = "Gradient"
    elif batchSize == 1: strDescType = "Stochastic"
    else: strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType +"descent - Stop: "
    if stopType == STOP_ITER: strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST: strStop = "cost change < {}".format(thresh)
    else: strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)),costs,c='r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + ' - Error vs. Iteration')
    plt.ioff()
    plt.show()
    return theta

'''计算精度'''
def predict(X,theta):
    return [1 if x >= 0.5 else 0 for x in model(X,theta)]

def main(pdData):
    pdData.insert(0, 'Ones', 1)  # DataFrame中插入第一列，值全为1
    orig_data = pdData.as_matrix()  # DataFrame转为矩阵
    cols = orig_data.shape[1]  # 读取列数

    X = orig_data[:, 0: cols - 1]  # 把结果切割
    y = orig_data[:, cols - 1: cols]
    theta = np.zeros([1, 3])  # 初始化𝜃集
    # print(theta.ravel())
    global n
    n = 100
    # runExpe(orig_data,theta,n,STOP_ITER,thresh=5000,alpha=0.000001)         #测试一，5000次梯度下降
    # runExpe(orig_data,theta,n,STOP_COST,thresh=0.000001,alpha=0.001)        #测试二，损失差阈值1E-6

    '''对比不同梯度下降方法'''
    # runExpe(orig_data,theta,1,STOP_ITER,thresh=5000,alpha=0.001)            #测试三，只迭代一个样本，减少迭代次数，增大学习率
    # runExpe(orig_data,theta,1,STOP_ITER,thresh=15000,alpha=0.000002)        #测试三，增加迭代次数，减小学习率

    '''Mini-batch descent'''
    # runExpe(orig_data,theta,16,STOP_ITER,thresh=15000,alpha=0.001)              #mini_batch,15000次迭代，学习率0.001
    '''数据浮动仍然很大，尝试对数据标准化处理（减均值，除以方差），使其均值在0附近，方差为1，标准正态分布'''

    scaled_data = orig_data.copy()
    scaled_data[:, 1:3] = pre.scale(orig_data[:, 1:3])
    print(scaled_data)
    # runExpe(scaled_data,theta,n,STOP_ITER,thresh=5000,alpha=0.001)

    # runExpe(scaled_data, theta, n, STOP_GRAD, thresh=0.02, alpha=0.001)     #迭代次数增加，效果更好
    theta1 = runExpe(scaled_data, theta, 16, STOP_GRAD, thresh=0.002 * 2, alpha=0.001)  # 少样本，较小学习率，迭代次数少，效率高
    scaled_X = scaled_data[:, 0:3]
    y = scaled_data[:, 3]
    predicitons = predict(scaled_X, theta1)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predicitons, y)]
    accuracy = sum(map(int, correct)) / len(correct)
    print('Accuracy = {}%'.format(accuracy * 100))

if __name__ == '__main__':
    path = 'data' + os.sep + 'LogiReg_data.txt'
    pdData = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])  # 并不是标准的数据源，第一行不是列名，所以将头部设为None
    # print(pdData.shape)
    # print(pdData.head())
    positive = pdData[pdData['Admitted'] == 1]  # 把样本分类存储
    negative = pdData[pdData['Admitted'] == 0]
    # print(positive)
    fix,ax = plt.subplots(figsize=(10,5))
    ax.scatter(positive['Exam1'],positive['Exam2'],s = 30,c = 'b',marker = 'o',label = 'Admitted')      #两个类别显示分布情况，看大致边界
    ax.scatter(negative['Exam1'],negative['Exam2'],s = 30,c = 'r',marker = 'x',label = 'Not Admitted')
    ax.set_xlable=('Exam1 Score')
    ax.set_ylable=('Exam2 Score')
    ax.legend()
    plt.show()

    main(pdData)