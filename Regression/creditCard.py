import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os


# print(data.head())

def print_classInfo(data):
    count_classes = pd.value_counts(data['Class'], sort=True,).sort_index()     #value_counts()，输出Series有几种类型，每种类型有多少，sort按数量大小排序，sort_index()
    count_classes.plot(kind='bar')
    plt.title('Fraud class histogram')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

'''样本数据特征值得差异太大，一般采取标准化或者归一化'''
'''标准化'''

from sklearn.preprocessing import StandardScaler

def Standard(data):
    # print(data['Amount'].values)
    data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))    #reshape(-1,x)根据原有的行列乘积，自动除以x，得到新的行
    data =data.drop(['Time','Amount'], axis = 1)
    # print(data.head())
    return data

def under_sample(data):
    '''样本数据分布不均匀，采用上采样和过采样处理'''
    X = data.iloc[ : , data.columns != 'Class']
    y = data.iloc[ : ,data.columns == 'Class']

    number_records_fraud = len(data[data.Class == 1])           #获取类别为1的样本个数
    fraud_indices = np.array(data[data.Class == 1].index)       #把样本中[Class]列值为1的index取出放入ndarry

    # normal_indices = data[data.Class == 0].index

    normal_indices = np.array(data[data.Class == 0].index)      #把样本中[Class]列值为0的index取出放入ndarry
    random_normal_indices = np.random.choice(normal_indices,number_records_fraud,replace=False) #从类别为0的样本中随机选择和样本为1样本一样数量的索引
    # random_normal_indices = np.array(random_normal_indices)
    under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])    #把处理后的两序列合并
    under_sample_data = data.iloc[under_sample_indices, : ]                         #使用合并后的序列对原dataFrame切片，得到归一化后的数据集
    X_underSample = under_sample_data.iloc[ : ,under_sample_data.columns != 'Class']
    y_underSample = under_sample_data.iloc[ : ,under_sample_data.columns == 'Class']

    print(len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
    print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
    print("Total number of transactions in resampled data: ", len(under_sample_data))
    return X_underSample, y_underSample, X, y

'''交叉验证，保证模型评价可靠性，减少离群点等异常数据对评价结果的影响'''
from sklearn.model_selection import train_test_split
def select_example(X, y, X_underSample, y_underSample):
    X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=0)     #对原始数据集切分为，测试训练集（训练样本，训练结果）和测试集（测试样本，测试结果）
    print("Number transactions train dataset: ", len(X_train))
    print("Number transactions test dataset: ", len(X_test))
    print("Total number of transactions: ", len(X_train)+len(X_test))

    X_train_underSample, X_test_underSample, y_train_underSample, y_test_underSample =train_test_split(X_underSample, y_underSample, test_size=0.3, random_state=0)     #对标准化和采样后的数据集做同样操作
    print("")
    print("Number transactions train dataset: ", len(X_train_underSample))
    print("Number transactions test dataset: ", len(X_test_underSample))
    print("Total number of transactions: ", len(X_train_underSample)+len(X_test_underSample))
    return X_train, X_test, y_train, y_test, X_train_underSample, X_test_underSample, y_train_underSample, y_test_underSample

'''recall相关'''
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report

def print_KFold_scores(x_train_data, y_train_data):
    fold = KFold(5,shuffle=False)         #后续验证中，按原始训练集长度切分成5份，不洗牌
    c_param_range = [0.01, 0.1, 1, 10, 100]
    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range
    print(results_table.head())
    j=0
    for c_param in c_param_range:
        print('-------------------------------------------')
        print('C parameter: ', c_param)
        print('-------------------------------------------')
        print('')

        '''交叉验证'''
        recall_accs = []
        for iteration, indices in enumerate(fold.split(x_train_data)):     #耙可遍历数据对象索引化，start为确定开始下标
            lr = LogisticRegression(C=c_param, penalty='l1', solver='liblinear')    #实例化模型，确定惩罚范数
            lr.fit(x_train_data.iloc[indices[0], :],y_train_data.iloc[indices[0], :].values.ravel())        #使用交叉验证序列化索引和之前切分好的训练集中4/5数据（X,y）训练模型
            y_pred_underSample = lr.predict(x_train_data.iloc[indices[1], :].values)    #使用训练集中的测试部分索引(1/5)预测结果
            recall_acc = recall_score((y_train_data.iloc[indices[1], :]).values, y_pred_underSample)   #使用预测结果和y_train中的（1/5）结果计算recall
            recall_accs.append(recall_acc)          #recall添加到recall_accs list中去
            print('Iteration：', iteration, '。 recall_score = ',recall_acc,'。')

        results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

    best_c = results_table.loc[results_table['Mean recall score'].astype('float64').idxmax()]['C_parameter']
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    print(results_table['Mean recall score'])
    return best_c



'''进行混淆矩阵分析，对采样处理后数据'''
import itertools
def plot_confusion_matrix(cm, classes, title='Confusin Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() /2
    # print(range(cm.shape[0]), range(cm.shape[1]))
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def show_Raw_result(best_c, X_train_undersample, y_train_undersample, X_test, y_test):
    lr = LogisticRegression(C=best_c, penalty='l1', solver='liblinear')
    lr.fit(X_train_undersample, y_train_undersample.values.ravel())
    y_pred = lr.predict(X_test.values)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    print("Recall metric in the testing dataset: ", cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))

    # Plot non-normalized confusion matrix
    class_names = [0, 1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Confusion matrix')
    plt.show()

'''阈值调整，误杀数下降，精度上升，但recall下降。实际应用中，需要考虑精度，recall，误杀数量选择阈值'''
def show_Sample_adjust_threshold(X_train_underSample, y_train_underSample, X_test_underSample, y_test_underSample):
    lr = LogisticRegression(C=0.01, penalty='l1', solver='liblinear')
    lr.fit(X_train_underSample, y_train_underSample.values.ravel())
    y_pred_underSample_proba = lr.predict_proba(X_test_underSample.values)
    print(y_pred_underSample_proba)
    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    plt.figure(figsize=(10,10))
    j = 1
    for i in threshold:
        y_test_predictions_high_recall = y_pred_underSample_proba[:, 1] > i
        plt.subplot(3,3,j)
        j += 1
        cnf_matrix = confusion_matrix(y_test_underSample,y_test_predictions_high_recall)
        np.set_printoptions(precision=2)
        print("Recall metric in the testing dataset: ",
              cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
        class_names = [0,1]
        plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >= %s'%i)
    plt.show()

if __name__ == '__main__':
    path = 'data' + os.sep + 'creditcard.csv'
    data = pd.read_csv(path)

    print_classInfo(data)

    data = Standard(data)
    X_underSample, y_underSample, X, y = under_sample(data)
    X_train, X_test, y_train, y_test, \
    X_train_underSample, X_test_underSample, \
    y_train_underSample, y_test_underSample = select_example(X, y, X_underSample,y_underSample)

    best_c = print_KFold_scores(X_train_underSample, y_train_underSample)

    show_Raw_result(best_c, X_train_underSample, y_train_underSample, X_test_underSample, y_test_underSample)
    show_Raw_result(best_c, X_train_underSample, y_train_underSample, X_test, y_test)

    bad_c = print_KFold_scores(X_train, y_train)
    show_Raw_result(bad_c, X_train, y_train, X_test, y_test)

    show_Sample_adjust_threshold(X_train_underSample, y_train_underSample, X_test_underSample, y_test_underSample)


    # best_c = print_KFold_scores(X_train_underSample, y_train_underSample)
