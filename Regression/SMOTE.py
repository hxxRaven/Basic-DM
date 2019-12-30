import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import os
from creditCard import print_KFold_scores,plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import time



def smote(features, labels):
    '''原始数据切分'''
    features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                                labels,
                                                                                test_size=0.2,
                                                                                random_state=0)
    overSample = SMOTE(random_state=0)      #实例化SMOTE对象
    os_features,os_labels=overSample.fit_sample(features_train,labels_train)    #使用SMOTE模型，创造新的数据集(正负样本数已经平衡# )
    '''把数据集重新dataframe化'''
    print(len(os_features))
    print(len(os_labels))
    os_features = pd.DataFrame(os_features)
    os_labels = pd.DataFrame(os_labels)
    return os_features, os_labels, features_test, labels_test

def show_overSample_info(os_features, os_labels, features_test, labels_test):
    t1 = time.time()
    best_c = print_KFold_scores(os_features,os_labels)
    t2 = time.time()


    lr = LogisticRegression(C = best_c, penalty = 'l1', solver='liblinear')
    lr.fit(os_features,os_labels.values.ravel())
    y_pred = lr.predict(features_test.values)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(labels_test,y_pred)
    np.set_printoptions(precision=2)
    print(len(os_labels[os_labels==1]),'\n')
    print(len(os_labels[os_labels==0]),'\n')
    print(t2 - t1,'\n')
    print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

    # Plot non-normalized confusion matrix
    class_names = [0,1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Confusion matrix')
    plt.show()

if __name__ == '__main__':
    path = 'data' + os.sep + 'creditcard.csv'
    credit_cards = pd.read_csv(path)

    columns = credit_cards.columns          #获得DataFarme的列索引

    # The labels are in the last column ('Class'). Simply remove it to obtain features columns
    features_columns = columns.delete(len(columns) - 1)         #删掉最后一个索引


    features = credit_cards[features_columns]
    labels = credit_cards['Class']
    os_features, os_labels, features_test, labels_test = smote(features, labels)
    show_overSample_info(os_features, os_labels, features_test, labels_test)
