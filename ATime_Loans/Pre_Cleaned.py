import pandas as pd
import os

path = 'data' + os.sep + 'loan_classifier.csv'
loans = pd.read_csv(path)

null_counts = loans.isnull().sum()  #看下每个列的缺失值的数量，去除缺失值多的那个特征
print(null_counts)
loans = loans.drop("pub_rec_bankruptcies", axis=1)

loans = loans.dropna(axis=0)    #去除有缺失值的样本（向下一次去除整条记录）

#由于sklearn只接受数值型的特征，所以对于Object要进行处理
print(loans.dtypes.value_counts())  #看下这些特征的值的类型

object_columns_df = loans.select_dtypes(include=['object']) #输出object的特征来列表
print(object_columns_df.iloc[0])

#对待可处理，可映射的object（先查看这些object的values情况）
cols = ['home_ownership', 'verification_status', 'emp_length', 'term', 'addr_state', 'purpose', 'title']
for c in cols:
    print(loans[c].value_counts())

#先处理verification_status, 对年份进行映射
mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}
loans = loans.replace(mapping_dict)

loans = loans.drop(["last_credit_pull_d",
                    "earliest_cr_line", "addr_state",
                    "title"], axis=1)   #去除这三个变化不规律的特征

loans['int_rate'] = loans['int_rate'].str.rstrip('%').astype('float')   #把百分比去除，强转为float
loans['revol_util'] = loans['revol_util'].str.rstrip('%').astype('float')

#使用get_dummies来编码str类型的数据
cat_columns = ["home_ownership", "verification_status", "emp_length", "purpose", "term"]
dummy_df = pd.get_dummies(loans[cat_columns])
loans = pd.concat([loans, dummy_df], axis=1)
loans = loans.drop(cat_columns, axis=1)
loans = loans.drop("pymnt_plan", axis=1)

loans.to_csv('data\cleaned_loans.csv')

