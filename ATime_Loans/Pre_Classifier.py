'''
贷款数据的建模分析
'''
import pandas as pd
import os

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

path = 'data' + os.sep + 'loans_2007.csv'

loans_2007 = pd.read_csv(path)
#数据预处理（设置丢弃阈值为总数一半，横向上，如果为空数据大于阈值则丢弃）
half_count = len(loans_2007) / 2
loans_2007.dropna(thresh=half_count, axis=1)
# loans_2007.drop(['desc', 'url'], axis=1)
loans_2007.to_csv('loan_2007.csv', index=False)

loans_2007.drop_duplicates()
print(loans_2007.head(5))
print(loans_2007.iloc[0])
print(loans_2007.shape[1])

#预处理（相近特征，保留一个就行）（无意义，意义不大）（已经确定的结果不应用来测试）（编码制式）
loans_2007 = loans_2007.drop(["id", "member_id", "funded_amnt", "funded_amnt_inv", "grade", "sub_grade", "emp_title", "issue_d"], axis=1)
loans_2007 = loans_2007.drop(["zip_code", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp"], axis=1)
loans_2007 = loans_2007.drop(["total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt"], axis=1)
print(loans_2007.iloc[0])
print(loans_2007.shape[1])
#第一步处理之后还剩下30多个特征


#特征列变化不大也可以清理掉
orig_columns = loans_2007.columns
drop_columns = []
for col in orig_columns:
    col_series = loans_2007[col].dropna().unique()
    if (len(col_series)) == 1:
        drop_columns.append(col)
loans_2007 = loans_2007.drop(drop_columns, axis=1)
print(drop_columns)


#使用Fully Paid和Charge Off来当做label
print(loans_2007['loan_status'].value_counts()) 
loans_2007 = loans_2007[(loans_2007['loan_status'] == 'Fully Paid') | (loans_2007['loan_status'] == 'Charged Off')]

status_replace = {'loan_status' : {'Fully Paid':1, 'Charged Off':0} }   #设置替换格式(get_dumplie函数也可以使用)
loans_2007 = loans_2007.replace(status_replace)
print(loans_2007.head(5))

loans_2007.to_csv('data\loan_classifier.csv', index=False)