'''
tsfresh
自动时间序列提取，机遇统计特性提取特征
'''


import matplotlib.pylab as plt
import seaborn as sns
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


#http://tsfresh.readthedocs.io/en/latest/text/quick_start.html

download_robot_execution_failures()
df, y = load_robot_execution_failures()
print(df.head())

#展示正常的id3数据和不正常的id20数据
df[df.id == 3][['time','F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z']].plot(x='time',title='Success example (id 3)', figsize=(12, 6))
df[df.id == 20][['time', 'F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z']].plot(x='time', title='Failure example (id 20)', figsize=(12, 6))


#提取特征,以id为分类条件，以time为排序标准
extraction_settings = ComprehensiveFCParameters()
X = extract_features(df,
                     column_id='id', column_sort='time',
                     default_fc_parameters=extraction_settings,
                     impute_function= impute)   #
print(X.head(5))
print(X.info())

#设置条件。过滤特征
X_filtered = extract_relevant_features(df, y,
                                       column_id='id', column_sort='time',
                                       default_fc_parameters=extraction_settings)
print(X_filtered.info())

#切分训练
X_train, X_test, X_filtered_train, X_filtered_test, y_train, y_test = train_test_split(X, X_filtered, y, test_size=0.4)

cl = DecisionTreeClassifier()
cl.fit(X_train, y_train)
print(classification_report(y_test, cl.predict(X_test)))
print(cl.n_features_)

cl2 = DecisionTreeClassifier()
cl2.fit(X_filtered_train, y_train)
print(classification_report(y_test, cl2.predict(X_filtered_test)))
print(cl2.n_features_)

