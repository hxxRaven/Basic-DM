import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

'''sklearndd的dataset中包含很多数据集，这里用加州房价数据实验'''
from sklearn.datasets.california_housing import fetch_california_housing
housing = fetch_california_housing()
print(housing.DESCR)
print(housing.data.shape)

from sklearn import tree
dtr = tree.DecisionTreeRegressor(max_depth=2)       #实例化树模型，树的最大深度制定下

'''
树模型参数:
1.criterion gini or entropy 评价标准，用熵还是GINI系数

2.splitter best or random 前者是在所有特征中找最好的切分点 后者是在部分特征中（数据量大的时候） 默认best

3.max_features None（所有），log2，sqrt，N 特征小于50的时候一般使用所有的

4.max_depth 数据少或者特征少的时候可以不管这个值，如果模型样本量多，特征也多的情况下，可以尝试限制下

5.min_samples_split 如果某节点的样本数少于min_samples_split，则不会继续再尝试选择最优特征来进行划分如果样本量不大，不需要管这个值。如果样本量数量级非常大，则推荐增大这个值。

6.min_samples_leaf 这个值限制了叶子节点最少的样本数，如果某叶子节点数目小于样本数，则会和兄弟节点一起被剪枝，如果样本量不大，不需要管这个值，大些如10W可是尝试下5

7.min_weight_fraction_leaf 这个值限制了叶子节点所有样本权重和的最小值，如果小于这个值，则会和兄弟节点一起被剪枝默认是0，就是不考虑权重问题。一般来说，如果我们有较多样本有缺失值，或者分类树样本的分布类别偏差很大，就会引入样本权重，这时我们就要注意这个值了。

8.max_leaf_nodes 通过限制最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。如果加了限制，算法会建立在最大叶子节点数内最优的决策树。如果特征不多，可以不考虑这个值，但是如果特征分成多的话，可以加以限制具体的值可以通过交叉验证得到。

9.class_weight 指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多导致训练的决策树过于偏向这些类别。这里可以自己指定各个样本的权重如果使用“balanced”，则算法会自己计算权重，样本量少的类别所对应的样本权重会高。

10.min_impurity_split 这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益，均方差，绝对差)小于这个阈值则该节点不再生成子节点。即为叶子节点 。

n_estimators:要建立树的个数
'''

Info = dtr.fit(housing.data[ : , [6, 7]], housing.target)  #指定数据特征和结果，这里数据指定不是切片而是list
print(Info)
# print(housing.feature_names)
# print(housing.data)
'''树的可视化需要先安装，graphviz'''

'''构建图数据对象'''
dot_data = tree.export_graphviz(dtr,
                                out_file = None,
                                feature_names= housing.feature_names[6:8],          #切记这里取切片
                                filled = True,
                                impurity = False,
                                rounded = True
)
'''导入pydoyplay包，转换dot_data数据，也可进行图数据保存'''
import pydotplus
graph = pydotplus.graph_from_dot_data(dot_data)     #把dot_data类型转为graph类型
graph.get_nodes()[7].set_fillcolor("#FFF2DD")       #获取7个节点，颜色风格为设置色

'''导入Iamge，主要为了jupyter显示'''
from IPython.display import Image
Image(graph.create_png())       #jupyter显示

graph.write_png('test.png')     #png显示

'''切分测试集和训练集'''
from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(
                                        housing.data, housing.target,
                                        test_size=0.1, random_state=42) #random_state指定每次随机结果一致
dtr1 = tree.DecisionTreeRegressor(random_state=42)

dtr1.fit(data_train, target_train)
dtr_score = dtr1.score(data_test, target_test)      #获取评价分数
print(dtr_score)

'''参数设置'''
# from sklearn.model_selection import GridSearchCV        #设计参数组合，寻找最合适的参数
# tree_param_grid = {'min_sample_split' : list((3, 6, 9)), 'n_estimators' : list((10, 50, 100))}  #构建字典参数对应范围
# grid = GridSearchCV(RandomForestRegressor(), param_grid=tree_param_grid, cv=5)      #选取评估算法，参数范围，cv进行几次交叉验证
# grid.fit(data_train, target_train)
# print(grid.scorer_)
from sklearn.model_selection import GridSearchCV
tree_param_grid = { 'min_samples_split': list((3,6,9)),'n_estimators':list((10,50,100))}
grid = GridSearchCV(RandomForestRegressor(),param_grid=tree_param_grid, cv=5)
grid.fit(data_train, target_train)
print(grid.cv_results_['mean_test_score'], grid.cv_results_['params'], grid.best_params_, grid.best_score_)

'''手动测试最优参数'''
rfr = RandomForestRegressor(min_samples_split=3,n_estimators = 100,random_state = 42)
rfr.fit(data_train, target_train)
rfr.score(data_test, target_test)
print(rfr.feature_importances_)

rfr_results = pd.Series(rfr.feature_importances_, index = housing.feature_names).sort_values(ascending = False)


