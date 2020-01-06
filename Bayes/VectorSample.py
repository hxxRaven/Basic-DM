'''第一种方法'''
from sklearn.feature_extraction.text import CountVectorizer
texts=["dog cat fish","dog cat cat","fish bird", 'bird']    #分词化之后的list() ,分开文章 空格分开词
cv = CountVectorizer()
cv_fit=cv.fit_transform(texts)

print(cv.get_feature_names())               #取出特征名和序号
print(cv_fit.toarray())                     #将我输入list向量化
print(cv_fit.toarray().sum(axis=0))         #向量求和

'''第二种方法，ngram_range可以让词与词按数量组合，生成新的特征（一般2维就够用）'''
from sklearn.feature_extraction.text import CountVectorizer
texts=["dog cat fish","dog cat cat","fish bird", 'bird']
cv = CountVectorizer(ngram_range=(1,4))
cv_fit=cv.fit_transform(texts)

print(cv.get_feature_names())
print(cv_fit.toarray())


print(cv_fit.toarray().sum(axis=0))