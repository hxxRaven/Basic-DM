import pandas as pd
import jieba
import os
import numpy as np

path = 'data' + os.sep + 'val.txt'
df_news = pd.read_table(path ,names=['category','theme','URL','content'], encoding='utf-8').astype(str)
df_news = df_news.dropna()

content = df_news.content.values.tolist()  #内容分类转为list才能分词
category = df_news.category.values.tolist()

content_afterCut = []
for line in content:
    sent = ''.join(line)    #list转str，才能进行分词
    current_segment = jieba.lcut(sent)
    if len(current_segment) > 1 and current_segment != '\n\r':
        content_afterCut.append(current_segment)

# print(content_afterCut[1000])
df_content = pd.DataFrame({'Content_After_Cut' : content_afterCut})     #再将每条新闻的分词处理后结果归纳为datafram
# print(df_content.head())

'''使用停用词表清洗'''
stopword_path = 'data' + os.sep + 'stopwords.txt'

'''不指定索引，指定分隔为指表，引号常量为3， 列名为stopwords '''
stopwords = pd.read_csv(stopword_path, index_col=False,
                        sep='\t', quoting=3 ,names=['stopword'], encoding='utf-8')

def drop_stopwords(contents, stopwords):    #content_clean获取清理的content。all_words获取语料库所有有效词。
    content_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        content_clean.append(line_clean)

    return content_clean, all_words

contents = df_content.Content_After_Cut.values.tolist()
stopwords = stopwords.stopword.values.tolist()

contents_clean,all_words = drop_stopwords(contents, stopwords)

df_train=pd.DataFrame({'contents_clean':contents_clean,'label':df_news['category']})
# print(df_train.tail())

print(df_train.label.unique())

label_mapping = {'汽车' : 1, '财经' : 2, '科技' : 3, '健康' : 4,
                 '体育' : 4, '教育' : 6, '文化' : 7, '军事' : 8, '娱乐' : 9, '时尚' : 10}
df_train['label'] = df_train['label'].map(label_mapping)       #映射规则替换

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values,
                                                    df_train['label'].values, random_state=1)

print(x_train)

'''将list of list转为list中包含字符串的模式，以进行向量化'''
words = []
for line_index in range(len(x_train)):
    try:
        words.append(' '.join(x_train[line_index]))
    except:
        print(line_index)
print(words[0])

'''贝叶斯模型构造'''
'''1.向量化——CountVectorizer'''
from sklearn.feature_extraction.text import CountVectorizer

#实例化模型，确定分析对象，最大特征，训练处一个词映射数字的规则
vector = CountVectorizer(analyzer=words, max_features=4000, lowercase=False)
vector.fit(words)

'''贝叶斯！！！ 训练模型'''
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vector.transform(words), y_train)    #运用上述规则转换words，y_train

'''保证训练集和预测集处理步骤一样'''
test_words = []
for line_index in range(len(x_test)):
    try:
        #x_train[line_index][word_index] = str(x_train[line_index][word_index])
        test_words.append(' '.join(x_test[line_index]))
    except:
         print (line_index)

print(classifier.score(vector.transform(test_words), x_test)) #精度

'''2.向量化——TfidfVectorizer'''
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer='word', max_features=4000,  lowercase = False)
vectorizer.fit(words)

classifier = MultinomialNB()
classifier.fit(vectorizer.transform(words), y_train)

classifier.score(vectorizer.transform(test_words), y_test)

