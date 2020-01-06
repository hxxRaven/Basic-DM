import pandas as pd
import jieba
import os
import numpy as np

path = 'data' + os.sep + 'val.txt'

df_news = pd.read_table(path ,names=['category','theme','URL','content'], encoding='utf-8').astype(str)
df_news = df_news.dropna()      #包含空值的样本直接drop（可以看出广告很多）
# print(df_news.head())

'''使用分词器'''
content = df_news.content.values.tolist()  #内容分类转为list才能分词
category = df_news.category.values.tolist()
URL = df_news.URL.values.tolist()
theme = df_news.theme.values.tolist()

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

df_content_clean = pd.DataFrame({'contents_clean' : contents_clean})
df_all_words = pd.DataFrame({'all_words' : all_words})
print(df_content_clean.head())

'''统计语料库（词与出现次数）'''

word_counts = df_all_words.groupby(by=['all_words'])['all_words'].agg(np.size)  #已经被拆分成Series了
word_counts = word_counts.to_frame()    #转回frame
word_counts.columns = ['count']         #列名
word_counts = word_counts.reset_index().sort_values(by=['count'],ascending=False)


# '''wordCloud展示'''
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import matplotlib
# font_path = 'data' + os.sep + 'simhei.tff'
# matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
#
# wordcloud=WordCloud(font_path, background_color="white",max_font_size=80)
# word_frequence = {x[0]:x[1] for x in word_counts.head(100).values}  #为词设置频数
# wordcloud=wordcloud.fit_words(word_frequence)
# plt.imshow(wordcloud)

'''TF-idf确定关键词'''
import jieba.analyse
index = 2400
print (df_news['content'][index])
content_afterCut_str = "".join(content_afterCut[index])
'''jieba.analyse.extract_tags(content_afterCut_str, topK=5, withWeight=False) 文本越长结果越好1'''
print ("  ".join(jieba.analyse.extract_tags(content_afterCut_str, topK=5, withWeight=False)))

'''LDA主题模式 格式要求：list of list [文章[文章内的词]]'''
from gensim import corpora, models, similarities
import gensim                                   #gensim文处理库包含word2vec等重要工具

dictionary = corpora.Dictionary(contents_clean)         #做映射表，做个字典（今天映射为1，明天映射为2等等）
corpus = [dictionary.doc2bow(sentence) for sentence in contents_clean]  #构建文集，用数字建立字典对应的文集

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20) #训练模型：确定映射，文集，主题数

# 循环打印，前5个关键词和系数。用tpoic控制循环，第一项为欲求结果

for topic in lda.print_topics(num_topics=20, num_words=5):
    print (topic[1])