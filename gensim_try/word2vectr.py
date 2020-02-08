from gensim.models import word2vec


import logging  
logging.basicConfig(format='%(asctime)s： %(levelname)s： %(message)s', level=logging.INFO)   #对打印进格式化
raw_sentences = ["the quick brown fox jumps over the lazy dogs", "yoyoyo you go home now to sleep"]
sentence = [s.split() for s in raw_sentences]   #切分例句
# print(sentence)
'''
min_count 在语料集中对基准词频的控制。如在较大语料集中，控制忽略一些只出现过一到两次的单词，就通过设置min_count(一般在0~100)
Size 神经网络层数设置，默认100
'''
model = word2vec.Word2Vec(sentence, min_count=1)    
print(model.similarity('dogs', 'you'))  #判断两个词的相似程度，靠近1则为相关w