'''求解 P（正确单词 | 输入单词）---> （ P（ 输入单词 | 正确单词）* P（ 正确单词）） / P( 输入单词占语料库概率 )

### 求解：argmaxc P(c|w) -> argmaxc P(w|c) P(c) / P(w) ###

* P(c), 文章中出现一个正确拼写词 c 的概率, 也就是说, 在英语文章中, c 出现的概率有多大
* P(w|c), 在用户想键入 c 的情况下敲成 w 的概率. 因为这个是代表用户会以多大的概率把 c 敲错成 w
* argmaxc, 用来枚举所有可能的 c 并且选取概率最大的    '''

import re, collections, os



def tolcs(text):         #把语料中的单词全部抽取出来, 转成小写, 并且去除单词中间的特殊符号
    return re.findall('[a-z]+', text.lower())

def train(features):    #防止出现语料库中不存在的词，导致p（c）为零，结果为0.将它概率初始化为1.然后再多次相加得到总出现次数
    model = collections.defaultdict(lambda : 1)     #初始化一个字典
    for f in features:          #features为list，包含所有单词
        model[f] += 1
    return model
'''编辑距离:
两个词之间的编辑距离定义为使用了几次插入(在词中插入一个单字母), 删除(删除一个单字母), 交换(交换相邻两个字母), 
替换(把一个字母换成另一个)的操作从一个词变到另一个词。

操作逻辑编辑距离为1的正确单词比编辑距离为2的优先级高, 而编辑距离为0的正确单词优先级比编辑距离为1的高.
'''
def distance1(word):
    n = len(word)
    print(word[0:0])
    print(word[1:])
    return set([word[0:i]+word[i+1:] for i in range(n)] +                     # deletion,实现从第一个开始删，删到最后一个
               [word[0:i]+word[i+1]+word[i]+word[i+2:] for i in range(n-1)] + # transposition，邻近字母调换顺序
               [word[0:i]+c+word[i+1:] for i in range(n) for c in alphabet] + # alteration，每个字母被字母表中字母替换
               [word[0:i]+c+word[i:] for i in range(n+1) for c in alphabet])    #insertion，插入字母表中的字母，从第一个前到最后一个后，两个for控制。

def distance2(word, NWORDS):
    return set(e2 for e1 in distance1(word) for e2 in distance1(e1))        #键距为2，先e1从键位为1生成，再将结果放回d1计算，得到艰巨为2

def better_distance2(word, NWORDS):
    return set(e2 for e1 in distance1(word) for e2 in distance1(e1) if e2 in NWORDS)    #仅选取正常单词

def know(words,NWORDS):
    return set(w for w in words if w in NWORDS)

def correct(word,NWORDS):          #按顺序判断返回候选词，在NWORDS出现最多的
    candidates = know([word],NWORDS) or know(distance1(word),NWORDS) or better_distance2(word,NWORDS) or [word]
    return max(candidates, key=lambda w: NWORDS[w])     #candidate中的两个字符串作为key，在NWORDS中按K查询次数，返回NWORDS中多的那个



if __name__ == '__main__':
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    path = 'data' + os.sep + 'big.txt'
    NWORDS = train(tolcs(open(path).read()))
    test = input('输入单词')
    print(test+'\n')
    print(correct(test,NWORDS))
