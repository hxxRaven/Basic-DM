import jieba
import jieba.analyse
import jieba.posseg as pseg
import codecs,sys
def cut_words(sentence):
    #print sentence
    print(" ".join(jieba.cut(sentence)).encode('utf-8'))
    return " ".join(jieba.cut(sentence)).encode('utf-8')
f=codecs.open('wiki_zh_cn','r',encoding="utf8")
target = codecs.open("zh.jian.wiki.seg.106.txt", 'w', encoding="utf8")
print ('open files')
line_num=1
line = f.readline()
while line:
    print('---- processing ', line_num, ' article----------------')
    line_seg = " ".join(jieba.cut(line))
    print(line_seg)
    target.writelines(line_seg)
    line_num = line_num + 1
    print(line)
    line = f.readline()
f.close()
target.close()
exit()
while line:
    curr = []
    for oneline in line:
        #print(oneline)
        curr.append(oneline)
    after_cut = map(cut_words, curr)
    target.writelines(after_cut)
    print ('saved',line_num,'articles')
    exit()
    line = f.readline1()
f.close()
target.close()

# python SegByJieba.py