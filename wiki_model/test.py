import codecs, sys, os

path1 = 'extracted' + os.sep + 'AA' + os.sep + 'wiki_00'
path2 = 'wiki_zh_cn'
f1 = codecs.open(path1, 'r', encoding='utf-8')
line = f1.readline()
print(line)

f2 = codecs.open(path2, 'r', encoding='utf-8')
line1 = f2.readline()
print(line1)