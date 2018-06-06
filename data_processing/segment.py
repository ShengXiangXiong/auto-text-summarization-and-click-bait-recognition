#!/user/bin/python
# coding:utf-8
__author__ = 'ShengXiang.X'
import jieba
import codecs
import opencc
from string import punctuation


#过滤英文
def isAlpha(word):
    try:
        return word.encode('ascii').isalpha()
    except UnicodeEncodeError:
        return False

#opencc繁体转简体，jieba中文分词
def trans_seg(input,output):
    # 加载停用词表
    stoppath='D:\Python_Space\Automatic_abstracting\data\哈工大停用词表.txt'
    stoplist = [line.strip() for line in codecs.open(stoppath, 'r', encoding='utf-8').readlines()]
    stopwords = {}.fromkeys(stoplist)
    # cc=opencc.OpenCC('t2s')
    i=0
    seg_text=[]
    with codecs.open(output,'w','utf-8') as wopen:
        print('开始...')
        with codecs.open(input,'r','utf-8') as ropen:
            while True:
                line=ropen.readline().strip()
                # i+=1
                # print('line '+str(i))
                text=''
                for cha in line.split():
                    if isAlpha(cha):
                        continue
                    # cha=cc.convert(cha)
                    text+=cha
                words=jieba.cut(text)
                seg=''
                for word in words:
                    if word not in stopwords:
                        if len(word)>1 and isAlpha(word)==False: #去掉长度小于1的词和英文
                            if word !='\t':
                                seg+=word+' '
                seg_text.append(seg)
        ls=len(seg_text)
        while(ls ):
            wopen.write(seg+'\n')
        print('结束!')
        return seg_text