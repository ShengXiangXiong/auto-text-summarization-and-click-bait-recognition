#!/user/bin/python
#coding:utf-8
__author__ = 'ShengXiang.X'

import jieba
import numpy as np
import matplotlib.pyplot as plt
import gensim
from sklearn.decomposition import PCA

model = gensim.models.KeyedVectors.load_word2vec_format('Word60.model.bin', binary=True)
np.seterr(all='warn')

# 句子中的stopwords
def create_stopwords():
    stop_list = [line.strip() for line in open("D:\Python_Space\Automatic_abstracting\data\哈工大停用词表.txt", 'r', encoding='utf-8').readlines()]
    return stop_list

def filter_symbols(sents):
    stopwords = create_stopwords() + ['。', ' ', '.']
    _sents = []
    for sentence in sents:
        stop_word = []
        for word in sentence:
            if word in stopwords:
               stop_word.append(word)
        for word in stop_word:
            sentence.remove(word)
        if sentence:
            _sents.append(sentence)
    return _sents


def filter_model(sents):
    _sents = []
    for sentence in sents:
        filter_word = []
        for word in sentence:
            if word not in model:
                filter_word.append(word)
        for word in filter_word:
            sentence.remove(word)
        if sentence:
            _sents.append(sentence)
    return _sents


def get_vec(sents):
    #当句子长度为0时直接返回
    if len(sents) == 0:
        return 0.0
    sents_vec = []
    #循环将句子中的每个词向量加入到句子二维词向量上去
    for sentence in sents:
        vec = model[sentence[0]]
        for word in sentence[1:]:
            vec = vec+model[word]
        sents_vec.append(vec/len(vec))
    return sents_vec

if __name__ == '__main__':
    s=[]
    sents = []
    s.append('小明喜欢打篮球')
    s.append('大雄爱踢足球')
    s.append('秋水共长天一色')
    s.append('落霞与孤鹜齐飞')
    s.append('李白是唐代著名浪漫主义诗人')
    s.append('杜工部是唐代伟大的现实主义诗人')
    s.append('象为齿焚蚌缘珠剖')
    s.append('梅因寒茂荷以暑清')
    for sent in s:
        sents.append([word for word in jieba.cut(sent) if word])
    #过滤掉停用词，即一些无意义的词
    sents = filter_symbols(sents)
    #过滤掉不在模型中的词汇
    sents = filter_model(sents)
    #生成词向量
    sents_vec = get_vec(sents)
    #PCA降维
    X_reduced = PCA(n_components=2).fit_transform(sents_vec)
    #可视化展示
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis([-0.5, 1.6, -0.5, 0.5])

    for i in [0,1]:
        ax.text(X_reduced[i][0], X_reduced[i][1], s[i], color='r')
    for i in [2,3]:
        ax.text(X_reduced[i][0], X_reduced[i][1], s[i], color='k')
    for i in [4,5]:
        ax.text(X_reduced[i][0], X_reduced[i][1], s[i], color='c')
    for i in [6,7]:
        ax.text(X_reduced[i][0], X_reduced[i][1], s[i], color='b')

    plt.show()