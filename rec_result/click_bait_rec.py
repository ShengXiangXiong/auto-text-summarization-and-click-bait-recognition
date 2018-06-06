#!/user/bin/python
#coding:utf-8
__author__ = 'ShengXiang.X'

import jieba
import numpy as np
import matplotlib.pyplot as plt
import gensim
from sklearn.decomposition import PCA
import text_summary

model = gensim.models.KeyedVectors.load_word2vec_format('D:\Python_Space\Automatic_abstracting\data\Word60.model.bin', binary=True)

def create_stopwords():
    stop_list = [line.strip() for line in open("D:\Python_Space\Automatic_abstracting\data\哈工大停用词表.txt", 'r', encoding='utf-8').readlines()]
    return stop_list

def get_vec_1(sents):
    if len(sents) == 0:
        return 0.0
    vec = model[sents[0]]
    for word in sents[1:]:
        vec = vec + model[word]
    return vec

def get_vec_n(sents):
    # 当句子长度为0时直接返回
    if len(sents) == 0:
        return 0.0
    # # 当句子长度为1时返回一维向量
    # if len(sents) == 1:
    #     vec = model[sents[0]]
    #     for word in sents[1:]:
    #         vec = vec + model[word]
    #     return vec
    # 循环将句子中的每个词向量加入到句子二维词向量上去
    sents_vec = []
    for sentence in sents:
        vec = model[sentence[0]]
        for word in sentence[1:]:
            vec = vec+model[word]
        sents_vec.append(vec/len(vec))
    return sents_vec


def filter_model_1(sents):
    filter_word = []
    for word in sents:
        if word not in model:
            filter_word.append(word)
    for word in filter_word:
        sents.remove(word)
    return sents

def filter_model_n(sents):
    # if len(sents) ==1:
    #     filter_word = []
    #     for word in sents:
    #         if word not in model:
    #             filter_word.append(word)
    #     for word in filter_word:
    #         sents.remove(word)
    #     return sents
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


def click_bait_recognition(title, text):
    # 生成摘要
    summary_text = text_summary.summarize(text)
    # 分词
    stopwords = create_stopwords()
    ts = [word for word in jieba.cut(title) if word not in stopwords]
    cs = []
    for sent in summary_text:
        cs.append([word for word in jieba.cut(sent) if word not in stopwords])

    # 过滤掉不在模型中的词汇
    ts = filter_model_1(ts)
    cs = filter_model_n(cs)
    # 生成词向量
    t_vec = get_vec_1(ts)
    s_vec = get_vec_n(cs)
    # 标题党识别
    max_sim = 0  # 设置初始最大相似度值
    for vec in s_vec:
        max_sim = max(max_sim, text_summary.cosine_similarity(t_vec, vec))
    return max_sim






if __name__ == '__main__':
    title = '我们需要恢复繁体字的使用吗？'
    text = ''
    with open("news3.txt", "r", encoding='utf-8') as myfile:
        text = myfile.read()
        text = text.replace('\n', '')
    # 生成摘要
    summary_text = text_summary.summarize(text)
    # 分词
    stopwords = create_stopwords()
    ts = [word for word in jieba.cut(title) if word not in stopwords]
    cs = []
    for sent in summary_text:
        cs.append([word for word in jieba.cut(sent) if word not in stopwords])

    #过滤掉不在模型中的词汇
    ts = filter_model_1(ts)
    cs = filter_model_n(cs)
    # 生成词向量
    t_vec = get_vec_1(ts)
    s_vec = get_vec_n(cs)
    # 标题党识别
    max_sim = 0  # 设置初始最大相似度值
    for vec in s_vec:
        max_sim = max(max_sim,text_summary.cosine_similarity(t_vec,vec))
    print('simiilarity:'+str(max_sim))