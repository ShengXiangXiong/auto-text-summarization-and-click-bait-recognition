#!/user/bin/python
#coding:utf-8
__author__ = 'ShengXiang.X'

import jieba
import math
from heapq import nlargest
from itertools import product, count
from gensim.models.deprecated.word2vec import Word2Vec
import gensim
import numpy as np

model = gensim.models.KeyedVectors.load_word2vec_format('D:\Python_Space\Automatic_abstracting\data\Word60.model.bin', binary=True)
# model = Word2Vec.load("Word60.model")
np.seterr(all='warn')


def cut_sentences(sentence):
    puns = frozenset(u'。！？')
    tmp = []
    for ch in sentence:
        tmp.append(ch)
        if puns.__contains__(ch):
            yield ''.join(tmp)
            tmp = []
    yield ''.join(tmp)


# 句子中的stopwords
def create_stopwords():
    stop_list = [line.strip() for line in open("D:\Python_Space\Automatic_abstracting\data\哈工大停用词表.txt", 'r', encoding='utf-8').readlines()]
    return stop_list


def two_sentences_similarity(sents_1, sents_2):
    '''
    计算两个句子的相似性
    :param sents_1:
    :param sents_2:
    :return:
    '''
    counter = 0
    for sent in sents_1:
        if sent in sents_2:
            counter += 1
    return counter / (math.log(len(sents_1) + len(sents_2)))


def create_graph(word_sent):
    """
    传入句子链表  返回句子之间相似度的图
    :param word_sent:
    :return:
    """
    num = len(word_sent)
    board = [[0.0 for _ in range(num)] for _ in range(num)]

    for i, j in product(range(num), repeat=2):
        if i != j:
            board[i][j] = compute_similarity_by_avg(word_sent[i], word_sent[j])
    return board


def cosine_similarity(vec1, vec2):
    '''
    计算两个向量之间的余弦相似度
    :param vec1:
    :param vec2:
    :return:
    '''
    tx = np.array(vec1)
    ty = np.array(vec2)
    cos1 = np.sum(tx * ty)
    cos21 = np.sqrt(sum(tx ** 2))
    cos22 = np.sqrt(sum(ty ** 2))
    cosine_value = cos1 / float(cos21 * cos22)
    return cosine_value


def compute_similarity_by_avg(sents_1, sents_2):
    '''
    对两个句子求平均词向量
    :param sents_1:
    :param sents_2:
    :return:
    '''
    if len(sents_1) == 0 or len(sents_2) == 0:
        return 0.0
    vec1 = model[sents_1[0]]
    for word1 in sents_1[1:]:
        vec1 = vec1 + model[word1]

    vec2 = model[sents_2[0]]
    for word2 in sents_2[1:]:
        vec2 = vec2 + model[word2]

    similarity = cosine_similarity(vec1 / len(sents_1), vec2 / len(sents_2))
    return similarity


def calculate_score(weight_graph, scores, i):
    """
    计算句子在图中的分数
    :param weight_graph:
    :param scores:
    :param i:
    :return:
    """
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0

    for j in range(length):
        fraction = 0.0
        denominator = 0.0
        # 计算分子
        fraction = weight_graph[j][i] * scores[j]
        # 计算分母
        for k in range(length):
            denominator += weight_graph[j][k]
            if denominator == 0:
                denominator = 1
        added_score += fraction / denominator
    # 算出最终的分数
    weighted_score = (1 - d) + d * added_score
    return weighted_score


def weight_sentences_rank(weight_graph):
    '''
    输入相似度的图（矩阵)
    返回各个句子的分数
    :param weight_graph:
    :return:
    '''
    # 初始分数设置为0.5
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]

    # 开始迭代
    while different(scores, old_scores):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        for i in range(len(weight_graph)):
            scores[i] = calculate_score(weight_graph, scores, i)
    return scores


def different(scores, old_scores):
    '''
    判断前后分数有无变化
    :param scores:
    :param old_scores:
    :return:
    '''
    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= 0.0001:
            flag = True
            break
    return flag


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


def summarize(text):
    tokens = cut_sentences(text)
    sentences = []
    sents = []
    for sent in tokens:
        sentences.append(sent)
        sents.append([word for word in jieba.cut(sent) if word])
    # 过滤掉停用词，即一些无意义的词
    sents = filter_symbols(sents)
    # 过滤掉不在模型中的词汇
    sents = filter_model(sents)
    # 生成摘要句的个数n
    n = math.ceil(len(sents)*0.1)
    # 构建文本图模型
    graph = create_graph(sents)

    scores = weight_sentences_rank(graph)
    sent_selected = nlargest(n, zip(scores, count()))
    sent_index = []
    for i in range(n):
        sent_index.append(sent_selected[i][1])
    return [sentences[i] for i in sent_index]


if __name__ == '__main__':
    with open("news.txt", "r", encoding='utf-8') as myfile:
        text = myfile.read()
        print(text)
        print('----------------------------------------提取摘要后：---------------------------------------')
        text = text.replace('\n', '')
        rs=summarize(text, 3)
        for tt in rs:
            print(str(tt))
    # s=[]
    # s.append('小明喜欢打篮球')
    # s.append('秋水共长天一色')
    # s.append('大雄爱踢足球')
    # s.append('落霞与孤鹜齐飞')
    # sents=[]
    # for sent in s:
    #     sents.append([word for word in jieba.cut(sent) if word])
    #
    # for i in range(len(s)):
    #     for j in range(i+1,len(s)):
    #             sim = compute_similarity_by_avg(sents[i], sents[j])
    #             print(s[i])
    #             print(s[j])
    #             print('句子相似程度为：'+str(sim))
    #             print('-----------------')