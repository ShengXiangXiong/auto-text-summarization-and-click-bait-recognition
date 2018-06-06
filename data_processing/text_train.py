#!/user/bin/python
# coding:utf-8
__author__ = 'ShengXiang.X'
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing
# from Automatic_abstracting.text_processing import segment
# import segment  //import同级目录报错
# pycharm不会将当前文件目录自动加入自己的sourse_path。
# 右键make_directory as-->Sources Root将当前工作的文件夹加入source_path就可以了。
from text_processing.segment import trans_seg

#利用gensim中的word2vec训练词向量
def word2vec(input,output):
    print('Start...')
    rawdata=input
    modelpath=output
    #vectorpath='E:\word2vec\vector'
    model=Word2Vec(LineSentence(rawdata),size=400,window=5,min_count=5,workers=multiprocessing.cpu_count())#参数说明，gensim函数库的Word2Vec的参数说明
    model.save(modelpath)
    #model.wv.save_word2vec_format(vectorpath,binary=False)
    print("Finished!")

#检验词向量训练效果
def wordsimilarity():
    model=Word2Vec.load(r'F:\Python_Space\Automatic_abstracting\raw_files\modeldata.model')
    semi=''
    try:
        semi=model.most_similar('重庆', topn=10)
    except KeyError:
        print('The word not in vocabulary!')

    #print(model[u'重庆'])#打印词向量
    for term in semi:
        print('%s,%s' %(term[0],term[1]))

if __name__=='__main__':
    # inp='zhwiki-latest-pages-articles.xml.bz2'
    # dataprocess(inp)
    inp='D:\Python_Space\Automatic_abstracting\data\wiki.zh.txt'
    output='D:\Python_Space\Automatic_abstracting\data\zhwiki_wordVec.model'
    # trans_seg(inp,output)
    # input = output
    # output = r'F:\Python_Space\Automatic_abstracting\data\modeldata.model'
    word2vec(inp,output)
    wordsimilarity()

