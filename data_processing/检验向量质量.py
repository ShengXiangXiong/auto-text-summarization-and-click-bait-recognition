#!/user/bin/python
# coding:utf-8
__author__ = 'ShengXiang.X'

from gensim.models.deprecated.word2vec import Word2Vec

#文本文件必须是utf-8无bom格式
mod = Word2Vec.load('Word60.model')
# mod = gensim.models.KeyedVectors.load_word2vec_format('Word60.model', unicode_errors='ignore')	#3个文件放在一起：Word60.model   Word60.model.syn0.npy   Word60.model.syn1neg.npy
fout = open("字词相似度.txt", 'w')

showWord=['重庆',
			'教授',
			'吃',
			'李白',
		   '天安门',
			'',]

for word in showWord:
	if word in mod.index2word:
		sim = mod.most_similar(word,topn=5)
		fout.write(word +'\n')
		for ww in sim:
			fout.write('\t\t\t' + ww[0] + '\t\t'  + str(ww[1])+'\n')
		fout.write('\n')
	else:
		fout.write(word + '\t\t\t——不在词汇表里' + '\n\n')

fout.close()  
