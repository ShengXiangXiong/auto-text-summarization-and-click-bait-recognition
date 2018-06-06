#!/user/bin/python
#coding:utf-8
__author__ = 'ShengXiang.X'

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gensim
from sklearn.decomposition import PCA
from gensim.models.deprecated.word2vec import Word2Vec
from mpl_toolkits.mplot3d import Axes3D

# load the word2vec model
# model = Word2Vec.load('Word60.model.bin')
model = gensim.models.KeyedVectors.load_word2vec_format('Word60.model.bin', binary=True)

# model = gensim.models.KeyedVectors.load_word2vec_format(str, binary=True)
# str = 'D:\Python_Space\Automatic_abstracting\data\Word60.model.bin'

vec = model.vectors

# reduce the dimension of word vector
X_reduced = PCA(n_components=2).fit_transform(vec)

# show some word(center word) and it's similar words
index01 = model.similar_by_word(u'重庆')
index1=[np.where(vec == model[u'重庆'])[0][0]]
index02 = model.similar_by_word(u'清华大学')
index2=[np.where(vec == model[u'清华大学'])[0][0]]
index03 = model.similar_by_word(u'爱因斯坦')
index3=[np.where(vec == model[u'爱因斯坦'])[0][0]]
index05 = model.similar_by_word(u'唐僧')
index5=[np.where(vec == model[u'唐僧'])[0][0]]

for ch1 in index01:
    index1.append(np.where(vec == model[ch1[0]])[0][0])

for ch2 in index02:
    index2.append(np.where(vec == model[ch2[0]])[0][0])

for ch3 in index03:
    index3.append(np.where(vec == model[ch3[0]])[0][0])

for ch5 in index05:
    index5.append(np.where(vec == model[ch5[0]])[0][0])

# plot the result
#zhfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis([-7.5,7.5,-10,7.5])

for i in index1:
    ax.text(X_reduced[i][0], X_reduced[i][1], model.index2word[i], color='r')

for i in index2:
    ax.text(X_reduced[i][0],X_reduced[i][1], model.index2word[i], color='b')

for i in index3:
    ax.text(X_reduced[i][0],X_reduced[i][1], model.index2word[i], color='g')

for i in index5:
    ax.text(X_reduced[i][0],X_reduced[i][1], model.index2word[i], color='c')

plt.show()


