#!/user/bin/python
#coding:utf-8
__author__ = 'ShengXiang.X'
import re
import codecs
import click_bait_rec

str = r'D:\Python_Space\Automatic_abstracting\data\news_tensite_xml.smarty.dat'
# file = codecs.open(str, 'w','utf-8')
# 设置数据解析格式
url_pat = '<url>(.*?)</url>'
title_pat = '<contenttitle>(.*?)</contenttitle>'
con_pat = '<content>(.*?)</content>'

with codecs.open(str, "r", encoding='utf-8') as myfile:
    text = myfile.read()
    url = re.findall(url_pat,text)
    title = re.findall(title_pat,text)
    context = re.findall(con_pat,text)

p = 0.35
str1 = '非标题党'
str2 = '标题党'
s = len(title)
# with codecs.open(r'D:\Python_Space\Automatic_abstracting\data\result.txt', "w", encoding='utf-8') as newsfile:
for i in range(s):
    wstr=''
    if len(context[i])!=0:
        sim = click_bait_rec.click_bait_recognition(title[i], context[i])
        if sim<p:
            str3 = str2
        else:
            str3 = str1
        # wstr = title[i]+'   '+title[i]+'    '+sim+'    '+str3
        # newsfile.write(wstr)
        print(title[i],'    ',url[i],'  ',sim,' ',str3)