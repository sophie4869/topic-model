#_*_coding: utf-8_*_
import time
import jieba
from jieba import Tokenizer as tk
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from collections import Counter,OrderedDict
import re
from pullword import pullword as pw
from sklearn.feature_extraction.text import TfidfVectorizer as tv, CountVectorizer as cv
from sklearn.cross_validation import train_test_split as tts
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from read import getText


# count bag of words cut by jieba 
# return array of features, vocabulary and original text
def jiebaCounter(max_features=5000,prefix="extraction-",begin=1, end=1,dictionary=""):
    # get stopwords
    sf = open('chi_,.txt','r')
    stopwords = [x.strip().decode('utf-8') for x in sf.read().split(',')]
    if dictionary=="":
        vectorizer=cv(max_features=max_features,stop_words=stopwords)#tokenizer=tokenizer)
    else:
        vocabulary=open(dictionary,'r').read().split("\n")
        vectorizer=cv(vocabulary=vocabulary,max_features=max_features,stop_words=stopwords)#tokenizer=tokenizer)
    d={}
    st=time.time()
    d,txt=getText(prefix=prefix,begin=begin,end=end)
    getdatatime=time.time()
    print getdatatime-st
    corpus={}
    for i in range(len(txt)):#d.items():
        #corpus.append(" ".join(jieba.cut(line.split(',')[0],cut_all=False)))
        corpus[i]=(' '.join(jieba.cut(txt[i],cut_all=False)))
    vect=vectorizer.fit_transform(corpus.values()).toarray()
    print vect.shape
    voc=vectorizer.get_feature_names()
    wordssum = vect.sum(axis=0)
    index=range(len(voc))
    index = [index for (y,x,index) in sorted(zip(wordssum,voc,index),reverse=True) if x not in stopwords]
    print time.time() - st
    voc_sorted = [voc[i] for i in index]
    print time.time()-getdatatime
    return vect,voc,txt




# segment document to less than 1000 words for pullword input
def segdoc(doc):
    s=doc
    endStr="。|！|？"
    p=re.compile(endStr)
    starts=[match.start() for match in re.finditer(p,s)]
    print starts
    starts = np.array(starts)
    outstr=[]
    for i in range(len(s)/3000+1):
        e=starts[np.where(starts<(i+1)*3000)[0][-1]] # index of last ./!/?
        outstr.append(s[:e+3])
        s=s[e+3:]
    return outstr

# return Ordered Dict
def pwCount(doc):
    #doc = sys.argv[1]
    seg=segdoc(doc)
    res=[]
    stt=time.time()
    for x in seg:
        res+=(pw(x.decode('utf-8'),threshold=0.7))
    print time.time()-stt
    #print res
    d=OrderedDict()
    for x in res:
        d[x[0]]=float(x[1])+ (d[x[0]] if x[0] in d else 0)
    #for k,v in d.items():
    #    print k,v
    #print len(d)
    return d

