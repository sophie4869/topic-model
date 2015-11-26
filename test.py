#_*_coding: utf-8_*_
import time
import jieba
from jieba import Tokenizer as tk
import pandas as pd
import numpy as np
import sys
from collections import Counter,OrderedDict
import re
from pullword import pullword as pw
from sklearn.feature_extraction.text import TfidfVectorizer as tv, CountVectorizer as cv
from sklearn.cross_validation import train_test_split as tts
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from read import getText


#jiebaCounter()
def jiebaCount(max_features=5000,prefix="extraction-",begin=1, end=1):
    sf=open('chi_,.txt','r')
    d,txt=getText(prefix=prefix,begin=begin,end=end)
    print "Data loaded."
    res=[]
    count=[]
    stopwords = [x.strip().decode('utf-8') for x in sf.read().split(',')]
    sw=stopwords
    print len(txt)
    st=time.time() 
    for i in range(len(txt)):
        r=" ".join(jieba.cut(txt[i])).split(" ")
        r=[x.strip() for x in r]
        r=filter(None,r)
        r=[x for x in r if not x in stopwords]
        res.append(r)
        count.append(Counter(r))
        #print i
    print "Counting cost "+str(time.time()-st)+" seconds."
    #sw = [x.strip() for x in sf.read().split(',')]
    #sw=[]
    #[sw.append(x) for x in stopwords if x not in sw]
    #print len(sw)
    #stopwords=sw
    #of=open('stopwords1.txt','w')
    #of.write(','.join(stopwords))
    #of.close()
    #for line in f.readlines():
    #    line=re.sub(r'\s','',line)#eine=line.strip()
    #    res+=[x.strip() for x in line.split("/")]
    for x in count[0].most_common(20):
        print x[0],x[1]
    print len(count),len(count[0])
    return count


def tfidf(max_features=5000,prefix="extraction-",begin=1, end=26):
    # get stopwords
    sf = open('chi_stopwords.txt','r')
    stopwords = [x.strip() for x in sf.read().split(',')]
    vectorizer=tv(max_features=max_features)#tokenizer=tokenizer)
    d={}
    st=time.time()
    d,txt=getText(prefix=prefix,begin=begin,end=end)
    getdatatime=time.time()
    print getdatatime-st
    corpus={}
    for i in range(len(txt)):#d.items():
        #corpus.append(" ".join(jieba.cut(line.split(',')[0],cut_all=False)))
        corpus[i]=(' '.join(jieba.cut(txt[i],cut_all=False)))
    tfidf=vectorizer.fit_transform(corpus.values()).toarray()
    print tfidf.shape
    voc=vectorizer.get_feature_names()
    wordssum = tfidf.sum(axis=0)
    index=range(len(voc))
    index = [index for (y,x,index) in sorted(zip(wordssum,voc,index),reverse=True) if x.encode('utf-8') not in stopwords] 
    print time.time() - st
    voc_sorted = [voc[i] for i in index] 
    tfidfret = []
    print time.time()-getdatatime
    return tfidf,voc,txt

#print "pullword:"
#st=time.time()
#f=open('test.txt')
#d=pwCount(f.read())
#print time.time()-st
#for x in OrderedDict(sorted(d.items(), key=lambda t:t[1])).items()[-20:]:
#    print x[0],x[1]
