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
import mxnet as mx
import json
sys.path.append("/home/tingyubi/20w/data/")

def tfidf(max_features=10000,path="/home/tingyubi/20w/data/",prefix="extraction-",begin=1, end=26):
    # get stopwords
    sf = open('chi_,.txt','r')
    stopwords = [x.strip() for x in sf.read().split(',')]
    # load data
    d={}
    st=time.time()
    d,txt=getText(prefix=prefix,begin=begin,end=end)
    getdatatime=time.time()
    print "Loading data cost "+str(getdatatime-st)+" seconds."
    # cut text
    corpus={}
    for i in range(len(txt)):#d.items():
        #corpus.append(" ".join(jieba.cut(line.split(',')[0],cut_all=False)))
        corpus[i]=(' '.join(jieba.cut(txt[i],cut_all=False)))
    # tfidf vectorizer
    vectorizer=tv(max_features=max_features)#tokenizer=tokenizer)
    tfidf=vectorizer.fit_transform(corpus.values()).toarray()
    print "Tfidf vectorizing cost "+str(time.time()-getdatatime)+" seconds."
    #print tfidf.shape
    voc=vectorizer.get_feature_names()
    # sorting vocabulary
    #wordssum = tfidf.sum(axis=0)
    #index=range(len(voc))
    #index = [index for (y,x,index) in sorted(zip(wordssum,voc,index),reverse=True) if x.encode('utf-8') not in stopwords] 
    #voc_sorted = [voc[i] for i in index] 
    # save to json file
    jsonfile = "tfidf_"+prefix+str(begin)+"_"+str(end)+".json"
    data={}
    data['vocabulary']=voc
    data['tfidf']=tfidf.tolist()
    with open(jsonfile,'w') as f:
        json.dump(data,f)
    f.close()
    return tfidf,voc,txt

def tfidf_iterator(batch_size,max_features=10000,path="/home/tingyubi/20w/data/",prefix="extraction-",begin=1,end=26):
    #tf,voc,txt = tfidf(max_features=max_features,path=path,prefix=prefix,begin=begin,end=end)
    jsonfile = "tfidf_"+prefix+str(begin)+"_"+str(end)+".json"
    with open(jsonfile,'r') as f:
        data = json.load(f)
    f.close()
    tf,voc = np.array(data['tfidf']), data['vocabulary']
    tf = tf / (np.max(tf,axis = 1)[:, None] + 1e-10)
    x_train,x_test=tts(tf,train_size=0.9,test_size=0.1)
    train_iter = mx.io.NDArrayIter(data=x_train,batch_size=batch_size,shuffle=True)
    test_iter = mx.io.NDArrayIter(data=x_test,batch_size=batch_size,shuffle=True)
    return train_iter,test_iter
    
#tr,te=tfidf_iterator(batch_size=1,max_features=10,end=1)
