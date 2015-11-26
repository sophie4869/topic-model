#_*_coding: utf-8_*_
import argparse
import time
import jieba
import sys
import re
from sklearn.feature_extraction.text import TfidfVectorizer as tv
from read import getText

#def tfidf(max_features=5000,prefix="extraction-",begin=1, end=26):
parser = argparse.ArgumentParser()
parser.add_argument("max_features",type=int,help="number of max features")
parser.add_argument("prefix",type=str,help="prefix of json files")
parser.add_argument("begin",type=int,help="begin code of json files")
parser.add_argument("end",type=int,help="end code of json files")
parser.add_argument("outputfile",type=str,help="output vocabulary file path")
args=parser.parse_args()

# get stopwords
sf = open('chi_,.txt','r')
stopwords = [x.strip().decode('utf-8') for x in sf.read().split(',')]

# load data
d={}
st=time.time()
d,txt=getText(prefix=args.prefix,begin=args.begin,end=args.end)
getdatatime=time.time()
print "Loading data cost "+ str(getdatatime-st)+" seconds."

# cut words
corpus={}
for i in range(len(txt)):#d.items():
    #corpus.append(" ".join(jieba.cut(line.split(',')[0],cut_all=False)))
    corpus[i]=(' '.join(jieba.cut(txt[i],cut_all=False)))

# tfidf
vectorizer=tv(max_features=args.max_features,stop_words=stopwords)#tokenizer=tokenizer)
tfidf=vectorizer.fit_transform(corpus.values()).toarray()
print tfidf.shape
voc=vectorizer.get_feature_names()
print "Tfidf calculating cost "+str(time.time() - getdatatime)+" seconds."

# sorting according to tfidf
wordssum = tfidf.sum(axis=0)
index = range(len(voc))
index = [index for (y,x,index) in sorted(zip(wordssum,voc,index),reverse=True)] #if x not in stopwords] 
voc_sorted = [voc[i] for i in index] 
f=open(args.outputfile,'w')
for x in voc_sorted:
    f.write(x.encode("utf-8")+"\n")
f.close()
#return tfidf,voc,txt

