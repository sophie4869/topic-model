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
from bow import jiebaCounter

def classifier():
    vect,voc,txt=jiebaCounter()
    # normalisation
    x=np.array(vect/(np.max(vect,axis=1)+1e-10))
    x_train,x_test,y_train,y_test=tts(x,y,test_size=0.25,train_size=0.75)
    clf=svm.LinearSVC()
    clf.fit(x_train,y_train)
    Cs=np.logspace(-5,0,10)
    clf_ = GridSearchCV(estimator=clf, param_grid=dict(C=Cs))
    clf_.fit(x_,y)
    print(clf_.best_params_)
    print("train accuracy:")
    print(np.sum(clf_.predict(x_train)==y_train)/float(len(y_train)))
    print("test accuracy:")
    print(np.sum(clf_.predict(x_test)==y_test)/float(len(y_test)))



