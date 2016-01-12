import sys
import os
import numpy as np
# code to automatically download dataset
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, "../../tests/python/common"))
#import get_data
import mxnet as mx
from sklearn.datasets import fetch_20newsgroups
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer as cv
import sklearn.preprocessing as pp

def raw_to_words(raw): # words only
    text = raw#bs(raw).get_text()
    letters = re.sub("[^a-zA-Z]"," ",text)
    sf = open('/home/tingyubi/20w/data/eng_n.txt','r')
    stops = [x.strip().decode('utf-8') for x in sf.read().split('\n')]
    #stops = set(stopwords.words("english"))
    words = letters.lower().split()
    meaningful = [w for w in words if not w in stops]
    return " ".join(meaningful)

def news_iterator(input_size,batchsize=100,alldata=True,label="y"):
    # Load some categories from the training set
    categories = [
        'alt.atheism',
        'talk.religion.misc',
    ]
    # load train data
    if(alldata):
        print("Loading 20 newsgroups dataset for 20 categories.")
        traindata = fetch_20newsgroups(subset='train',  remove=('headers','footers','quotes'),categories=None)
    else:
        print("Loading 20 newsgroups dataset for 2 categories.")
        traindata = fetch_20newsgroups(subset='train',  remove=('headers','footers','quotes'),categories=categories)
    # preprocessing
    words=traindata.data
    train_words=[]
    for i in range(0,len(traindata.data)):
        train_words.append( raw_to_words(traindata.data[i]) )

    # train iterator
    vectorizer=cv(analyzer="word",max_features=input_size, stop_words='english')
    train_features=vectorizer.fit_transform(train_words).toarray()
    vocabulary=vectorizer.get_feature_names() # feature name
    x = train_features.astype(np.float64)
    X_normalized = x / (np.max(x, axis=1)[:, None] + 1e-10)
    #ss = pp.StandardScaler(with_mean=False).fit(x)
    #X_normalized = ss.transform(x)
    y = traindata.target
    if(label=="x"):
        train_dataiter = mx.io.NDArrayIter(data=X_normalized, label = X_normalized, batch_size=batchsize, shuffle = True)
    else:
        train_dataiter = mx.io.NDArrayIter(data=X_normalized, label = y, batch_size=batchsize, shuffle = True)
    
    # load test data
    if alldata :
        testdata = fetch_20newsgroups(subset='test',  categories=None, remove=('headers','footers','quotes'))
    else:
        testdata = fetch_20newsgroups(subset='test',  categories=categories, remove=('headers','footers','quotes'))
    test_words=[]
    for i in range(0,len(testdata.data)):
        test_words.append( raw_to_words(testdata.data[i]) )

    # test iterator
    test_features=vectorizer.transform(test_words).toarray()
    x = test_features.astype(np.float64)
    #X_normalized = ss.transform(x)
    X_normalized = x / (np.max(x, axis=1)[:, None] + 1e-10)
    y = testdata.target
    if(label=="y"):
        val_dataiter = mx.io.NDArrayIter(data=X_normalized, label = y,  batch_size=batchsize, shuffle = True)
    else:
        val_dataiter = mx.io.NDArrayIter(data=X_normalized, label = X_normalized,  batch_size=batchsize, shuffle = True)

    return (train_dataiter, val_dataiter,vocabulary)

def mnist_iterator(batch_size, input_shape):
    """return train and val iterators for mnist"""
    # download data
    #get_data.GetMNIST_ubyte()
    flat = False if len(input_shape) == 3 else True

    train_dataiter = mx.io.MNISTIter(
        image="data/train-images-idx3-ubyte",
        label="data/train-labels-idx1-ubyte",
        input_shape=input_shape,
        batch_size=batch_size,
        shuffle=True,
        flat=flat)

    val_dataiter = mx.io.MNISTIter(
        image="data/t10k-images-idx3-ubyte",
        label="data/t10k-labels-idx1-ubyte",
        input_shape=input_shape,
        batch_size=batch_size,
        flat=flat)

    return (train_dataiter, val_dataiter)


def news_iterator_raw(input_size,batchsize):
    # Load some categories from the training set
    categories = [
        'alt.atheism',
        'talk.religion.misc',
    ]
    print("Loading 20 newsgroups dataset for 20 categories.")
    #print(categories)
    traindata = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'),categories=None)
    tfidf_vectorizer = cv(max_df=0.95, min_df=2, max_features=input_size,
                                   stop_words='english')
    train = tfidf_vectorizer.fit_transform(traindata.data)

    # bag of words
    vocabulary=tfidf_vectorizer.get_feature_names() # feature name
    x = train.astype(np.float64).toarray()

    train_x = x / (np.sum(x, axis=1)[:, None] + 1e-10)
    train_y = traindata.target

    testdata = fetch_20newsgroups(subset='test', categories=None, remove=('headers','footers','quotes'))
    test_features=tfidf_vectorizer.transform(testdata.data)

    x = test_features.astype(np.float64).toarray()
    test_x = x / (np.sum(x, axis=1)[:, None] + 1e-10)
    test_y = testdata.target
    return (train_x, train_y, test_x, test_y, vocabulary)


def test_iterator(batchsize,input_size):
    a = np.array([[2,1]]*5)
    train = mx.io.NDArrayIter(data=a, batch_size=batchsize)
    return train,0,0