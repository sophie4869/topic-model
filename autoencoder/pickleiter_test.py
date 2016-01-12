import sys
sys.path.append("../")
sys.path.append("../bow_text_cls")
import mxnet_ as mx
from tfidf import tfidf_iterator
import numpy

datapath="../data/tfidf_extraction-1_26.paths"
with open(datapath) as f:
    paths = f.read().split("\n")
Pickleiter = mx.io.PickleIter(batch_size=100,datapath=paths,shuffle=False)
NDiter = tfidf_iterator()[0]
for i in range(10000):
    data1 = Pickleiter.next().data[0].asnumpy()
    data2 = NDiter.next().data[0].asnumpy()
    #print data1[0][28],Pickleiter.cursor, data2[0][28], NDiter.cursor
    assert data1 == data2, "data not equal"
