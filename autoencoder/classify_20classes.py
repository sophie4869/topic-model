import mxnet as mx
import sys
sys.path.append("../")
sys.path.append("../../autoencoder")
import logging
import numpy as np
from autoencoder import AutoEncoderModel
#from visualize import visualize
from data_all import news_iterator
#ae_model = AutoEncoderModel(mx.gpu(0), [5000,100], pt_dropout=0.5) 
ae_model = AutoEncoderModel(mx.gpu(3),[5000,100],internal_act='sigmoid', output_act='sigmoid', sparseness_penalty=1e-4, pt_dropout=0)
logging.basicConfig(level=logging.DEBUG) 
#ae_model.load('../../autoencoder/news_20classes_small.arg')#classes_small.arg') #news_20_ltest.arg
ae_model.load('news_20classes_small_1e-4_non-neg.arg')
batch_size = 100

fea_sym = ae_model.loss.get_internals()#[3]
logging.info(fea_sym.list_outputs())
output=fea_sym['sparse_encoder_0_output']
fc3 = mx.symbol.FullyConnected(data=output, num_hidden=20)
softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
#logging.info(softmax.list_arguments())

args=ae_model.args
datashape=(100,5000)
train, val, _ = news_iterator(input_size = 5000,batchsize=100)

#fc = softmax.get_internals()
#logging.info(fc.list_arguments())
args_shape,ow,aw = softmax.get_internals().infer_shape(data=datashape)
#logging.info(args_shape)
args['fullyconnected0_weight']=mx.nd.zeros(args_shape[2])
args['fullyconnected0_bias']=mx.nd.zeros(args_shape[1])
#logging.info(args)
lr=10
wd=0.00000
logging.info(lr)
print(wd)
model = mx.model.FeedForward(ctx=mx.gpu(), num_epoch=100, symbol=softmax,learning_rate = lr,
                                         arg_params=args,#ae_model.args,# aux_params=ae_model.aux_params,
                                         allow_extra_params=True,wd=wd)
model.fit(X = train, eval_data = val)
