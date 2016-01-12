import sys
sys.path.append('../bow_text_cls')
import mxnet as mx
#from data_all import news_iterator, mnist_iterator
import numpy as np
import logging
import copy
#from tfidf import tfidf_iterator
from tfidf import tfidf_iterator_labelisx
batch_size = 128

def l2_norm(label, pred):
    return np.mean(np.square(label.asnumpy()-pred.asnumpy()))/2.0
def SGD(weight, grad, lr=0.1, grad_norm=batch_size):
    weight[:] -= lr * grad / batch_size

if __name__ == '__main__':
    # set to INFO to see less information during training
    logging.basicConfig(level=logging.DEBUG)
    train, val, voc = tfidf_iterator_labelisx(batch_size)
    num_gpu = 1
    gpus = [mx.gpu(i) for i in range(num_gpu)]
    dev=mx.gpu()
    #train, val, voc = news_iterator(5000, 128, "x")
    istack = 0
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(name='encoder_%d'%istack, data=data, num_hidden=20)
    sig1 = mx.symbol.Activation(data=fc1, act_type='sigmoid')
    sparse1 = mx.symbol.IdentityAttachKLSparseReg(data=sig1, penalty=1e-4, sparseness_target=0.1)
    fc2 = mx.symbol.FullyConnected(name='decoder_%d'%istack, data=sparse1, num_hidden=10000)
    loss = mx.symbol.LinearRegressionOutput(data=fc2, name='softmax')

    print loss.get_internals().list_outputs()
    model = mx.model.FeedForward(ctx=gpus, symbol=loss, num_epoch=10,
                             learning_rate=10, momentum=0.9, wd=0)

    model.fit(X=train, #eval_data=val,
          eval_metric=mx.metric.CustomMetric(l2_norm))

    print loss.get_internals().list_outputs()
    print loss.get_internals().list_arguments()  
    print model.arg_params['encoder_0_weight'].shape
    for k in range(20):
        words_index=np.argsort(model.arg_params['encoder_0_weight'].asnumpy()[k,:])
        logging.info('Topic %d' % k)
        logging.info([voc[i] for i in words_index[-20:]])

    weight = model.arg_params['encoder_0_weight'].asnumpy()
    print float(np.sum(weight==0)) / (weight.shape[0] * weight.shape[1])
"""
    for k in range(2):
        data_shape=(1,5000)
        arg_names = fc2.list_arguments() 
        print("arg_names:")
        print(arg_names)
        arg_shapes, output_shapes, aux_shapes = fc2.infer_shape(data=data_shape)

        arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
        grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
        reqs = ["write" for name in arg_names]
        
        arg_map = dict(zip(arg_names, arg_arrays))
        grad_map = dict(zip(arg_names, grad_arrays))

        encoder = loss.get_internals()['encoder_0_output']
        for name in arg_names:
            if "weight" in name or "bias" in name:
                arg_map[name][:] = model.arg_params[name]

        # print encoder.list_arguments()
        # print [k for k in model.arg_params.iterkeys()]
        encoder_model = encoder.bind(ctx=dev, args=arg_map, args_grad=grad_map, grad_req=reqs, aux_states=model.aux_params)
        for i in range(1000):
            encoder_model.forward(is_train = True)
            out_grad = np.ones(encoder_model.outputs[0].shape) / 100
            out_grad[:, k] = -1
            encoder_model.backward([mx.nd.array(out_grad, ctx=dev)])
            #print grad_map['data'].asnumpy()
            SGD(arg_map['data'], grad_map['data'], 10)
        #print encoder_model.outputs[0].asnumpy()
        data=arg_map['data'][:].asnumpy()[0]
        words_index=np.argsort(data)
        #logging.info(words_index[-20:])
        logging.info('Topic %d' % k)
        logging.info([voc[i] for i in words_index[-20:]])
"""