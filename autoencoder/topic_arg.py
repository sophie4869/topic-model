import mxnet as mx
import sys
sys.path.append("../autoencoder")
from autoencoder import AutoEncoderModel
from data_all import news_iterator, mnist_iterator
from tfidf import tfidf_iterator
import numpy as np
import logging
import copy

batch_size = 128

def l2_norm(label, pred):
    return np.mean(np.square(label.asnumpy()-pred.asnumpy()))/2.0
def SGD(weight, grad, lr=1, grad_norm=batch_size):
   # print grad.asnumpy()
    weight[:] -= lr * grad / batch_size
    #w = weight.asnumpy()
    #w *= w > 0

    #weight[:] = np.clip(weight.asnumpy(), 0, 1)

if __name__ == '__main__':
    # set to INFO to see less information during training
    logging.basicConfig(level=logging.DEBUG)
    voc_count = 10000
    num_gpu = 1
    batch_size = 100
    gpus = [mx.gpu(i) for i in range(num_gpu)]
    dev=mx.gpu()

#    train, val, voc = news_iterator(voc_count, 100)
    vocfile = "/home/tingyubi/20w/data/tfidf_extraction-1_26.voc"
    with open(vocfile,'r') as f:
        voc = f.read().decode('utf-8').split(" ")
    """
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(name='encoder_%d'%istack, data=data, num_hidden=20)
    sig1 = mx.symbol.Activation(data=fc1, act_type='sigmoid')
    sparse1 = mx.symbol.SparseReg(data=sig1, penalty=1e-3, sparseness_target=0.1)
    fc2 = mx.symbol.FullyConnected(name='decoder_%d'%istack, data=sparse1, num_hidden=5000)
    loss = mx.symbol.LinearRegressionOutput(data=fc2, name='softmax')
"""
    aem = AutoEncoderModel(mx.gpu(3),[voc_count,1000,200],internal_act='sigmoid', output_act='sigmoid', sparseness_penalty=1e-4, pt_dropout=0)
    aem.load('/home/tingyubi/20w/autoencoder/20w_1000_200_1e-4_non-neg.arg')
    #print aem.loss.get_internals().list_outputs()
    fc2 = aem.encoder
    print aem.loss.get_internals().list_outputs()
    #print aem.loss.get_internals().list_arguments()
    model = aem
    #print model.arg_params['encoder_0_weight'].shape

    """
    for k in range(2):
        words_index=np.argsort(model.args['encoder_0_weight'].asnumpy()[k,:])
        logging.info('Topic %d' % k)
        logging.info([voc[i] for i in words_index[-20:]])
    """

    for k in range(200):
   #     words_index=np.argsort(model.args['encoder_0_weight'].asnumpy()[k,:])
   #     logging.info('Topic %d' % k)
   #     logging.info([voc[i] for i in words_index[-50:]])


        data_shape=(1,voc_count)
        arg_names = fc2.list_arguments()
        #print("arg_names:")
        #print(arg_names)
        aux_names = fc2.list_auxiliary_states()
        arg_shapes, output_shapes, aux_shapes = fc2.infer_shape(data=data_shape)

        arg_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
        grad_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in arg_shapes]
        aux_arrays = [mx.nd.zeros(shape, ctx=dev) for shape in aux_shapes]
        reqs = ["write" for name in arg_names]

        arg_map = dict(zip(arg_names, arg_arrays))
        grad_map = dict(zip(arg_names, grad_arrays))
        aux_map = dict(zip(aux_names, aux_arrays))

        encoder = aem.loss.get_internals()['encoder_1_output']
#        encoder = aem.loss.get_internals()['encoder_1_output']
        for name in arg_names:
            if "weight" in name:
                arg_map[name][:] = model.args[name]
        for name in aux_names:
            aux_map[name][:] = model.auxs[name]

        data = np.zeros(arg_map['data'].shape)
        arg_map['data'][:] = data
        encoder_model = encoder.bind(ctx=dev, args=arg_map, args_grad=grad_map, grad_req=reqs, aux_states=aux_map)
        #"""
        ### grad = -1
        # print encoder.list_arguments()
        # print [k for k in model.arg_params.iterkeys()]
        for i in range(1000):
        #    print arg_map['data'].asnumpy()
            encoder_model.forward(is_train = False)
            out_grad = np.ones(encoder_model.outputs[0].shape) / 200
            out_grad[:, k] = -1
            #print encoder_model.outputs[0].shape
            encoder_model.backward([mx.nd.array(out_grad, ctx=dev)])
#            print grad_map['encoder_2_weight'].asnumpy()
            #print encoder_model.outputs.shape
            #print encoder_model.loss.get_internals().outputs[0].shape
            #print grad_map['decoder_2_weight'].asnumpy()

            #print arg_map['data'].asnumpy()
#            print np.sum(grad_map['encoder_1_weight'].asnumpy())
           # print grad_map['encoder_2_weight'].asnumpy()
           # print arg_map['encoder_2_weight'].asnumpy()
#            print grad_map['encoder_1_bias'].asnumpy()
            #print grad_map['data'].asnumpy()
            SGD(arg_map['data'], grad_map['data'], 1e-2)
        #print arg_map['encoder_0_weight'].asnumpy()
#        print encoder_model.outputs[0].asnumpy()
        #print encoder_model.outputs[0].asnumpy()
        data=arg_map['data'][:].asnumpy()[0]
        words_index=np.argsort(data)
        print words_index.shape
#        logging.info(words_index[-50:])
        print('Topic %d' % k)
        words = [voc[i] for i in words_index[-50:]]
        word = ', '.join(words)
        print word

        ###
        #"""
        """
        ###
        y = np.array([[0]*100])
        y[0][k]=1
        data = mx.nd.array([[0.1]*5000]*100)
        print arg_map['data'].shape
        arg_map['data'][:] = data
        out_grad = mx.nd.zeros(encoder_model.outputs[0].shape,ctx=dev)
        for i in range(1000):
            encoder_model.forward()
            theta = encoder_model.outputs[0].asnumpy()
            out_grad[:]=theta - y
            encoder_model.backward([out_grad])
            SGD(arg_map['data'], grad_map['data'])
            #print grad_map['data'].asnumpy()
        data = arg_map['data'][:].asnumpy()[0]
        words_index = np.argsort(data)
        logging.info(words_index[-10:])
        logging.info([voc[i] for i in words_index[-20:]])
        ###
        """

