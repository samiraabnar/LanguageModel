import theano
import theano.tensor as T
import numpy as np
import nltk

import sys
sys.path.append('../../')

from LSTM.src.LSTMLayer import *
from LSTM.src.FullyConnectedLayer import *
from LSTM.src.OutputLayer import *
from LSTM.src.WordEmbeddingLayer import *


from Util.util.data.DataPrep import *
from Util.util.file.FileUtil import *
from Util.util.nnet.LearningAlgorithms import *


class Theano_LSTM_LM(object):

    def __init__(self,input_dim,hidden_dims,output_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.random_state = np.random.RandomState(23455)
        self.learning_rate = learning_rate

        self.build_model()


    def build_model(self):
        x = T.matrix('x').astype(theano.config.floatX)
        next_x = T.matrix('n_x').astype(theano.config.floatX)

        self.layers = {}

        self.layers[0] = LSTMLayer(input=x,
                                             input_dim=self.input_dim,
                                             output_dim=self.hidden_dims[0],
                                             outer_output_dim=self.output_dim,
                                             random_state=self.random_state,layer_id="_0")

        params = self.layers[0].params
        cost = T.sum(T.sqrt(T.sum((self.layers[0].hidden_state - next_x) ** 2,axis=1)))
        grads = T.grad(cost,params)
        updates = [(param_i,param_i - self.learning_rate * grad_i) for param_i,grad_i in zip(params,grads)]
        self.learning_step = theano.function([x,next_x],cost,updates=updates)
        next_gen = T.argmax(self.layers[0].hidden_state)
        self.predict = theano.function([x],[next_gen])



    def train(self):
        with open("../../LSTM/data/sentiment/trainsentence_and_label_binary.txt", 'r') as filedata:
            data = filedata.readlines()

        tokenized_sentences_with_labels = []
        for sent in data:
            tokenized = nltk.tokenize(sent)
            tokenized_sentences_with_labels.append((int(tokenized[0]), tokenized[1:]))




if __name__ == '__main__':
    with open("../../LSTM/data/sentiment/trainsentence_and_label_binary.txt", 'r') as filedata:
        data = filedata.readlines()
    char_list = []
    tokenized_sentences_with_labels = []
    for sent in data:
        tokenized_sentences_with_labels.append((int(sent[0]), sent[1:]))
        for s in sent[1:]:
            if s not in char_list:
                char_list.append(s)
    print(char_list)
    dic = {}
    for i in np.arange(len(char_list)):
        dic[char_list[i]] = i

    one_hot_vector = np.eye(len(char_list))
    one_hot_sents = []
    for sent in tokenized_sentences_with_labels:
        one_hot_sent = [ one_hot_vector[dic[c]] for c in sent[1]]
        one_hot_sents.append(one_hot_sent)

    model = Theano_LSTM_LM(len(char_list),[len(char_list)],1)
    for sent in one_hot_sent:
        forward_sent = sent[1:]
        model.learning_step(np.asarray(sent[0:len(sent) - 1],np.float32),np.asarray(sent[1:],np.float32))

    first_char = char_list[np.random.randint(0,len(char_list))]

    current_char = [first_char]
    sent += current_char[0]

    while current_char not in ['.','!','?']:
        current_char = char_list[model.predict(current_char)]
        sent += current_char[0]
        print(sent)







