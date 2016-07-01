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

import itertools
from six.moves import cPickle


class Theano_LSTM_LM(object):

    def __init__(self,input_dim,hidden_dims,output_dim, learning_rate=0.1):
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
                                             random_state=self.random_state,layer_id="_0",bptt_truncate=5)

        params = self.layers[0].params
        params += self.layers[0].output_params
        cost = T.sum(T.nnet.categorical_crossentropy(self.layers[0].output,next_x))

        grads = T.grad(cost,params)
        updates = [(param_i,param_i - self.learning_rate * grad_i) for param_i,grad_i in zip(params,grads)]
        self.learning_step = theano.function([x,next_x],cost,updates=updates)
        next_gen = T.argmax(self.layers[0].output,axis=1)
        self.predict = theano.function([x],[next_gen])



    def train(self):
        with open("../../LSTM/data/sentiment/trainsentence_and_label_binary.txt", 'r') as filedata:
            data = filedata.readlines()

        tokenized_sentences_with_labels = []
        for sent in data:
            tokenized = nltk.tokenize(sent.lower())
            tokenized_sentences_with_labels.append((int(tokenized[0]), tokenized[1:]))




if __name__ == '__main__':
    sys.setrecursionlimit(10000)

    with open("../../LSTM/data/sentiment/trainsentence_and_label_binary.txt", 'r') as filedata:
        data = filedata.readlines()
    char_list = []
    sentences_with_labels = []
    for sent in data:
        sent = sent.lower()
        sentences_with_labels.append((int(sent[0]), sent[1:]))

    tokenized_sentences = [ nltk.word_tokenize(sent[1]) for sent in sentences_with_labels]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

    unknown_token = 'UNK'
    vocabulary_size = 4000
    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))


    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    print("\nExample sentence: '%s'" % sentences_with_labels[0][1])
    print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])




    model = Theano_LSTM_LM(vocabulary_size,[100],vocabulary_size)
    onehot = np.eye(vocabulary_size)

    for e in np.arange(1,100):
        iter = 0
        for i in np.random.permutation(len(X_train)):
            sent = [ onehot[w] for w in X_train[i]]
            f_sent = [ onehot[w] for w in y_train[i]]
            cost = model.learning_step(np.asarray(sent,np.float32),
                                np.asarray(f_sent,np.float32))
            iter += 1
            print(str(iter)+": "+str(cost))


        for i in np.arange(1):
            first_word = np.random.randint(0,vocabulary_size - 1)

            current_input = [first_word]
            sent = index_to_word[current_input[0]]

            l = 0
            while sent[-1] not in ['.','!','?'] and l < 100:
                #print(current_input)
                p_word = model.predict(np.asarray([onehot[cw] for cw in current_input],dtype=np.float32))
                #print(p_char[0])
                #print(p_char[0][-1])
                current_input.append(p_word[0][-1])
                sent += " " + index_to_word[p_word[0][-1]]
                l += 1

            print(sent)

        modelfile = "LM"+str(iter)
        with open(modelfile, "wb") as f:
            cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)






