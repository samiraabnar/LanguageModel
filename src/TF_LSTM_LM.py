import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

import numpy as np

class TF_LSTM(object):
    def __init__(self):
        self.layers = {}

    def build_model(self):
        # tf Graph input
        x = tf.placeholder("float", [None, n_steps, n_input])
        y = tf.placeholder("float", [None, n_classes])

        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        self.layers["LSTM"] = rnn_cell(10,forget_bias=1.0)
        self.outputs,self.states = rnn.rnn(self.layers["LSTM"],self.x, dtype=tf.float32)


if __name__ == '__main__':
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
        initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
        cell = tf.nn.rnn_cell.LSTMCell(
            num_units, use_peepholes=True, cell_clip=0.0, initializer=initializer)
        inputs = max_length * [
            tf.placeholder(tf.float32, shape=(batch_size, input_size))]
        outputs, _ = tf.nn.rnn(cell, inputs, dtype=tf.float32)
        assertEqual(len(outputs), len(inputs))
        for out in outputs:
            self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

        tf.initialize_all_variables().run()
        input_value = np.random.randn(batch_size, input_size)
        values = sess.run(outputs, feed_dict={inputs[0]: input_value})

    for value in values:
        # if cell c is clipped to 0, tanh(c) = 0 => m==0
        assertAllEqual(value, np.zeros((batch_size, num_units)))