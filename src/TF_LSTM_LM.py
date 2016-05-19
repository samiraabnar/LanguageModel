import tensorflow as tf

class TF_LSTM_LM(object):
    def __init__(self):
        lstm = rnn_cell.BasicLSTMCell(100)
        state = tf.zeros([batch_size, lstm.state_size])