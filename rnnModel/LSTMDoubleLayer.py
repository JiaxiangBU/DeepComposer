import tensorflow as tf
import numpy as np


class lstm_double:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.first_cell = tf.contrib.rnn.LSTMCell(input_dim)
        self.second_cell = tf.contrib.rnn.LSTMCell(input_dim)

    def get_dynamicRNN(self, batch_size):
        first_zero_state = self.first_cell.zero_state(batch_size, tf.float32)
        batch_x = tf.placeholder(tf.float32, [None, batch_size, self.input_dim])
        seq_length = tf.placeholder(tf.int32, [batch_size])

        first_out, _ = tf.nn.dynamic_rnn(self.first_cell,
                                         batch_x,
                                         sequence_length=seq_length,
                                         initial_state=first_zero_state,
                                         time_major=True,
                                         scope='first_layer')

        second_zero_state = self.second_cell.zero_state(batch_size, tf.float32)

        second_out, _ = tf.nn.dynamic_rnn(self.second_cell,
                                          first_out,
                                          sequence_length=seq_length,
                                          initial_state=second_zero_state,
                                          time_major=True,
                                          scope='second_layer')

        return batch_x, seq_length, second_out

    def LSTM_step(self, first_state, second_state):
        #first_state = self.first_cell.zero_state(batch_size, tf.float32)
        #second_state = self.second_cell.zero_state(batch_size, tf.float32)

        inputs = tf.placeholder(tf.float32, [None, self.input_dim])

        with tf.variable_scope("pred") as scope:
            out, first_state = self.first_cell(inputs, first_state)
            scope.reuse_variables()
            out, second_state = self.second_cell(out, second_state)

        return inputs, first_state, second_state, out

