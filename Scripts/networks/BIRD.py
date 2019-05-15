#! /user/bin/evn python3
# -*- coding:utf8 -*-

"""
BIRD
======

The Details of BIRD.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://github.com/GuoxiuHe
@copyright: "Copyright (c) 2018 Guoxiu He. All Rights Reserved"
"""

import os, sys
import numpy as np
import tensorflow as tf
tf.set_random_seed(1234)

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
rootdir = '/'.join(curdir.split('/')[:4])
PRO_NAME = 'BIRD'
prodir = rootdir + '/Research/' + PRO_NAME
sys.path.insert(0, prodir)

from utils.utility import show_layer_info_with_memory

from Scripts.Network import Network
from TensorFlow.layers.BPTRUCell import BPTRUCell

class BIRD(Network):
    """
    Implements the main BIRD.
    """
    def __init__(self, memory=0.5, **kwargs):

        Network.__init__(self, memory=memory)
        self.model_name = 'BIRD'

    def _embed(self, word_embedding_matrix=np.array([None])):
        """
        The embedding layer, question and passage share embeddings
        """

        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):

            if word_embedding_matrix.any() == None:
                self.word_embeddings = tf.get_variable(
                    'word_embeddings_random',
                    shape=(self.nb_words, self.embedding_dim),
                    initializer=self.initializer,
                    trainable=False
                )
            else:
                self.word_embeddings = tf.get_variable(
                    'word_embeddings',
                    shape=(self.nb_words, self.embedding_dim),
                    initializer=tf.constant_initializer(word_embedding_matrix),
                    trainable=False
                )

        self.logger.info("get embed.")

    def _inference(self):
        """
        encode sentence information
        """
        with tf.variable_scope('prepare'):
            # batch_size, title_size, title_len
            self.embedded = tf.nn.embedding_lookup(self.word_embeddings, self.input_x)
            # batch_size, query_size, title_size, title_len, embedding_dim
            self.embedded_i = tf.nn.embedding_lookup(self.word_embeddings, self.input_x_i)
            # batch_size, query_size, query_len, embedding_dim
            self.embedded_q = tf.nn.embedding_lookup(self.word_embeddings, self.input_x_q)

            self.embedded_transfer = tf.keras.layers.Dense(self.embedding_dim, activation='tanh')(self.embedded)
            self.embedded_i_transfer = tf.keras.layers.Dense(self.embedding_dim, activation='tanh')(self.embedded_i)
            self.embedded_q_transfer = tf.keras.layers.Dense(self.embedding_dim, activation='tanh')(self.embedded_q)

            if self.dropout_rate:
                self.embedded_transfer = tf.nn.dropout(self.embedded_transfer, self.dropout_keep_prob)
            show_layer_info_with_memory('embedded_transfer', self.embedded_transfer, self.logger)

            if self.dropout_rate:
                self.embedded_i_transfer = tf.nn.dropout(self.embedded_i_transfer, self.dropout_keep_prob)
            show_layer_info_with_memory('embedded_i_transfer', self.embedded_i_transfer, self.logger)

            if self.dropout_rate:
                self.embedded_q_transfer = tf.nn.dropout(self.embedded_q_transfer, self.dropout_keep_prob)
            show_layer_info_with_memory('embedded_q_transfer', self.embedded_q_transfer, self.logger)

        with tf.variable_scope('encoding'):

            self.sample_size, self.query_size, self.query_len, _ = tf.unstack(tf.shape(self.embedded_q_transfer))

            # batch_size * query_size, query_len, embedding_dim
            self.word_level_query_input = tf.reshape(self.embedded_q_transfer, [
                self.sample_size * self.query_size,
                self.query_len,
                self.embedding_dim
            ])

            # batch_size * query_size, embedding_dim
            self.word_level_query_output = tf.keras.layers.GlobalAveragePooling1D()(self.word_level_query_input)

            if self.dropout_rate:
                self.word_level_query_output = tf.nn.dropout(self.word_level_query_output, self.dropout_keep_prob)
            show_layer_info_with_memory('word_level_query_output', self.word_level_query_output, self.logger)

            # Final Representation of QUERY
            # batch_size, query_size, embedding_dim
            self.sent_level_query = tf.reshape(self.word_level_query_output, [
                self.sample_size,
                self.query_size,
                self.embedding_dim
            ])

            self.sample_size, self.query_size, self.title_size, self.title_len, _ = tf.unstack(
                tf.shape(self.embedded_i_transfer))

            # batch_size, query_size, 1, 1, embedding_dim
            self.sent_level_query_rely = tf.expand_dims(tf.expand_dims(self.sent_level_query, axis=2), axis=2)

            # batch_size, query_size, title_size, title_len, embedding_dim
            self.sent_level_query_rely = tf.tile(self.sent_level_query_rely,
                                                 multiples=[1, 1, self.title_size, self.title_len, 1])

            # batch_size * query_size * title_size * title_len, embedding_dim
            self.sent_level_query_rely = tf.reshape(self.sent_level_query_rely, [
                self.sample_size * self.query_size * self.title_size * self.title_len,
                self.embedding_dim
            ])

            # batch_size * query_size * title_size * title_len, embedding_dim
            self.word_level_title_input = tf.reshape(self.embedded_i_transfer, [
                self.sample_size * self.query_size * self.title_size * self.title_len,
                self.embedding_dim
            ])

            # batch_size * query_size * title_size * title_len
            self.title_word_atten = tf.reduce_sum(tf.multiply(self.sent_level_query_rely,
                                                              self.word_level_title_input),
                                                  axis=-1)

            # batch_size * query_size * title_size * title_len, 1
            self.title_word_atten = tf.reshape(tf.nn.softmax(tf.reshape(self.title_word_atten,
                                                                        [self.sample_size * self.query_size * self.title_size,
                                                                         self.title_len]),
                                                             axis=-1),
                                               [self.sample_size * self.query_size * self.title_size * self.title_len,
                                                1]
                                               )

            # batch_size * query_size * title_size * title_len, embedding_dim
            self.word_level_title_input_atten = tf.multiply(self.word_level_title_input, self.title_word_atten)

            # batch_size * query_size * title_size, title_len, embedding_dim
            self.word_level_title_input_atten = tf.reshape(self.word_level_title_input_atten, [
                self.sample_size * self.query_size * self.title_size,
                self.title_len,
                self.embedding_dim
            ])

            # batch_size * query_size * title_size, embedding_dim
            self.word_level_title_output = tf.reduce_sum(self.word_level_title_input_atten, axis=1)

            # batch_size * query_size * title_size, embedding_dim
            self.sent_level_query_rely_2 = tf.reshape(tf.tile(tf.expand_dims(self.sent_level_query,
                                                                             axis=2),
                                                              [1, 1, self.title_size, 1]),
                                                      [self.sample_size * self.query_size * self.title_size,
                                                       self.embedding_dim])

            # batch_size * query_size * title_size, 1
            self.word_level_title_output_atten = tf.reshape(
                tf.nn.softmax(tf.reshape(tf.reduce_sum(tf.multiply(self.word_level_title_output,
                                                                   self.sent_level_query_rely_2),
                                                       axis=-1),
                                         [self.sample_size * self.query_size, self.title_size]),
                              axis=-1),
                [self.sample_size * self.query_size * self.title_size, 1])

            # batch_size * query_size * title_size, embedding_dim
            self.title_level_title_input_atten = tf.multiply(self.word_level_title_output,
                                                             self.word_level_title_output_atten)

            # batch_size * query_size, title_size, embedding_dim
            self.title_level_title_input = tf.reshape(self.title_level_title_input_atten,
                                                      [self.sample_size * self.query_size,
                                                       self.title_size,
                                                       self.embedding_dim
                                                       ])

            # batch_size * query_size, embedding_dim
            self.title_level_title_output = tf.reduce_sum(self.title_level_title_input, axis=1)

            if self.dropout_rate:
                self.title_level_title_output = tf.nn.dropout(self.title_level_title_output, self.dropout_keep_prob)
            show_layer_info_with_memory('title_level_title_output', self.title_level_title_output, self.logger)

            # Final Representation of TITLE
            # batch_size, query_size, embedding_dim
            self.query_level_title = tf.reshape(self.title_level_title_output, [
                self.sample_size,
                self.query_size,
                self.embedding_dim
            ])

            # batch_size, query_size, embedding_dim
            self.combine_gate = tf.layers.dense(tf.concat([self.sent_level_query, self.query_level_title], axis=-1),
                                                self.embedding_dim,
                                                activation=tf.nn.sigmoid,
                                                kernel_initializer=self.initializer)

            # batch_size, query_size, embedding_dim
            self.query_level = tf.multiply((1 - self.combine_gate), self.query_level_title) + \
                               tf.multiply(self.combine_gate, self.sent_level_query)

            fw_rnn_cell = BPTRUCell(self.rnn_dim)
            fw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(fw_rnn_cell,
                                                        input_keep_prob=self.dropout_keep_prob,
                                                        output_keep_prob=self.dropout_keep_prob)
            bw_rnn_cell = BPTRUCell(self.rnn_dim)
            bw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(bw_rnn_cell,
                                                        input_keep_prob=self.dropout_keep_prob,
                                                        output_keep_prob=self.dropout_keep_prob)

            outputs, output_states = \
                tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell, self.query_level, dtype=tf.float32)

            fw_state, bw_state = output_states
            fw_c, fw_h = fw_state
            bw_c, bw_h = bw_state

            self.cell_memory = tf.concat([fw_c, bw_c], -1)

            self.rnn_outputs = tf.concat(outputs, -1)

            self.last_hidden = self.rnn_outputs[:, -1, :]

            self.last_hidden_reply = tf.tile(tf.expand_dims(self.last_hidden, 1),
                                             [1, self.query_size, 1])

            self.numerator = tf.reduce_sum(tf.multiply(self.last_hidden_reply,
                                                       self.rnn_outputs),
                                           axis=-1)

            self.denominator = tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(self.last_hidden_reply),
                                                                 axis=-1)),
                                           tf.sqrt(tf.reduce_sum(tf.square(self.rnn_outputs),
                                                                 axis=-1)))

            # batch_size, query_size
            self.last_hidden_atten = tf.expand_dims(tf.nn.softmax(tf.sigmoid(tf.divide(self.numerator,
                                                                                       self.denominator)),
                                                                  axis=-1),
                                                    axis=-1)

            # batch_size, embedding_dim
            self.output_attentive = tf.reduce_sum(tf.multiply(self.last_hidden_atten,
                                                              self.rnn_outputs),
                                                  axis=1)

            # self.pooling = tf.concat([self.hidden_state_attentive, self.cell_memory], axis=-1)
            self.pooling = tf.concat([self.output_attentive, self.cell_memory], -1)
            # self.pooling = self.hidden_state

            # if self.dropout_rate:
            #     self.pooling = tf.nn.dropout(self.pooling, self.dropout_keep_prob)

            show_layer_info_with_memory('Pooling', self.pooling, self.logger)

        with tf.variable_scope("output"):

            self.dense = tf.keras.layers.Dense(self.dense_dim, activation='relu')(self.pooling)

            if self.dropout_rate:
                self.dense = tf.nn.dropout(self.dense, self.dropout_keep_prob)

            self.logits = tf.keras.layers.Dense(self.nb_classes)(self.dense)

            show_layer_info_with_memory('Logits', self.logits, self.logger)

            self.output = tf.argmax(self.logits, axis=-1)
            show_layer_info_with_memory('output', self.output, self.logger)

            self.proba = tf.nn.softmax(self.logits, axis=-1)[-1]
            show_layer_info_with_memory('proba', self.proba, self.logger)

        self.logger.info("BIRD inference.")

        return self.logits


if __name__ == '__main__':
    network = BIRD()
    network.build_graph()
