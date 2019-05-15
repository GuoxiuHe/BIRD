#! /user/bin/evn python3
# -*- coding:utf8 -*-

"""
BPTRUCell
======

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@copyright: "Copyright (c) 2018 Guoxiu He. All Rights Reserved"
"""

import os, sys, time, pickle, logging, codecs, json, argparse
import numpy as np
import tensorflow as tf
import random
tf.set_random_seed(1234)

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
rootdir = '/'.join(curdir.split('/')[:4])
PRO_NAME = 'BIRD'
prodir = rootdir + '/Research/' + PRO_NAME
sys.path.insert(0, prodir)

from tensorflow.python.ops.rnn_cell import LayerRNNCell
from tensorflow.python.keras.utils import tf_utils


class BPTRUCell(LayerRNNCell):
    def __init__(self, num_units):
        """Initialize the parameters for an BPTRU cell.

            Args:
              num_units: int, The number of units in the BPTRU cell.
            """
        super(BPTRUCell, self).__init__()
        self._num_units = num_units

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    @tf_utils.shape_type_conversion
    def build(self, inputs_shape):
        x_size = inputs_shape[1]
        self.W_u = tf.get_variable('W_u', [x_size, 4 * self.output_size])
        self.W_c = tf.get_variable('W_c', [self.output_size, self.output_size])
        self.W_h = tf.get_variable('W_h', [self.output_size, self.output_size])
        self.b_f = tf.get_variable('b_f', [self._num_units])
        self.v_f = tf.get_variable('v_f', [self._num_units])
        self.b_r = tf.get_variable('b_r', [self._num_units])
        self.v_r = tf.get_variable('v_r', [self._num_units])
        self.b_o = tf.get_variable('b_o', [self._num_units])
        self.built = True

    def call(self, x, state):
        c, h = state

        xh = tf.matmul(x, self.W_u)
        x1, x2, x3, x4 = tf.split(xh, 4, 1)

        f = tf.sigmoid(x1 + self.v_f * c + self.b_f)
        r = tf.sigmoid(x2 + self.v_r * h + self.b_r)
        o = tf.sigmoid(tf.matmul(c, self.W_c) + tf.matmul(h, self.W_h) + self.b_o)

        new_c = tf.tanh(f * c + (1 - f) * x3 + o * h)
        new_h = tf.tanh(r * h + (1 - r) * x4 + o * c)

        return new_h, (new_c, new_h)