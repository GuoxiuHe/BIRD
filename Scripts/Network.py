#! /user/bin/evn python3
# -*- coding:utf8 -*-

"""
Network
======

The root Class for antispam of secure.

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

from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from utils.utility import show_layer_info, show_layer_info_with_memory
from utils.utility import print_all_variables, print_trainable_variables
from utils.utility import get_a_p_r_f, get_a_p_r_f_more, get_a_p_r_f_more_more

class Network(object):
    """
    Implements the main AntiSpam Secure Network.
    """
    def __init__(self, maxlen=150, nb_classes=2, nb_words=200000,
                 embedding_dim=200, dense_dim=200, rnn_dim=100, cnn_filters=200, dropout_rate=0.5,
                 learning_rate=0.001, weight_decay=0, optim_type='adam', batch_size=64,
                 gpu='0', memory=0.5,
                 **kwargs):

        # logging
        self.logger = logging.getLogger("AntiSpam_Secure")

        # data config
        self.maxlen = maxlen
        self.nb_classes = nb_classes
        self.nb_words = nb_words
        self.nb_users = 1000000
        self.nb_query_words = 1000000

        # network config
        self.embedding_dim = embedding_dim
        self.dense_dim = dense_dim
        self.rnn_dim = rnn_dim
        self.cnn_filters = cnn_filters
        self.dropout_rate = dropout_rate

        # initializer
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.word_initializer = tf.random_normal_initializer(stddev=0.1, seed=1234)
        self.zero_initializer = tf.zeros_initializer()

        # optimizer config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optim_type = optim_type
        self.batch_size = batch_size

        self.model_name = 'Network'
        self.data_name = 'PPDD'

        # gpu config
        self.gpu = gpu
        self.memory = memory

        # session config and generate a session
        if self.memory > 0:
            num_threads = os.environ.get('OMP_NUM_THREADS')
            self.logger.info("Memory use is %s."%self.memory)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=float(self.memory))
            config = tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads)
            self.sess = tf.Session(config=config)
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

    # set all hyperparameters
    # set nb words
    def set_nb_words(self, nb_words):
        self.nb_words = nb_words
        self.logger.info("set nb_words.")

    def set_nb_query_words(self, nb_query_words):
        self.nb_query_words = nb_query_words
        self.logger.info("set nb_query_words.")

    def set_nb_users(self, nb_users):
        self.nb_users = nb_users
        self.logger.info("set nb_users.")

    # set data name
    def set_data_name(self, data_name):
        self.data_name = data_name
        self.logger.info("set data_name.")

    # set model name
    def set_name(self, model_name):
        self.model_name = model_name
        self.logger.info("set model_name.")

    # set data config
    def set_from_data_config(self, data_config):
        self.nb_classes = data_config['nb_classes']
        # self.nb_words = data_config['nb_words'] + 1
        self.logger.info("set from data_config.")

    # set model config
    def set_from_model_config(self, model_config):
        self.embedding_dim = model_config['embedding_dim']
        self.rnn_dim = model_config['rnn_dim']
        self.dense_dim = model_config['dense_dim']
        self.cnn_filters = model_config['cnn_filters']
        self.dropout_rate = model_config['dropout_rate']
        self.optim_type = model_config['optimizer']
        self.learning_rate = model_config['learning_rate']
        self.weight_decay = model_config['weight_decay']
        self.logger.info("set from model_config.")

    # set all config from json
    def set_from_json(self, network_config):
        self.embedding_dim = network_config['embedding_dim']
        self.rnn_dim = network_config['rnn_dim']
        self.dense_dim = network_config['dense_dim']
        self.dropout_rate = network_config['dropout_rate']
        self.nb_classes = network_config['nb_classes']
        self.optim_type = network_config['optimizer']
        self.logger.info("set from json.")

    # build graph
    def build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()  # start time
        self._setup_placeholders()  # end-to-end input and output
        self._embed()  # embedding: pre-train is better
        self._inference()  # details of model
        self._compute_loss()  # loss
        self._create_train_op()  # training
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))  # create time
        print_trainable_variables(output_detail=True, logger=self.logger)  # print trainable variables
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])  # print num of all params
        self.logger.info('There are {} parameters in the model'.format(param_num))

        # save info
        self.save_dir = prodir + '/Scripts/weights/' + self.data_name + '/' + self.model_name + '/'

        if not os.path.exists(prodir + '/Scripts/weights/'):
            os.mkdir(prodir + '/Scripts/weights/')

        if not os.path.exists(prodir + '/Scripts/weights/' + self.data_name):
            os.mkdir(prodir + '/Scripts/weights/' + self.data_name)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        self.input_x_i = tf.placeholder(tf.int32, [None, None, None, None], name="input_x_i")
        self.input_x_q = tf.placeholder(tf.int32, [None, None, None], name="input_x_q")
        self.input_y = tf.placeholder(tf.int32, [None, self.nb_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.logger.info("setup placeholders.")

    def _embed(self, word_embedding_matrix=np.array([None])):
        """
        The embedding layer, question and passage share embeddings
        """

        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            if word_embedding_matrix.any() == None:
                self.word_embeddings = tf.concat([tf.get_variable('word_embeddings_zero',
                                                                  shape=(1, self.embedding_dim),
                                                                  initializer=self.zero_initializer,
                                                                  trainable=False),
                                                  tf.get_variable('word_embeddings_random',
                                                                  shape=(self.nb_words-1, self.embedding_dim),
                                                                  initializer=self.initializer,
                                                                  trainable=False)
                                                  ], axis=0)
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
            self.embedded = tf.nn.embedding_lookup(self.word_embeddings, self.input_x)
            self.embedded_i = tf.nn.embedding_lookup(self.word_embeddings, self.input_x_i)
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

            self.sample_size, self.query_size, self.title_size, self.title_len, _ = tf.unstack(tf.shape(self.embedded_i_transfer))

            self.word_level_title_input = tf.reshape(self.embedded_i_transfer, [
                self.sample_size * self.query_size * self.title_size,
                self.title_len,
                self.embedding_dim
            ])
            show_layer_info_with_memory('word_level_title_input', self.word_level_title_input, self.logger)

            self.word_level_title_output = tf.keras.layers.GlobalAveragePooling1D()(self.word_level_title_input)
            if self.dropout_rate:
                self.word_level_title_output = tf.nn.dropout(self.word_level_title_output, self.dropout_keep_prob)
            show_layer_info_with_memory('word_level_title_output', self.word_level_title_output, self.logger)

            self.title_level_title_input = tf.reshape(self.word_level_title_output, [
                self.sample_size * self.query_size,
                self.title_size,
                self.embedding_dim
            ])
            show_layer_info_with_memory('title_level_title_input', self.title_level_title_input, self.logger)

            self.title_level_title_output = tf.keras.layers.GlobalAveragePooling1D()(self.title_level_title_input)
            if self.dropout_rate:
                self.title_level_title_output = tf.nn.dropout(self.title_level_title_output, self.dropout_keep_prob)
            show_layer_info_with_memory('title_level_title_output', self.title_level_title_output, self.logger)

            self.query_level_title_input = tf.reshape(self.title_level_title_output, [
                self.sample_size,
                self.query_size,
                self.embedding_dim
            ])
            show_layer_info_with_memory('query_level_title_input', self.query_level_title_input, self.logger)

            self.query_level_title_output = tf.keras.layers.GlobalAveragePooling1D()(self.query_level_title_input)
            if self.dropout_rate:
                self.query_level_title_output = tf.nn.dropout(self.query_level_title_output, self.dropout_keep_prob)
            show_layer_info_with_memory('query_level_title_output', self.query_level_title_output, self.logger)

            self.sample_size, self.query_size, self.query_len, _ = tf.unstack(tf.shape(self.embedded_q_transfer))

            self.word_level_query_input = tf.reshape(self.embedded_q_transfer, [
                self.sample_size * self.query_size,
                self.query_len,
                self.embedding_dim
            ])
            show_layer_info_with_memory('word_level_query_input', self.word_level_query_input, self.logger)

            self.word_level_query_output = tf.keras.layers.GlobalAveragePooling1D()(self.word_level_query_input)
            if self.dropout_rate:
                self.word_level_query_output = tf.nn.dropout(self.word_level_query_output, self.dropout_keep_prob)
            show_layer_info_with_memory('word_level_query_output', self.word_level_query_output, self.logger)

            self.sent_level_query_input = tf.reshape(self.word_level_query_output, [
                self.sample_size,
                self.query_size,
                self.embedding_dim
            ])
            show_layer_info_with_memory('sent_level_query_input', self.sent_level_query_input, self.logger)

            self.sent_level_query_output = tf.keras.layers.GlobalAveragePooling1D()(self.sent_level_query_input)
            if self.dropout_rate:
                self.sent_level_query_output = tf.nn.dropout(self.sent_level_query_output, self.dropout_keep_prob)
            show_layer_info_with_memory('sent_level_query_output', self.sent_level_query_output, self.logger)

            self.pooling = tf.concat([self.query_level_title_output, self.sent_level_query_output], -1)

            show_layer_info_with_memory('Pooling', self.pooling, self.logger)

        with tf.variable_scope("output"):

            self.dense = tf.keras.layers.Dense(self.dense_dim, activation='relu')(self.pooling)
            if self.dropout_rate:
                self.dense = tf.nn.dropout(self.dense, self.dropout_keep_prob)
            show_layer_info_with_memory('dense', self.dense, self.logger)

            self.logits = tf.keras.layers.Dense(self.nb_classes)(self.dense)

            show_layer_info_with_memory('Logits', self.logits, self.logger)

            self.output = tf.argmax(self.logits, axis=-1)
            show_layer_info_with_memory('output', self.output, self.logger)

            self.proba = tf.nn.softmax(self.logits, axis=-1)[-1]
            show_layer_info_with_memory('proba', self.proba, self.logger)

        self.logger.info("base model inference.")

        return self.logits

    def _compute_loss(self):
        """
        The loss function
        """

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y,
                                                       logits=self.logits))
        self.logger.info("Calculate Loss.")

        self.all_params = tf.trainable_variables()
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss
            self.logger.info("Add L2 Loss.")

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """

        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)

    def _get_a_p_r_f(self, input_y, prediction, category):
        target_class = np.array(np.argmax(input_y, axis=-1))
        pre_class = np.array(np.argmax(prediction, axis=-1))
        accuracy, precision, recall, f1score = get_a_p_r_f(target_class, pre_class, category)
        return accuracy, precision, recall, f1score

    def _get_a_p_r_f_more(self, input_y, prediction, category):
        target_class = np.array(np.argmax(input_y, axis=-1))
        pre_class = np.array(np.argmax(prediction, axis=-1))

        self.logger.info('True 0 in data %s.'%np.sum(np.array(target_class) == 0))
        self.logger.info('True 1 in data %s.'%np.sum(np.array(target_class) == 1))

        self.logger.info('Predict 0 in data %s.'%np.sum(np.array(pre_class) == 0))
        self.logger.info('Predict 1 in data %s.'%np.sum(np.array(pre_class) == 1))
        accuracy, precision, recall, f1score, f0_5score, f2score = get_a_p_r_f_more(target_class, pre_class, category)
        return accuracy, precision, recall, f1score, f0_5score, f2score

    def train(self, data_generator, load_feature_label,
              train_data, val_data, test_data, word_index, query_word_index, user_index, black_user_set,
              dropout_rate, nb_classes, epochs, train_size, batch_size,
              if_evaluate=None, save_best=True, *args, **kwargs):

        steps_per_epoch = int(train_size / batch_size + 1)
        log_num_per_epoch = 5
        self.logger.info('All steps per epoch for train are %d.'%steps_per_epoch)

        max_val_f1 = 0

        for epoch in range(epochs):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            start_t = int(time.time())

            counter, total_loss = 0, 0.0

            all_logits = None
            all_Y = None

            for Item, X, X_Q, User_Y, X_Y, Y, Length in data_generator(train_data, word_index,
                                                                       query_word_index, user_index,
                                                                       black_user_set, nb_classes, batch_size):
                feed_dict = {self.input_x: Item,
                             self.input_x_i: X,
                             self.input_x_q: X_Q,
                             self.input_y: Y,
                             self.dropout_keep_prob: dropout_rate}
                _, loss, logits = self.sess.run([self.train_op, self.loss, self.logits], feed_dict)

                if epoch == 0 and counter == 0:
                    embedding = self.sess.run(self.word_embeddings)
                    self.logger.info('embedding:\n %s' % embedding)

                if counter == 0:
                    all_logits = logits
                    all_Y = Y
                else:
                    all_logits = np.concatenate([all_logits, logits], axis=0)
                    all_Y = np.concatenate([all_Y, Y], axis=0)

                total_loss += loss

                counter += 1
                if counter >= steps_per_epoch:
                    embedding = self.sess.run(self.word_embeddings)
                    self.logger.info('embedding:\n %s' % embedding)

                    accuracy, precision, recall, f1score, f0_5score, f2score = \
                        self._get_a_p_r_f_more(input_y=all_Y, prediction=all_logits, category=1)
                    self.logger.info(
                        "Epoch %d\tBatch %d\tTrain Loss:%.4f\tAcc:%.4f\tPrecision:%.4f\tRecall:%.4f\tF1Score:%.4f\tF0_5Score:%.4f\tF2Score:%.4f"
                        % (epoch, counter, total_loss/float(counter), accuracy,
                           precision, recall, f1score, f0_5score, f2score)
                    )
                    self.logger.info('Epoch time: %sh: %sm: %ss'%(int((int(time.time()) - start_t) / 3600),
                                                                  int((int(time.time()) - start_t) % 3600 / 60),
                                                                  int((int(time.time()) - start_t) % 3600 % 60)
                                                                  ))
                    break

                if counter % int(steps_per_epoch/log_num_per_epoch) == 0:
                    accuracy, precision, recall, f1score, f0_5score, f2score = \
                        self._get_a_p_r_f_more(input_y=all_Y, prediction=all_logits, category=1)
                    self.logger.info(
                        "Epoch %d\tBatch %d\tTrain Loss:%.4f\tAcc:%.4f\tPrecision:%.4f\tRecall:%.4f\tF1Score:%.4f\tF0_5Score:%.4f\tF2Score:%.4f"
                        % (epoch, counter, total_loss / float(counter), accuracy,
                           precision, recall, f1score, f0_5score, f2score)
                    )
                    self.logger.info('Spend time: %sh: %sm: %ss' % (int((int(time.time()) - start_t) / 3600),
                                                                    int((int(time.time()) - start_t) % 3600 / 60),
                                                                    int((int(time.time()) - start_t) % 3600 % 60)
                                                                    ))

            self.logger.info("#" * 20)
            if if_evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))

                if val_data is not None:
                    loss, accuracy, precision, recall, f1score, f0_5score, f2score = \
                        self._evaluate_batch(load_feature_label, val_data, word_index, query_word_index, user_index, black_user_set, nb_classes, batch_size)
                    self.logger.info(
                        "Evaluate Loss:%.4f\tAcc:%.4f\tPrecision:%.4f\tRecall:%.4f\tF1Score:%.4f\tF0_5Score:%.4f\tF2Score:%.4f"
                        % (loss, accuracy, precision, recall, f1score, f0_5score, f2score)
                    )
                    self.logger.info('Evaluate time: %sh: %sm: %ss' % (int((int(time.time()) - start_t) / 3600),
                                                                    int((int(time.time()) - start_t) % 3600 / 60),
                                                                    int((int(time.time()) - start_t) % 3600 % 60)
                                                                    ))
                    # self.save(self.save_dir, self.model_name + '_epoch' + str(epoch))
                    if f1score >= max_val_f1 and save_best:
                        max_val_f1 = f1score
                        self.save(self.save_dir, self.model_name)
                else:
                    self.logger.warning('No dev data is loaded for evaluation in the dataset!')

                if test_data is not None:
                    loss, accuracy, precision, recall, f1score, f0_5score, f2score = \
                        self._evaluate_batch(load_feature_label, test_data, word_index, query_word_index, user_index, black_user_set, nb_classes, batch_size)
                    self.logger.info(
                        "Test Loss:%.4f\tAcc:%.4f\tPrecision:%.4f\tRecall:%.4f\tF1Score:%.4f\tF0_5Score:%.4f\tF2Score:%.4f"
                        % (loss, accuracy, precision, recall, f1score, f0_5score, f2score)
                    )
                    self.logger.info('Test time: %sh: %sm: %ss' % (int((int(time.time()) - start_t) / 3600),
                                                                    int((int(time.time()) - start_t) % 3600 / 60),
                                                                    int((int(time.time()) - start_t) % 3600 % 60)
                                                                    ))
                else:
                    self.logger.warning('No dev data is loaded for evaluation in the dataset!')

            self.logger.info('Epoch Done time: %sh: %sm: %ss' % (int((int(time.time()) - start_t) / 3600),
                                                                 int((int(time.time()) - start_t) % 3600 / 60),
                                                                 int((int(time.time()) - start_t) % 3600 % 60)
                                                                 ))
            self.logger.info("#" * 20)

        self.save(self.save_dir, self.model_name + '_last_epoch')

    def _evaluate_batch(self, load_feature_label, test_data, word_index, query_word_index, user_index, black_user_set, nb_classes, batch_size=32):
        counter, total_loss = 0, 0.0
        self.logger.info("Start to evaluate on batch.")

        all_logits = None
        all_Y = None

        batch_size = batch_size

        for Item, X, X_Q, User_Y, X_Y, Y, Length in load_feature_label(test_data, word_index=word_index,
                                                                       query_word_index=query_word_index,
                                                                       user_index=user_index,
                                                                       black_user_set=black_user_set,
                                                                       batch_size=batch_size):

            feed_dict = {self.input_x: Item,
                         self.input_x_i: X,
                         self.input_x_q: X_Q,
                         self.input_y: Y,
                         self.dropout_keep_prob: 1.0}

            loss, logits = self.sess.run([self.loss, self.logits], feed_dict)

            if counter == 0:
                all_logits = logits
                all_Y = Y
            else:
                all_logits = np.concatenate([all_logits, logits], axis=0)
                all_Y = np.concatenate([all_Y, Y], axis=0)

            total_loss += loss

            counter += 1

        accuracy, precision, recall, f1score, f0_5score, f2score = \
            self._get_a_p_r_f_more(input_y=all_Y, prediction=all_logits, category=1)

        return total_loss/float(counter), accuracy, precision, recall, f1score, f0_5score, f2score

    def evaluate_batch(self, load_feature_label, test_data, word_index, query_word_index, user_index, black_user_set, nb_classes, batch_size=32):
        start_t = int(time.time())
        self.logger.info("*"*20)
        self.logger.info("Start to evaluate on batch.")

        loss, accuracy, precision, recall, f1score, f0_5score, f2score, \
            = self._evaluate_batch(load_feature_label, test_data, word_index, query_word_index,
                                   user_index, black_user_set, nb_classes, batch_size=32)

        self.logger.info("Evaluate Loss on Test Data:%.4f\tAcc:%.4f\tPrecision:%.4f\tRecall:%.4f\tF1Score:%.4f\tF0_5score:%.4f\tF2_score:%.4f"
                         % (loss, accuracy, precision, recall, f1score, f0_5score, f2score))
        self.logger.info('Evaluate time: %sh: %sm: %ss' % (int((int(time.time()) - start_t) / 3600),
                                                           int((int(time.time()) - start_t) % 3600 / 60),
                                                           int((int(time.time()) - start_t) % 3600 % 60)
                                                           ))

        return loss, accuracy, precision, recall, f1score, f0_5score, f2score

    # 把所有session结合起来判断item的类别
    def predict_items(self, load_feature_label_inference, test_data,
                      word_index, query_word_index, user_index,
                      black_user_set, nb_classes, batch_size=32,
                      write=False, step=5):
            start_t = int(time.time())
            self.logger.info("*" * 20)
            self.logger.info("Start to evaluate on batch.")

            item_feature_dict = load_feature_label_inference(test_data, word_index=word_index,
                                                             query_word_index=query_word_index,
                                                             user_index=user_index,
                                                             black_user_set=black_user_set)

            self.logger.info("len of items %d." % len(item_feature_dict))

            item_dict = {}

            count = 0

            bugger = 0

            rate = 0.1

            predict_result_dict = {}

            for item, feature_label in item_feature_dict.items():

                feature = feature_label[0]
                label = feature_label[1]
                title = feature_label[2]

                item_dict[item] = label

                Item, X, X_Q, X_Y = feature[0], feature[1], feature[2], feature[3]

                num = 0

                output = None

                for i in range(0, len(Item), batch_size):
                    batch_Item = Item[i: i + batch_size]
                    batch_X = X[i: i + batch_size]
                    batch_X_Q = X_Q[i: i + batch_size]

                    if len(np.shape(batch_Item)) != 2 or len(np.shape(batch_X)) != 4 or len(np.shape(batch_X_Q)) != 3:
                        continue

                    feed_dict = {self.input_x: batch_Item,
                                 self.input_x_i: batch_X,
                                 self.input_x_q: batch_X_Q,
                                 self.dropout_keep_prob: 1.0}

                    batch_output = self.sess.run([self.output], feed_dict)

                    if num == 0:
                        output = batch_output[0]
                    else:
                        output = np.concatenate([output, batch_output[0]], axis=0)

                    num += 1

                try:
                    result = np.sum(output)
                    predict_result_dict[item] = result

                except:
                    count += 1
                    bugger += 1
                    continue

                count += 1

                process = float(count / len(item_feature_dict))
                if process > rate:
                    self.logger.info("Done process: %s."%process)
                    rate += 0.1
            print("There are %d bugger."%bugger)

            new_predict_result_dict = sorted(predict_result_dict.items(), key=lambda item: item[1], reverse=True)

            predict_label = []
            true_label = []

            for k, v in new_predict_result_dict:
                predict_label.append(1 if v > 0 else 0)
                true_label.append(item_dict[k])

            accuracy, precision, recall, f1score, f0_5score, f2score, map_score, ndcg_score = \
                get_a_p_r_f_more_more(np.array(true_label), np.array(predict_label), 1)
            self.logger.info(
                "Results: %.4f\tAcc:%.4f\tPrecision:%.4f\tRecall:%.4f\tF1Score:%.4f\tF0_5score:%.4f\tF2_score:%.4f\tmap_score:%.4f\tndcg_score:%.4f"
                % (0.0, accuracy, precision, recall, f1score, f0_5score, f2score, map_score, ndcg_score))

            self.logger.info('Evaluate time: %sh: %sm: %ss' % (int((int(time.time()) - start_t) / 3600),
                                                               int((int(time.time()) - start_t) % 3600 / 60),
                                                               int((int(time.time()) - start_t) % 3600 % 60)
                                                               ))

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        print(os.path.join(model_dir, model_prefix))
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))


if __name__ == '__main__':
    network = Network()
    network.build_graph()
