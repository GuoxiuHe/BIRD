#! /user/bin/evn python3
# -*- coding:utf8 -*-

"""
Main
======

This module prepares and runs the whole system.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@copyright: "Copyright (c) 2018 Guoxiu He. All Rights Reserved"
"""

import os, sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
rootdir = '/'.join(curdir.split('/')[:4])
PRO_NAME = 'BIRD'
prodir = rootdir + '/Research/' + PRO_NAME
sys.path.insert(0, prodir)

import logging, argparse, time
import tensorflow as tf
tf.set_random_seed(1234)

import json
from utils.data_loader import Data_Loader

from Scripts.Network import Network
from Scripts.networks.BIRD import BIRD

CONFIG_ROOT = prodir + '/Scripts/config/'

def main():

    start_t = time.time()
    # Obtain arguments from system
    parser = argparse.ArgumentParser('Params for Anti Spam')
    parser.add_argument('--phase', default='train',
                        help='Phase: Can be train or predict, the default value is train.')
    parser.add_argument('--data_name', default='secure',
                        help='Data_Name: The data you will use.')
    parser.add_argument('--model_name', default='network',
                        help='Model_Name: The model you will use.')
    parser.add_argument('--model_path', default='None',
                        help='Model_Path: The model path you will load.')
    parser.add_argument('--memory', default='0.5',
                        help='Memory: The gpu memory you will use.')
    parser.add_argument('--gpu', default='0',
                        help='GPU: Which gpu you will use.')
    parser.add_argument('--log_path', default='./logs/',
                        help='path of the log file. If not set, logs are printed to console.')
    parser.add_argument('--suffix', default='',
                        help='suffix for differentiate log.')

    args = parser.parse_args()

    logger = logging.getLogger("AntiSpam_Secure")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    now_time = '_'.join(time.asctime(time.localtime(time.time())).split(' ')[:3])

    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    # log_path = ''
    if len(args.suffix)>0:
        log_path = args.log_path + args.model_name + '.' + args.data_name + \
                   '.' + args.phase + '.' + now_time + args.suffix + '.log'
    else:
        log_path = args.log_path + args.model_name + '.' + args.data_name + \
                   '.' + args.phase + '.' + now_time + '.log'

    if os.path.exists(log_path):
        os.remove(log_path)

    if args.log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # print('Running with args : {}'.format(args))
    logger.info('Running with args : {}'.format(args))

    # get object named data_loader
    data_loader = Data_Loader()

    # print('Load data_set and vocab...')
    logger.info('Load data_set and vocab...')

    data_config_path = CONFIG_ROOT + 'data/config.' + args.data_name + '.json'
    model_config_path = CONFIG_ROOT + 'model/config.' + args.model_name + '.json'
    data_config = data_loader.load_config(data_config_path)
    model_config = data_loader.load_config(model_config_path)

    logger.info('Data config is %s' %data_config)
    logger.info('Model config is %s' %model_config)

    # Get config param
    model_name = model_config['model_name']
    batch_size = model_config['batch_size']
    epochs = model_config['epochs']
    dropout_rate = model_config['dropout_rate']

    data_name = data_config['data_name']
    train_size = data_config['train_size']
    nb_classes = data_config['nb_classes']
    word_index_path = data_loader.data_root + 'dictionary/' + data_name + '/word_index.json'
    if_evaluate = model_config['evaluate']

    train_set_path = data_loader.data_root + data_name + '/train_list.txt'
    val_set_path = data_loader.data_root + data_name + '/val_list.txt'
    test_set_path = data_loader.data_root + data_name + '/test_list.txt'
    online_test_set_path = data_loader.data_root + data_name + '/online_test_1.txt'
    online_test_set_path2 = data_loader.data_root + data_name + '/online_test_2.txt'

    memory = float(args.memory)
    logger.info("Memory in train %s."%memory)

    # Get word index dictionary
    with open(word_index_path, 'r') as fp:
        word_index = json.load(fp)

    # Get Network Framework
    if model_name == 'Network':
        network = Network(memory=memory)
    elif model_name == 'BIRD':
        network = BIRD(memory=memory)
    else:
        logger.info("The error comes from model selection.")
        return

    # Set param for netwrok
    network.set_nb_words(len(word_index) + 1)
    network.set_nb_query_words(0)
    network.set_nb_users(0)
    network.set_data_name(data_name)
    network.set_name(model_name+args.suffix)
    network.set_from_model_config(model_config)
    network.set_from_data_config(data_config)

    network.build_graph()

    logger.info('All values in the Network are %s' % network.__dict__)

    if args.phase == 'train':
        logger.info("Start train.")

        if args.model_path != 'None':
            network.saver.restore(network.sess, args.model_path)

        train(network, data_loader.session_group_user_query_item_generator_hiera,
              data_loader.load_session_group_user_query_item_label_hiera,
              train_set_path, val_set_path, test_set_path, word_index, None, None, None,
              dropout_rate, nb_classes, epochs, train_size, batch_size,
              if_evaluate=if_evaluate, save_best=True)

        logger.info("#" * 20)
        evaluate_batch(network, data_loader.load_session_group_user_query_item_label_hiera, val_set_path,
                       word_index, None, None, None, nb_classes, batch_size)

        logger.info("#" * 20)
        evaluate_batch(network, data_loader.load_session_group_user_query_item_label_hiera, test_set_path,
                       word_index, None, None, None, nb_classes, batch_size)

    elif args.phase == 'evaluate_val':
        logger.info("#"*20)
        evaluate_batch(network, data_loader.load_session_group_user_query_item_label_hiera, val_set_path,
                       word_index, None, None, None, nb_classes, batch_size)

    elif args.phase == 'evaluate_test':
        logger.info("#" * 20)
        evaluate_batch(network, data_loader.load_session_group_user_query_item_label_hiera, test_set_path,
                       word_index, None, None, None, nb_classes, batch_size)

    elif args.phase == 'predict_items':
        logger.info("#" * 20)
        logger.info("evaluate predict items.")
        predict_items(network, data_loader.load_session_group_user_query_item_label_inference, online_test_set_path,
                      word_index, None, None, None, nb_classes, batch_size, is_last=False)

    elif args.phase == 'predict_items2':
        logger.info("#" * 20)
        logger.info("evaluate predict items.")
        predict_items(network, data_loader.load_session_group_user_query_item_label_inference, online_test_set_path2,
                      word_index, None, None, None, nb_classes, batch_size, is_last=False)

    else:
        logger.info("The error comes from phase Selection.")
        return

    logger.info('The whole program spends time: %sh: %sm: %ss' % (int((int(time.time()) - start_t) / 3600),
                                                                  int((int(time.time()) - start_t) % 3600 / 60),
                                                                  int((int(time.time()) - start_t) % 3600 % 60)
                                                                  )
                )


def train(network, data_generator, load_feature_label,
          train_data, val_data, test_data, word_index, query_word_index, user_index, black_user_set,
          dropout_rate, nb_classes, epochs, train_size, batch_size,
          if_evaluate=None, save_best=True, *args, **kwargs):

    if 'load_feature_label_inference' in kwargs:
        network.train(data_generator, load_feature_label,
                      train_data, val_data, test_data, word_index, query_word_index, user_index, black_user_set,
                      dropout_rate, nb_classes, epochs, train_size, batch_size,
                      if_evaluate=if_evaluate, save_best=save_best,
                      load_feature_label_inference=kwargs['load_feature_label_inference'])
    else:
        network.train(data_generator, load_feature_label,
                      train_data, val_data, test_data, word_index, query_word_index, user_index, black_user_set,
                      dropout_rate, nb_classes, epochs, train_size, batch_size,
                      if_evaluate=if_evaluate, save_best=save_best)

    network.restore(network.save_dir, network.model_name)

    network.evaluate_batch(load_feature_label, test_data,
                           word_index, query_word_index, user_index, black_user_set,
                           nb_classes, batch_size)


def evaluate_batch(network, load_feature_label, val_data,
                   word_index, query_word_index, user_index, black_user_set,
                   nb_classes, batch_size):

    network.restore(network.save_dir, network.model_name)
    print("Restore successfully.")

    network.evaluate_batch(load_feature_label, val_data,
                           word_index, query_word_index, user_index, black_user_set,
                           nb_classes, batch_size)


def predict_items(network, load_feature_label, val_data,
                  word_index, query_word_index, user_index, black_user_set,
                  nb_classes, batch_size, is_last=False):

    if is_last:
        network.restore(network.save_dir, network.model_name + '_last_epoch')
    else:
        network.restore(network.save_dir, network.model_name)

    print("Restore successfully.")

    network.predict_items(load_feature_label, val_data,
                          word_index, query_word_index, user_index, black_user_set,
                          nb_classes, batch_size)


if __name__ == '__main__':
    main()
