#! /user/bin/evn python3
# -*- coding:utf8 -*-

"""
utility
======

A function for utility.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@copyright: "Copyright (c) 2018 Guoxiu He. All Rights Reserved"
"""

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

import sys, traceback, resource
import tensorflow as tf
import numpy as np
import math
import re

import random
random.seed(1234)

import codecs


_NEG_INF = -1e9
_MIN_INF = 1e-15


def show_layer_info(layer_name, layer_out, logger=None):
    if logger:
        logger.info('[layer]: %s\t[shape]: %s'
                    % (layer_name, str(layer_out.get_shape().as_list())))
    else:
        print('[layer]: %s\t[shape]: %s'
              % (layer_name, str(layer_out.get_shape().as_list())))


def show_layer_info_with_memory(layer_name, layer_out, logger=None):
    if logger:
        logger.info('[layer]: %s\t[shape]: %s \n%s'
                    % (layer_name, str(layer_out.get_shape().as_list()), show_memory_use()))
    else:
        print('[layer]: %s\t[shape]: %s \n%s'
              % (layer_name, str(layer_out.get_shape().as_list()), show_memory_use()))


def shape(tensor):
    if None in tensor.get_shape().as_list():
        return tf.shape(tensor)
    else:
        return tensor.get_shape().as_list()


def show_memory_use():
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        rusage_denom = rusage_denom * rusage_denom
    ru = resource.getrusage(resource.RUSAGE_SELF)
    total_memory = 1. * (ru.ru_maxrss + ru.ru_ixrss +
                         ru.ru_idrss + ru.ru_isrss) / rusage_denom
    strinfo = "\x1b[33m [Memory] Total Memory Use: %.4f MB \t Resident: %ld Shared: %ld UnshareData: %ld UnshareStack: %ld \x1b[0m" % \
              (total_memory, ru.ru_maxrss, ru.ru_ixrss, ru.ru_idrss, ru.ru_isrss)
    return strinfo


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (
            class_str, traceback.format_exception(*sys.exc_info())))


def import_object(import_str, *args, **kwargs):
    return import_class(import_str)(*args, **kwargs)


def import_module(import_str):
    __import__(import_str)
    return sys.modules[import_str]


def img2array(image):
    return img_to_array(image)


def get_a_p_r_f(target, predict, category):
    idx = np.array(range(len(target)))
    _target = set(idx[target == category])
    _predict = set(idx[predict == category])
    true = _target & _predict
    accuracy = float(np.sum(np.array(target) == np.array(predict))) / float(len(idx))
    precision = len(true) / float(len(_predict) + _MIN_INF)
    recall = len(true) / float(len(_target) + _MIN_INF)
    f1_score = precision * recall * 2 / (precision + recall + _MIN_INF)
    return accuracy, precision, recall, f1_score


def get_a_p_r_f_more_more(target, predict, category):
    idx = np.array(range(len(target)))
    _target = set(idx[target == category])
    _predict = set(idx[predict == category])
    true = _target & _predict
    accuracy = float(np.sum(np.array(target)==np.array(predict)))/float(len(idx))
    precision = len(true) / float(len(_predict) + _MIN_INF)
    recall = len(true) / float(len(_target) + _MIN_INF)
    f1_score = precision * recall * 2 / (precision + recall + _MIN_INF)
    f0_5_score = precision * recall * (1+0.5*0.5) / (0.5*0.5*precision + recall + _MIN_INF)
    f2_score = precision * recall * (1+2*2) / (2*2*precision + recall + _MIN_INF)
    map_score = map(target, predict)
    ndcg_score = ndcg(target, predict)
    return accuracy, precision, recall, f1_score, f0_5_score, f2_score, map_score, ndcg_score


def get_a_p_r_f_more(target, predict, category):
    idx = np.array(range(len(target)))
    _target = set(idx[target == category])
    _predict = set(idx[predict == category])
    true = _target & _predict
    accuracy = float(np.sum(np.array(target)==np.array(predict)))/float(len(idx))
    precision = len(true) / float(len(_predict) + _MIN_INF)
    recall = len(true) / float(len(_target) + _MIN_INF)
    f1_score = precision * recall * 2 / (precision + recall + _MIN_INF)
    f0_5_score = precision * recall * (1+0.5*0.5) / (0.5*0.5*precision + recall + _MIN_INF)
    f2_score = precision * recall * (1+2*2) / (2*2*precision + recall + _MIN_INF)
    return accuracy, precision, recall, f1_score, f0_5_score, f2_score


def map(y_true, y_pred):
    c = zip(y_true, y_pred)
    ipos = 0.
    s = 0.
    for i, (g, p) in enumerate(c):
        if g > 0:
            ipos += 1.
            if p > 0:
                s += ipos / (1. + i)
    if ipos == 0:
        return 0.
    else:
        return s / ipos


def ndcg(y_true, y_pred):
    c = zip(y_true, y_pred)
    ipos = 0.
    dcg = 0.
    idcg = 0.

    for i, (g, p) in enumerate(c):
        if p > 0:
            ipos += 1
            dcg += (pow(2, g) - 1) / math.log2(i+2)

    for i in range(int(np.sum(y_true))):
        idcg += (pow(2, 1) - 1) / math.log2(i+2)

    if ipos == 0 or idcg == 0:
        return 0.
    else:
        return dcg / idcg


def ndcg_tmp(y_true, y_pred, k=None):
    if not k:
        k = len(y_pred)
    s = 0.
    c = zip(y_true, y_pred)
    c_g = sorted(c, key=lambda x: x[0], reverse=True)
    c_p = sorted(c, key=lambda x: x[1], reverse=True)
    idcg = np.zeros([k], dtype=np.float32)
    dcg = np.zeros([k], dtype=np.float32)
    for i, (g, p) in enumerate(c_g):
        if g > 0:
            idcg[i:] += (math.pow(2., g) - 1.) / math.log(2. + i)
        if i >= k:
            break
    for i, (g, p) in enumerate(c_p):
        if g > 0:
            dcg[i:] += (math.pow(2., g) - 1.) / math.log(2. + i)
        if i >= k:
            break
    for idx, v in enumerate(idcg):
        if v == 0.:
            dcg[idx] = 0.
        else:
            dcg[idx] /= v
    return dcg


def asarray(sequence):
    return np.asarray(sequence, dtype=np.int32)


def split_data(feature, target, split_rate=0.2):
    num = len(feature)
    idx = range(num)
    random.shuffle(idx)
    test_feature = feature[idx[:int(num*split_rate)]]
    test_target = target[idx[:int(num*split_rate)]]
    dev_feature = feature[idx[int(num*split_rate): int(num*2*split_rate)]]
    dev_target = target[idx[int(num*split_rate): int(num*2*split_rate)]]
    train_feature = feature[idx[int(num*2*split_rate):]]
    train_target = target[idx[int(num*2*split_rate):]]
    print('splited successfully!')
    return train_feature, train_target, dev_feature, dev_target, test_feature, test_target


def get_pos_encoding_matrix(max_len, embedding_dim):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / embedding_dim) for j in range(embedding_dim)]
        if pos != 0 else np.zeros(embedding_dim)
            for pos in range(max_len)
    ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
    return pos_enc


def get_position_encoding_matrix(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
        length: Sequence length.
        hidden_size: Size of the
        min_timescale: Minimum scale that will be applied at each position
        max_timescale: Maximum scale that will be applied at each position

    Returns:
        Tensor with shape [length, hidden_size]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


def get_embedding_matrix(embedding_path, word_index, embedding_dims):
    print('Preparing embedding matrix')

    # nb_words = min(max_features, len(word_index)) + 1
    nb_words = len(word_index) + 1
    # embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    numpy_rng = np.random.RandomState(4321)
    embedding_matrix = numpy_rng.uniform(low=-0.05, high=0.05, size=(nb_words, embedding_dims))
    embeddings_from_file = {}
    miss_count = 0
    with codecs.open(embedding_path, encoding='utf8') as embedding_file:
        embedding_file.readline()
        while True:
            try:
                line = embedding_file.readline()
                if not line:
                    break
                fields = line.strip().split(' ')
                word = fields[0]
                vector = np.array(fields[1:], dtype='float32')
                embeddings_from_file[word] = vector
            except:
                miss_count += 1
            # print(line)

    print('num of embedding file is ', len(embeddings_from_file))
    count = 0
    embedding_matrix[0] = embeddings_from_file['</s>']
    for word, i in word_index.items():
        if word in embeddings_from_file:
            embedding_matrix[i] = embeddings_from_file[word]
            count += 1
        # else:
            # print(word)
    print('miss word embedding is', miss_count)
    print('nb words is', nb_words)
    print('num of word embeddings:', count)
    print('Null word embeddings: %d' % (nb_words - count))

    return embedding_matrix


def convert_sequence(text, filter_json=None):
    sequence = text_to_word_sequence(text)
    if filter_json == None:
        return ' '.join(sequence)
    new_sequence = [term for term in sequence if term in filter_json]
    return ' '.join(new_sequence)


def get_all_para(file_path, save_path):
    fw = open(save_path, 'a')
    with codecs.open(file_path, encoding='utf8') as fp:
        while True:
            line = fp.readline()
            if not line:
                fw.close()
                return
            text = line.strip().split('\t')[-1]
            text = convert_sequence(text.encode('utf8'))
            fw.write(text+' ')


def convert(line):
    line = line.strip().split()
    return line


def get_len(line):
    return len(line)


def get_tokenizer(text, num_words=None):
    if num_words == None:
        tokenizer = Tokenizer(num_words=200000)
    else:
        tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(text)
    return tokenizer


def get_sequences(tokenizer, text):
    sequences = tokenizer.texts_to_sequences(text)
    return sequences


def get_padded_sequences(sequences, maxlen=None):
    if maxlen == None:
        return pad_sequences(sequences, maxlen=100)
    else:
        return pad_sequences(sequences, maxlen)


def get_categorical(labels, num_classes=5):
    print("start to categorical...")
    return to_categorical(labels, num_classes)


def sent_seg(sentence):
    temp = re.split(u'[.!?。！？……；;\n\s]+', sentence)
    result = []
    for i in temp:
        if i.strip():
            result.append(i.strip())
    return result


def run_sent_split():
    sentence = u'我爱我的祖国。我的祖国是中国.我叫贺国秀……你觉得你可以叫我国修么？就是这样!'
    result = sent_seg(sentence.strip())
    print(result)


def get_padding(x, padding_value=0):
    """Return float tensor representing the padding values in x.

    Args:
        x: int tensor with any shape
        padding_value: int value that

    Returns:
        flaot tensor with same shape as x containing values 0 or 1.
          0 -> non-padding, 1 -> padding
    """
    with tf.name_scope("padding"):
        return tf.to_float(tf.equal(x, padding_value))


def get_padding_bias(x):
    """Calculate bias tensor from padding values in tensor.

    Bias tensor that is added to the pre-softmax multi-headed attention logits,
    which has shape [batch_size, num_heads, length, length]. The tensor is zero at
    non-padding locations, and -1e9 (negative infinity) at padding locations.

    Args:
        x: int tensor with shape [batch_size, length]

    Returns:
        Attention bias tensor of shape [batch_size, 1, 1, length].
    """
    with tf.name_scope("attention_bias"):
        padding = get_padding(x)
        attention_bias = padding * _NEG_INF
        attention_bias = tf.expand_dims(
            tf.expand_dims(attention_bias, axis=1), axis=1
        )
    return attention_bias


def pad_sentence(sentence, maxlen, padding_value=''):
    if len(sentence) >= maxlen:
        return sentence[-maxlen:]
    else:
        return [padding_value]*(maxlen-len(sentence))+sentence


def pad_paragraph(paragraph, maxlen, padding_value=''):
    if len(paragraph) >= maxlen:
        return paragraph[-maxlen:]
    else:
        return [[padding_value]*maxlen]*(maxlen-len(paragraph))+paragraph


def print_trainable_variables(output_detail, logger):
    total_parameters = 0
    parameters_string = ""

    for variable in tf.trainable_variables():
        logger.info("%s" %(variable.name))
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d\n" % (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d\n" % (variable.name, str(shape), variable_parameters))

    if logger:
        if output_detail:
            logger.info('\n' + parameters_string)
        logger.info("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))
    else:
        if output_detail:
            print('\n' + parameters_string)
        print("Total %d variables, %s params" % (len(tf.trainable_variables()), "{:,}".format(total_parameters)))


def print_all_variables(output_detail, logger=None):
    total_parameters = 0
    parameters_string = ""

    for variable in tf.all_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
        if len(shape) == 1:
            parameters_string += ("%s %d\n" % (variable.name, variable_parameters))
        else:
            parameters_string += ("%s %s=%d\n" % (variable.name, str(shape), variable_parameters))

    if logger is not None:
        if output_detail:
            logger.info('\n' + parameters_string)
        logger.info("Total %d variables, %s params" % (len(tf.all_variables()), "{:,}".format(total_parameters)))
    else:
        if output_detail:
            print('\n' + parameters_string)
        print("Total %d variables, %s params" % (len(tf.all_variables()), "{:,}".format(total_parameters)))


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.to_float(learning_rate_warmup_steps)
        step = tf.to_float(tf.train.get_or_create_global_step())

        learning_rate *= (hidden_size ** -0.5)
        # Apply linear warmup
        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        # Apply rsqrt decay
        learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

        # Create a named tensor that will be logged using the logging hook.
        # The full name includes variable and names scope. In this case, the name
        # is model/get_train_op/learning_rate/learning_rate
        tf.identity(learning_rate, "learning_rate")

    return learning_rate


def save_model(sess, signature, path):
    builder = tf.saved_model.builder.SavedModelBuilder(path)
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict': signature
        },
        legacy_init_op=tf.saved_model.main_op.main_op()
    )
    builder.save()


def find_top_atten(para, index):
    '''

    :param para: 段落的二维矩阵，其中第一维是句子，第二维是词
    :param index: 句子的下标
    :return: 段落中对应下标的句子
    '''
    return ''.join(para[index])


def KL(P, Q):
    P = np.array(P)
    Q = np.array(Q)

    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P+epsilon
    Q = Q+epsilon

    divergence = np.sum(P*np.log(P/Q))
    return divergence


if __name__ == "__main__":
    pass

