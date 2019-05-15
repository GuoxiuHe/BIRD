#! /user/bin/evn python3
# -*- coding:utf8 -*-

"""
data loader
======

A class for data loader.

@author: Guoxiu He
@contact: guoxiu.he@whu.edu.cn
@site: https://frankblood.github.io
@copyright: "Copyright (c) 2018 Guoxiu He. All Rights Reserved"
"""

import os, sys, shutil, copy

curdir = os.path.dirname(os.path.abspath(__file__))
print(curdir)
sys.path.insert(0, os.path.dirname(curdir))
rootdir = '/'.join(curdir.split('/')[:4])
PRO_NAME = 'BIRD'
prodir = rootdir + '/Research/' + PRO_NAME
sys.path.insert(0, prodir)

import random, json, pickle, argparse
random.seed(1234)
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re

class Data_Loader(object):
    def __init__(self):
        self.data_root = prodir + '/Dataset/'

    def load_session_group_user_query_item_label_hiera(self, data_path, word_index,
                                                       query_word_index, user_index,
                                                       black_user_set, nb_classes=2,
                                                       batch_size=20, **kwargs):
        Item = []
        X, Y = [], []
        X_Q = []
        User_Y, X_Y = [], []
        Length = []

        count = 0
        num = 0

        query_maxlen = 0
        title_maxsize = 0
        title_maxlen = 0
        query_maxsize = 0

        with open(data_path, 'r') as fp:
            while True:
                line = fp.readline()

                if not line:
                    if len(X_Q) > 0:

                        new_X_Q = list(map(lambda x: pad_sequences(x, query_maxlen), X_Q))
                        new_X = list(map(lambda x: pad_sequences(list(map(lambda y: pad_sequences(y,
                                                                                                  title_maxlen),
                                                                          x)),
                                                                 title_maxsize),
                                         X))
                        query_maxsize = 20 if query_maxsize > 20 else query_maxsize

                        yield pad_sequences(Item), \
                              pad_sequences(new_X, query_maxsize), \
                              pad_sequences(new_X_Q, query_maxsize), \
                              to_categorical(User_Y, num_classes=nb_classes), \
                              to_categorical(X_Y, num_classes=nb_classes), \
                              to_categorical(Y, num_classes=nb_classes), \
                              np.array(Length)
                    break

                try:
                    line = json.loads(line.strip())
                except:
                    print(line)
                    continue

                if len(line) != 5:
                    print(line)
                    continue

                user_id = line['user_id']
                item_id = line['item_id']
                title = line['title']
                session = line['session']
                label = line['label']

                item_title = []
                title_seq = []
                query_seq = []

                for word in title.strip().split():
                    if re.match('^[A-Za-z0-9]+$', word):
                        continue
                    try:
                        item_title.append(word_index[word])
                    except:
                        pass
                Item.append(item_title)

                for pv in session:
                    query = pv[0]
                    items = pv[1]

                    query_id = []

                    for word in query.strip().split():
                        if re.match('^[A-Za-z0-9]+$', word):
                            continue
                        try:
                            query_id.append(word_index[word])
                        except:
                            pass
                    query_seq.append(query_id)
                    query_maxlen = len(query_id) if len(query_id) > query_maxlen else query_maxlen

                    title_tmp = []
                    for title in items:
                        if len(title) == 0:
                            continue
                        title_id = []
                        for word in title.strip().split():
                            if re.match('^[A-Za-z0-9]+$', word):
                                continue
                            try:
                                title_id.append(word_index[word])
                            except:
                                # title_id.append(0)
                                pass
                        title_tmp.append(title_id)
                        title_maxlen = len(title_id) if len(title_id) > title_maxlen else title_maxlen
                    title_maxsize = len(title_tmp) if len(title_tmp) > title_maxsize else title_maxsize
                    title_seq.append(title_tmp)
                query_maxsize = len(title_seq) if len(title_seq) > query_maxsize else query_maxsize

                User_Y.append([1])

                X.append(title_seq)
                X_Q.append(query_seq)

                if label == 'black_list':
                    X_Y.append([1])
                    Y.append([1])
                else:
                    X_Y.append([0])
                    Y.append([0])

                Length.append(len(title_seq))

                count += 1

                if count == batch_size:
                    new_X_Q = list(map(lambda x: pad_sequences(x, query_maxlen), X_Q))
                    new_X = list(map(lambda x: pad_sequences(list(map(lambda y: pad_sequences(y,
                                                                                              title_maxlen),
                                                                      x)),
                                                             title_maxsize),
                                     X))

                    query_maxsize = 20 if query_maxsize > 20 else query_maxsize

                    yield pad_sequences(Item), \
                          pad_sequences(new_X, query_maxsize), \
                          pad_sequences(new_X_Q, query_maxsize), \
                          to_categorical(User_Y, num_classes=nb_classes), \
                          to_categorical(X_Y, num_classes=nb_classes), \
                          to_categorical(Y, num_classes=nb_classes), \
                          np.array(Length)

                    Item = []
                    X, Y = [], []
                    X_Q = []
                    User_Y, X_Y = [], []
                    Length = []

                    query_maxlen = 0
                    title_maxsize = 0
                    title_maxlen = 0
                    query_maxsize = 0
                    count = 0

                    num += 1

    def load_session_group_user_query_item_label_inference(self, data_path, word_index,
                                                           query_word_index, user_index,
                                                           black_user_set, nb_classes=2, **kwargs):

        item_feature_dict = {}

        count = 0

        query_maxlen = 0
        title_maxsize = 0
        title_maxlen = 0
        query_maxsize = 0

        with open(data_path, 'r') as fp:
            while True:
                line = fp.readline()

                if not line:
                    new_item_feature_dict = {}
                    num = 0
                    for k, v in item_feature_dict.items():
                        query_maxlen = 20 if query_maxlen > 20 else query_maxlen
                        title_maxsize = 20 if title_maxsize > 20 else title_maxsize
                        title_maxlen = 20 if title_maxlen > 20 else title_maxlen
                        query_maxsize = 20 if query_maxsize > 20 else query_maxsize

                        try:

                            new_Item = pad_sequences(v[0][0], title_maxsize)
                            new_X = pad_sequences(list(map(lambda x: pad_sequences(list(map(lambda y: pad_sequences(y,
                                                                                                                    title_maxlen),
                                                                                            x)),
                                                                                   title_maxsize),
                                                           v[0][1])),
                                                  query_maxsize)

                            new_X_Q = pad_sequences(list(map(lambda x: pad_sequences(x,
                                                                                     query_maxlen),
                                                             v[0][2])),
                                                    query_maxsize)

                            new_X_Y = to_categorical(v[0][3], num_classes=nb_classes)
                            new_item_feature_dict[k] = [[new_Item, new_X, new_X_Q, new_X_Y], v[1], v[2]]
                        except:
                            num += 1
                            # print(v)

                    print("There are %d samples have problem."%num)
                    return new_item_feature_dict

                try:
                    line = json.loads(line.strip(), encoding='utf8')
                except:
                    print(line)
                    continue

                if len(line) != 5:
                    print(line)
                    continue

                user_id = line['user_id']
                item_id = line['item_id']
                title = line['title']
                session = line['session']
                label = line['label']

                item_title = []
                title_seq = []
                query_seq = []

                for word in title.strip().split():
                    if re.match('^[A-Za-z0-9]+$', word):
                        continue
                    try:
                        item_title.append(word_index[word])
                    except:
                        # item_title.append(0)
                        pass

                for pv in session:
                    query = pv[0]
                    items = pv[1]

                    query_id = []

                    for word in query.strip().split():
                        if re.match('^[A-Za-z0-9]+$', word):
                            continue
                        try:
                            query_id.append(word_index[word])
                        except:
                            # query_id.append(0)
                            pass
                    query_seq.append(query_id)
                    query_maxlen = len(query_id) if len(query_id) > query_maxlen else query_maxlen

                    title_tmp = []
                    for title in items:
                        if len(title) == 0:
                            continue
                        title_id = []
                        for word in title.strip().split():
                            if re.match('^[A-Za-z0-9]+$', word):
                                continue
                            try:
                                title_id.append(word_index[word])
                            except:
                                # title_id.append(0)
                                pass
                        title_tmp.append(title_id)
                        title_maxlen = len(title_id) if len(title_id) > title_maxlen else title_maxlen
                    title_maxsize = len(title_tmp) if len(title_tmp) > title_maxsize else title_maxsize
                    title_seq.append(title_tmp)

                query_maxsize = len(query_seq) if len(query_seq) > query_maxsize else query_maxsize

                if item_id in item_feature_dict and label == 'black_list':
                    item_feature_dict[item_id][0][0].append(item_title)
                    item_feature_dict[item_id][0][1].append(title_seq)
                    item_feature_dict[item_id][0][2].append(query_seq)
                    item_feature_dict[item_id][0][3].append([1])

                elif item_id in item_feature_dict and label == 'white_list':
                    item_feature_dict[item_id][0][0].append(item_title)
                    item_feature_dict[item_id][0][1].append(title_seq)
                    item_feature_dict[item_id][0][2].append(query_seq)
                    item_feature_dict[item_id][0][3].append([0])

                elif item_id not in item_feature_dict and label == 'black_list':
                    item_feature_dict[item_id] = [[[item_title],
                                                   [title_seq],
                                                   [query_seq],
                                                   [[1]]
                                                   ],
                                                  1,
                                                  title]

                else:
                    item_feature_dict[item_id] = [[[item_title],
                                                   [title_seq],
                                                   [query_seq],
                                                   [[0]]
                                                   ],
                                                  0,
                                                  title]

                count += 1

    def session_group_user_query_item_generator_hiera(self, data_path, word_index,
                                                      query_word_index, user_index, black_user_set,
                                                      nb_classes, batch_size=400,
                                                      shuffle=True, **kwargs):

        Item = []
        X, Y = [], []
        X_Q = []
        User_Y, X_Y = [], []
        Length = []

        count = 0

        fp = open(data_path, 'r')
        print("Open successfully.")

        query_maxlen = 0
        title_maxsize = 0
        title_maxlen = 0
        query_maxsize = 0

        while True:
            line = fp.readline()
            if not line:
                fp.close()
                fp = open(data_path, 'r')

            try:
                line = json.loads(line.strip(), encoding='utf8')
            except:
                print(line)
                continue

            if len(line) != 5:
                print(line)
                continue

            user_id = line['user_id']
            item_id = line['item_id']
            title = line['title']
            session = line['session']
            label = line['label']

            item_title = []
            title_seq = []
            query_seq = []

            for word in title.strip().split():
                if re.match('^[A-Za-z0-9]+$', word):
                    continue
                try:
                    item_title.append(word_index[word])
                except:
                    # item_title.append(0)
                    pass
            Item.append(item_title)

            for pv in session:
                query = pv[0]
                items = pv[1]

                query_id = []

                for word in query.strip().split():
                    if re.match('^[A-Za-z0-9]+$', word):
                        continue
                    try:
                        query_id.append(word_index[word])
                    except:
                        # query_id.append(0)
                        pass
                query_seq.append(query_id)
                query_maxlen = len(query_id) if len(query_id) > query_maxlen else query_maxlen

                title_tmp = []
                for title in items:
                    if len(title) == 0:
                        continue
                    title_id = []
                    for word in title.strip().split():
                        if re.match('^[A-Za-z0-9]+$', word):
                            continue
                        try:
                            title_id.append(word_index[word])
                        except:
                            # title_id.append(0)
                            pass
                    title_tmp.append(title_id)
                    title_maxlen = len(title_id) if len(title_id) > title_maxlen else title_maxlen
                title_maxsize = len(title_tmp) if len(title_tmp) > title_maxsize else title_maxsize
                title_seq.append(title_tmp)

            query_maxsize = len(query_seq) if len(query_seq) > query_maxsize else query_maxsize

            User_Y.append([1])

            X.append(title_seq)
            X_Q.append(query_seq)

            if label == 'black_list':
                X_Y.append([1])
                Y.append([1])
            else:
                X_Y.append([0])
                Y.append([0])

            Length.append(len(title_seq))

            count += 1

            if count == batch_size:
                new_X_Q = list(map(lambda x: pad_sequences(x, query_maxlen), X_Q))
                new_X = list(map(lambda x: pad_sequences(list(map(lambda y: pad_sequences(y,
                                                                                          title_maxlen),
                                                                  x)),
                                                         title_maxsize),
                                 X))

                query_maxsize = 20 if query_maxsize > 20 else query_maxsize

                yield pad_sequences(Item), \
                      pad_sequences(new_X, query_maxsize), \
                      pad_sequences(new_X_Q, query_maxsize), \
                      to_categorical(User_Y, num_classes=nb_classes), \
                      to_categorical(X_Y, num_classes=nb_classes), \
                      to_categorical(Y, num_classes=nb_classes), \
                      np.array(Length)

                Item = []
                X, Y = [], []
                X_Q = []
                User_Y, X_Y = [], []
                Length = []

                query_maxlen = 0
                title_maxsize = 0
                title_maxlen = 0
                query_maxsize = 0
                count = 0

    def load_config(self, config_path):
        with open(config_path, 'r') as fp:
            return json.load(fp)

    def load_json(self, json_path):
        with open(json_path, 'r') as fp:
            return json.load(fp)

    def save_to_json(self, json_data, json_path):
        with open(json_path, "w") as f:
            json.dump(json_data, f, ensure_ascii=False)

    def load_pickle(self, pickle_path):
        with open(pickle_path, 'rb+') as fp:
            return pickle.load(fp)

    def save_to_pickle(self, pickle_data, pickle_path):
        with open(pickle_path, "wb+") as f:
            pickle.dump(pickle_data, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='read_data', help='Please enter what function you will use.')
    args = parser.parse_args()
    if args.phase == 'pass':
        pass


if __name__ == '__main__':
    main()