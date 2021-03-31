#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 17:22 
# ide： PyCharm
import keras
import pandas as pd
import numpy as np
import random
import os

from bert4keras.snippets import DataGenerator, sequence_padding
from keras.utils import to_categorical
# from keras_bert import Tokenizer
from utils.common_tools import save_json, load_json
from tqdm import tqdm


def fit_process(self, embedding_type, path, embed, rate=1, shuffle=True):
    data = pd.read_csv(path)
    ques = data['ques'].tolist()
    label = data['label'].tolist()
    ques = [str(q).upper() for q in ques]
    label = [str(l).upper() for l in label]
    if shuffle:
        ques = np.array(ques)
        label = np.array(label)
        indexs = [ids for ids in range(len(label))]
        random.shuffle(indexs)
        ques, label = ques[indexs].tolist(), label[indexs].tolist()
    # 如果label2index存在则不转换了
    if not os.path.exists(self.path_fast_text_model_l2i_i2l):
        label_set = set(label)
        count = 0
        label2index = {}
        index2label = {}
        for label_one in label_set:
            label2index[label_one] = count
            index2label[count] = label_one
            count = count + 1

        l2i_i2l = {}
        l2i_i2l['l2i'] = label2index
        l2i_i2l['i2l'] = index2label
        save_json(l2i_i2l, self.path_fast_text_model_l2i_i2l)
    else:
        l2i_i2l = load_json(self.path_fast_text_model_l2i_i2l)

    len_ql = int(rate * len(ques))
    if len_ql <= 500:  # sample时候不生效,使得语料足够训练
        len_ql = len(ques)

    x = []
    print("ques to index start!")
    ques_len_ql = ques[0:len_ql]
    for i in tqdm(range(len_ql)):
        que = ques_len_ql[i]
        que_embed = embed.sentence2idx(que)
        x.append(que_embed)  # [[], ]
    label_zo = []
    print("label to onehot start!")
    label_len_ql = label[0:len_ql]
    for j in tqdm(range(len_ql)):
        label_one = label_len_ql[j]
        label_zeros = [0] * len(l2i_i2l['l2i'])
        label_zeros[l2i_i2l['l2i'][label_one]] = 1
        label_zo.append(label_zeros)

    count = 0
    if embedding_type in ['bert', 'albert']:
        x_, y_ = np.array(x), np.array(label_zo)
        x_1 = np.array([x[0] for x in x_])
        x_2 = np.array([x[1] for x in x_])
        x_all = [x_1, x_2]
        return x_all, y_
    elif embedding_type == 'xlnet':
        count += 1
        if count == 1:
            x_0 = x[0]
            print(x[0][0][0])
        x_, y_ = x, np.array(label_zo)
        x_1 = np.array([x[0][0] for x in x_])
        x_2 = np.array([x[1][0] for x in x_])
        x_3 = np.array([x[2][0] for x in x_])
        if embed.trainable:
            x_4 = np.array([x[3][0] for x in x_])
            x_all = [x_1, x_2, x_3, x_4]
        else:
            x_all = [x_1, x_2, x_3]
        return x_all, y_
    else:
        x_, y_ = np.array(x), np.array(label_zo)
        return x_, y_


def fig_generator_process():
    pass


# 重写tokenizer
# class OurTokenizer(Tokenizer):
#     def __init__(self, token_dict):
#         self.vocab_size = len(token_dict)
#         super().__init__(token_dict)
#
#     def _tokenize(self, text):
#         R = []
#         for c in text:
#             if c in self._token_dict:
#                 R.append(c)
#             elif self._is_space(c):
#                 R.append('[unused1]')  # 用[unused1]来表示空格类字符
#             else:
#                 R.append('[UNK]')  # 不在列表的字符用[UNK]表示
#         return R
#
#     def encode(self, first, second=None, max_length=None, algo_code=None):
#         if not algo_code:
#             return super().encode(first, second=None, max_len=None)
#         else:
#             first_tokens = self._tokenize(first)
#             second_tokens = self._tokenize(second) if second is not None else None
#             self._truncate(first_tokens, second_tokens, max_length)
#             token_ids = self._convert_tokens_to_ids(first_tokens)
#             return token_ids


# 让每条文本的长度相同，用0填充
def seq_padding(X, padding=0, max_len: int = 0):
    if max_len:
        ML = max_len
    else:
        L = [len(x) for x in X]
        ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


from keras.utils.np_utils import to_categorical
# DataGenerator只是一种为了节约内存的数据方式
class datagenerator(DataGenerator):
    def __init__(self, data, l2i, tokenizer, batch_size, maxlen=128):
        super().__init__(data, batch_size=batch_size)
        self.l2i = l2i
        self.maxlen = maxlen
        self.tokenizer = tokenizer

    def __iter__(self,random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (label, text) in self.sample(random):
        # for is_end, (text,label) in self.sample(random):
            # print(text,"*************")
            # print(label,"=================")
            token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.maxlen)
            # print(len(token_ids),"999")
            # print(len(segment_ids),"888")
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            # batch_labels.append([int(self.l2i.get(str(label)))])
            batch_labels.append([int(label)])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids,length=self.maxlen)
                batch_segment_ids = sequence_padding(batch_segment_ids,length=self.maxlen)
                batch_labels = sequence_padding(batch_labels)  # 多分类
                # batch_labels = to_categorical(batch_labels)  # 二分类
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# DataGenerator只是一种为了节约内存的数据方式
class MyDataGenerator:
    def __init__(self, data, l2i, tokenizer, categories, maxlen=128, batch_size=32, shuffle=True):
        self.data = data
        self.l2i = l2i
        self.batch_size = batch_size
        self.categories = categories
        self.maxlen = maxlen
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            if self.shuffle:
                np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[1][:self.maxlen]
                x1, x2 = self.tokenizer.encode(text,max_length=self.maxlen)  # token_ids, segment_ids
                y = self.l2i.get(str(d[0]))
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    X1, X2, Y = [], [], []


def evaluate(data, predict):
    total, right = 0., 0.
    for x_true, y_true in tqdm(data):
    # for x_true, y_true in data:
        y_pred = predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self, model, model_path, valid_generator, test_generator):
        self.best_val_acc = 0.
        self.model = model
        self.model_path = model_path
        self.valid_generator = valid_generator
        self.test_generator = test_generator

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(self.valid_generator, self.model.predict)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.model.save_weights(self.model_path)
        # test_acc = evaluate(self.test_generator,self.model.predict)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )

    def on_train_end(self, logs=None):
        if self.test_generator:
            test_acc = evaluate(self.test_generator, self.model.predict)
            print(
                u'best_val_acc: %.5f, test_acc: %.5f\n' %
                (self.best_val_acc, test_acc)
            )
