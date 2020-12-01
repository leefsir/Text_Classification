#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 17:22 
# ide： PyCharm
import pandas as pd
import numpy as np
import random
import os

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