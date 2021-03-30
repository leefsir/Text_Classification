#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2021/3/29 14:13 
# ide： PyCharm


from __future__ import print_function, division
import os
import sys

rootPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(rootPath)
import jieba
import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import extend_with_piecewise_linear_lr
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, concatenate, Conv1D, Lambda, SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam
from bert4keras.tokenizers import Tokenizer
from basis_framework.basis_graph_last import BasisGraph
from configs.path_config import CORPUS_ROOT_PATH, WoNeZha_MODEL_PATH
from utils.common_tools import data2csv, data_preprocess, split, json_data_process, txt_data_process
from utils.data_process import datagenerator, Evaluator

from keras.engine import Layer


class NonMaskingLayer(Layer):
    """
    fix convolutional 1D can't receive masked input, detail: https://github.com/keras-team/keras/issues/4978
    thanks for https://github.com/jacoxu
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


class BertGraph(BasisGraph):
    def __init__(self, params={}, Train=False):
        if not params.get('model_code'):
            params['model_code'] = 'classifier'
        self.filters = params.get('filters', [3, 4, 5])  # 卷积核大小
        self.filters_num = params.get('filters_num', 300)  # 核数
        super().__init__(params, Train)


    def data_process(self, sep='\t'):
        """
        数据处理
        :return:
        """
        self.index2label, self.label2index, self.labels, train_data = json_data_process(self.train_data_path)
        # self.index2label, self.label2index, self.labels, train_data = txt_data_process(self.train_data_path)
        print(self.label2index,"1111111")
        self.num_classes = len(self.index2label)
        if self.valid_data_path:
            _, _, _, valid_data = json_data_process(self.valid_data_path)
            # _, _, _, valid_data = txt_data_process(self.valid_data_path)
        else:
            train_data, valid_data = split(train_data, self.split)
        if self.test_data_path:
            _, _, _, test_data = json_data_process(self.test_data_path)
            # _, _, _, test_data = txt_data_process(self.test_data_path)
        else:
            test_data = []
        self.train_generator = datagenerator(train_data, self.label2index, self.tokenizer, self.batch_size,
                                             self.max_len)
        self.valid_generator = datagenerator(valid_data, self.label2index, self.tokenizer, self.batch_size,
                                             self.max_len)
        self.test_generator = datagenerator(test_data, self.label2index, self.tokenizer, self.batch_size,
                                            self.max_len)

    def build_model(self):
        bert = build_transformer_model(
            config_path=self.bert_config_path,
            checkpoint_path=self.bert_checkpoint_path,
            model="nezha",
            return_keras_model=False,
            sequence_length=self.max_len,
        )

        x = Lambda(lambda x: x, output_shape=lambda s: s)(bert.model.output)
        x = SpatialDropout1D(rate=self.dropout)(x)
        print(x.shape)
        conv_pools = []
        # 词窗大小分别为3,4,5
        for filter in self.filters:
            cnn = Conv1D(self.filters_num, filter, padding='same', strides=1, activation='relu')(x)
            cnn = MaxPooling1D(pool_size=self.max_len - filter + 1)(cnn)
            print(cnn.shape)
            conv_pools.append(cnn)
        # 合并三个模型的输出向量
        cnn = concatenate(conv_pools, axis=-1)
        print(cnn.shape)
        flat = Flatten()(cnn)
        drop = Dropout(self.dropout)(flat)
        output = Dense(self.num_classes, activation=self.activation)(drop)
        self.model = Model(bert.model.input, output)
        print(self.model.summary(150))

    def predict(self, text):
        token_ids, segment_ids = self.tokenizer.encode(text)
        pre = self.model.predict([[token_ids], [segment_ids]])
        res = self.index2label.get(str(np.argmax(pre[0])))
        return res

    def compile_model(self):
        # 派生为带分段线性学习率的优化器。
        # 其中name参数可选，但最好填入，以区分不同的派生优化器。
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.model.compile(loss=self.loss,
                           optimizer=AdamLR(lr=self.learning_rate, lr_schedule={
                               1000: 1,
                               2000: 0.1
                           }),
                           metrics=self.metrics, )

    def train(self):
        # 保存超参数
        evaluator = Evaluator(self.model, self.model_path, self.valid_generator, self.test_generator)

        # 模型训练
        self.model.fit_generator(
            self.train_generator.forfit(),
            steps_per_epoch=len(self.train_generator),
            epochs=self.epoch,
            callbacks=[evaluator],
        )


if __name__ == '__main__':
    params = {
        'model_code': 'iflytek_public_wonezhacnn',
        'train_data_path': CORPUS_ROOT_PATH + '/iflytek_public/train.json',
        'valid_data_path': CORPUS_ROOT_PATH + '/iflytek_public/dev.json',
        # 'test_data_path': CORPUS_ROOT_PATH + '/iflytek_public/test.txt',
        'batch_size': 16,
        'max_len': 256,
        'epoch': 10,
        'learning_rate': 1e-4,
        'gpu_id': 0,
    }
    # params = {
    #     'model_code': 'thuc_news_bertcnn1',
    #     'train_data_path': CORPUS_ROOT_PATH + '/thuc_news/train.txt',
    #     'valid_data_path': CORPUS_ROOT_PATH + '/thuc_news/dev.txt',
    #     'test_data_path': CORPUS_ROOT_PATH + '/thuc_news/test.txt',
    #     'batch_size': 64,
    #     'max_len': 30,
    #     'epoch': 10,
    #     'learning_rate': 1e-4,
    #     'gpu_id': 1,
    # }
    bertModel = BertGraph(params, Train=True)
    bertModel.train()
else:
    params = {
        'model_code': 'iflytek_public_wonezhacnn',  # 此处与训练时code保持一致
        'gpu_id': 1,
    }
    bertModel = BertGraph(params)
