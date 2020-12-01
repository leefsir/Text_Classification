#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 15:13 
# ide： PyCharm

from __future__ import print_function, division

import os
import keras_bert
from keras_bert import load_trained_model_from_checkpoint
from keras.layers import Dense, Lambda, Flatten
from keras.models import Model
from keras.optimizers import Adam

from basis_framework.basis_graph import BasisGraph
from configs.path_config import BERT_MODEL_PATH, MODEL_ROOT_PATH
from utils.common_tools import token_process
from utils.data_process import OurTokenizer, DataGenerator


class BertGraph(BasisGraph):
    def __init__(self, parameters):
        self.bert_config_path = os.path.join(BERT_MODEL_PATH, 'bert_config.json')
        self.bert_checkpoint_path = os.path.join(BERT_MODEL_PATH, 'bert_model.ckpt')
        self.bert_vocab_path = os.path.join(BERT_MODEL_PATH, 'vocab.txt')
        model_code = parameters.get('model_code', '1')
        model_dir = os.path.join(MODEL_ROOT_PATH, model_code)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        self.path_hyper_parameters = os.path.join(model_dir, 'params.json')
        self.model_path = os.path.join(model_dir, 'best_model.weights')
        self.tensorboard_path = os.path.join(model_dir, 'logs')
        # self.tokenizer = keras_bert.Tokenizer(self.bert_vocab_path)
        self.token_dict = token_process(self.bert_vocab_path)
        self.tokenizer = OurTokenizer(self.token_dict)
        super().__init__(parameters)

    def data_process(self):
        """
        数据处理
        :return:
        """
        # TODO

    def model_create(self):
        bert_model = load_trained_model_from_checkpoint(self.bert_config_path,
                                                        self.bert_checkpoint_path,
                                                        self.max_len,
                                                        self.trainable, )
        output_layer = Lambda(lambda x: x[:, 0])(bert_model.output)  # 取出[cls]层对应的向量来做分类
        pre = Dense(self.categories, activation=self.activation)(output_layer)  # 全连接层激活函数分类
        self.model = Model(bert_model.inputs, pre)
        self.model_compile()

    def model_compile(self):
        self.model.compile(loss=self.loss,
                           optimizer=Adam(self.lr, decay=self.decay_rate),
                           metrics=self.metrics, )
        print(self.model.summary(150))

    def fit_generator(self):
        train_D = DataGenerator(self.train_data, self.tokenizer, self.categories, self.max_len, self.batch_size,
                                shuffle=True)
        valid_D = DataGenerator(self.valid_data, self.tokenizer, self.categories, self.max_len, self.batch_size,
                                shuffle=True)
        test_D = DataGenerator(self.test_data, self.tokenizer, self.categories, self.max_len, self.batch_size,
                               shuffle=True)

        # 模型训练
        self.model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=self.epoch,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=self.callback(),
        )

