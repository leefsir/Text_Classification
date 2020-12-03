#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 15:13 
# ide： PyCharm

from __future__ import print_function, division

import os

from keras.layers import Dense, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint

from basis_framework.basis_graph import BasisGraph
from configs.path_config import BERT_MODEL_PATH, MODEL_ROOT_PATH
from utils.common_tools import token_process, save_json
from utils.data_process import OurTokenizer, DataGenerator


class BertGraph(BasisGraph):
    def __init__(self, parameters):
        self.bert_config_path = os.path.join(BERT_MODEL_PATH, 'bert_config.json')
        self.bert_checkpoint_path = os.path.join(BERT_MODEL_PATH, 'bert_model.ckpt')
        self.bert_vocab_path = os.path.join(BERT_MODEL_PATH, 'vocab.txt')
        model_code = parameters.get('model_code', 'bert')
        model_dir = os.path.join(MODEL_ROOT_PATH, model_code)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        self.path_parameters = os.path.join(model_dir, 'params.json')
        self.model_path = os.path.join(model_dir, 'best_model.weights')
        self.index2label_path = os.path.join(model_dir, 'index2label.json')
        self.tensorboard_path = os.path.join(model_dir, 'logs')
        self.token_dict = token_process(self.bert_vocab_path)
        self.tokenizer = OurTokenizer(self.token_dict)
        self.is_training = parameters.get('model_env_parameters', {}).get('is_training', False)  # 是否训练, 保存时候为Flase,方便预测
        self.parameters = parameters
        if not self.is_training:self.predict_process()

        super().__init__(self.parameters)


    def model_create(self):
        bert_model = load_trained_model_from_checkpoint(self.bert_config_path,
                                                        self.bert_checkpoint_path,
                                                        seq_len=None,
                                                        )
        # x1_in = Input(shape=(None,))
        # x2_in = Input(shape=(None,))
        # output = bert_model([x1_in, x2_in])
        output = bert_model(bert_model.inputs)
        output_layer = Lambda(lambda x: x[:, 0])(output)  # 取出[cls]层对应的向量来做分类
        pre = Dense(self.categories, activation=self.activation)(output_layer)  # 全连接层激活函数分类
        self.model = Model(bert_model.inputs, pre)
        print(self.model.summary(150))
        if self.is_training:self.model_compile()


    def model_compile(self):
        self.model.compile(loss=self.loss,
                           optimizer=Adam(self.lr),
                           metrics=self.metrics, )

    def fit_generator(self):
        # 保存超参数
        self.parameters['hyper_parameters']['is_training'] = False  # 预测时候这些设为False
        self.parameters['model_env_parameters']['trainable'] = False
        save_json(jsons=self.i2l, json_path=self.index2label_path)
        save_json(jsons=self.parameters, json_path=self.path_parameters)
        train_D = DataGenerator(self.train_data, self.l2i, self.tokenizer, self.categories, self.max_len,
                                self.batch_size,
                                shuffle=True)
        valid_D = DataGenerator(self.valid_data, self.l2i, self.tokenizer, self.categories, self.max_len,
                                self.batch_size,
                                shuffle=True)
        # test_D = DataGenerator(self.test_data, self.l2i,self.tokenizer, self.categories, self.max_len, self.batch_size,
        #                        shuffle=True)

        # 模型训练
        self.model.fit_generator(
            train_D.__iter__(),
            steps_per_epoch=len(train_D),
            epochs=self.epoch,
            validation_data=valid_D.__iter__(),
            validation_steps=len(valid_D),
            callbacks=self.callback(),
        )

