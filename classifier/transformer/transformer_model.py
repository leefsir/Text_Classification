#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 15:13 
# ide： PyCharm

from __future__ import print_function, division

import os

from keras.layers import Dense, Input, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam

from basis_framework.basis_graph import BasisGraph
from configs.path_config import MODEL_ROOT_PATH
from layers_utils.transformer_utils.embedding import EmbeddingRet
from layers_utils.transformer_utils.non_mask_layer import NonMaskingLayer
from layers_utils.transformer_utils.transformer import build_encoders
from layers_utils.transformer_utils.triangle_position_embedding import TriglePositiomEmbedding
from utils.common_tools import save_json
from utils.data_process import DataGenerator


class TransformerEncodeGraph(BasisGraph):
    def __init__(self, parameters):
        self.encoder_num = parameters["hyper_parameters"].get('encoder_num', 2)
        self.head_num = parameters["hyper_parameters"].get('head_num', 6)
        self.hidden_dim = parameters["hyper_parameters"].get('hidden_dim', 3072)
        self.attention_activation = parameters["hyper_parameters"].get('attention_activation', 'relu')
        self.feed_forward_activation = parameters["hyper_parameters"].get('feed_forward_activation', 'relu')
        self.use_adapter = parameters["hyper_parameters"].get('use_adapter', False)
        self.adapter_units = parameters["hyper_parameters"].get('adapter_units', 768)
        self.adapter_activation = parameters["hyper_parameters"].get('adapter_activation', 'relu')

        # 构建文件路径
        model_code = parameters.get('model_code', 'bert')
        model_dir = os.path.join(MODEL_ROOT_PATH, model_code)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        self.path_parameters = os.path.join(model_dir, 'params.json')
        self.model_path = os.path.join(model_dir, 'best_model.weights')
        self.index2label_path = os.path.join(model_dir, 'index2label.json')
        self.tensorboard_path = os.path.join(model_dir, 'logs')
        self.is_training = parameters.get('model_env_parameters', {}).get('is_training', False)  # 是否训练, 保存时候为Flase,方便预测
        self.parameters = parameters
        if not self.is_training: self.predict_process()

        super().__init__(self.parameters)

    def model_create(self):
        # 构建网络层
        from basis_framework.embedding import RandomEmbedding as Embeddings

        self.word_embedding = Embeddings(hyper_parameters=self.hyper_parameters)
        if os.path.exists(self.path_fineture) and self.trainable:
            self.word_embedding.model.load_weights(self.path_fineture)
            print("load path_fineture ok!")
        encoder_input = Input(shape=(self.max_len,), name='Encoder-Input')
        encoder_embed_layer = EmbeddingRet(input_dim=self.word_embedding.vocab_size,
                                           output_dim=self.word_embedding.embed_size,
                                           mask_zero=False,
                                           weights=None,
                                           trainable=self.trainable,
                                           name='Token-Embedding', )
        encoder_embedding = encoder_embed_layer(encoder_input)
        encoder_embed = TriglePositiomEmbedding(mode=TriglePositiomEmbedding.MODE_ADD,
                                                name='Encoder-Embedding', )(encoder_embedding[0])
        encoded_layer = build_encoders(encoder_num=self.encoder_num,
                                       input_layer=encoder_embed,
                                       head_num=self.head_num,
                                       hidden_dim=self.hidden_dim,
                                       attention_activation=self.activation,
                                       feed_forward_activation=self.activation,
                                       dropout_rate=self.dropout,
                                       trainable=self.trainable,
                                       use_adapter=self.use_adapter,
                                       adapter_units=self.adapter_units,
                                       adapter_activation=self.adapter_activation,
                                       )
        encoded_layer = NonMaskingLayer()(encoded_layer)
        encoded_layer_flat = Flatten()(encoded_layer)
        encoded_layer_drop = Dropout(self.dropout)(encoded_layer_flat)

        pre = Dense(self.categories, activation=self.activation)(encoded_layer_drop)  # 全连接层激活函数分类
        self.model = Model(encoder_input, pre)
        print(self.model.summary(150))
        if self.is_training: self.model_compile()

    def model_compile(self):
        self.model.compile(loss=self.loss,
                           optimizer=Adam(self.lr),
                           metrics=self.metrics, )

    def fit_generator(self):
        # 保存超参数
        self.parameters['parameters']['is_training'] = False  # 预测时候这些设为False
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
