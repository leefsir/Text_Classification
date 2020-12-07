#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 11:06 
# ide： PyCharm
import os

import numpy as np
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

from utils.common_tools import data_preprocess, data2csv, load_json, token_process
from utils.data_process import OurTokenizer


class BasisGraph():
    def __init__(self, parameters: {}):
        """
        基础模型框架构建
        :param self.hyper_parameters: 模型超参数
        """
        self.hyper_parameters = parameters.get('hyper_parameters', {})
        self.model_env_parameters = parameters.get('model_env_parameters', {})

        # 初始化环境参数
        self.gpu_memory_fraction = self.model_env_parameters.get('gpu_memory_fraction', None)  # gpu使用占比
        self.trainable = self.model_env_parameters.get('trainable', None)  # 预训练语言模型是否参与训练微调
        # self.is_training = self.model_env_parameters.get('is_training', False)  # 是否训练, 保存时候为Flase,方便预测
        self.gpu_id = self.model_env_parameters.get('gpu_id', None)  # 使用的gpu_id

        # 初始化模型超参数
        self.epoch = self.hyper_parameters.get('epoch', 10)  # 训练伦次
        self.batch_size = self.hyper_parameters.get('batch_size', 32)  # 批次数量
        self.max_len = self.hyper_parameters.get('max_len', 128)  # 最大文本长度
        # self.categories = self.hyper_parameters.get('categories', 2)  # 文本类别数量
        self.embed_size = self.hyper_parameters.get('embed_size', 300)  # 词向量编码维度
        self.dropout = self.hyper_parameters.get('dropout', 0.5)  # dropout层系数，丢失率控制
        self.decay_step = self.hyper_parameters.get('decay_step', 100)  # 衰减步数
        self.decay_rate = self.hyper_parameters.get('decay_rate', 0.99)  # 衰减系数
        self.lr = self.hyper_parameters.get('lr', 1e-4)  # 学习率
        self.patience = self.hyper_parameters.get('patience', 2)  # 早停计数
        self.activation = self.hyper_parameters.get('activation', 'softmax')  # 分类激活函数,softmax或者signod
        self.loss = self.hyper_parameters.get('loss',
                                              'categorical_crossentropy')  # 损失函数, mse, categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy等
        self.metrics = self.hyper_parameters.get('metrics',
                                                 [
                                                     'accuracy'])  # acc, binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, sparse_top_k_categorical_accuracy

        self.path_fineture = self.hyper_parameters.get('path_fineture',
                                                       "path_fineture")  # embedding层保存地址, 例如静态词向量、动态词向量、微调bert层等
        self.train_data_path = self.hyper_parameters.get('train_data_path')
        if self.is_training and not self.train_data_path: raise Exception("No training data!")
        self.valid_data_path = self.hyper_parameters.get('valid_data_path')
        self.token_dict = token_process(self.vocab_path)
        if not self.token_dict : raise Exception("No token_dict!")
        self.tokenizer = OurTokenizer(self.token_dict)
        self.parameters = parameters
        self._set_gpu_id(self.gpu_id)  # 设置训练的GPU_ID
        if self.is_training: self.data_process()
        self.model_create()
        if not self.is_training:
            self.load_model()

    def _set_gpu_id(self, gpu_id):
        """指定使用的GPU显卡id"""
        if gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    def data_process(self, sep='\t'):
        """
        数据处理
        :return:
        """
        if '.csv' not in self.train_data_path:
            self.train_data_path = data2csv(self.train_data_path, sep)
        self.i2l, self.l2i, self.train_data = data_preprocess(self.train_data_path)
        print(self.l2i)
        self.categories = len(self.i2l)
        if self.valid_data_path:
            if '.csv' not in self.valid_data_path:
                self.valid_data_path = data2csv(self.valid_data_path, sep)
            _, _, self.valid_data = data_preprocess(self.valid_data_path)
        else:
            split = int(len(self.train_data) * 0.8)
            self.train_data = self.train_data[:split]
            self.valid_data = self.train_data[split:]

    def predict_process(self, sep='\t'):
        """
        预测模型参数处理
        :return:
        """
        self.i2l = load_json(self.index2label_path)
        self.categories = len(self.i2l)
        self.parameters = load_json(self.path_parameters)

    def model_create(self):
        """
        模型框架搭建
        :return:
        """
        raise NotImplementedError

    def model_compile(self):
        """
        模型编译，添加loss，优化器，评价函数
        :return:
        """
        raise NotImplementedError

    def fit(self, x_train, y_train, x_dev, y_dev):
        """
        模型编译，添加loss，优化器，评价函数
        :return:
        """
        # # 保存超参数
        # self.parameters['hyper_parameters']['is_training'] = False  # 预测时候这些设为False
        # self.parameters['model_env_parameters']['trainable'] = False
        # self.parameters['hyper_parameters']['dropout'] = 0.0
        #
        # save_json(jsons=self.hyper_parameters, json_path=self.path_hyper_parameters)
        # # if self.is_training and os.path.exists(self.model_path):
        # #     print("load_weights")
        # #     self.model.load_weights(self.model_path)
        # # 训练模型
        # self.model.fit(x_train, y_train, batch_size=self.batch_size,
        #                epochs=self.epoch, validation_data=(x_dev, y_dev),
        #                shuffle=True,
        #                callbacks=self.callback())
        # # 保存embedding, 动态的
        # # if self.trainable:
        # #     self.word_embedding.model.save(self.path_fineture)
        raise NotImplementedError

    def fit_generator(self):
        """
        模型编译，添加loss，优化器，评价函数
        :return:
        """
        raise NotImplementedError

    def predict(self, text):
        token_ids, segment_ids = self.tokenizer.encode(text)
        # print(token_ids)
        pre = self.model.predict([[token_ids], [segment_ids]])
        # print(pre)
        print(self.i2l.get(str(np.argmax(pre[0]))))

    def callback(self):
        """
        回调函数：评价函数，早停函数，模型保存，tensorboard保存可在此构建
        :return:
        """
        tensorboard = TensorBoard(log_dir=self.tensorboard_path, batch_size=self.batch_size, update_freq='batch')
        earlystopp = EarlyStopping(monitor='val_loss', mode='min', min_delta=1e-8, patience=self.patience)
        checkpoint = ModelCheckpoint(monitor='val_loss', mode='min', filepath=self.model_path, verbose=1,
                                     save_best_only=True, save_weights_only=True)
        return [tensorboard, earlystopp, checkpoint]

    def load_model(self):
        self.model.load_weights(self.model_path)
