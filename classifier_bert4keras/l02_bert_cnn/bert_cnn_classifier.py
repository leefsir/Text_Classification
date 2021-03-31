#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 15:13
# ide： PyCharm

from __future__ import print_function, division
import os
import sys

import time
from bert4keras.snippets import sequence_padding
from sklearn.metrics import classification_report

rootPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(rootPath)
import numpy as np
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import extend_with_piecewise_linear_lr
from keras.layers import Dense, Dropout, Flatten, MaxPooling1D, concatenate, Conv1D, Lambda, SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam

from basis_framework.basis_graph_last import BasisGraph
from configs.path_config import CORPUS_ROOT_PATH
from utils.common_tools import data2csv, data_preprocess, split
from utils.data_process import datagenerator,Evaluator


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
        if '.csv' not in self.train_data_path:
            self.train_data_path = data2csv(self.train_data_path, sep)
        self.index2label, self.label2index, self.labels, train_data = data_preprocess(self.train_data_path)
        self.num_classes = len(self.index2label)
        if self.valid_data_path:
            if '.csv' not in self.valid_data_path:
                self.valid_data_path = data2csv(self.valid_data_path, sep)
            _, _, _, valid_data = data_preprocess(self.valid_data_path)
        else:
            train_data, valid_data = split(train_data, self.split)
        if self.test_data_path:
            if '.csv' not in self.test_data_path:
                self.test_data_path = data2csv(self.test_data_path, sep)
            _, _, _, test_data = data_preprocess(self.test_data_path)
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
        token_ids = sequence_padding([token_ids], length=self.max_len)
        segment_ids = sequence_padding([segment_ids], length=self.max_len)
        pre = self.model.predict([token_ids, segment_ids])
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
    def data_score(self, text_path):
        time_start = time.time()
        # 测试集的准确率
        if '.csv' not in text_path:
            text_path = data2csv(text_path, sep='\t')
        _, _, _, test_data = data_preprocess(text_path)
        y_pred = []
        y_true = []
        for label, text in test_data:
            y_true.append(self.index2label[str(label)])
            token_ids, segment_ids = self.tokenizer.encode(text, max_length=self.max_len)  # maxlen 新版本
            token_ids = sequence_padding([token_ids], length=self.max_len)
            segment_ids = sequence_padding([segment_ids], length=self.max_len)
            pred = self.model.predict([token_ids, segment_ids])
            pred = np.argmax(pred[0])
            y_pred.append(self.index2label[str(pred)])

        print("data pred ok!")
        # 评估
        target_names = [str(label) for label in self.labels]
        report_predict = classification_report(y_true, y_pred,
                                               target_names=target_names, digits=9)
        print(report_predict)
        print("耗时:" + str(time.time() - time_start))

if __name__ == '__main__':
    params = {
        'model_code': 'thuc_news_bertcnn',
        'train_data_path': CORPUS_ROOT_PATH + '/thuc_news/train.txt',
        'valid_data_path': CORPUS_ROOT_PATH + '/thuc_news/dev.txt',
        'test_data_path': CORPUS_ROOT_PATH + '/thuc_news/test.txt',
        'batch_size': 128,
        'max_len': 30,
        'epoch': 4,
        'learning_rate': 1e-4,
        'gpu_id': 0,
    }
    # bertModel = BertGraph(params, Train=True)
    # bertModel.train()
    bertModel = BertGraph(params)
    data_path = CORPUS_ROOT_PATH + '/thuc_news/test.csv'
    print(bertModel.data_score(data_path))
else:
    params = {
        'model_code': 'thuc_news_bertcnn',  # 此处与训练时code保持一致
        'gpu_id': 1,
    }
    bertModel = BertGraph(params)
