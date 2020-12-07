#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 17:13 
# ide： PyCharm
import os
import sys

rootPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(rootPath)
print(rootPath)
from configs.path_config import CORPUS_ROOT_PATH, PATH_EMBEDDING_RANDOM_CHAR
from classifier.transformer.transformer_model import TransformerEncodeGraph
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
# train_data_path = CORPUS_ROOT_PATH +'/baidu_qa_2019/baike_qa_train.csv'
train_data_path = CORPUS_ROOT_PATH + '/thuc_news/train.txt'
valid_data_path = CORPUS_ROOT_PATH + '/thuc_news/dev.txt'
# valid_data_path = CORPUS_ROOT_PATH +'/baidu_qa_2019/baike_qa_valid.csv'
parameters = {
    'model_code': 'thuc_news_transformer',
    'hyper_parameters': {
        'train_data_path': train_data_path,
        'valid_data_path': valid_data_path,
        'vocab_path': PATH_EMBEDDING_RANDOM_CHAR,
        'batch_size': 128,
        'max_len': 30,  # 句子最大长度, 固定 推荐20-50
        'embed_size': 768,  # 字/词向量维度
        'vocab_size': 20000,  # 这里随便填的，会根据代码里修改
        'trainable': True,  # embedding是静态的还是动态的
        'dropout': 0.1,  # 随机失活, 概率
        'decay_step': 100,  # 学习率衰减step, 每N个step衰减一次
        'decay_rate': 0.99,  # 学习率衰减系数, 乘法
        'epochs': 50,  # 训练最大轮次
        'patience': 5,  # 早停,2-3就好
        'lr': 1e-4,  # 学习率, 对训练会有比较大的影响, 如果准确率一直上不去,可以考虑调这个参数
        'l2': 1e-9,  # l2正则化
        'droupout_spatial': 0.25,
        'encoder_num': 1,
        'head_num': 12,
        'hidden_dim': 3072,
        'attention_activation': 'relu',
        'feed_forward_activation': 'relu',
        'use_adapter': False,
        'adapter_units': 768,
        'adapter_activation': 'relu',
    },
    'model_env_parameters': {'is_training': True, },
}
bertModel = TransformerEncodeGraph(parameters)
bertModel.fit_generator()
