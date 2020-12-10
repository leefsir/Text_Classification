#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 17:13 
# ide： PyCharm
import os
import sys

from classifier_keras.l02_albert.albert_model import AlbertGraph

rootPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(rootPath)
print(rootPath)
from configs.path_config import CORPUS_ROOT_PATH

# train_data_path = CORPUS_ROOT_PATH +'/baidu_qa_2019/baike_qa_train.csv'
train_data_path = CORPUS_ROOT_PATH + '/thuc_news/train.txt'
valid_data_path = CORPUS_ROOT_PATH + '/thuc_news/dev.txt'
# valid_data_path = CORPUS_ROOT_PATH +'/baidu_qa_2019/baike_qa_valid.csv'
parameters = {
    'model_code': 'thuc_news',
    'hyper_parameters': {
        'train_data_path': train_data_path,
        'valid_data_path': valid_data_path,
        'batch_size': 256,
        'epoch': 1,
    },
    'model_env_parameters': {'is_training': True, },
}
bertModel = AlbertGraph(parameters)
bertModel.fit_generator()
