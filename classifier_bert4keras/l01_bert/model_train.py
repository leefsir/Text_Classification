#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 17:13 
# ide： PyCharm
import os,sys

from classifier_bert4keras.l01_bert.bert_mode import BertGraph

rootPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(rootPath)
print(rootPath)
from configs.path_config import CORPUS_ROOT_PATH
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
# train_data_path = CORPUS_ROOT_PATH +'/baidu_qa_2019/baike_qa_train.csv'
train_data_path = CORPUS_ROOT_PATH +'/thuc_news/train.txt'
valid_data_path = CORPUS_ROOT_PATH +'/thuc_news/dev.txt'
test_data_path = CORPUS_ROOT_PATH +'/thuc_news/test.txt'
# valid_data_path = CORPUS_ROOT_PATH +'/baidu_qa_2019/baike_qa_valid.csv'
parameters = {
'model_code':'thuc_news_bert',
'hyper_parameters':{
    'train_data_path':train_data_path,
    'valid_data_path':valid_data_path,
    'test_data_path':test_data_path,
    'batch_size':128,
    'max_len':30,
    'epoch':10,
    'lr':1e-5,
},
'model_env_parameters':{'is_training':True,},
}
bertModel = BertGraph(parameters)
bertModel.fit_generator()