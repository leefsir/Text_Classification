#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 17:14 
# ide： PyCharm
from classifier.l01_bert.bert_model import BertGraph

parameters = {
'model_code':'thuc_news',
'model_env_parameters':{'is_training':False},
}
bertModel = BertGraph(parameters)

while True:
    text = input()
    res = bertModel.predict(text)
    print(res)