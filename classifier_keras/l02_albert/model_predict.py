#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 17:14 
# ide： PyCharm
from classifier_keras.l02_albert.albert_model import AlbertGraph

parameters = {
    'model_code': 'thuc_news',
    'model_env_parameters': {'is_training': False},
}
bertModel = AlbertGraph(parameters)

while True:
    text = input()
    res = bertModel.predict(text)
    print(res)
