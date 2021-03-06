#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： liwfeng
# datetime： 2020/12/1 16:52 
# ide： PyCharm
import codecs
import json
import os
import random
import re
import jieba
import pandas as pd
import numpy as np

from configs.path_config import CORPUS_ROOT_PATH


def json_data_process(path):
    index2label,ret_data = {},[]
    # with open(path, 'r', encoding='utf-8') as fj:
    #     json_data = json.load(fj)
    json_data = open(path)
    # json_data = json.load(open(path, encoding='utf-8'))
    for line in json_data.readlines():
        data = json.loads(line)
        index = data.get("label")
        label = data.get("label_des")
        sentence = data.get("sentence")
        ret_data.append([index,sentence])
        if not index2label.get(index):index2label[index]=label
    label2index = {l: i for i, l in index2label.items()}
    labels = list(label2index.keys())
    return index2label,label2index,labels,ret_data

def txt_data_process(path):
    index2label,ret_data = {},[]
    json_data = open(path)
    for line in json_data.readlines():
        line = line.strip()
        line = line.split('\t')
        index = line[1]
        label = line[1]
        ret_data.append([line[0],line[1]])
        if not index2label.get(index):index2label[index]=label
    label2index = {l: i for i, l in index2label.items()}
    labels = list(label2index.keys())
    return index2label,label2index,labels,ret_data



def data2csv(data_path, sep):
    # 训练数据、测试数据和标签转化为模型输入格式
    label = []
    content = []
    with open(data_path, 'r', encoding='utf8') as f:
        contents = f.readlines()
        for line in contents:
            line = line.split(sep)
            label.append(line[1].strip())
            content.append(line[0].strip())
    data = {}
    data['label'] = label
    data['content'] = content
    data_path = ''.join(data_path.split('.')[:-1]) + '.csv'
    pd.DataFrame(data).to_csv(data_path, index=False)
    return data_path


def txt_read(file_path, encode_type='utf-8'):
    """
      读取txt文件，默认utf8格式
    :param file_path: str, 文件路径
    :param encode_type: str, 编码格式
    :return: list
    """
    list_line = []
    try:
        file = open(file_path, 'r', encoding=encode_type)
        while True:
            line = file.readline()
            line = line.strip()
            if not line:
                break
            list_line.append(line)
        file.close()
    except Exception as e:
        print(str(e))
    finally:
        return list_line


def txt_write(list_line, file_path, type='w', encode_type='utf-8'):
    """
      txt写入list文件
    :param listLine:list, list文件，写入要带"/n"
    :param filePath:str, 写入文件的路径
    :param type: str, 写入类型, w, a等
    :param encode_type:
    :return:
    """
    try:
        file = open(file_path, type, encoding=encode_type)
        file.writelines(list_line)
        file.close()

    except Exception as e:
        print(str(e))


def extract_chinese(text):
    """
      只提取出中文、字母和数字
    :param text: str, input of sentence
    :return:
    """
    chinese_exttract = ''.join(re.findall(u"([/u4e00-/u9fa5A-Za-z0-9@._])", text))
    return chinese_exttract


def read_and_process(path):
    """
      读取文本数据并
    :param path:
    :return:
    """
    # with open(path, 'r', encoding='utf-8') as f:
    #     lines = f.readlines()
    #     line_x = [extract_chinese(str(line.split(",")[0])) for line in lines]
    #     line_y = [extract_chinese(str(line.split(",")[1])) for line in lines]
    #     return line_x, line_y

    data = pd.read_csv(path)
    ques = data["ques"].values.tolist()
    labels = data["label"].values.tolist()
    line_x = [extract_chinese(str(line).upper()) for line in labels]
    line_y = [extract_chinese(str(line).upper()) for line in ques]
    return line_x, line_y


def preprocess_label_ques(path):
    x, y, x_y = [], [], []
    x_y.append('label,ques/n')
    with open(path, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            try:
                line_json = json.loads(line)
            except:
                break
            ques = line_json['title']
            label = line_json['category'][0:2]
            line_x = " ".join(
                [extract_chinese(word) for word in list(jieba.cut(ques, cut_all=False, HMM=True))]).strip().replace(
                '  ', ' ')
            line_y = extract_chinese(label)
            x_y.append(line_y + ',' + line_x + '/n')
    return x_y


def save_json(jsons, json_path):
    """
      保存json，
    :param json_: json
    :param path: str
    :return: None
    """
    with open(json_path, 'w', encoding='utf-8') as fj:
        fj.write(json.dumps(jsons, ensure_ascii=False))
    fj.close()


def load_json(path):
    """
      获取json，只取第一行
    :param path: str
    :return: json
    """
    with open(path, 'r', encoding='utf-8') as fj:
        model_json = json.loads(fj.readlines()[0])
    return model_json


def delete_file(path):
    """
        删除一个目录下的所有文件
    :param path: str, dir path
    :return: None
    """
    for i in os.listdir(path):
        # 取文件或者目录的绝对路径
        path_children = os.path.join(path, i)
        if os.path.isfile(path_children):
            if path_children.endswith(".h5") or path_children.endswith(".json"):
                os.remove(path_children)
        else:  # 递归, 删除目录下的所有文件
            delete_file(path_children)


def token_process(vocab_path):
    """
    数据处理
    :return:
    """
    # 将词表中的词转换为字典
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


def data_preprocess(data_path, label='label',usecols=['content','label']):
    """
    处理数据返回类别标签转换字典
    :param data_path:
    :return:
    """
    df = pd.read_csv(data_path,usecols=usecols).dropna()
    label_unique = df[label].unique().tolist()
    data = df.values.tolist()
    i2l = {i: str(v) for i, v in enumerate(label_unique)}
    l2i = {str(v): i for i, v in enumerate(label_unique)}
    return i2l, l2i,label_unique, data


def split(train_data, sep=0.8):
    data_len = len(train_data)
    indexs = list(range(data_len))
    random.shuffle(indexs)
    sep = int(data_len * sep)
    train_data, valid_data = [train_data[i] for i in indexs[:sep]], [train_data[i] for i in
                                                                     indexs[sep:]]
    # self.train_data ,self.valid_data = [self.train_data[i] for i in indexs[:sep]],[self.train_data[i] for i in indexs[sep:]]
    return train_data, valid_data

if __name__ == '__main__':
    # data_preprocess('E:/lwf_practice/Text_Classification/corpus/baidu_qa_2019/baike_qa_train.csv')
    json_data_process(CORPUS_ROOT_PATH + '/iflytek_public/train.json')
