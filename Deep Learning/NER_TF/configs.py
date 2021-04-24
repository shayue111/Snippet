# -*- coding: utf-8 -*-
# @Time    : 2020/9/24 下午10:49
# @Author  : wangt
# @Email   : wt0504@zju.edu.cn
# @File    : configs.py

import datetime
import os

""" 设置输入输出文件夹 """

TRAIN_PATH = '../data/round1_train/'
TEST_PATH = '../data/round1_test/'

today = datetime.date.today().strftime('%m%d')
PRED_PATH = '../pred_data/' + today

if not os.path.exists(PRED_PATH):
    os.mkdir(PRED_PATH)

""" 实体类型 """

entities = [
    'DRUG',
    'DRUG_INGREDIENT',
    'DISEASE',
    'SYMPTOM',
    'SYNDROME',
    'DISEASE_GROUP',
    'FOOD',
    'FOOD_GROUP',
    'PERSON_GROUP',
    'DRUG_GROUP',
    'DRUG_DOSAGE',
    'DRUG_TASTE',
    'DRUG_EFFICACY',
]

entity2labels = {entity: [entity + '_S', entity + '_M', entity + '_E'] for entity in entities}

""" 实体标签个数 """

label_size = 0
labels = []

for x in entity2labels.values():
    label_size += len(x)
    labels.extend(x)

# 加上O
labels.append('O')
label_size += 1

""" 实体标签对id的字典以及反之 """

labels2id = {label: id_ for id_, label in enumerate(labels)}
id2labels = {id_: label for label, id_ in labels2id.items()}

""" 读取字典 """
vocab = []
with open('../resource/vocab.txt', 'r') as reader_:
    for ch in reader_:
        vocab.append(ch[:-1])
    vocab.append(' ')

vocab2id = {ch: id_ for id_, ch in enumerate(vocab, start=1)}
