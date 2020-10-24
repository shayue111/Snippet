# -*- coding: utf-8 -*-
# @Time    : 2020/9/26 上午11:22
# @Author  : wangt
# @Email   : wt0504@zju.edu.cn
# @File    : predict.py
from configs import *
from model import *
from tqdm import tqdm
from utils import write_ann_file, cut_long_text, ernie_tokenizer

""" 设置模型与优化器 """
model = BilstmCRF(label_size)

""" 设置模型路径 """
ckpt = tf.train.Checkpoint(model=model)
ckpt.restore(tf.train.latest_checkpoint('../resource/model_1019/'))

""" 预测测试集 """
test_files = [x for x in os.listdir(TEST_PATH)]

for file in tqdm(test_files):
    with open(TEST_PATH + file, 'r') as reader:
        text = reader.read()
    cut_texts, _ = cut_long_text(text, [])

    pred_labels = []

    for cut_text in cut_texts:
        # cut_text的前后是有cls与sep两个token的
        text_ids = ernie_tokenizer.encode(list(cut_text))
        text_ids = tf.expand_dims(tf.constant(text_ids), axis=0)
        attention_mask = tf.math.not_equal(text_ids, 0)
        pred_label = model(text_ids, attention_mask)
        # 掐掉头尾的cls与sep
        pred_label = pred_label[0].numpy().tolist()[1:-1]
        pred_labels.extend(pred_label)

    write_ann_file(file_name=file.split('.')[0], text=text, pred_labels=pred_labels)
