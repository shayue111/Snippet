# -*- coding: utf-8 -*-
# @Time    : 2020/9/24 下午10:55
# @Author  : wangt
# @Email   : wt0504@zju.edu.cn
# @File    : utils.py

from configs import *
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
ernie_tokenizer = BertTokenizer.from_pretrained('../resource/ernie_torch/')


def open_train_files(file_path):
    """
    从指定的路径中读取文本信息与标注信息
    ----------------------------------
    :param file_path (str) <- 文本路径,无后缀
    :returns text, tag (str, dict) <- 读取到的文本字符串，以及保存标注信息的字典
    """
    # 打开原始文档
    with open(file_path + '.txt', 'r') as reader:
        text = reader.read()

    # 打开标注文档
    with open(file_path + '.ann', 'r') as reader:
        tags = reader.read()

    """ 将tag整理成[(实体类别, 起始位置, 终止位置), ...]的形式,并且根据位置的先后顺序进行排序 """

    tags = [x.strip().split('\t')[1] for x in tags.split('\n')]
    tags = [x.split() for x in tags]
    tags = [(x[0], eval(x[1]), eval(x[2])) for x in tags]
    tags = sorted(tags, key=lambda x: x[1])

    return text, tags


def generate_instance(text, tags):
    """
    对一组输入的文本与标注信息产生模型需要的样本
    """
    # 初始化文本的类别至O，之后对相应位置的类别进行修改
    labels_ = ['O'] * len(text)

    # 对于每个实体，都有B、M、E三种情况，B代表实体的首字，M代表实体的中间字，E代表实体的末尾。
    # 由于本项目中的实体不会出现单个字的情况，因此不定义S
    for tag in tags:
        # entity是实体类别，spos是起始位置，epos是终止位置，左开右闭
        entity, spos, epos = tag
        for pos in range(spos, epos):
            if pos == spos:
                labels_[pos] = entity2labels[entity][0]
            elif pos + 1 == epos:
                labels_[pos] = entity2labels[entity][2]
            else:
                labels_[pos] = entity2labels[entity][1]

    text_ids = [vocab2id[ch] for ch in text]
    label_ids = [labels2id[cate] for cate in labels_]

    return text_ids, label_ids


def generate_train_dataset():
    """
    通过训练数据产生训练集
    --------------------
    :return (list) <- 列表元素为元组,每个元组又包含两个元素,一为文本id,二为实体标签id
    """
    train_dataset = []
    files = set([file.split('.')[0] for file in os.listdir(TRAIN_PATH)])
    for file in files:
        text, tags = open_train_files(TRAIN_PATH + file)
        train_dataset.append(generate_instance(text, tags))
    return train_dataset


# 有bug的方法,先已修正,用下面的代替
# def get_pred_entities(pred_labels):
#     """
#     针对预测的实体编号序列,将它们转化为[(实体类型, 起始位置, 终止位置), ...]的形式返回
#     --------------------------------------------------------------------------
#     :param pred_labels (numpy.array) <- 模型预测出的序列, shape = [d, ]
#     :return (list) <- 其中元素为(实体类型, 起始位置, 终止位置)的元组
#     """
#     pred_entities_with_pos = []
#
#     find_entity = False
#     spos = None
#     entity = None
#     for pos, label in enumerate(pred_labels):
#         # 说明此处是实体
#         if label != 39:
#             if find_entity is False:
#                 # 将标志置为True,并保存当前位置,这是第一次找到实体
#                 # 接下来要去判定这个实体是什么时候结束.
#                 find_entity = True
#                 spos = pos
#                 entity = id2labels[label][:-2]
#         elif label != 0:
#             if find_entity is True:
#                 epos = pos
#                 pred_entities_with_pos.append((entity, spos, epos))
#                 # 全部恢复
#                 find_entity = False
#                 spos = None
#                 entity = None
#     return pred_entities_with_pos


""" 订正这个方法, 否则遇到实体相邻的时候,对于实体的获取会出错 """


# todo: 测试这个方法,感觉会报错(出现死循环)
def get_pred_entities(pred_labels):
    """
    针对预测的实体编号序列,将它们转化为[(实体类型, 起始位置, 终止位置), ...]的形式返回
    --------------------------------------------------------------------------
    :param pred_labels (numpy.array) <- 模型预测出的序列, shape = [d, ]
    :return (list) <- 其中元素为(实体类型, 起始位置, 终止位置)的元组
    """
    pred_entities_with_pos = []
    seq_len = len(pred_labels)

    start_pos = 0

    while start_pos < seq_len:
        label = pred_labels[start_pos]

        if label != 39:
            # 找到了实体的开始位置
            spos = start_pos
            entity = id2labels[label][:-2]
            entity_e = entity + '_E'

            epos = start_pos + 1
            while epos < seq_len and pred_labels[epos] != labels2id[entity_e] and pred_labels[epos] != labels2id['O'] \
                    and id2labels[pred_labels[epos]][:-2] == entity:
                epos += 1

            # 这个实体预测没有结尾
            if epos == seq_len:
                break
            # 这个实体预测的有问题,跳过
            elif pred_labels[epos] == labels2id['O']:
                start_pos = epos + 1
                continue

            # 找到了实体,将其进行保存
            pred_entities_with_pos.append((entity, spos, epos + 1))

            # 更新下一个实体开始寻找的位置
            start_pos = epos + 1
        else:
            start_pos += 1

    return pred_entities_with_pos


def write_ann_file(file_name, text, pred_labels):
    """
    通过模型预测文本获得实体标签后,将其按照固定格式写入文件夹
    ---------------------------------------------------
    :param file_name (str) <- 文件名
    :param text (str) <- 预测的文本
    :param pred_labels (numpy.array) <- [d, ], 文本预测结果
    :return: None
    """
    pred_entities_with_pos = get_pred_entities(pred_labels)
    with open(PRED_PATH + '/' + file_name + '.ann', 'w') as writer:
        for idx, item in enumerate(pred_entities_with_pos, start=1):
            key = 'T' + repr(idx)
            entity, spos, epos = item
            entity_name = text[spos: epos]
            tmp = ' '.join((entity, repr(spos), repr(epos)))
            s = '\t'.join((key, tmp, entity_name)) + '\n'
            writer.write(s)


def get_strict_f1(pred_y, valid_y):
    """
    对预测结果计算f1指标
    -------------------
    :param pred_y (tf.Tensor) <- 模型预测结果
    :param valid_y (tf.Tensor) <- 人工标注结果
    :return (float, float, float) <- precision, recall, f1
    """

    pred_entities_with_pos = get_pred_entities(pred_y.numpy())
    valid_entities_with_pos = get_pred_entities(valid_y.numpy())

    if pred_entities_with_pos == [] and valid_entities_with_pos == []:
        return 1, 1, 1

    pred_pos_dict = {repr(tup[1]) + '_' + repr(tup[-1]): tup[0] for tup in pred_entities_with_pos}
    valid_pos_dict = {repr(tup[1]) + '_' + repr(tup[-1]): tup[0] for tup in valid_entities_with_pos}

    cnt = 0
    # 记录在预测出的实体中,有多少是和标注结果的一样
    for k, v in pred_pos_dict.items():
        if valid_pos_dict.get(k) == v:
            cnt += 1

    if cnt == 0:
        return 0, 0, 0
    p = cnt / len(pred_pos_dict)
    r = cnt / len(valid_pos_dict)
    f1 = 2 * p * r / (p + r)
    return p, r, f1


""" 于09.27增加,为了测试bert编码,并且建模测试 """


################################################################################################


def generate_instance_with_bert(text, tags):
    """
    对一组输入的文本与标注信息产生模型需要的样本
    """
    # 初始化文本的类别至O，之后对相应位置的类别进行修改
    labels_ = ['O'] * len(text)

    # 对于每个实体，都有B、M、E三种情况，B代表实体的首字，M代表实体的中间字，E代表实体的末尾。
    # 由于本项目中的实体不会出现单个字的情况，因此不定义S
    for tag in tags:
        # entity是实体类别，spos是起始位置，epos是终止位置，左开右闭
        entity, spos, epos = tag
        for pos in range(spos, epos):
            if pos == spos:
                labels_[pos] = entity2labels[entity][0]
            elif pos + 1 == epos:
                labels_[pos] = entity2labels[entity][2]
            else:
                labels_[pos] = entity2labels[entity][1]

    # text的前后会被加上cls和sep,同理,其对应的label的前后也要加上'O'
    text_ids = tokenizer.encode(list(text))
    label_ids = [labels2id[cate] for cate in ['O'] + labels_ + ['O']]

    return text_ids, label_ids


def generate_train_dataset_with_bert():
    """
    通过训练数据产生训练集
    --------------------
    :return (list) <- 列表元素为元组,每个元组又包含两个元素,一为文本id,二为实体标签id
    """
    train_dataset = []
    files = set([file.split('.')[0] for file in os.listdir(TRAIN_PATH)])
    for file in files:
        text, tags = open_train_files(TRAIN_PATH + file)
        # 由于bert的输入只能是512个字符,在数据集中有长文本.因此考虑对所有文本进行句号打断,
        # 具体方法在cut_long_text中进行实现
        cut_texts, cut_tags = cut_long_text(text, tags)
        for cut_text, cut_tag in zip(cut_texts, cut_tags):
            train_dataset.append(generate_instance_with_bert(cut_text, cut_tag))
    return train_dataset


################################################################################################
""" 于09.27增加,为了测试bert编码,并且建模测试 """

""" 于10.08增加,为了测试ernie编码,并且建模测试 """


################################################################################################
def generate_instance_with_ernie(text, tags):
    """
    对一组输入的文本与标注信息产生模型需要的样本
    """
    # 初始化文本的类别至O，之后对相应位置的类别进行修改
    labels_ = ['O'] * len(text)

    # 对于每个实体，都有B、M、E三种情况，B代表实体的首字，M代表实体的中间字，E代表实体的末尾。
    # 由于本项目中的实体不会出现单个字的情况，因此不定义S
    for tag in tags:
        # entity是实体类别，spos是起始位置，epos是终止位置，左开右闭
        entity, spos, epos = tag
        for pos in range(spos, epos):
            if pos == spos:
                labels_[pos] = entity2labels[entity][0]
            elif pos + 1 == epos:
                labels_[pos] = entity2labels[entity][2]
            else:
                labels_[pos] = entity2labels[entity][1]

    # text的前后会被加上cls和sep,同理,其对应的label的前后也要加上'O'
    text_ids = ernie_tokenizer.encode(list(text))
    label_ids = [labels2id[cate] for cate in ['O'] + labels_ + ['O']]

    return text_ids, label_ids


def generate_train_dataset_with_ernie():
    """
    通过训练数据产生训练集
    --------------------
    :return (list) <- 列表元素为元组,每个元组又包含两个元素,一为文本id,二为实体标签id
    """
    train_dataset = []
    files = set([file.split('.')[0] for file in os.listdir(TRAIN_PATH)])
    for file in files:
        text, tags = open_train_files(TRAIN_PATH + file)
        # 由于bert的输入只能是512个字符,在数据集中有长文本.因此考虑对所有文本进行句号打断,
        # 具体方法在cut_long_text中进行实现
        cut_texts, cut_tags = cut_long_text(text, tags)
        for cut_text, cut_tag in zip(cut_texts, cut_tags):
            train_dataset.append(generate_instance_with_ernie(cut_text, cut_tag))
    return train_dataset


################################################################################################
# todo: 使用空格切割长文本
""" 于09.27增加,为了测试bert编码,并且建模测试 """


def cut_long_text(text, tags):
    """
    对文本以句号为切分点进行切割,保留句号.同时切分出对应的实体,并更新实体的位置
    -------------------------------------------------------------------
    :params text (str) <- 长文本
    :params tags (list) <- 排序后的实体对,其中的元素为元组,每个元组包含3个元素,分别为实体类别,实体起始与终止位置
    """
    # 用于保存切分出的部分文本
    cut_texts = []
    # 用于保存切出文本对应的实体
    cut_tags = []
    # 记录上一次切分文本的位置
    last_text_pos = 0
    # 记录上一次切分实体的位置
    tag_start_pos = 0

    # 以。为切分标记
    for pos, ch in enumerate(text):
        if ch == '。':
            # 切分文本
            cut_text = text[last_text_pos: pos + 1]  # 保留。
            cut_texts.append(cut_text)

            # 切分实体
            cut_tag = []
            for tag_pos in range(tag_start_pos, len(tags)):
                tag = tags[tag_pos]
                if tag[-1] <= pos:
                    # 改变tag中实体的位置信息
                    new_tag = (tag[0], tag[1] - last_text_pos, tag[2] - last_text_pos)
                    cut_tag.append(new_tag)

                    # 如果tag_pos到了实体的最后一个，且已经进行更新了,之后将不会有实体再参与更新,
                    # 因此,将tag_start_pos进行相应的设置
                    if tag_pos + 1 == len(tags):
                        tag_start_pos = len(tags)
                else:
                    # 如果没有进上面的分支，说明当前实体的位置超过了截取出的文本。更新状态，进行下一轮。
                    tag_start_pos = tag_pos
                    break

            cut_tags.append(cut_tag)

            # 更新文本切分位置
            last_text_pos = pos + 1

    # 针对最后一段文本处理，因为整个长文本可能不会以句号结尾
    if last_text_pos != len(text):
        cut_text = text[last_text_pos:]
        cut_tag = []
        for tag_pos in range(tag_start_pos, len(tags)):
            tag = tags[tag_pos]
            new_tag = (tag[0], tag[1] - last_text_pos, tag[2] - last_text_pos)
            cut_tag.append(new_tag)
        cut_texts.append(cut_text)
        cut_tags.append(cut_tag)

    return cut_texts, cut_tags


def merge(cut_texts, cut_tags):
    """
    对切割的文本及对应的实体进行合并
    ------------------------------
    :param cut_texts (list) <- 待合并的文本,元素是str
    :param cut_tags (list) <- 待合并的实体,元素是tuple;元组的形式是(实体类别, 起始位置, 终止位置)
    :return (str, list) <- 合并后的文本及其对应的实体列表
    """
    last_text_len = 0
    tags = []
    for cut_text, cut_tag in zip(cut_texts, cut_tags):
        for tag in cut_tag:
            new_tag = (tag[0], tag[1] + last_text_len, tag[2] + last_text_len)
            tags.append(new_tag)
        last_text_len += len(cut_text)
    text = "".join(cut_texts)
    return text, tags
