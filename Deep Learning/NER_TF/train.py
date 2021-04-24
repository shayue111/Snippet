# -*- coding: utf-8 -*-
# @Time    : 2020/9/24 下午10:55
# @Author  : wangt
# @Email   : wt0504@zju.edu.cn
# @File    : train.py

from utils import *
from model import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# 从训练文件中生成数据集列表
dataset = generate_train_dataset_with_ernie()
BATCH_SIZE = 128
EPOCHES = 10

tf.print("整体数据集的大小为: ", len(dataset))


# 产生数据集
def generator():
    for text, tag in dataset:
        yield text, tag


# 生成tf.data类型的结构
tf_dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.int32, tf.int32))

""" 切分训练集与测试集 """
step_size = len(dataset) // BATCH_SIZE
train_size = int(step_size * 0.8)

""" 设置模型与优化器 """
optimizer = tf.keras.optimizers.Adam(1e-3, decay=1e-2)
optimizer_crf = tf.keras.optimizers.Adam(1e-1, decay=1e-2)
model = BilstmCRF(label_size)

""" 设置模型路径 """
ckpt = tf.train.Checkpoint(model=model)
ckpt.restore(tf.train.latest_checkpoint('../resource/model_1019/'))
ckpt_manager = tf.train.CheckpointManager(
    ckpt,
    '../resource/model_1019/',
    checkpoint_name='model.ckpt',
    max_to_keep=10
)


@tf.function(experimental_relax_shapes=True)
def train_step(batch_x, batch_y):
    attention_mask = tf.math.not_equal(batch_x, 0)
    with tf.GradientTape() as tape:
        log_likelihood = model(batch_x, attention_mask, batch_y, True)
        loss_ = -tf.reduce_mean(log_likelihood)
    gradients = tape.gradient(loss_, model.trainable_variables)
    # 取出最后一个gradient,它是crf中transition matrix的参数.对它需要更新得快一些.
    optimizer.apply_gradients(zip(gradients[:-1], model.trainable_variables[:-1]))
    optimizer_crf.apply_gradients(zip([gradients[-1]], [model.trainable_variables[-1]]))
    return loss_


@tf.function(experimental_relax_shapes=True)
def valid_step_v2(batch_x, batch_y):
    # 获取mask
    attention_mask = tf.math.not_equal(batch_x, 0)
    batch_pred = model(batch_x, attention_mask)

    # 计算标注与预测的比较
    inter = tf.math.equal(batch_y, batch_pred)
    h1 = tf.math.not_equal(batch_y, 39)
    h2 = tf.math.not_equal(batch_pred, 39)
    inter = tf.math.logical_and(tf.math.logical_and(inter, attention_mask), tf.math.logical_and(h1, h2))
    inter = tf.reduce_sum(tf.cast(inter, dtype=tf.int16))

    # 计算
    S = tf.math.logical_and(tf.math.not_equal(batch_pred, 39), attention_mask)
    S = tf.reduce_sum(tf.cast(S, dtype=tf.int16))

    G = tf.math.logical_and(tf.math.not_equal(batch_y, 39), attention_mask)
    G = tf.reduce_sum(tf.cast(G, dtype=tf.int16))

    P = inter / S
    P = tf.where(tf.math.is_inf(P), 0.0, P)
    P = tf.where(tf.math.is_nan(P), 0.0, P)

    R = inter / G
    R = tf.where(tf.math.is_inf(R), 0.0, R)
    R = tf.where(tf.math.is_nan(R), 0.0, R)

    F1 = 2 * P * R / (P + R)
    F1 = tf.where(tf.math.is_inf(F1), 0.0, F1)
    F1 = tf.where(tf.math.is_nan(F1), 0.0, F1)

    return tf.reduce_mean(P), tf.reduce_mean(R), tf.reduce_mean(F1)


""" 开始训练 """

last_batch_f1 = tf.Variable(0.)
last_loss = 9999
for epoch in tf.range(EPOCHES):
    # 形成batch并且对长度较短的文本用0填充
    tf_dataset = tf_dataset.shuffle(len(dataset))
    tf_dataset_batch = tf_dataset.padded_batch(
        batch_size=BATCH_SIZE,
        padded_shapes=([None], [None]),
        padding_values=(
            tf.constant(0, dtype=tf.int32),
            tf.constant(39, dtype=tf.int32)
        ),
        drop_remainder=False
    )

    train_tf_dataset = tf_dataset_batch.take(train_size)
    valid_tf_dataset = tf_dataset_batch.skip(train_size)

    for step, (train_X, train_Y) in tf_dataset_batch.enumerate(start=1):
        train_loss = train_step(train_X, train_Y)

        if step % 10 == 0:
            batch_p, batch_r, batch_f1 = tf.Variable(0.), tf.Variable(0.), tf.Variable(0.)
            for valid_X, valid_Y in tf_dataset_batch:
                p, r, f1 = valid_step_v2(valid_X, valid_Y)
                batch_p = batch_p + p / tf.constant(step_size, dtype=tf.float32)
                batch_r = batch_r + r / tf.constant(step_size, dtype=tf.float32)
                batch_f1 = batch_f1 + f1 / tf.constant(step_size, dtype=tf.float32)

            tf.print("epoch = ", epoch, " p = ", batch_p, " r = ", batch_r, " f1 = ", batch_f1,
                     " train_loss = ", train_loss)

            if epoch == 0 and step == 10:
                last_batch_f1 = batch_f1
            elif batch_f1 > last_batch_f1:
                last_batch_f1 = batch_f1
                ckpt_manager.save()
                tf.print("batch_f1 = {}, 模型已保存!".format(batch_f1))