import tensorflow as tf
import tensorflow_addons as tfa
from transformers import TFBertModel

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class BilstmCRF(tf.keras.Model):
    def __init__(self, label_size):
        super(BilstmCRF, self).__init__(self)
        self.embedding = TFBertModel.from_pretrained(
            '../resource/ernie_torch/',
            from_pt=True,
            output_hidden_states=True
        )
        # 设置Embedding层不训练
        self.embedding.trainable = False

        # CNN
        self.conv1 = tf.keras.layers.Conv1D(filters=192, kernel_size=1, padding="same", activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(filters=192, kernel_size=2, padding="same", activation='relu')
        self.conv3 = tf.keras.layers.Conv1D(filters=192, kernel_size=3, padding="same", activation='relu')
        self.conv4 = tf.keras.layers.Conv1D(filters=192, kernel_size=4, padding="same", activation='relu')

        self.dropout = tf.keras.layers.Dropout(0.4)

        # 双向LSTM
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))

        # CRF层参数
        self.transition = tf.Variable(tf.initializers.GlorotNormal()(shape=(label_size, label_size)))
        
        self.dense = tf.keras.layers.Dense(label_size, name='dense_out')
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inputs, attention_mask, targets=None, training=None):
        """
        :param inputs (tensor) <-
        :param attention_mask (tensor) <-
        :param targets (tensor) <-
        :param training (bool) <-
        :return:
        """
        _, _, hidden_states = self.embedding(inputs, attention_mask=tf.cast(attention_mask, dtype=tf.int32))
        embedding_out = tf.concat(hidden_states[-3: -1], axis=-1)
        embedding_out = self.layer_norm1(embedding_out)
        conv1_out = self.conv1(embedding_out)
        conv2_out = self.conv2(embedding_out)
        conv3_out = self.conv3(embedding_out)
        conv4_out = self.conv4(embedding_out)
        conv_out = tf.concat((conv1_out, conv2_out, conv3_out, conv4_out), axis=-1)
        conv_out = self.layer_norm2(conv_out)
        bilstm_out = self.bilstm(conv_out, mask=attention_mask)
        bilstm_out = self.dropout(bilstm_out, training=training)
        dense_out = self.dense(bilstm_out)

        # seq_len是batch中每个文本的真实长度
        seqs_len = tf.reduce_sum(tf.cast(tf.math.not_equal(inputs, 0), dtype=tf.int16), axis=1)

        if targets is not None:
            """ 训练过程 """
            log_likelihood, self.transition = tfa.text.crf_log_likelihood(dense_out, targets, seqs_len, self.transition)
            return log_likelihood
        else:
            """ 预测过程，需要解码 """
            decode_tags, best_score = tfa.text.crf_decode(dense_out, self.transition, seqs_len)
            return decode_tags