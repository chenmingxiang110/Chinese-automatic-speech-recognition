import tensorflow as tf
import numpy as np
from layerNormedGRU import layerNormedGRU

class model:

    def __init__(self, num_class, topk_paths = 10):
        self.xs = tf.placeholder(tf.float32, [None, 1000, 161])
        self.ys = tf.sparse_placeholder(tf.int32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.isTrain = tf.placeholder(tf.bool, name='phase')

        xs_input = tf.expand_dims(self.xs, 3)

        conv1 = self._nn_conv_bn_layer(xs_input, 'conv_1', [11, 41, 1, 32], [3, 2])
        conv2 = self._nn_conv_bn_layer(conv1, 'conv_2', [11, 21, 32, 64], [1, 2])
        conv_out = tf.reshape(conv2, [-1, 334, 41*64])
        biRNN1 = self._biRNN_bn_layer(conv_out, 'biRNN_1', 1024)
        biRNN2 = self._biRNN_bn_layer(biRNN1, 'biRNN_2', 1024)
        biRNN3 = self._biRNN_bn_layer(biRNN2, 'biRNN_3', 1024)
        biRNN4 = self._biRNN_bn_layer(biRNN3, 'biRNN_4', 1024)
        biRNN5 = self._biRNN_bn_layer(biRNN4, 'biRNN_5', 1024)

        self.phonemes = tf.layers.dense(biRNN5, num_class)

        # Notes: tf.nn.ctc_loss performs the softmax operation for you, so
        # inputs should be e.g. linear projections of outputs by an LSTM.
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.ys, inputs=self.phonemes, sequence_length=self.seq_len,
                                                  ignore_longer_outputs_than_inputs=True, time_major=False))

        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.7, beta2=0.9)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -400., 400.), var) for grad, var in gvs if grad is not None]
            self.train_op = optimizer.apply_gradients(capped_gvs)

        self.prediction, log_prob = tf.nn.ctc_beam_search_decoder(tf.transpose(self.phonemes,[1,0,2]), self.seq_len, top_paths=topk_paths, merge_repeated=False)

        self.loss_summary = tf.summary.scalar("loss", self.loss)
        self.merged = tf.summary.merge_all()

    def _nn_conv_bn_layer(self, inputs, scope, shape, strides):
        with tf.variable_scope(scope):
            W_conv = tf.get_variable("W", shape=shape, initializer=tf.contrib.layers.xavier_initializer())
            h_conv = tf.nn.conv2d(inputs, W_conv, strides=[1, strides[0], strides[1], 1], padding='SAME', name="conv2d")
            b = tf.get_variable("bias" , shape=[shape[3]], initializer=tf.contrib.layers.xavier_initializer())
            h_bn = tf.layers.batch_normalization(h_conv+b, training = self.isTrain)
            h_relu = tf.nn.relu6(h_bn, name="relu6")
            return h_relu

    def _biRNN_bn_layer(self, input, scope, hidden_units, cell = "LayerNormedGRU"):
        with tf.variable_scope(scope):
            if cell == 'GRU':
                fw_cell = tf.nn.rnn_cell.GRUCell(hidden_units, activation=tf.nn.relu, name = 'fw_cell')
                bw_cell = tf.nn.rnn_cell.GRUCell(hidden_units, activation=tf.nn.relu, name = 'bw_cell')
            elif cell == 'LSTM':
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units, activation=tf.nn.relu, name = 'fw_cell')
                bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units, activation=tf.nn.relu, name = 'bw_cell')
            elif cell == 'vanila':
                fw_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_units, activation=tf.nn.relu, name = 'fw_cell')
                bw_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_units, activation=tf.nn.relu, name = 'bw_cell')
            elif cell == 'LayerNormedGRU':
                with tf.variable_scope('fw_cell'):
                    fw_cell = layerNormedGRU(hidden_units, activation=tf.nn.relu)
                with tf.variable_scope('bw_cell'):
                    bw_cell = layerNormedGRU(hidden_units, activation=tf.nn.relu)
            else:
                raise ValueError("Invalid cell type: "+str(cell))

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input, dtype=tf.float32, scope="bi_dynamic_rnn")
            # output_fw_bn = tf.layers.batch_normalization(output_fw, training = self.isTrain, name = 'output_fw_bn')
            # output_bw_bn = tf.layers.batch_normalization(output_bw, training = self.isTrain, name = 'output_bw_bn')
            # bilstm_outputs_concat_1 = tf.concat([output_fw_bn, output_bw_bn], 2)
            bilstm_outputs_concat_1 = tf.concat([output_fw, output_bw], 2)
            return bilstm_outputs_concat_1

    def train(self, sess, learning_rate, xs, ys):
        _, loss, summary = sess.run([self.train_op, self.loss, self.merged], feed_dict = {self.isTrain: True, self.learning_rate: learning_rate, self.seq_len: np.ones(xs.shape[0])*334, self.xs: xs, self.ys: ys})
        return loss, summary

    def get_loss(self, sess, xs, ys):
        loss = sess.run(self.loss, feed_dict = {self.isTrain: False, self.seq_len: np.ones(xs.shape[0])*334, self.xs: xs, self.ys: ys})
        return loss

    def predict(self, sess, xs):
        prediction = sess.run(self.prediction, feed_dict = {self.isTrain: False, self.seq_len: np.ones(xs.shape[0])*334, self.xs: xs})
        return prediction
