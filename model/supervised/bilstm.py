import tensorflow as tf
import numpy as np



class BiLSTM:
    def __init__(self, train_data, dev_data, eval_data, glove_embedding, n_hidden, seq_length, learning_rate, epochs,
                 batch_size, vocab_size, embedding_size, n_tag):
        self.train_data = train_data
        self.dev_data = dev_data
        self.eval_data = eval_data
        self.glove_embedding = glove_embedding
        self.n_hidden = n_hidden
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_tag = n_tag

    def model(self):
        with tf.name_scope("input"):
            X = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_length], name="inputs")
            Y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.n_tag], name="targets")

        with tf.variable_scope("embedding"):
            glove_initializer = tf.constant_initializer(self.glove_embedding)
            glove_embeddings = tf.get_variable(name="glove_embeddings", shape=[self.vocab_size, self.embedding_size],
                                               initializer=glove_initializer, trainable=False)
            word_embeddings = tf.nn.embedding_lookup(glove_embeddings, X)

        with tf.variable_scope("weight"):
            W = tf.Variable(tf.random_normal(shape=[2*self.n_hidden, self.n_tag]))
            b = tf.Variable(tf.random_normal(shape=[self.n_tag]))

        with tf.variable_scope("bilstm"):
            cell_fw = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
            cell_bw = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, word_embeddings, dtype=tf.float32)
            outputs = tf.concat(outputs, 2)
            outputs = tf.reshape(outputs, shape=[-1, self.seq_length, 2*self.n_hidden])

        with tf.variable_scope("activation"):
            ffn = tf.nn.xw_plus_b(outputs, W, b)
            ffn = tf.reshape(ffn, [-1, ])
