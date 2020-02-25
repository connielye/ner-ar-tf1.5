import tensorflow as tf
import numpy as np

class BiLSTMCRF:

    def __init__(self, train_data, dev_data, eval_data, batch_size, seq_length, word_length, char_embedding_size,
                 embedding_size, n_tag, n_char_hidden, n_hidden, glove_embedding, vocab_size, char_size, learning_rate, epochs, output_dir):
        self.train_data = train_data
        self.dev_data = dev_data
        self.eval_data =eval_data
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.word_length = word_length
        self.char_embedding_size = char_embedding_size
        self.embedding_size = embedding_size
        self.n_tag = n_tag
        self.n_char_hidden = n_char_hidden
        self.n_hidden = n_hidden
        self.glove_embedding = glove_embedding
        self.vocab_size = vocab_size
        self.char_size = char_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_dir = output_dir


    def model(self):

        with tf.name_scope('input'):
            X = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_length], name='inputs') #[batch_size, seq_length]
            Y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_length], name='labels') #[batch_size, seq_length]
            L = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='length') #[batch_size]
            C = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_length, self.word_length], name="characters")

        with tf.variable_scope('glove_embedding'):
            glove_initializer = tf.constant_initializer(self.glove_embedding)
            glove_embeddings = tf.get_variable(name="glove_word_embedding", shape=[self.vocab_size, self.glove_embedding],initializer=glove_initializer, trainable=False)
            word_embeddings = tf.nn.embedding_lookup(glove_embeddings, X) #[batch_size, seq_length, embedding_size]

        with tf.name_scope('embeddings'):
            characters = tf.Variable(tf.random_normal([self.char_size, self.char_embedding_size], -0.1, 0.1))
            char_embeddings = tf.nn.embedding_lookup(characters, C) #[batch_size, seq_length, word_length, char_embedding_size]
            char_embeddings = tf.reshape(char_embeddings, [-1, self.word_length, self.char_embedding_size]) #[batch_size*seq_length, word_length, char_embedding_size]
            cell_fw_char = tf.nn.rnn_cell.LSTMCell(self.n_char_hidden) #[n_char_hidden]
            cell_bw_char = tf.nn.rnn_cell.LSTMCell(self.n_char_hidden) #[n_char_hidden]
            _, ((_,output_state_fw), (_,output_state_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw_char, cell_bw_char, char_embeddings, dtype=tf.float32) #each [batch_size*seq_length, n_char_hidden]
            outputs_char = tf.concat([output_state_fw, output_state_bw], 1) #[batch_size*seq_length, 2*n_char_hidden]
            outputs_char = tf.reshape(outputs_char, [-1, self.seq_length, 2*self.n_char_hidden]) #[batch_size, seq_length, 2*n_char_hidden]
            embeddings = tf.concat([word_embeddings, outputs_char], 2) #[batch_size, seq_length, embedding_size+2*n_char_hidden]

        with tf.variable_scope('weights'):
            W = tf.Variable(tf.random_normal(shape=[self.n_hidden*2, self.n_tag])) #[2*n_hidden, n_tag]
            b = tf.Variable(tf.random_normal(shape=[self.n_tag])) #[n_tag]

        with tf.variable_scope('bi-lstm'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(self.n_hidden) #[n_hidden]
            cell_bw = tf.nn.rnn_cell.LSTMCell(self.n_hidden) #[n_hidden]

            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embeddings, dtype=tf.float32) #(output_fw, output_bw), each [batch_size, seq_length, n_hidden]
            outputs = tf.concat(outputs, 2) #[batch_size, seq_length, 2*n_hidden]
            outputs = tf.reshape(outputs, [-1, 2*self.n_hidden]) #[seq_length*batch_size, 2*n_hidden]

        with tf.variable_scope('activation'):
            ffn = tf.nn.xw_plus_b(outputs, W, b) #[seq_length*batch_size, n_tag]
            ffn = tf.reshape(ffn, shape=[-1, self.seq_length, self.n_tag]) #[batch_size, seq_length, n_tag]

        with tf.variable_scope('linear-crf'):
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(ffn, Y, L)
            cost = tf.reduce_mean(-log_likelihood)

        with tf.variable_scope('predict'):
            decode_sequence, decode_score = tf.contrib.crf.crf_decode(ffn, transition_params, L)
            decode_sequence = tf.cast(decode_sequence[0], dtype=tf.int32, name='outputs')

        return X, Y, L, ffn, cost, decode_sequence, decode_score


    def generate_batch(self):
        batches = []
        batch_number = len(self.train_data)//self.batch_size
        for number in range(batch_number):
            batch = self.train_data[number*self.batch_size:(number+1)*self.batch_size]
            inputs_batch, targets_batch, lengths_batch = [], [], []
            for (input, target, length) in batch:
                inputs_batch.append(input)
                targets_batch.append(target)
                lengths_batch.append(length)
            batches.append((inputs_batch, targets_batch, lengths_batch))
        return batches


    def train(self):
        batches = self.generate_batch()
        X, Y, L, ffn, cost, decode_sequence, decode_score = self.model()

        optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(cost)
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)

            for epoch in range(self.epochs):
                total_loss = 0

                for(inputs_batch, targets_batch, lengths_batch) in batches:
                    _, loss = session.run([optimizer, cost], feed_dict={X: inputs_batch, Y:targets_batch, L:lengths_batch})
                    total_loss += loss

                if (epoch+1)%5 == 0:
                    average_loss = total_loss/len(batches)
                    print("Epoch:", "%04d"%(epoch+1), "cost=", "{:.6f}".format(average_loss))

                    dev_a_count = 0
                    dev_t_count = 0
                    for(input, target, length) in self.dev_data:
                        dev_input = []
                        dev_input.append(input)
                        dev_length = []
                        dev_length.append(length)
                        sequence, score = session.run([decode_sequence, decode_score], feed_dict={X:input, L: length})
                        for i in range(length[0]):
                            if sequence[0][i] == target[i]:
                                dev_a_count += 1
                        dev_t_count += length[0]
                    accuracy = float(dev_a_count/dev_t_count)
                    print("Epoch=", "%4d"%(epoch+1), "dev accuracy=", "{.6f}".format(accuracy))

            val_a_count = 0
            val_t_count = 0
            for (input, target, length) in self.eval_data:
                eval_input = []
                eval_input.append(input)
                eval_length = []
                eval_length.append(length)
                sequence, score = session.run([decode_sequence, decode_score], feed_dict={X:input, L:length})
                for i in range(length[0]):
                    if sequence[0][i] == target[i]:
                        val_a_count += 1
                    val_t_count += length[0]
            accuracy = val_a_count/val_t_count
            print("eval accuracy=", "{.6f}".format(accuracy))

            builder = tf.saved_model.builder.SavedModelBuilder(self.output_dir)
            signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=tf.saved_model.utils.build_tensor_info(X),
                                                                               outputs=tf.saved_model.utils.build_tensor_info(decode_sequence),
                                                                               method_name="tensorflow/serving/predict")
            builder.add_meta_graph_and_variables(session, tags=[tf.saved_model.tag_constants.SERVING], signature_def_map={tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature},
                                                 strip_default_attrs=True)
            builder.save()


def preprocess(data_dir, glove_dir, label_dir, char_dir):

    with open(data_dir, 'r') as data_file:
        lines = data_file.readlines()

    data_size = len(lines)
    sentences = []
    tags = []
    lengths = []
    for line in lines:
        tokens = line.rstrip().split("\t")
        sentences.append(tokens[0].rstrip().split(" "))
        tags.append(tokens[1].rstrip().split(" "))
        lengths.append([len(tokens[1].rstrip().split(" "))])

    with open(label_dir, 'r') as label_file:
        label_lines = label_file.readlines()

    labels = []
    for label_line in label_lines:
        labels.append(label_line.rstrip())
    label_dict = {w:i for i, w in enumerate(labels)}

    with open(char_dir, 'r') as char_file:
        char_lines = char_file.readlines()

    chars = []
    for char_line in char_lines:
        chars.append(char_line.rstrip())
    char_dict = {w:i for i, w in enumerate(chars)}

    with open(glove_dir, 'r') as glove_file:
        glove_lines = glove_file.readlines()

    print("building embeddings and dictionary ...")
    vocab = []
    vectors = []
    for glove_line in glove_lines:
        tokens = glove_line.rstrip().split(' ')
        vocab.append(tokens[0])
        vectors.append(np.asarray(tokens[1:]))

    vectors.insert(0, np.random.randn(100))
    vectors.append(np.random.randn(100))
    embeddings = np.asarray(vectors)

    vocab.insert(0, '<PAD>')
    vocab.append('<UNK>')

    vocab_size = len(vocab)
    dictionary = {w: i for i, w in enumerate(vocab)}

    print("building input dataset ...")

    datasets = []
    for i in range(data_size):
        input = [dictionary[token] for token in sentences[i]]
        charactors = [[char_dict[c]] for word in sentences[i] for c in word]
        target = [label_dict[tag] for tag in tags[i]]
        if len(input) < 128:
            for i in range(len(input), 128):
                input.append(0)
                target.append(label_dict["O"])
        datasets.append((np.asarray(input), np.asarray(target), np.asarray(lengths[i]), np.asarray(charactors)))

    print("train test splitting ...")
    np.random.shuffle(datasets)
    train_set = datasets[: int(data_size*0.6)]
    dev_set = datasets[int(data_size*0.6): int(data_size*0.8)]
    val_set = datasets[int(data_size*0.8):]

    return train_set, dev_set, val_set, vocab_size, embeddings



