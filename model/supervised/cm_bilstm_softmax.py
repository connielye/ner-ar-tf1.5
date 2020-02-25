import tensorflow as tf
import numpy as np


class BiLSTMSoftmax:
    def __init__(self, train_data, dev_data, eval_data, batch_size, seq_length, word_length, char_embedding_size,
                 embedding_size, n_tag, n_char_hidden, n_hidden, glove_embedding, vocab_size, char_size, learning_rate,
                 epochs, output_dir):
        self.train_data = train_data
        self.dev_data = dev_data
        self.eval_data = eval_data
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

        with tf.name_scope("input"):
            X = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_length], name="inputs")
            Y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_length], name="targets")
            L = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name="lengths")
            C = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_length, self.word_length], name="characters")

        with tf.variable_scope("glove_embedding"):
            glove_initializer = tf.constant_initializer(self.glove_embedding)
            glove_embeddings = tf.get_variable(name="glove_embeddings", shape=[self.vocab_size, self.embedding_size], initializer=glove_initializer, trainable=False)
            word_embeddings = tf.nn.embedding_lookup(glove_embeddings, X)

        with tf.variable_scope("embeddings"):
            characters = tf.Variable(tf.random_normal([self.char_size, self.char_embedding_size], -0.1, 0.1))
            character_embeddings = tf.nn.embedding_lookup(characters, C)
            character_embeddings = tf.reshape(character_embeddings, [-1, self.word_length, self.char_embedding_size])
            char_cell_fw = tf.nn.rnn_cell.LSTMCell(self.n_char_hidden)
            char_cell_bw = tf.nn.rnn_cell.LSTMCell(self.n_char_hidden)
            _, ((_, output_fw_state), (_, output_bw_state)) = tf.compat.v1.nn.bidirectional_dynamic_rnn(char_cell_fw, char_cell_bw, character_embeddings, dtype=tf.float32)
            output_char = tf.concat([output_fw_state, output_bw_state], 1)
            output_char = tf.reshape(output_char, [-1, self.seq_length, 2*self.n_char_hidden])
            embeddings = tf.concat([word_embeddings, output_char], 2)

        with tf.variable_scope("weights"):
            W = tf.Variable(tf.random_normal([2*self.n_hidden, self.n_tag]))
            b = tf.Variable(tf.random_normal([self.n_tag]))

        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
            cell_bw = tf.nn.rnn_cell.LSTMCell(self.n_hidden)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, embeddings, dtype=tf.float32)
            outputs = tf.concat(outputs, 2)
            outputs = tf.reshape(outputs, [-1, self.n_hidden*2])

        with tf.variable_scope("activation"):
            ffn = tf.nn.xw_plus_b(outputs, W, b)
            ffn = tf.reshape(ffn, [-1, self.seq_length, self.n_tag])

        with tf.variable_scope("softmax"):
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=ffn, labels=Y)
            mask = tf.sequence_mask(L)
            cost = tf.reduce_mean(tf.boolean_mask(cost, mask))
            preds = tf.cast(tf.arg_max(ffn, -1), dtype=tf.int32, name="output")
        return X, Y, L, C, cost, preds


    def generate_batch(self):
        batches = []
        batch_number = len(self.train_data)//self.batch_size
        for number in range(batch_number):
            batch = self.train_data[number*self.batch_size:(number+1)*self.batch_size]
            inputs_batch, targets_batch, length_batch, characters_batch = [], [], [], []
            for (input, target, length, characters) in batch:
                inputs_batch.append(input)
                targets_batch.append(target)
                characters_batch.append(characters)
            batches.append((inputs_batch, targets_batch, np.asarray(length), characters_batch))
        return batches

    def train(self):
        batches = self.generate_batch()
        X, Y, L, C, cost, preds = self.model()

        optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(cost)
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)

            for epoch in range(self.epochs):
                total_loss = 0
                print ("start training ......")
                for(inputs_batch, targets_batch, lengths_batch, characters_batch) in batches:
                    _, loss = session.run([optimizer, cost], feed_dict={X:inputs_batch, Y:targets_batch, L:lengths_batch, C:characters_batch})
                    total_loss+=loss

                if (epoch+1)%5 == 0:
                    average_loss = total_loss/len(batches)
                    print("Epoch:", "%04d"%(epoch+1), "cost=", "{:.6f}".format(average_loss))

                    dev_a_count = 0
                    dev_t_count = 0
                    for (input, target, length, character) in self.dev_data:
                        dev_input = []
                        dev_input.append(input)
                        dev_character = []
                        dev_character.append(character)
                        predict = session.run([preds], feed_dict={X:dev_input, L:np.asarray(length), C: dev_character})
                        for i in range(length[0]):
                            if predict[0][i] == target[i]:
                                dev_a_count += 1
                            dev_t_count+=1
                    dev_accu = float(dev_a_count/dev_t_count)
                    print("Epoch=", "%4d"%(epoch+1), "dev accuracy=", "{:.6f}".format(dev_accu))
            print("start validating...")
            val_a_count = 0
            val_t_count = 0
            for (input, target, length, character) in self.eval_data:
                eval_input = []
                eval_input.append(input)
                eval_character =[]
                eval_character.append(character)
                predict = session.run([preds], feed_dict={X:eval_input, L:np.asarray(length), C: eval_character})
                for i in range(length[0]):
                    if predict[0][0][i] == target[i]:
                        val_a_count+=1
                    val_t_count+=1
            val_accu = float(val_a_count/val_t_count)
            print("eval accuracy=", "{:.6f}".format(val_accu))

            builder = tf.saved_model.builder.SavedModelBuilder(self.output_dir)
            signature = tf.saved_model.signature_def_utils.build_signature_def(inputs={"inputs": tf.saved_model.utils.build_tensor_info(X), "lengths": tf.saved_model.utils.build_tensor_info(L),
                                                                                       "characters": tf.saved_model.utils.build_tensor_info(C)},
                                                                               outputs={"outputs":tf.saved_model.utils.build_tensor_info(preds)},)
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
        sentences.append(tokens[0].lower().rstrip().split(" "))
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
    char_size = len(char_dict)

    with open(glove_dir, 'r') as glove_file:
        glove_lines = glove_file.readlines()

    print("building embeddings and dictionary ...")
    vocab = []
    vectors = []
    for glove_line in glove_lines:
        tokens = glove_line.rstrip().split(' ')
        vocab.append(tokens[0])
        vectors.append(np.asarray([float(val) for val in tokens[1:]]))

    vectors.insert(0, np.random.randn(50))
    vectors.append(np.random.randn(50))
    embeddings = np.asarray(vectors)

    vocab.insert(0, '<PAD>')
    vocab.append('<UNK>')

    vocab_size = len(vocab)
    dictionary = {w: i for i, w in enumerate(vocab)}

    print("building input dataset ...")

    datasets = []
    for i in range(data_size):
        input = []
        for tok in sentences[i]:
            if tok in vocab:
                input.append(dictionary[tok])
            else:
                input.append(dictionary['<UNK>'])
        characters = []
        for word in sentences[i]:
            word_char = [char_dict[c] for c in word]
            if len(word_char) < 100:
                for m in range(len(word_char), 100):
                    word_char.append(28)
            characters.append(np.asarray(word_char))
        target = [label_dict[tag] for tag in tags[i]]
        if len(input) < 128:
            for j in range(len(input), 128):
                input.append(0)
                target.append(label_dict["O"])
                char_pad = np.asarray([28 for i in range(100)])
                characters.append(char_pad)
        datasets.append((np.asarray(input), np.asarray(target), lengths[i], np.asarray(characters)))

    print("train test splitting ...")
    np.random.shuffle(datasets)
    train_set = datasets[: int(data_size*0.6)]
    dev_set = datasets[int(data_size*0.6): int(data_size*0.8)]
    val_set = datasets[int(data_size*0.8):]

    return train_set, dev_set, val_set, vocab_size, char_size, embeddings



data_dir = "/home/bingxin/Downloads/tdata/ner/test/input.tsv"
glove_dir = "/home/bingxin/Downloads/tdata/ner/test/glove.6B.50d.txt"
label_dir = "/home/bingxin/Downloads/tdata/ner/test/labels.txt"
char_dir = "/home/bingxin/Downloads/tdata/ner/test/characters.txt"
output_dir = "/home/bingxin/Downloads/tdata/ner/test/tmp"
train_set, dev_set, val_set, vocab_size, char_size, embeddings = preprocess(data_dir, glove_dir, label_dir, char_dir)
model = BiLSTMSoftmax(train_set, dev_set, val_set, 1, 128, 100, 100, 50, 27, 1024, 1024, embeddings, vocab_size, char_size, 0.001, 1, output_dir)
model.train()