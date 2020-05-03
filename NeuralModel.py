import tensorflow as tf
import numpy as np
import pickle


class Model(tf.keras.Model):
    def __init__(self, vocab_size):

        """
        The Model class predicts the next words in a sequence.
        Feel free to initialize any variables that you find necessary in the constructor.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, emnbedding_size

        self.vocab_size = vocab_size
        self.rnn_size = 256
        self.window_size = 4
        self.embedding_size = 40
        self.batch_size = 64
        self.learning_rate = 0.01

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # - use tf.keras.layers.Dense for feed forward layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN

        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.window_size)
        self.lstm = tf.keras.layers.LSTM(self.rnn_size, return_sequences=True, return_state=True)
        # self.dense1 = tf.keras.layers.Dense(self.embedding_size, activation='softmax')
        self.dense = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, the final_state(s) of the rnn

        -Note 1: If you use an LSTM, the final_state will be the last two outputs of calling the rnn.
        If you use a GRU, it will just be the second output.

        -Note 2: You only need to use the initial state during generation. During training and testing it can be None.
        """
        # print(np.shape(inputs))
        mid = self.embedding_layer(inputs)
        # print(np.shape(mid))
        output, state1, state2 = self.lstm(mid)
        probs = self.dense(output)
        # print(np.shape(probs))
        # print(probs)
        return (probs, (state1, state2))


    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy
        # print(np.shape(labels), np.shape(probs))
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=model.learning_rate)

    indices = range(0, len(train_labels))
    shuffledIndices = tf.random.shuffle(indices)
    inputs = tf.gather(train_inputs, shuffledIndices)
    labels = tf.gather(train_labels, shuffledIndices)

    for i in range(0, len(train_labels), model.batch_size):
        if len(train_labels) - i < 0:
            break

        with tf.GradientTape() as tape:
            probs, _ = model.call(inputs[i:i+model.batch_size], None)
            losso = model.loss(probs, labels[i:i+model.batch_size])
        gradients = tape.gradient(losso, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # print(losso)
    pass

def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set

    Note: perplexity is exp(total_loss/number of predictions)

    """
    total = 0
    count = 0
    for i in range(0, len(test_labels), model.batch_size):
        if len(test_labels) - i < 0:
            break
        probs, _ = model.call(test_inputs[i:i+model.batch_size], None)
        total += model.loss(probs, test_labels[i:i+model.batch_size])
        count += 1
    return(np.exp(total/count))

def calc_individual_perplexity(model, test_inputs, test_labels):
    losses = np.zeros((len(test_labels)))
    for i in range(len(test_labels)):
        probs, _ = model.call(test_inputs[i], None)
        losses[i] = model.loss(probs, test_labels[i])
    return(np.exp(losses))

def generate_sentence(word1, length, vocab, model):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    This is only for your own exploration. What do the sequences your RNN generates look like?

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    reverse_vocab = {idx:word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits,previous_state = model.call(next_input,previous_state)
        out_index = np.argmax(np.array(logits[0][0]))

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))



def main():
    # TO-DO: Pre-process and vectorize the data
    training, testing, word2id = get_data("data/train.txt", "data/test.txt")
    # print(training)
    print(word2id)
    # HINT: Please note that you are predicting the next word at each timestep, so you want to remove the last element
    # from train_x and test_x. You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see very, very small perplexities.

    # TO-DO:  Separate your train and test data into inputs and labels
    train_x = []
    train_y = []
    for i in range(0, len(training) - 21, 20):
        train_x.append(training[i:i+20])
        train_y.append(training[i+1:i+21])
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    test_x = []
    test_y = []
    for i in range(0, len(testing) - 21, 20):
        test_x.append(testing[i:i+20])
        test_y.append(testing[i+1:i+21])
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    # TODO: initialize model and tensorflow variables
    model = Model(len(word2id))

    # TODO: Set-up the training step
    # print("Training")
    # start = timer()
    train(model, train_x, train_y)
    # end = timer()
    # print("time:", end-start)

    # # TODO: Set up the testing steps
    # print("Testing")
    print(test(model, test_x, test_y))

    # print("batch_size:", model.batch_size)
    # print("embedding_size:", model.embedding_size)
    # print("learning_rate:", model.learning_rate)

    # Print out perplexity


if __name__ == '__main__':
    # main()
    w2id = get_ptb_w2id("data/ptb.csv")
    # print(w2id)
    training_data = get_ptb_data(w2id)

    pickle.dump(w2id, open("data/w2id.p", "wb"))
    pickle.dump(training_data, open("data/ptb_training_id.p", "wb"))
