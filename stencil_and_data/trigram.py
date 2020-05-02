import tensorflow as tf
import numpy as np
from preprocess import get_data

# from timeit import default_timer as timer


class Model(tf.keras.Model):
    def __init__(self, vocab_size):

        """
        The Model class predicts the next words in a sequence,
        Feel free to initialize any variables that you find necessary in the constructor.

        vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize vocab_size, emnbedding_size
        self.vocab_size = vocab_size
        self.embedding_size = 50
        self.batch_size = 128

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        hiddenLayerSize = vocab_size
        self.E = tf.Variable(tf.random.truncated_normal([vocab_size, self.embedding_size], stddev=0.1))
        self.W = tf.Variable(tf.random.truncated_normal([self.embedding_size * 2, hiddenLayerSize], stddev=0.1))
        self.b = tf.Variable(tf.random.truncated_normal([hiddenLayerSize], stddev=0.1))

        # self.W1 = tf.Variable(tf.random.truncated_normal([hiddenLayerSize, vocab_size], stddev=.1))
        # self.b1 = tf.Variable(tf.random.truncated_normal([vocab_size], stddev=.1))
        # self.W2 = tf.Variable(tf.random.normal([hiddenLayerSize, hiddenLayerSize], stddev=.1, dtype=tf.float32))
        # self.b2 = tf.Variable(tf.random.normal([hiddenLayerSize], stddev=.1, dtype=tf.float32))
        # self.W3 = tf.Variable(tf.random.normal([hiddenLayerSize, hiddenLayerSize], stddev=.1, dtype=tf.float32))
        # self.b3 = tf.Variable(tf.random.normal([hiddenLayerSize], stddev=.1, dtype=tf.float32))

    def call(self, inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        :param inputs: word ids of shape (batch_size, 2)

        :return: prbs: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """
        # print(np.shape(inputs))
        embedding1 = tf.nn.embedding_lookup(self.E, inputs[:, 0])
        embedding2 = tf.nn.embedding_lookup(self.E, inputs[:, 1])
        # print(np.shape(embedding1), " ", np.shape(embedding2))
        embedding = tf.concat([embedding1, embedding2], 1)
        # print(np.shape(embedding))

        # print(np.shape(self.W))
        logits = tf.matmul(embedding, self.W) + self.b
        # print(np.shape(logits))

        # dense1 = tf.matmul(logits, self.W1) + self.b1

        # dense2 = tf.nn.relu(tf.matmul(dense1, self.W2) + self.b2)
        #
        # dense3 = tf.matmul(dense2, self.W3) + self.b3

        # print(np.shape(dense3))
        return tf.nn.softmax(logits)

    def loss_function(self,logits,labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction
        :param prbs: a matrix of shape (batch_size, vocab_size)
        :return: the loss of the model as a tensor of size 1
        """
        return tf.math.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=False))


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.
    Remember to shuffle your inputs and labels - ensure that they are shuffled in the same order.
    Also you should batch your input and labels here.
    :param model: the initilized model to use for forward and backward pass
    :param train_input: train inputs (all inputs for training) of shape (num_inputs,2)
    :param train_input: train labels (all labels for training) of shape (num_inputs,)
    :return: None
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01) # TODO: try 0.001

    indices = range(0, len(train_labels))
    shuffledIndices = tf.random.shuffle(indices)
    inputs = tf.gather(train_inputs, shuffledIndices)
    labels = tf.gather(train_labels, shuffledIndices)

    # inputs = train_inputs
    # labels = train_labels

    for i in range(0, len(train_labels), model.batch_size):
        if len(train_labels) - i < 0:
            break

        with tf.GradientTape() as tape:
            # print(np.shape(inputs[i:i+model.batch_size]))
            probs = model.call(inputs[i:i+model.batch_size])
            loss = model.loss_function(probs, labels[i:i+model.batch_size])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_input, test_labels):
    """
    Runs through all test examples.
    And test input should be batched here as well.
    :param model: the trained model to use for prediction
    :param test_input: train inputs (all inputs for testing) of shape (num_inputs,2)
    :param test_input: train labels (all labels for testing) of shape (num_inputs,)

    :returns: the perplexity of the test set
    """
    total = 0
    count = 0
    for i in range(0, len(test_labels), model.batch_size):
        if len(test_labels) - i < 0:
            break
        probs = model.call(test_input[i:i+model.batch_size])
        total += model.loss_function(probs, test_labels[i:i+model.batch_size])
        count += 1
    return(np.exp(total/count))

def generate_sentence(word1, word2, length, vocab,model):
    """
    Given initial 2 words, print out predicted sentence of target length.

    :param word1: string, first word
    :param word2: string, second word
    :param length: int, desired sentence length
    :param vocab: dictionary, word to id mapping
    :param model: trained trigram model

    """
    reverse_vocab = {idx:word for word, idx in vocab.items()}
    output_string = np.zeros((1,length), dtype=np.int)
    output_string[:,:2] = vocab[word1], vocab[word2]

    for end in range(2,length):
        start = end - 2
        output_string[:, end] = np.argmax(model(output_string[:,start:end]), axis=1)
    text = [reverse_vocab[i] for i in list(output_string[0])]

    print(" ".join(text))

def main():
    # TODO: Pre-process and vectorize the data using get_data from preprocess
    training, testing, word2id = get_data("data/train.txt", "data/test.txt")

    # TO-DO: Separate your train and test data into inputs and labels
    train_inputs = np.zeros((len(training) - 2, 2), dtype=int)
    train_labels = np.zeros((len(training) - 2), dtype=int)
    for i in range(len(training) - 2):
        train_inputs[i] = ([training[i], training[i+1]])
        train_labels[i] = (training[i+2])

    test_inputs = np.zeros((len(testing) - 2, 2), dtype=int)
    test_labels = np.zeros((len(testing) - 2), dtype=int)
    for i in range(len(testing) - 2):
        test_inputs[i] = ([testing[i], testing[i+1]])
        test_labels[i] = (testing[i+2])


    # TODO: initialize model and tensorflow variables
    model = Model(len(word2id))

    # print("Before training:")
    # start = timer()
    # print(test(model, test_inputs, test_labels))
    # end = timer()
    # print("time:", end-start)

    # TODO: Set-up the training step
    # print("Training")
    # start = timer()
    train(model, train_inputs, train_labels)
    # end = timer()
    # print("time:", end-start)

    # TODO: Set up the testing steps
    # print("Testing")
    results = test(model, test_inputs, test_labels)
    print(results)

    # Print out perplexity
    # generate_sentence("Ronald", "is", 10, word2id, model)
    # generate_sentence("Soviet", "Union", 10, word2id, model)

if __name__ == '__main__':
    main()
