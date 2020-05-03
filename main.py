from Preprocess import *
from SymbolicModel import SymbolicModel
from NeuralModel import Model, train, generate_sentence, test

import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from signal import signal, SIGINT
from sys import exit

def exit_handler(signal_received, frame):
    # Handle any cleanup here
    print('Early Exit, Saving Entries')
    if sym is not None:
        probs = sym.out_probs
        prods = sym.out_prods
        labels = lab[:len(probs)]
        inputs = inp[:len(probs)]
        visualize_results(probs, labels, "early-exit")
        export_results(probs, labels, inputs, prods)
    exit(0)

def visualize_results(probs, labels, title):
    y = np.random.rand(len(probs))
    plt.scatter(probs, y, c=labels, alpha=0.9)
    plt.savefig("results/%s" % title)
    if SHOW:
        plt.show()

def export_neural_results(probs, labels, lens):
    with open("results/neur_res.txt", "w") as filehandle:
        for i in range(len(probs)):
            filehandle.write("%s\t%s\t%s\n" % (probs[i], labels[i], lens[i]))

def export_results(probs, labels, inputs, prods):
    sents = list(map(lambda x: " ".join(x), inputs))
    with open("results/res.txt", "w") as filehandle:
        for i in range(len(probs)):
            filehandle.write("%s\t%s\t%s\t%s\n" % (sents[i], labels[i], probs[i], prods[i]))

TESTING = False
SHOW = True
sym = None
lab = None

def symbo():
    if not TESTING:
        signal(SIGINT, exit_handler)
    t = time.time()
    # grammar = create_pcfg_from_treebank(pickle_it=True, log_it=True, filename="treebank_full", full=True)
    if TESTING:
        grammar = pickle.load(open("pickled-vars/treebank-grammar.p", "rb"))
    else:
        grammar = pickle.load(open("pickled-vars/treebank_full-grammar.p", "rb"))
    print("Grammar loaded in %.1fs" % (time.time()-t))

    t = time.time()
    # parser = create_viterbi_parser(grammar, pickle_it=True, filename="viterbi_full")
    if TESTING:
        parser = pickle.load(open("pickled-vars/viterbi-parser.p", "rb"))
    else:
        parser = pickle.load(open("pickled-vars/viterbi_full-parser.p", "rb"))
    print("Parser loaded in %.1fs" % (time.time()-t))

    t = time.time()
    train_inputs, train_labels, test_inputs, test_labels = load_cola()
    print("CoLA loaded in %.1fs" % (time.time()-t))

    global sym
    sym = SymbolicModel(grammar, parser)
    t = time.time()
    train_inputs, train_labels = sym.filter_coverage(train_inputs, train_labels)
    test_inputs, test_labels = sym.filter_coverage(test_inputs, test_labels)
    print("Examples filtered for coverage in %.1fs" % (time.time()-t))

    global lab
    global inp
    t = time.time()
    lab = train_labels[:20]
    inp = train_inputs[:20]
    probs, prods = sym.produce_normalized_log_probs(inp, 'sum-norm')
    print("Calculated sentence probabilities in %.1fs" % (time.time()-t))
    # probs, prods = sym.produce_normalized_log_probs(["John John John John .".split()])
    if TESTING:
        print(probs)

    visualize_results(probs, lab, "fig5")
    export_results(probs, lab, inp, prods)

def neuro():
    # w2id = get_ptb_w2id("data/ptb.csv")
    # training = get_ptb_data(w2id)
    #
    # pickle.dump(w2id, open("pickled-vars/w2id.p", "wb"))
    # pickle.dump(training, open("pickled-vars/ptb_training_id.p", "wb"))

    t = time.time()
    training_data = pickle.load(open("pickled-vars/ptb_training_id.p", "rb"))
    w2id = pickle.load(open("pickled-vars/w2id.p", "rb"))
    print("Loaded PTB data in %.1fs" % (time.time()-t))

    # print(len(training_data))
    training = training_data[:80000]
    testing = training_data[80000:100000]

    w_size = 4

    t = time.time()
    train_x = []
    train_y = []
    for i in range(0, len(training) - (w_size+1), w_size):
        train_x.append(training[i:i+w_size])
        train_y.append(training[i+1:i+(w_size+1)])
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    test_x = []
    test_y = []
    for i in range(0, len(testing) - (w_size+1), w_size):
        test_x.append(testing[i:i+w_size])
        test_y.append(testing[i+1:i+(w_size+1)])
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    print("Created training and testing data in %.1fs" % (time.time()-t))

    model = Model(len(w2id))

    # Set-up the training step
    print("Training")
    t = time.time()
    train(model, train_x, train_y)
    print("Training completed in %.1fs" % (time.time()-t))

    print(test(model, test_x, test_y))

    train_inputs, train_labels, test_inputs, test_labels = load_cola()
    train_inputs_id = make_words_into_ids(train_inputs, w2id)

    new_inputs, new_labels = filter_window_size(train_inputs_id, train_labels, w_size)
    # print(len(new_inputs[1]))
    print(len(new_inputs))

    cola_train_x = []
    cola_train_y = []
    cola_lens = []

    for ex in new_inputs[:1000]:
        t_x = []
        t_y = []
        # print(len(ex))
        cola_lens.append(len(ex))
        for i in range(0, len(ex) - (w_size), w_size):
            t_x.append(ex[i:i+w_size])
            t_y.append(ex[i+1:i+(w_size+1)])
        cola_train_x.append(np.array(t_x))
        # print(len(t_x))
        cola_train_y.append(np.array(t_y))

    cola_train_x = np.array(cola_train_x)
    cola_train_y = np.array(cola_train_y)

    # print(np.shape(cola_train_x))
    # print(cola_train_x[0])
    # print(cola_train_y[0])
    # print(np.shape(cola_train_y))

    pps = []
    for i in range(len(cola_train_x)):
        pps.append(test(model, cola_train_x[i], cola_train_y[i]))
    # print(pps)
    # print(new_labels[:1000])

    visualize_results(pps, new_labels[:1000], "neur1")
    export_neural_results(pps, new_labels[:1000], cola_lens)


    # pickle.dump(model, open("pickled-vars/neural_model.p", "wb"))

    # print(generate_sentence("john", 6, w2id, model))
    # print(generate_sentence("the", 6, w2id, model))
    # print(generate_sentence("executive", 6, w2id, model))

    # print("batch_size:", model.batch_size)
    # print("embedding_size:", model.embedding_size)
    # print("learning_rate:", model.learning_rate)


def testing():
    train_inputs, train_labels, test_inputs, test_labels = load_cola()
    print(train_inputs)
    # print(len(train_labels), len(test_labels))
    # print(test_inputs[5:10])
    # avg_sent_len(train_inputs, train_labels)
    grammar = pickle.load(open("pickled-vars/treebank-grammar.p", "rb"))
    get_embeddings("data/ptb.csv")

if __name__ == "__main__":
    neuro()
    # testing()
    # symbo()
