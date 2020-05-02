from Preprocess import *
from SymbolicModel import SymbolicModel

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
    if TESTING:
        plt.show()

def export_results(probs, labels, inputs, prods):
    sents = list(map(lambda x: " ".join(x), inputs))
    with open("results/res.txt", "w") as filehandle:
        for i in range(len(probs)):
            filehandle.write("%s\t%s\t%s\t%s\n" % (sents[i], labels[i], probs[i], prods[i]))

TESTING = False
sym = None
lab = None

def main():
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
    lab = train_labels[:30]
    inp = train_inputs[:30]
    probs, prods = sym.produce_normalized_log_probs(inp, 'log')
    print("Calculated sentence probabilities in %.1fs" % (time.time()-t))
    # probs, prods = sym.produce_normalized_log_probs(["John John John John .".split()])
    if TESTING:
        print(probs)

    visualize_results(probs, lab, "fig5")
    export_results(probs, lab, inp, prods)

def test():
    train_inputs, train_labels, test_inputs, test_labels = load_cola()
    print(len(train_labels), len(test_labels))
    print(test_inputs[5:10])

if __name__ == "__main__":
    main()
