from Preprocess import *
from SymbolicModel import SymbolicModel

import time
import pickle
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(probs, labels, title):
    y = np.random.rand(len(probs))
    plt.scatter(probs, y, c=labels, alpha=0.9)
    plt.savefig("results/%s" % title)
    plt.show()

def export_results(probs, labels, inputs):
    sents = list(map(lambda x: " ".join(x), inputs))
    with open("results/res.txt", "w") as filehandle:
        for i in range(len(probs)):
            filehandle.write("%s\t%s\t%s\n" % (sents[i], labels[i], probs[i]))

TESTING = True

def main():
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

    sym = SymbolicModel(grammar, parser)
    t = time.time()
    train_inputs, train_labels = sym.filter_coverage(train_inputs, train_labels)
    test_inputs, test_labels = sym.filter_coverage(test_inputs, test_labels)
    print("Examples filtered for coverage in %.1fs" % (time.time()-t))

    t = time.time()
    probs = sym.produce_normalized_log_probs(train_inputs[:3])
    labels = train_labels[:3]
    print("Calculated sentence probabilities in %.1fs" % (time.time()-t))
    # probs = sym.produce_normalized_log_probs(["John John John John .".split()])
    if TESTING:
        print(probs)

    visualize_results(probs, labels, "fig4")
    export_results(probs, labels, train_inputs[:3])

def test():
    train_inputs, train_labels, test_inputs, test_labels = load_cola()
    print(len(train_labels), len(test_labels))
    print(test_inputs[5:10])

if __name__ == "__main__":
    main()
