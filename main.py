from Preprocess import *
from SymbolicModel import SymbolicModel

import pickle
import matplotlib.pyplot as plt
import numpy as np

def visualize_results(probs, labels):
    y = np.random.rand(len(probs))
    plt.scatter(probs, y, c=labels, alpha=0.9)
    plt.xscale("log")
    plt.xlim(10**-30, 1)
    plt.show()

def main():
    # grammar = create_pcfg_from_treebank(pickle_it=True, log_it=True, filename="treebank_full", full=True)
    grammar = pickle.load(open("pickled-vars/treebank_full-grammar.p", "rb"))
    print("Grammar loaded")
    # parser = create_viterbi_parser(grammar, pickle_it=True, filename="viterbi_full")
    parser = pickle.load(open("pickled-vars/viterbi_full-parser.p", "rb"))
    print("Parser loaded")

    sym = SymbolicModel(grammar, parser)
    # s = sym.produce_normalized_probs(["John John John the the the the the".split()])
    # print(s)
    #
    train_inputs, train_labels, test_inputs, test_labels = load_cola()
    print("Cola loaded")

    # lbls = test_labels[0:30]
    # probs = sym.produce_normalized_probs(test_inputs[0:30])
    probs = sym.produce_normalized_probs(["John John John John John .".split()])
    print(probs)
    # visualize_results(probs, lbls)

if __name__ == "__main__":
    main()
