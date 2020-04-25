from Preprocess import *
from SymbolicModel import SymbolicModel

def main():
    # create_pcfg_from_treebank(pickle_it=True, log_it=True, filename="treebank")
    grammar = pickle.load(open("pickled-vars/treebank-no-transformations-grammar.p", "rb"))
    print("Grammar loaded")
    # create_viterbi_parser(grammar, pickle_it=True, filename="viterbi")
    parser = pickle.load(open("pickled-vars/viterbi-no-transformations-parser.p", "rb"))
    print("Parser loaded")

    sym = SymbolicModel(grammar, parser)

    train_inputs, train_labels, test_inputs, test_labels = load_cola()
    print("Cola loaded")
    
    print(sym.produce_normalized_probs(test_inputs[15:30]))
    print(test_labels[15:30])

if __name__ == "__main__":
    main()
