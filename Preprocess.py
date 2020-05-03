import sys, time
import pickle
from functools import reduce

import nltk
from nltk import Nonterminal, induce_pcfg
from nltk.corpus import treebank, ptb
from nltk.parse import ViterbiParser
import numpy as np

var_dir = "pickled-vars/"
data_dir = "data/"

def save_grammar_cleartext(grammar, filename):
    f = open("%s%s-pcfg.txt" % (var_dir, filename), "w")
    f.write(str(grammar))
    f.close()

def save_lexicon_cleartext(grammar, filename):
    f = open("%s%s-lexicon.txt" % (var_dir, filename), "w")
    f.write("\n".join(filter(lambda x: "'" in x, str(grammar).split("\n"))))

def create_pcfg_from_treebank(pickle_it=False, log_it=False, filename="treebank", full=False):
    """
    Creates a PCFG from the Penn Treebank dataset using induce_pcfg
    Optional pickling of this PCFG in pickled-vars/
    """
    if full:
        tb = ptb
    else:
        tb = treebank
    productions = []
    flat_trees = 0
    for item in tb.fileids(): # Goes through all trees
        for tree in tb.parsed_sents(item):
            if tree.height() == 2:  # Gets rid of flat trees
                # print("####Tree not collected#####")
                flat_trees += 1
                continue
            # print(" ".join(tree.leaves()))    # This should print the sentences
            # perform optional tree transformations, e.g.:
            # tree.collapse_unary(collapsePOS = False)# Remove branches A-B-C into A-B+C
            # tree.chomsky_normal_form(horzMarkov = 2)# Remove A->(B,C,D) into A->B,C+D->D
            productions += tree.productions()
    print("%s Flat trees purged" % flat_trees)

    S = Nonterminal('S')
    grammar = induce_pcfg(S, productions)
    if pickle_it:
        pickle.dump(grammar, open("%s%s-grammar.p" % (var_dir, filename), "wb"))
    if log_it:
        save_grammar_cleartext(grammar, filename)
        save_lexicon_cleartext(grammar, filename)
    return grammar

def create_viterbi_parser(grammar, pickle_it=False, filename="viterbi"):
    parser = ViterbiParser(grammar)
    parser.trace(0)
    if pickle_it:
        pickle.dump(parser, open("%s%s-parser.p" % (var_dir, filename), "wb"))
    return parser

# def generate_prod_tree(grammar):
#     grammar_prods = grammar.productions()
#     g_tree = dict()
#     for p in grammar_prods:
#         if p.lhs() in g_tree:
#             g_tree.update({g_tree[p.lhs]})


#####################   Neural Preprocessing Below   ##########################

def get_ptb_w2id(embedding_file):
    ptb_emb = open(embedding_file, "r")
    ptb_str = ptb_emb.read()
    ptb_ln = ptb_str.split('\n')
    ptb_ln = ptb_ln[1:-1]
    vocabSet = set(list(map(lambda x: x.split(',')[0][1:-1], ptb_ln)))
    vocabDict = {word:i for i, word in enumerate(vocabSet)}
    ptb_emb.close()
    return vocabDict

def get_ptb_data(w2id):
    all_words = []
    for item in ptb.fileids():
        all_words.extend(list(map(str.lower, ptb.words(item))))
    # print(all_words)
    all_words_id = []
    for w in all_words:
        id = w2id.get(w)
        if id == None:
            id = w2id.get("<unk>")
        all_words_id.append(id)
    return all_words_id

def make_words_into_ids(inputs, w2id):
    def list_words_to_id(all_words):
        all_words_id = []
        for w in all_words:
            id = w2id.get(w)
            if id == None:
                id = w2id.get("<unk>")
            all_words_id.append(id)
        return all_words_id
    outputs = []
    for all_words in inputs:
        outputs.append(list_words_to_id(all_words))
    return np.array(outputs)

# def get_embeddings(embedding_file):
#     emb = open(embedding_file, "r")
#     emb_str = emb.read().split("\n")
#     emb_str = list(map(lambda x: x.split(','), emb_str))
#     emb_str = emb_str[:][1:]
#     print(emb_str[0])


#####################  Dataset Preprocessing Below   ##########################

def load_cola():
    def get_train_test(f):
        inputs = []
        labels = []
        l = f.readline()
        while l is not "":
            line_splt = l.split("\t")
            labels.append(int(line_splt[1]))
            example_tokens = line_splt[3].split(" ")
            example_tokens[-1] = example_tokens[-1][:-1]    # Gets rid of the \n at the end of each example
            if example_tokens[-1][-1] == '.' or \
            example_tokens[-1][-1] == '!' or \
            example_tokens[-1][-1] == '?':
                eos = example_tokens[-1][-1]
                example_tokens[-1] = example_tokens[-1][:-1]# Separates punctuation at the end of the line
                example_tokens.append(eos)                  # Adds punctuation as a separate token
            inputs.append(example_tokens)
            l = f.readline()
        return inputs, labels

    tr = open("%scola-raw_in_domain_train.tsv" % data_dir)
    te = open("%scola-raw_in_domain_dev.tsv" % data_dir)
    # tr = open("%scola-tokenized_in_domain_train.tsv" % data_dir)
    # te = open("%scola-tokenized_in_domain_dev.tsv" % data_dir)
    train_inputs, train_labels = get_train_test(tr)
    test_inputs, test_labels = get_train_test(te)
    # validation_inputs, validation_labels = get_train_test(val)

    return train_inputs, train_labels, test_inputs, test_labels

def avg_sent_len(inputs, labels):
    t_count = 0
    f_count = 0
    t_sum = 0
    f_sum = 0

    for i in range(len(inputs)):
        if labels[i] == 1:
            t_count += 1
            t_sum += len(inputs[i])
        if labels[i] == 0:
            f_count += 1
            f_sum += len(inputs[i])
    print("Average length of grammatical sentences: %s" % (t_sum / t_count))
    print("Average length of ungrammatical sentences: %s" % (f_sum / f_count))

def filter_window_size(inputs, labels, w_size):
    new_inputs = []
    new_labels = []
    for i in range(len(inputs)):
        if len(inputs[i]) > w_size:
            new_inputs.append(inputs[i])
            new_labels.append(labels[i])
    return new_inputs, new_labels
