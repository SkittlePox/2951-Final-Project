import nltk
from nltk import nonterminals, Nonterminal, Production, CFG, PCFG
from nltk.corpus import treebank, brown
from nltk import treetransforms
from nltk import induce_pcfg
from nltk.parse import pchart
import sys, time
from nltk import tokenize
from nltk.grammar import toy_pcfg1
from nltk.parse import pchart
from nltk.parse import ViterbiParser
from functools import reduce
import pickle

def CFG_Section():
    # Create some nonterminals
    # S, NP, VP, PP = nonterminals('S, NP, VP, PP')
    # N, V, P, Det = nonterminals('N, V, P, Det')
    # VP_slash_NP = Nonterminal('VP/NP')
    #
    # print('Some nonterminals:', [S, NP, VP, PP, N, V, P, Det, VP_slash_NP])
    # print('S.symbol() =>', S.symbol())
    #
    # print(Production(S, [NP]))


    grammar = CFG.fromstring("""
      S -> NP VP
      PP -> P NP
      NP -> Det N | NP PP
      VP -> V NP | VP PP
      Det -> 'a' | 'the'
      N -> 'dog' | 'cat'
      V -> 'chased' | 'sat'
      P -> 'on' | 'in'
    """)

    print('A Grammar:', grammar)
    print('grammar.start()   =>', grammar.start())
    print('grammar.productions() =>')
    # Use string.replace(...) is to line-wrap the output.
    print(grammar.productions())

    print('\nCoverage of input words by a grammar:')
    try:
        grammar.check_coverage(['a','dog'])
        print("All words covered")
    except:
        print("Strange")
    try:
        print(grammar.check_coverage(['a','toy']))
    except:
        print("Some words not covered")


def PCFG_Section():
    toy_pcfg1 = PCFG.fromstring("""
        S -> NP VP [1.0]
        NP -> Det N [0.5] | NP PP [0.25] | 'John' [0.1] | 'I' [0.15]
        Det -> 'the' [0.8] | 'my' [0.2]
        N -> 'man' [0.5] | 'telescope' [0.5]
        VP -> VP PP [0.1] | V NP [0.7] | V [0.2]
        V -> 'ate' [0.35] | 'saw' [0.65]
        PP -> P NP [1.0]
        P -> 'with' [0.61] | 'under' [0.39]
    """)

    pcfg_prods = toy_pcfg1.productions()

    pcfg_prod = pcfg_prods[2]
    print('A PCFG production:', pcfg_prod)
    print('pcfg_prod.lhs()  =>', pcfg_prod.lhs())
    print('pcfg_prod.rhs()  =>', pcfg_prod.rhs())
    print('pcfg_prod.prob() =>', pcfg_prod.prob())

    # extract productions from three trees and induce the PCFG
    print("Induce PCFG grammar from treebank data:")

    productions = []
    for item in treebank.fileids()[:2]:
      for tree in treebank.parsed_sents(item):
        print(" ".join(tree.leaves()))
        # perform optional tree transformations, e.g.:
        # tree.collapse_unary(collapsePOS = False)# Remove branches A-B-C into A-B+C
        # tree.chomsky_normal_form(horzMarkov = 2)# Remove A->(B,C,D) into A->B,C+D->D
        productions += tree.productions()

    S = Nonterminal('S')
    grammar = induce_pcfg(S, productions)
    print(grammar)    # This is a PCFG

    ### Parsing section below ###

    print("\nParse sentence using induced grammar:")

    parser = pchart.InsideChartParser(grammar)
    parser.trace(3)

    sent = treebank.parsed_sents('wsj_0001.mrg')[0].leaves()
    print(sent)

    # for parse in parser.parse(sent):
    #   print(parse)

def Parser_Section():
    demos = [('I saw John through the telescope', toy_pcfg1)]
    sent, grammar = demos[0]
    # print(grammar)

    # Tokenize the sentence.
    tokens = sent.split()
    parser = ViterbiParser(grammar)

    parser.trace(0) # Use this to change verbosity
    t = time.time()
    parses = parser.parse_all(tokens)
    print("Time:", time.time()-t)

    if parses:
        lp = len(parses)
        p = reduce(lambda a,b:a+b.prob(), parses, 0.0)
    else:
        p = 0

    print("Probability:", p)

def br():
    print(brown.words()[:10])

def main():
    # print(nltk.corpus.treebank.parsed_sents('wsj_0001.mrg')[0])
    # nltk.corpus.treebank.parsed_sents('wsj_0001.mrg')[0].draw()

    # print("Induce PCFG grammar from treebank data:")
    #
    productions = []
    print(len(treebank.fileids()))
    for item in treebank.fileids(): # Goes through all trees
      for tree in treebank.parsed_sents(item):
        # perform optional tree transformations, e.g.:
        tree.collapse_unary(collapsePOS = False)# Remove branches A-B-C into A-B+C
        tree.chomsky_normal_form(horzMarkov = 2)# Remove A->(B,C,D) into A->B,C+D->D
        productions += tree.productions()
    # #
    # # print(type(productions[0]))
    # #
    S = Nonterminal('S')
    grammar = induce_pcfg(S, productions)
    # # # print(grammar)    # This is a PCFG
    # pickle.dump(grammar, open("tbank-grammar.p", "wb"))
    # t = time.time()
    # grammar = pickle.load(open("tbank-grammar.p", "rb"))
    # textf = open("lexicon.txt", "w")
    # n = textf.write(str(reduce(lambda a, b: a + "\n" + b, list(filter(lambda x: "'" in x, str(grammar).split("\n"))))))
    # textf.close()
    # print(time.time()-t)
    parser = ViterbiParser(grammar)
    # pickle.dump(parser, open("cky-parser.p", "wb"))
    # parser = pickle.load(open("cky-parser.p", "rb"))
    parser.trace(0)
    sent = "John will join the board"
    tokens = sent.split()

    try:
        grammar.check_coverage(tokens)
        print("All words covered")
        parses = parser.parse_all(tokens)
        if parses:
            lp = len(parses)
            print(lp)
            print(parses[0].label())
            # parses[0].draw()
            p = reduce(lambda a,b:a+b.prob(), list(filter(lambda x: x.label() == 'S', parses)), 0.0)
        else:
            p = 0

        print("Probability:", p)
    except:
        print("Some words not covered")

if __name__ == "__main__":
    # Parser_Section()
    # main()
    PCFG_Section()
    # br()
