from functools import reduce
import numpy as np
from progress.bar import Bar

class SymbolicModel:
    def __init__(self, grammar, parser):
        self.grammar = grammar
        self.parser = parser
        self.out_probs = None
        self.out_prods = None
        self.grammar_prods = grammar.productions()

    def get_prob(self, prod):
        for i in self.grammar_prods:
            if prod.lhs() == i.lhs() and prod.rhs() == i.rhs():
                return i.prob()
        return 0

    def produce_normalized_log_probs(self, inputs, loss_func='log-mult'):
        self.out_probs = []
        self.out_prods = []
        bar = Bar('Parsing Sentences', max=len(inputs), suffix='[%(index)d / %(max)d] %(eta_td)s')

        # print(grammar_prods[0].lhs(), grammar_prods[0].rhs())
        for input in inputs:
            try:
                self.grammar.check_coverage(input)
                p = 10**-200
                parses = self.parser.parse_all(input)
                if parses:
                    # print(len(parses))
                    parses[0].draw()
                    productions = parses[0].productions()
                    prod_number = len(productions)
                    self.out_prods.append(prod_number)
                    # print(prod_number)
                    # print(self.grammar)
                    # print(productions[0])
                    # print(self.get_prob(productions[0]))
                    if loss_func == 'sum-norm':
                        summ = 0
                        for pr in productions:
                            summ += self.get_prob(pr)**2
                        summ = np.array(summ)
                        self.out_probs.append(-(1/summ).mean())
                        bar.next()
                        continue
                    p += reduce(lambda a,b:a+b.prob(), parses, 0.0)
                    if loss_func == 'log-norm':
                        self.out_probs.append(np.log(p/prod_number))
                    if loss_func == 'lin-log-norm':
                        self.out_probs.append(np.log(p)/prod_number)
                    if loss_func == 'prod-root':
                        self.out_probs.append(p**(1/prod_number))
                    elif loss_func == 'log':
                        self.out_probs.append(np.log(p))
                    elif loss_func == 'log-mult':
                        self.out_probs.append(np.log(p*prod_number))
                    elif loss_func == 'linear':
                        self.out_probs.append(p)
            except:
                self.out_prods.append(None)
                self.out_probs.append(None)
            bar.next()
        bar.finish()
        return self.out_probs, self.out_prods

    def filter_coverage(self, inputs, labels):
        c_inputs = []
        c_labels = []
        for i in range(len(inputs)):
            input = inputs[i]
            try:
                self.grammar.check_coverage(input)
                c_inputs.append(input)
                c_labels.append(labels[i])
            except:
                pass
        print("Coverage Rate: %.2f" % (100.0*len(c_inputs)/len(inputs)))
        return c_inputs, c_labels

    def accuracy_proto(inputs, labels, prob_threshold, loss_func='linear'):
        """
        inputs: a list of inputs (which are lists of words)
        labels: a list of 0 or 1 for each input
        :returns: ???
        """
        self.out_probs = []
        self.accurate = []
        bar = Bar('Testing Sentences', max=len(inputs), suffix='[%(index)d / %(max)d] %(eta_td)s')
        for input in inputs:
            try:
                self.grammar.check_coverage(input)
                p = 10**-50
                parses = self.parser.parse_all(input)
                if parses:
                    # print(len(parses))
                    # parses[0].draw()
                    prod_number = len(parses[0].productions())
                    # print(prod_number)
                    p += reduce(lambda a,b:a+b.prob(), list(filter(lambda x: x.label() == 'S', parses)), 0.0)
                    if loss_func == 'log-norm':
                        self.out_probs.append(np.log(p/prod_number))
                    elif loss_func == 'log':
                        self.out_probs.append(np.log(p))
                    elif loss_func == 'log-mult':
                        self.out_probs.append(np.log(p*prod_number))
                    elif loss_func == 'linear':
                        self.out_probs.append(p)

                    self.accurate.append(self.out_probs[-1] > prob_threshold)
            except:
                self.out_prods.append(None)
                self.out_probs.append(None)
            bar.next()
        bar.finish()
        count = 0.0
        for i in range(len(labels)):
            if labels[i] == self.accurate[i]:
                count += 1.0
        return count / len(labels)
