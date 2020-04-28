from functools import reduce
import numpy as np
from progress.bar import Bar

class SymbolicModel:
    def __init__(self, grammar, parser):
        self.grammar = grammar
        self.parser = parser

    def produce_normalized_log_probs(self, inputs):
        out_probs = []
        bar = Bar('Parsing Sentences', max=len(inputs), suffix='[%(index)d / %(max)d] sentences')
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
                    out_probs.append(np.log(p/prod_number))
            except:
                out_probs.append(None)
            bar.next()
        bar.finish()
        return out_probs

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

    def accuracy_proto(inputs, labels):
        """
        inputs: a list of inputs (which are lists of words)
        labels: a list of 0 or 1 for each input
        :returns: ???
        """
        pass
