from functools import reduce

class SymbolicModel:
    def __init__(self, grammar, parser):
        self.grammar = grammar
        self.parser = parser

    def produce_normalized_probs(self, inputs):
        out_probs = []
        for input in inputs:
            try:
                self.grammar.check_coverage(input)
                p = 0.0
                parses = self.parser.parse_all(input)
                if parses:
                    # parses[0].draw()
                    prod_number = len(parses[0].productions())
                    p = reduce(lambda a,b:a+b.prob(), list(filter(lambda x: x.label() == 'S', parses)), 0.0)
                out_probs.append(p/prod_number)
            except:
                out_probs.append(None)
        return out_probs

    def accuracy_proto(inputs, labels):
        """
        inputs: a list of inputs (which are lists of words)
        labels: a list of 0 or 1 for each input
        :returns: ???
        """
        pass
