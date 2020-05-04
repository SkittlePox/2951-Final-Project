import matplotlib.pyplot as plt
import numpy as np

def load_neural_results(filename):
    f = open(filename, "r")
    line = f.readline()
    results = []
    while line is not "":
        line = f.readline()[:-1]
        if line == '':
            break
        splt = line.split('\t')
        splt[1] = float(splt[1])
        splt[2] = int(splt[2])
        splt[3] = int(splt[3])
        results.append(splt)
    return results

def load_symb_results(filename):
    f = open(filename, "r")
    line = f.readline()
    results = []
    while line is not "":
        line = f.readline()[:-1]
        if line == '':
            break
        splt = line.split('\t')
        splt[1] = int(splt[1])
        splt[2] = float(splt[2])
        splt[3] = int(splt[3])
        results.append(splt)
    return results

def neur_accuracy(examples, threshold):
    total = 0
    ungram = 0
    ungram_total = 0
    gram = 0
    gram_total = 0
    for ex in examples:
        if ex[2] == 1:
            gram_total += 1
            if ex[1] < threshold:
                total += 1
                gram += 1
        if ex[2] == 0:
            ungram_total += 1
            if ex[1] > threshold:
                total += 1
                ungram += 1
    return total/len(examples), gram/gram_total, ungram/ungram_total

def symb_accuracy(examples, threshold):
    total = 0
    ungram = 0
    ungram_total = 0
    gram = 0
    gram_total = 0
    for ex in examples:
        if ex[1] == 1:
            gram_total += 1
            if ex[2] > threshold:
                total += 1
                gram += 1
        if ex[1] == 0:
            ungram_total += 1
            if ex[2] < threshold:
                total += 1
                ungram += 1
    return total/len(examples), gram/gram_total, ungram/ungram_total

def neur_threshold_graph(examples):
    tot = []
    gram = []
    ungram = []
    thr = []
    for thresh in range(20, 1000, 20):
        thr.append(thresh)
        t, g, u = neur_accuracy(examples, thresh)
        tot.append(t)
        gram.append(g)
        ungram.append(u)
    plt.plot(thr, tot)
    plt.plot(thr, gram, "g")
    plt.plot(thr, ungram)
    plt.plot(281,0.516, "k+", markersize=15)

    plt.title("Neural Training Threshold Accuracies")
    plt.xlabel("Threshold Value")
    plt.ylabel("Accuracy")
    plt.legend(("Overall", "Grammatical", "Ungrammatical"))
    plt.show()

def symb_threshold_graph(examples):
    tot = []
    gram = []
    ungram = []
    thr = []
    for thresh in np.arange(-0.3, -1.0, -0.02):
        thr.append(thresh)
        t, g, u = symb_accuracy(examples, thresh)
        tot.append(t)
        gram.append(g)
        ungram.append(u)
    plt.plot(thr, tot)
    plt.plot(thr, gram, "g")
    plt.plot(thr, ungram)
    plt.plot(-0.635,0.508, "k+", markersize=15)

    plt.title("Symbolic Training Threshold Accuracies")
    plt.xlabel("Threshold Value")
    plt.ylabel("Accuracy")
    plt.legend(("Overall", "Grammatical", "Ungrammatical"))
    plt.show()

def main():
    neur_format = ["sentence", "perplexity", "CoLA Label", "Sentence Length"]
    symb_format = ["sentence", "CoLA Label", "prob prediction", "Number of Productions"]
    neur_test = load_neural_results("results_analysis/neur-res-test.txt")
    neur_train = load_neural_results("results_analysis/neur-res-train.txt")
    symb_train = load_symb_results("results_analysis/symb-sum-norm-train.txt")
    symb_test = load_symb_results("results_analysis/symb-sum-norm-test.txt")

    # print(neur_accuracy(neur_train, 281))
    # neur_threshold_graph(neur_train)
    # print(symb_train[0])
    print(symb_accuracy(symb_test, -0.635))
    # symb_threshold_graph(symb_train)

if __name__ == '__main__':
    main()
