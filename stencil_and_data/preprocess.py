import tensorflow as tf
import numpy as np

def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of train (1-d list or array with training words in vectorized/id form), test (1-d list or array with testing words in vectorized/id form), vocabulary (Dict containg index->word mapping)
    """

    trainFile = open(train_file, "r")
    testFile = open(test_file, "r")

    trainStr = trainFile.read()
    testStr = testFile.read()

    trainArr = trainStr.split()
    testArr = testStr.split()

    vocabSet = set(trainArr)
    vocabDict = {word:i for i, word in enumerate(vocabSet)}

    trainArrID = list(map(lambda x: vocabDict[x], trainArr))
    testArrID = list(map(lambda x: vocabDict[x], testArr))

    testFile.close()
    trainFile.close()
    return (trainArrID, testArrID, vocabDict)
