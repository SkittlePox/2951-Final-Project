from preprocess import get_data

def main():
    training, test, word2id = get_data("data/train.txt", "data/test.txt")
    print(len(training))

if __name__ == "__main__":
    main()
