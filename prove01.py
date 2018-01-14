from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from numpy import genfromtxt
import numpy as np


# getopts from https://gist.github.com/dideler/2395703
def getopts(argv):
    """Collect command-line options in a dictionary"""
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by removing script name
    return opts


class HardCodedModel:
    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(0)
        return np.array(predictions)


class HardCodedClassifier:
    def fit(self, X, y):
        return HardCodedModel()


def load_csv(filename):
    """Reads a CSV file into data and target arrays. Last column == target"""
    data = genfromtxt(filename, delimiter=',')[:, :-1]
    target = genfromtxt(filename, delimiter=',', usecols=(-1), dtype=str)
    uniqueItems = np.unique(target)
    numericTarget = []
    for item in target:
        # Convert strings to numbers
        numericTarget.append(np.where(uniqueItems == item)[0][0])
    return data, np.array(numericTarget)


def main():
    from sys import argv
    myargs = getopts(argv)
    if '-f' in myargs:
        print("Using provided CSV: {}".format(myargs['-f']))
        data, target = load_csv(myargs['-f'])
    else:
        iris = datasets.load_iris()
        data = iris.data
        target = iris.target

    data_train, data_test, target_train, target_test = \
        train_test_split(data,
                         target,
                         test_size=0.3,
                         train_size=0.7,
                         shuffle=True)
    # classifier = GaussianNB()
    classifier = HardCodedClassifier()
    model = classifier.fit(data_train, target_train)

    score = accuracy_score(target_test, model.predict(data_test))

    output = "Accuracy: {0:.1f}%".format(score*100)
    print(output)


if __name__ == '__main__':
    main()
