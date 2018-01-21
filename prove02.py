#!/usr/bin/env python

import time
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from numpy import genfromtxt
import numpy as np


# Timer class from:
# http://preshing.com/20110924/timing-your-code-using-pythons-with-statement/
class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


# getopts from https://gist.github.com/dideler/2395703
def getopts(argv):
    """Collect command-line options in a dictionary"""
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            if argv[0] == '-':
                pass
            elif argv[0] == "-h":
                opts[argv[0]] = ''
            else:
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


class MyNeighborsModel:
    def __init__(self, train_data, train_target, k):
        self.train_data = train_data
        self.train_target = train_target
        self.k = k

    def predict(self, inputs):

        # Bring data into current namespace
        data = self.train_data
        targets = self.train_target 
        k = int(self.k)

        # Initialize predictions array and set up loop count
        inputSize = np.shape(inputs)[0]
        predictions = np.zeros(inputSize)

        # Loop through each item in inputs
        for x in range(inputSize):
            # Compute distances between training data and inputs
            distances = np.sum((data-inputs[x, :])**2, axis=1)

            # Add target data to distance array
            # distances = np.append(np.vstack(distances), targets, axis=1)
            distances = np.append(np.vstack(distances), np.vstack(targets), axis=1)

            # Sort array by distances and extract k nearest targets
            # Snippet from https://stackoverflow.com/a/2828121
            nearest_targets = distances[distances[:, 0].argsort()][:k, -1]

            # If all the nearest neighbors are unique, choose the closest
            # neighbor. Otherwise, choose the most frequent neighbor.
            if np.shape(np.unique(nearest_targets))[0] == k:
                predictions[x] = nearest_targets[0]
            else:
                counts = np.bincount(nearest_targets.astype(int))
                predictions[x] = np.argmax(counts)

        return predictions


class MyNeighborsClassifier:
    def fit(self, X, y, k):
        return MyNeighborsModel(X, y, k)


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


def load_dataset(index):
    if index == 1:
        print("Using iris dataset\n")
        dataset = datasets.load_iris()
    elif index == 2:
        print("Using digits dataset\n")
        dataset = datasets.load_digits()
    elif index == 3:
        print("Using wine dataset\n")
        dataset = datasets.load_wine()
    elif index == 4:
        print("Using breast cancer dataset\n")
        dataset = datasets.load_breast_cancer()
    else:
        print("Invalid dataset")
        exit(-1)
    return dataset


def main():
    from sys import argv
    myargs = getopts(argv)
    if '-f' in myargs:
        print("Using provided CSV: {}".format(myargs['-f']))
        data, target = load_csv(myargs['-f'])
    elif '-h' in myargs or '--help' in myargs:
        print("Usage: python prove02.py [-h|--help] [-f filename]" +
              " [-d|--dataset <1-4>]")
        exit()
    else:
        if '--dataset' in myargs:
            dataset = load_dataset(int(myargs['--dataset']))
        elif '-d' in myargs:
            dataset = load_dataset(int(myargs['-d']))
        else:
            print("Select dataset:\n\t1. Iris\n\t2. Digits")
            selection = input("\t3. Wine\n\t4. Breast Cancer\n>> ")
            selection = int(selection)
            dataset = load_dataset(selection)

        data = dataset.data
        target = dataset.target

    data_train, data_test, target_train, target_test = \
        train_test_split(data,
                         target,
                         test_size=0.3,
                         train_size=0.7,
                         shuffle=True)
    
    neighbors = 3

    # Test our classifier
    classifier = MyNeighborsClassifier()

    with Timer() as runtime:
        model = classifier.fit(data_train, target_train, neighbors)

        score = accuracy_score(target_test, model.predict(data_test))

    print("Homemade Classifier:")
    output = "Accuracy: {0:.2f}%\nRuntime: {1:.5f} seconds\n". \
        format(score*100, runtime.interval)

    print(output)

    # Test sklearn kNN classifier
    classifier2 = KNeighborsClassifier(n_neighbors=neighbors)

    with Timer() as runtime2:
        model2 = classifier2.fit(data_train, target_train)

        score2 = accuracy_score(target_test, model2.predict(data_test))

    print("sklearn Classifier:")
    output2 = "Accuracy: {0:.2f}%\nRuntime: {1:.5f} seconds". \
        format(score2*100, runtime2.interval)

    print(output2)


if __name__ == '__main__':
    main()
