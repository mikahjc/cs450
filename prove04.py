#!/usr/bin/env python

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from numpy import genfromtxt
import numpy as np
import pandas as pd
import pprint


class Dataset:
    def __init__(self, dataset):
        if type(dataset) is not pd.DataFrame:
            raise("Dataset initializer was not pandas DataFrame")
        if "target" not in list(dataset):
            raise("Dataset initializer did not contain target column")
        self.dataset = dataset

    @property
    def data(self):
        return self.dataset.drop("target", axis=1)

    @property
    def target(self):
        return self.dataset["target"]

    @property
    def all(self):
        return self.dataset
    
    
class MyDecisionTreeClassifier:
    def fit(self, data, target):
        return DecisionTreeModel(make_tree(data, target))


class DecisionTreeModel:
    def __init__(self, tree):
        self.tree = tree

    def show(self):
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(self.tree)

    def visit(self, row, node):
        if type(node) == dict:
            for key, subnode in node.items():
                if type(subnode) == dict and row[key] in subnode:
                    return self.visit(row, subnode[row[key]])
                else:
                    return self.visit(row, subnode)
        else:
            return node

    def predict(self, X):
        tree = self.tree
        predictions = []
        for index, row in X.iterrows():
            predictions.append(self.visit(row, tree))
        return np.array(predictions)


def calc_entropy(target):
    entropy = 0
    items, item_counts = np.unique(target, return_counts=True)
    for count in item_counts:
        entropy -= (count/sum(item_counts)) * \
                   np.log2(count/sum(item_counts))
    return entropy


def calc_gain(target, feature):
    table = np.vstack([feature, target]).T
    table = pd.DataFrame(table)
    gain = calc_entropy(target)
    items, item_counts = np.unique(feature, return_counts=True)
    items = dict(zip(items, item_counts))
    for item in items:
        subtable = table.loc[table[0].isin([item])]
        subitems, subitem_counts = np.unique(subtable.iloc[:, -1], 
                                             return_counts=True)
        subitems = dict(zip(subitems, subitem_counts))
        subgain = 0
        for subcount in subitems:
            subgain -= (subitems[subcount]/sum(subitem_counts)) * \
                        np.log2(subitems[subcount]/sum(subitem_counts))
        gain -= (sum(subitem_counts)/sum(item_counts)) * subgain
    return gain


def make_tree(data, targets):
    ''' data should be a DataFrame with names attached '''
    def mode(x): return x.mode() if len(x) > 2 else np.array(x)
    if np.shape(targets.agg(mode))[0] == 0:
        default = np.NaN
    else:
        default = targets.agg(mode)[0]

    if data.empty:
        return default
    elif np.shape(targets.value_counts())[0] == 1:
        # Only one remaining target option
        return targets.iloc[0]
    else:
        # Choose feature that results in lowest entropy
        best_entropy = calc_entropy(targets)
        for feature in data:
            resulting_entropy = calc_gain(targets, data[feature])
            if resulting_entropy <= best_entropy:
                best_feature = data[feature].name
        tree = {best_feature: {}}

        # Get list of values in best feature
        values = np.unique(data[best_feature])

        for value in values:
            newData = data[data[best_feature] == value]
            newTargets = targets[data[best_feature] == value]
            newData = newData.drop(best_feature, axis=1)
            subtree = make_tree(newData, newTargets)
            tree[best_feature][value] = subtree
        return tree


def load_lenses():
    names = ["linenum", "age", "prescription", "astig", "tear", "target"]
    dataset = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-"
                          "databases/lenses/lenses.data",
                          header=None,
                          delimiter="\s+",
                          names=names)
    dataset = dataset.drop("linenum", axis=1)
    return Dataset(dataset)


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


def load_csv(filename):
    """Reads a CSV file into data and target arrays. Assumes last column == target"""
    data = genfromtxt(filename, delimiter=',')[:, :-1]
    target = genfromtxt(filename, delimiter=',', usecols=(-1), dtype=str)
    uniqueItems = np.unique(target)
    numericTarget = []
    for item in target:
        # Convert strings to numbers
        numericTarget.append(np.where(uniqueItems == item)[0][0])
    return data, np.array(numericTarget)


def load_dataset(index):
    return load_lenses()
    if index == 1:
        print("Using cars dataset\n")
        # target, data = load_cars()
    elif index == 2:
        print("Using diabetes dataset\n")
        # target, data = load_pima()
    elif index == 3:
        print("Using mpg dataset\n")
        # target, data = load_mpg()
    else:
        print("Invalid dataset")
        exit(-1)
    # return target, data


def main():
    dataset = load_lenses()

    from sys import argv
    myargs = getopts(argv)
    if '-h' in myargs or '--help' in myargs:
        print("Usage: python prove04.py [-h|--help]")
        exit()

    data_train, data_test, target_train, target_test = \
        train_test_split(dataset.data,
                         dataset.target,
                         test_size=0.3,
                         train_size=0.7,
                         shuffle=True)
    
    classifier = MyDecisionTreeClassifier()
    model = classifier.fit(data_train, target_train)

    model.show()

    score = accuracy_score(target_test, model.predict(data_test))

    print("Accuracy: {:.2f}%".format(score*100))

    clf = DecisionTreeClassifier()
    score = cross_val_score(clf, dataset.data, dataset.target, cv=4)

    print("Accuracy: {:.2f}% (+/- {:.2f})".format(score.mean() * 100, score.std() *2 ))


if __name__ == '__main__':
    main()
