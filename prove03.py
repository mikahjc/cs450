#!/usr/bin/env python

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from numpy import genfromtxt
import numpy as np
import pandas as pd


def load_cars():
    names = ["buying",
             "maint",
             "doors",
             "persons",
             "trunk",
             "safety",
             "target"]
    dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning"
                          + "-databases/car/car.data",
                          header=None,
                          names=names)

    # All fields contain some data that we want to make numeric. This dict
    # contains all the values we want to change them to.
    cleanup_nums = {"buying":   {"vhigh": 3, "high": 2, "med": 1, "low": 0},
                    "maint":    {"vhigh": 3, "high": 2, "med": 1, "low": 0},
                    "doors":    {"5more": 5},
                    "persons":  {"more": 6},
                    "trunk":    {"small": 0, "med": 1, "big": 2},
                    "safety":   {"low": 0, "med": 1, "high": 3},
                    "target":   {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}}
    dataset.replace(cleanup_nums, inplace=True)

    # Split data into arrays and return
    target = np.array(dataset["target"])
    data = np.array(dataset.drop("target", axis=1))
    return target, data


def load_pima():
    names = ["times_pregnant",
             "plasma",
             "blood_pressure",
             "skin_fold",
             "insulin",
             "bmi",
             "pedigree",
             "age",
             "target"]
    dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning"
                          + "-databases/pima-indians-diabetes/pima-indians"
                          + "-diabetes.data",
                          header=None,
                          names=names)
    # We don't want to drop rows with invalid data, since this will severely
    # reduce our dataset. Instead, we'll use the imputer class from sklearn
    # to attempt to fill in the data.
    dataset[["times_pregnant",
             "plasma",
             "blood_pressure",
             "skin_fold",
             "insulin"]] = dataset[["times_pregnant",
                                    "plasma",
                                    "blood_pressure",
                                    "skin_fold",
                                    "insulin"]].replace(0, np.NaN)
    values = dataset.values
    imputer = Imputer()
    new_values = imputer.fit_transform(values)

    # Imputer returns numpy arrays, so slice them and return.
    target = new_values[:, -1]
    data = new_values[:, :-1]
    return target, data


def load_mpg():
    names = ["mpg",
             "cylinders",
             "displacement",
             "horsepower",
             "weight",
             "acceleration",
             "model-year",
             "origin",
             "car-name"]
    dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning"
                          + "-databases/auto-mpg/auto-mpg.data",
                          header=None,
                          sep="\s+",
                          names=names,
                          na_values='?')

    # Drop car-name column. Probably impossible to use in machine learning.
    # Unless standardized, which it does not appear to be.
    dataset = dataset.drop("car-name", axis=1)

    dataset = pd.get_dummies(dataset, columns=['origin'])

    # Drop rows with incomplete data. We retain 98% of our data, so
    # I find it acceptable to do so.
    dataset.dropna(inplace=True)

    # Convert to numpy arrays and return
    target = np.array(dataset["mpg"])
    data = np.array(dataset.drop("mpg", axis=1))
    return target, data


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
        print("Using cars dataset\n")
        target, data = load_cars()
    elif index == 2:
        print("Using diabetes dataset\n")
        target, data = load_pima()
    elif index == 3:
        print("Using mpg dataset\n")
        target, data = load_mpg()
    else:
        print("Invalid dataset")
        exit(-1)
    return target, data


def main():
    from sys import argv
    myargs = getopts(argv)
    if '-h' in myargs or '--help' in myargs:
        print("Usage: python prove03.py [-h|--help]" +
              " [-d|--dataset <1-3>]")
        exit()
    else:
        if '--dataset' in myargs:
            target, data = load_dataset(int(myargs['--dataset']))
        elif '-d' in myargs:
            target, data = load_dataset(int(myargs['-d']))
        else:
            print("Select dataset:\n\t1. Cars\n\t2. Diabetes")
            selection = input("\t3. MPG\n>> ")
            selection = int(selection)
            target, data = load_dataset(selection)

    data_train, data_test, target_train, target_test = \
        train_test_split(data,
                         target,
                         test_size=0.3,
                         train_size=0.7,
                         shuffle=True)
    
    if selection == 1:
        classifier = svm.SVC(kernel="rbf", C=1)
        scoring = None
        scores = cross_val_score(classifier, data, target, cv=10)
    elif selection == 2:
        classifier = svm.SVC(kernel="linear", C=1)
        scores = cross_val_score(classifier, data, target, cv=10)
        scoring = None
    elif selection == 3:
        classifier = linear_model.LinearRegression()
        scoring = "neg_mean_squared_error"
        scores = cross_val_score(classifier, data, target, cv=10, scoring=scoring)

    print("Accuracy: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std() * 2))


if __name__ == '__main__':
    main()
