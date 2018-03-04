#!/usr/sbin/python

import sys
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weights, hasBias=True, biasValue=-1):
        self.weights = copy.copy(weights)
        self.hasBias = hasBias
        self.biasValue = biasValue

    def simulate(self, inputs):
        sigma = 0
        if np.shape(self.weights)[0] != \
           (np.shape(inputs)[0] + int(self.hasBias)):
            raise IndexError("Input size not equal to weights size")
        if self.hasBias:
            inputs = np.insert(inputs, 0, self.biasValue)
        self.lastInputs = inputs
        for weight, value in zip(self.weights, inputs):
            sigma += weight * value
        self.lastResult = sigmoid(sigma)
        return self.lastResult

    def newWeights(self, weights):
        self.newWeights = np.array(copy.copy(weights))

    def replaceWeights(self):
        if not hasattr(self, "newWeights"):
            raise RuntimeError("no new weights to replace with")
        self.weights = np.array(copy.copy(self.newWeights))
        self.newWeights = None
        del self.newWeights

    def __repr__(self):
        i = 0 + (not self.hasBias)
        retval = ""
        for weight in self.weights:
            retval += "Weight {}: {}\n".format(i, weight)
            i += 1
        return retval


class PerceptronLayer():
    def __init__(self,
                 numberOfNodes,
                 inputSize,
                 useBias=True,
                 biasValue=-1):
        self.usingBias = useBias
        weights = np.zeros(inputSize + int(useBias))
        nodes = []
        for _i in range(numberOfNodes):
            for j in range(inputSize + int(useBias)):
                weights[j] = random.uniform(-1, 1)
            nodes.append(Neuron(weights, useBias, biasValue))
        self.nodes = np.array(nodes)

    def __repr__(self):
        retval = "Using bias node: {}\n".format(self.usingBias)
        i = 0
        for node in self.nodes:
            retval += "Node {}:\n{}\n".format(i, node)
            i += 1
        return retval

    def simulate(self, inputs):
        results = []
        for node in self.nodes:
            results.append(node.simulate(inputs))
        return np.array(results)


class Perceptron:
    ''' Default init settings create a single layer perceptron '''
    def __init__(self,
                 inputSize,
                 hiddenLayers=0,
                 hiddenNodes=0,
                 outputNodes=1,
                 useBias=True,
                 biasValue=-1):
        self.layers = []
        self.inputSize = inputSize
        nodeNumIsList = False
        if type(hiddenNodes) is not int:
            nodeNumIsList = True
            nodeNumList = hiddenNodes
        for _i in range(hiddenLayers):

            if nodeNumIsList:
                hiddenNodes = nodeNumList[_i]
                if _i != 0:
                    inputSize = nodeNumList[_i - 1]

            self.layers.append(PerceptronLayer(hiddenNodes,
                                               inputSize,
                                               useBias,
                                               biasValue))
            inputSize = hiddenNodes

        self.outputLayer = PerceptronLayer(outputNodes,
                                           inputSize,
                                           useBias,
                                           biasValue)

    def __repr__(self):
        index = 1
        retval = ""

        for layer in self.layers:
            retval += "Layer {}:\n{}".format(index, layer)
            index += 1
        retval += "Output Layer:\n{}".format(self.outputLayer)

        return retval

    def simulate(self, inputValues):
        if np.shape(inputValues)[0] != self.inputSize:
            raise ValueError("input size does not match perceptron")

        for layer in self.layers:
            inputValues = layer.simulate(inputValues)
        return self.outputLayer.simulate(inputValues)


class PerceptronModel:
    def fit(self,
            trainingData,
            trainingTargets,
            hiddenLayers=0,
            hiddenNodes=0,
            outputNodes=1,
            useBias=True,
            biasValue=-1,
            learningRate=0.1,
            patience=250,
            maxIterations=1000,
            stopTrainingAt=95):
        self.learningRate = learningRate
        self.perceptron = Perceptron(np.shape(trainingData)[1],
                                     hiddenLayers,
                                     hiddenNodes,
                                     outputNodes,
                                     useBias,
                                     biasValue)
        targetArray = np.zeros(outputNodes)
        accuracyHistory = []
        for _i in range(maxIterations):
            score = accuracy_score(trainingTargets, self.predict(trainingData))
            accuracyHistory.append(score*100)
            sys.stdout.write("\rTraining iteration {} - Accuracy: {:.1f}".format(_i, score*100))
            sys.stdout.flush()
            bestScore = 0
            backup = copy.deepcopy(self.perceptron)
            for row, target in zip(trainingData, trainingTargets):
                self.perceptron.simulate(row)
                targetArray[target] = 1
                self.updateWeights(targetArray)
                targetArray[target] = 0

            if score > bestScore:
                bestScore = score
            if np.std(accuracyHistory[-patience:]) < 0.1 and _i > patience:
                print("\nEnding early due to no learning progress")
                break
            if _i > 2:
                if bestScore > accuracyHistory[-1]:
                    backup = copy.deepcopy(self.perceptron)
            if accuracyHistory[-1] > stopTrainingAt:
                print("\nTarget accuracy reached")
                break

        lastPerceptron = copy.deepcopy(self.perceptron)
        self.perceptron = backup
        score = accuracy_score(trainingTargets, self.predict(trainingData))
        if accuracyHistory[-1] < score:
            print("Backup model was better than last perceptron, restoring...")
            self.perceptron = backup
        else:
            self.perceptron = lastPerceptron
        return accuracyHistory

    def __repr__(self):
        return self.perceptron

    def updateWeights(self, targets):
        perceptron = self.perceptron

        # Update output layer first
        for node, target in \
                zip(perceptron.outputLayer.nodes, targets):
            node.error = node.lastResult * \
                         (1 - node.lastResult) * \
                         (node.lastResult - target)
            newWeights = []
            for weight, value in zip(node.weights, node.lastInputs):
                newWeights.append(weight -
                                  (self.learningRate * node.error * value))
            node.newWeights(newWeights)

        # Update remaining layers
        nextLayer = perceptron.outputLayer
        for layer in perceptron.layers:
            index = 0
            for node in layer.nodes:
                sigma = 0
                for outerNode in nextLayer.nodes:
                    sigma += outerNode.weights[index] * outerNode.error

                node.error = node.lastResult * (1 - node.lastResult) * sigma
                newWeights = []
                for weight, value in zip(node.weights, node.lastInputs):
                    newWeights.append(weight -
                                      (self.learningRate * node.error * value))
                node.newWeights(newWeights)
                index += 1
            nextLayer = layer

        for node in perceptron.outputLayer.nodes:
            node.replaceWeights()

        for layer in perceptron.layers:
            for node in layer.nodes:
                node.replaceWeights()

    def predict(self, data):
        if not hasattr(self, "perceptron"):
            raise RuntimeError("model has not been trained")
        results = []
        for row in data:
            rowResult = self.perceptron.simulate(row)
            results.append(np.argmax(rowResult))
        return np.array(results)


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
    target = new_values[:, -1].astype(int)
    data = new_values[:, :-1]
    return target, data


def main():
    # Constants for easy customization
    irisStop = 90
    irisHidden = 2
    irisNodes = (4, 3)
    irisRate = 0.1
    irisPatience = 250
    pimaStop = 80
    pimaHidden = 1
    pimaNodes = 7
    pimaRate = 0.2
    pimaPatience = 50

    iris = load_iris()

    data_train, data_test, target_train, target_test = \
        train_test_split(iris.data,
                         iris.target,
                         test_size=0.3,
                         train_size=0.7,
                         shuffle=True)

    scaler1 = MinMaxScaler()
    data_train = scaler1.fit_transform(data_train)
    data_test = scaler1.transform(data_test)

    # Fit code goes here
    model = PerceptronModel()
    irisHistory = model.fit(data_train,
                            target_train,
                            hiddenLayers=irisHidden,
                            hiddenNodes=irisNodes,
                            learningRate=irisRate,
                            outputNodes=3,
                            patience=irisPatience,
                            stopTrainingAt=irisStop)

    score = accuracy_score(target_test, model.predict(data_test))
    print("\nIris accuracy: {0:.1f}%".format(score*100))

    plt.plot(irisHistory)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Iris Final Accuracy: {:.1f}".format(score*100))
    plt.savefig("irisTrainingProgress.png")

    plt.clf()

    pima_target, pima_data = load_pima()

    data_train, data_test, target_train, target_test = \
        train_test_split(pima_data,
                         pima_target,
                         test_size=0.3,
                         train_size=0.7,
                         shuffle=True)

    scaler2 = MinMaxScaler()
    data_train = scaler2.fit_transform(data_train)
    data_test = scaler2.transform(data_test)

    # Fit code goes here
    model2 = PerceptronModel()
    pimaHistory = model2.fit(data_train,
                             target_train,
                             hiddenLayers=pimaHidden,
                             hiddenNodes=pimaNodes,
                             outputNodes=2,
                             learningRate=pimaRate,
                             patience=pimaPatience,
                             stopTrainingAt=pimaStop)

    score = accuracy_score(target_test, model2.predict(data_test))
    print("\nPima accuracy: {0:.1f}%".format(score*100))

    plt.plot(pimaHistory)
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Pima Final Accuracy: {:.1f}".format(score*100))
    plt.savefig("pimaTrainingProgress.png")


if __name__ == '__main__':
    main()
