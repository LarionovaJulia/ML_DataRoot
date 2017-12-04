import csv
import random
import math
import numpy as np

def loadCsv(filename):
    return np.genfromtxt(filename, delimiter = ',')

def splitDataset(dataset, splitRatio):
    trainSize = int(dataset.shape[0] * splitRatio)
    indices = np.random.permutation(dataset.shape[0])
    training_idx, test_idx = indices[:trainSize], indices[trainSize:]
    training, test = dataset[training_idx,:], dataset[test_idx,:]
    return training, test

def separateByClass(dataset):
    return {
        1: dataset[np.where(dataset[:, -1]==1), :],
        0: dataset[np.where(dataset[:, -1]==0), :]
    }

def summarize(dataset):
    means = dataset.mean(axis=1)[0][:-1]
    stds = dataset.std(axis=1, ddof=1)[0][:-1]
    return means, stds

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
    return np.exp(-(x - mean)**2/(2 * stdev**2)) / (np.sqrt(2 * np.pi) * stdev)

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        means = classSummaries[0]
        stds  = classSummaries[1]
        probabilities[classValue] = np.prod(calculateProbability(inputVector[:-1], means, stds))
    return probabilities

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def main():
    filename = 'data.csv'
    splitRatio = 0.67
    dataset = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
    summaries = summarizeByClass(trainingSet)
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%'.format(accuracy))

main()

