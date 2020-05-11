# program to implement KNN algorithm with random values and its power calculations

import math
from random import shuffle


def ReadData(filename):
    f = open(filename, 'r')
    lines = f.read().splitlines()
    f.close()
    features = lines[0].split(', ')[:-1]
    items = []
    for i in range(1, len(lines)):
        line = lines[i].split(', ')
        itemFeatures = {'Class': line[-1]}
        for j in range(len(features)):
            f = features[j]
            v = float(line[j])
            itemFeatures[f] = v
        items.append(itemFeatures)
    shuffle(items)
    print("Pulled items from Dataset")
    for i in items:
        print(i)
    return items


def EuclideanDistance(x, y):
    S = 0
    for key in x.keys():
        S += math.pow(x[key]-y[key], 2)
    return math.sqrt(S)


def ClaculateNeighborsClass(neighbors, k):
    count = {}
    for i in range(k):
        if neighbors[i][1] not in count:
            count[neighbors[i][1]] = 1
        else:
            count[neighbors[i][1]] += 1
    return count


def FindMax(Dict):
    maximum = -1
    classification = ''
    for key in Dict.keys():
        if Dict[key] > maximum:
            maximum = Dict[key]
            classification=key
    return (classification, maximum)


def Classify(nitem, k, items):
    neighbors = []
    for item in items:
        distance = EuclideanDistance(nitem, item)
        neighbors = UpdateNeighbors(neighbors, item, distance, k)
    count = ClaculateNeighborsClass(neighbors, k)
    return FindMax(count)


def UpdateNeighbors(neighbors, item, distance, k,):
    if len(neighbors) < k:
        neighbors.append([distance, item['Class']])
        neighbors = sorted(neighbors)
    else:
        if neighbors[-1][0] > distance:
            neighbors[-1] = [distance, item['Class']]
            neighbors = sorted(neighbors)
    return neighbors


def K_FoldValidation(K, k, items):
    if K > len(items):
        return -1
    correct = 0
    total = len(items) * (K - 1)
    l = int(len(items) / k)
    for i in range(K):
        trainingSet = items[i * l:(i + 1) * l]
        testSet = items[:i * l] + items[(i + 1) * l:]
        for item in testSet:
            itemClass = item['Class']
            itemFeatures = {}
            for key in item:
                if key != 'Class':
                    itemFeatures[key] = item[key]
            guess = Classify(itemFeatures, k, trainingSet)[0]
            if guess == itemClass:
                correct += 1
    accuracy = correct / float(total)
    return accuracy


def Evaluate(K, k, items, iterations):
    accuracy = 0
    for i in range(iterations):
        shuffle(items)
        accuracy += K_FoldValidation(K, k, items)
    print("Accuracy : ", accuracy / float(iterations))


# driver code
items = ReadData('data.txt')
Evaluate(4, 4, items, 60)
