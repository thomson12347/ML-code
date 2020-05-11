# program for implementation of KNN algorithm on Breast Cancer Data

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("K is set to be a value less than total voting groups !")
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance,group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result, distances, votes


def read_train():
    df = pd.read_csv('breast-cancer-wisconsin.data')
    df.replace("?", -99999, inplace=True)
    df.drop(['Id'], 1, inplace=True)
    full_data = df.astype(float).values.tolist()

    random.shuffle(full_data)

    test_size = 0.2
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(test_size * len(full_data))]
    test_data = full_data[-int(test_size * len(full_data)):]
    print(train_data[:10])
    print(test_data[:10])

    for i in train_data:
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        test_set[i[-1]].append(i[:-1])
    return train_set, test_set


def accuracy_calc(train_set, test_set):
    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, distances, votes = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            total += 1
    print("Correct Predictions : ", correct)
    print("Total : ", total)
    print("Accuracy : ", correct / total)
    return distances


def plotting(distances):
    col = lambda val: 0 if val == 2 else 1
    # random.shuffle(distances)
    for i, j in enumerate(distances):
        l = lambda j: 0 if (j == 2) else 1
        plt.scatter(i, j[1],
                    c=np.atleast_2d(ListedColormap(('black', 'red'))(l(j[1]))), label=j[1])
    plt.show()


# driver code
train_set, test_set = read_train()
distances = accuracy_calc(train_set, test_set)
plotting(distances)