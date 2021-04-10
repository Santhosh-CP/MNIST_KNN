import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split



def distance(p1, p2):
    return np.sum((p2 - p1) ** 2) ** 0.5


def kNearestNeighbors(X, y, queryPoint, k=5):
    m = X.shape[0]  # Here, it's 100
    distances = []

    for i in range(m):
        d = distance(queryPoint, X[i])
        distances.append((d, y[i]))

    distances = sorted(distances)
    distances = np.array(distances)[:k]

    labels = distances[:, 1]

    newLabels, values = np.unique(labels, return_counts=True)
    prediction = newLabels[np.argmax(values)]

    return prediction

def main():
    queryPoint = np.array([0, -5])

    data = pd.read_csv("train.csv")
    mnist = data.values
    X = mnist[:, :-1]
    y = mnist[:, -1]
    print("Out")
    plt.imshow(X[789].reshape(28,28))
    plt.show()
    print("fullut")

    # Splitting into train and test
    X_train = X[0:15000]
    y_train = y[0:15000]
    X_test = X[15000:20000]
    y_test = y[15000:20000]


main()