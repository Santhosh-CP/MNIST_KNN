import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from keras.datasets import mnist
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_blobs


def distance(p1, p2):
    return np.sum((p2 - p1) ** 2) ** 0.5


def kNearestNeighbors(X, y, queryPoint, k=5):
    m = X.shape[0]  # Here, it's 100
    Distances = []

    for i in range(m):
        d = distance(queryPoint, X[i])
        Distances.append(d)

    Distances = sorted(Distances)[:k]

    return Distances

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=2)
query_point = np.array([0, -5])
dist = kNearestNeighbors(X, y, query_point)
print(dist)