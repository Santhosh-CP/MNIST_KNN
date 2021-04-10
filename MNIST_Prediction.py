import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def distance(p1, p2):
    return np.sum((p2 - p1) ** 2) ** 0.5


def manualKNN(X, y, queryPoint, k=5):
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


def  displayAccuracy(X_train, y_train, X_test, y_test):
    predictions = []
    num = len(y_test)
    correct = 0  # Number of correct predictions

    for i in range(num):
        prediction = manualKNN(X_train, y_train, X_test[i])
        if int(prediction) == int(y_test[i]):
            correct += 1
        print(f"i={i} Correct={correct}")
    print(f"Number of correct predictions = {correct}")

    

def libraryKNN(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracyScore = accuracy_score(y_test, predictions)
    print(f"Accuracy Score of KNN Model = {accuracyScore}")

    return knn


def randomSample(model, X_test, y_test):
    num = len(y_test)

    x = random.randint(0, num-1)
    print(f"Random Value = {x}")
    print(f"Predicted Value = {model.predict(x)[0]}")
    plt.imshow(X_test.reshape(28, 28))
    plt.show()

def main():
    # Reading the MNIST dataset
    data = pd.read_csv("train.csv")
    print(data.shape)
    mnist = data.values
    X = mnist[:, :-1]
    y = mnist[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=50)
    model = libraryKNN(X_train, X_test, y_train, y_test)
    randomSample(model, X_test, y_test)

main()