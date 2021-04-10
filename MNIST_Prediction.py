import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from keras.datasets import mnist
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

(train_X, train_y), (test_X, test_y) = mnist.load_data()

df = pd.DataFrame(train_X)
print(df.head())
df.fillna(128)
df.sort_values(by="0.530")