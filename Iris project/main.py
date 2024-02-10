#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

SEED = 0

iris_dataset = load_iris()

print(iris_dataset["feature_names"])
print(iris_dataset["data"][0])

# splitting the dataset

X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=SEED)

print("X_train shape: {}\ny_train shape: {}".format(X_train.shape, y_train.shape))

# Before create a ML model we can visualize the data of the dataset
# we can plot it with scatter plot but we have only 2 dimension

# Plot the dataframe with pair plot, which looks at all possible pairs of features.
# First create a Pandas dataframe
iris_pandas = pd.DataFrame(iris_dataset["data"], columns=iris_dataset["feature_names"])
print(iris_pandas.head(5))

sns.pairplot(iris_pandas, hue=y_train, diag_kind="hist", markers=["o"])