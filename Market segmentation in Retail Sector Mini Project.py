#!/usr/bin/env python
# coding: utf-8

# ### Market Segmentation in Retail Sector
#
# This code performs market segmentation in the retail sector using K-Means clustering. It loads a CSV file containing customer data, explores the data, and performs clustering to segment the customers into different groups based on their satisfaction and loyalty.

# #### Import the relevant libraries
#
# We import the following libraries:
#
# * `pandas` for data manipulation
# * `numpy` for numerical computations
# * `matplotlib` and `seaborn` for data visualization
# * `sklearn` for clustering

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn import preprocessing

# #### Load the data
#
# We load a CSV file named `MarketData.csv` into a pandas DataFrame using `pd.read_csv`.

data = pd.read_csv ('MarketData.csv')

# #### Data Exploration
#
# We perform some preliminary data exploration:
#
# * We plot a scatter plot of `Satisfaction` vs `Loyalty` to visualize the data.
# * We plot a histogram of `Age` to visualize the age distribution.
# * We plot a pie chart of `Gender` to visualize the gender distribution.

def plot_scatter(data, x, y):
    """
    Plots a scatter plot of `x` vs `y` for the given data.

    Parameters:
    data (pandas.DataFrame): The data to plot.
    x (str): The name of the x-axis variable.
    y (str): The name of the y-axis variable.

    Returns:
    None
    """
    plt.scatter(data[x], data[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

def plot_histogram(data, column):
    """
    Plots a histogram of the given column for the given data.

    Parameters:
    data (pandas.DataFrame): The data to plot.
    column (str): The name of the column to plot.

    Returns:
    None
    """
    plt.hist(data[column], bins=20)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_pie(data, column):
    """
    Plots a pie chart of the given column for the given data.

    Parameters:
    data (pandas.DataFrame): The data to plot.
    column (str): The name of the column to plot.

    Returns:
    None
    """
    labels = data[column].value_counts().index
    sizes = data[column].value_counts().values
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()

plot_scatter(data, 'Satisfaction', 'Loyalty')
plot_histogram(data, 'Age')
plot_pie(data, 'Gender')

# #### Feature Selection
#
# We select the first two columns of the data (`Satisfaction` and `Loyalty`) as features for clustering.

x = data.iloc[:, 0:2]

# #### Clustering
#
# We perform K-Means clustering with 4 clusters using `KMeans` from `sklearn`. We fit the model to the selected features and predict the cluster labels.

def kmeans_clustering(x, k):
    """
    Performs K-Means clustering with the given number of clusters on the given data.

    Parameters:
    x (pandas.DataFrame): The data to cluster.
    k (int): The number of
