
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn import preprocessing

# Load the data
data = pd.read_csv('MarketData.csv')

# Define functions for plotting
def plot_scatter(data, x, y):
    plt.scatter(data[x], data[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

def plot_histogram(data, column):
    plt.hist(data[column], bins=20)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_pie(data, column):
    labels = data[column].value_counts().index
    sizes = data[column].value_counts().values
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()

# Data exploration
plot_scatter(data, 'Satisfaction', 'Loyalty')
plot_histogram(data, 'Age')
plot_pie(data, 'Gender')

# Feature selection
x = data.iloc[:, 0:2]

# Clustering
def kmeans_clustering(x, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    labels = kmeans.labels_
    return labels

labels = kmeans_clustering(x, 4)

# Add cluster labels to data
data['Cluster'] = labels

# Visualize clustered data
plt.scatter(data['Satisfaction'], data['Loyalty'], c=data['Cluster'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()
