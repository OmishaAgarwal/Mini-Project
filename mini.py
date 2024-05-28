import csv
import math

# Load the data
data = []
with open('MarketData.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        data.append(row)

# Convert data to float
for i in range(len(data)):
    for j in range(len(data[i])):
        data[i][j] = float(data[i][j])

# Define functions for plotting
def plot_scatter(data, x, y):
    for i in range(len(data)):
        print(f"({data[i][x]}, {data[i][y]})")

def plot_histogram(data, column):
    freq = {}
    for i in range(len(data)):
        if data[i][column] in freq:
            freq[data[i][column]] += 1
        else:
            freq[data[i][column]] = 1
    for key, value in freq.items():
        print(f"{key}: {value}")

def plot_pie(data, column):
    freq = {}
    for i in range(len(data)):
        if data[i][column] in freq:
            freq[data[i][column]] += 1
        else:
            freq[data[i][column]] = 1
    for key, value in freq.items():
        print(f"{key}: {value} ({value/len(data)*100:.2f}%)")

# Data exploration
plot_scatter(data, 0, 1)
plot_histogram(data, 2)
plot_pie(data, 3)

# Feature selection
x = [[row[0], row[1]] for row in data]

# Clustering
def euclidean_distance(x, y):
    return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def kmeans_clustering(x, k):
    # Initialize centroids randomly
    centroids = [x[i] for i in range(k)]
    labels = [0] * len(x)
    while True:
        new_labels = []
        for i in range(len(x)):
            distances = [euclidean_distance(x[i], centroid) for centroid in centroids]
            label = distances.index(min(distances))
            new_labels.append(label)
        if new_labels == labels:
            break
        labels = new_labels
        centroids = [[0, 0] for _ in range(k)]
        counts = [0] * k
        for i in range(len(x)):
            centroids[labels[i]][0] += x[i][0]
            centroids[labels[i]][1] += x[i][1]
            counts[labels[i]] += 1
        for i in range(k):
            centroids[i][0] /= counts[i]
            centroids[i][1] /= counts[i]
    return labels

labels = kmeans_clustering(x, 4)

# Add cluster labels to data
for i in range(len(data)):
    data[i].append(labels[i])

# Predict loyalty score based on cluster label
loyalty_scores = [0, 0.5, 0.8, 1]

# Add loyalty scores to data
for i in range(len(data)):
    data[i].append(loyalty_scores[data[i][4]])

# Visualize clustered data with loyalty scores
for i in range(len(data)):
    print(f"({data[i][0]}, {data[i][1]}) - Cluster {data[i][4]} - Loyalty Score {data[i][5]}")
