import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing

# Load the data
data = []
with open('MarketData.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        data.append([float(x) for x in row])

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(data, columns=['Satisfaction', 'Loyalty', 'Age', 'Gender'])

# Plot the data
plt.scatter(df['Satisfaction'], df['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

# Plot the Age
plt.xticks([i for i in range(0, 101, 5)])
plt.hist(df['Age'])

# Plot the Gender
li = [0, 0]
for i in df['Gender']:
    li[i] += 1
labels = ['Male', 'Female']
colors = ['Turquoise', 'Orange']
plt.pie(li, labels=labels, colors=colors, autopct="%1.1f%%", shadow=True)

# Select the features
x = df[['Satisfaction', 'Loyalty']]

# Clustering
kmeans = KMeans(4)
kmeans.fit(x)

# Clustering results
clusters = x.copy()
clusters['cluster_pred'] = kmeans.fit_predict(x)

# Standardize the variables
x_scaled = preprocessing.scale(x)

# Take advantage of the Elbow method
wcss = []
for i in range(1, 10):
    # Cluster solution with i clusters
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

# Explore clustering solutions and select the number of clusters
kmeans_new = KMeans(5)
kmeans_new.fit(x_scaled)
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)

# Plot the final result
plt.scatter(clusters_new['Satisfaction'], clusters_new['Loyalty'], c=clusters_new['cluster_pred'], cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')