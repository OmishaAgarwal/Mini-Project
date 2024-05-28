import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Set up Streamlit configuration
st.set_page_config(page_title='Market Data Analysis and Clustering', layout='wide')

# Custom CSS styles
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .header {
        font-size: 2em;
        color: #333;
        text-align: center;
    }
    .subheader {
        font-size: 1.5em;
        color: #555;
    }
    .sidebar .sidebar-content {
        background: #f0f0f0;
    }
    .uploadedFile {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit setup
st.markdown('<div class="header">Market Data Analysis and Clustering</div>', unsafe_allow_html=True)
st.write("This app performs data analysis and clustering on market data.")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)

    # Define functions for plotting
    def plot_scatter(data, x, y):
        fig, ax = plt.subplots()
        ax.scatter(data[x], data[y])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        st.pyplot(fig)

    def plot_histogram(data, column):
        fig, ax = plt.subplots()
        ax.hist(data[column], bins=20)
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    def plot_pie(data, column):
        fig, ax = plt.subplots()
        labels = data[column].value_counts().index
        sizes = data[column].value_counts().values
        ax.pie(sizes, labels=labels, autopct='%1.1f%%')
        ax.axis('equal')
        st.pyplot(fig)

    # Data exploration
    st.markdown('<div class="subheader">Data Exploration</div>', unsafe_allow_html=True)
    plot_scatter(data, 'Satisfaction', 'Loyalty')
    plot_histogram(data, 'Age')
    plot_pie(data, 'Gender')

    # Feature selection
    x = data[['Satisfaction', 'Loyalty']]

    # Clustering
    def kmeans_clustering(x, k):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(x)
        labels = kmeans.labels_
        return labels

    st.markdown('<div class="subheader">KMeans Clustering</div>', unsafe_allow_html=True)
    k = st.slider('Select number of clusters for KMeans', 2, 10, 4)
    labels = kmeans_clustering(x, k)

    # Add cluster labels to data
    data['Cluster'] = labels

    # Visualize clustered data
    st.markdown('<div class="subheader">Clustering Result</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots()
    scatter = ax.scatter(data['Satisfaction'], data['Loyalty'], c=data['Cluster'], cmap='viridis')
    ax.set_xlabel('Satisfaction')
    ax.set_ylabel('Loyalty')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    st.pyplot(fig)

    # Predict loyalty score based on cluster label
    loyalty_scores = np.linspace(0, 1, k)

    # Add loyalty scores to data
    data['Loyalty Score'] = data['Cluster'].apply(lambda x: loyalty_scores[x])

    # Display clustered data with loyalty scores
    st.markdown('<div class="subheader">Clustered Data with Loyalty Scores</div>', unsafe_allow_html=True)
    st.write(data)
else:
    st.markdown('<div class="uploadedFile">Please upload a CSV file to proceed.</div>', unsafe_allow_html=True)
