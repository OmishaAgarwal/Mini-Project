#!/usr/bin/env python
# coding: utf-8

# # Market Segmentation in Retail Sector

# ## Import the relevant libraries

# In[4]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# ## Load the data

# In[5]:


data = pd.read_csv ('MarketData.csv')


# In[6]:


data


# ## Plot the data

# Create a preliminary plot to see if you can spot something

# In[7]:


plt.scatter(data['Satisfaction'],data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()


# ## Plot the Age 

# In[8]:


plt.xticks([i for i in range(0,101,5)])
plt.hist(data['Age'])
plt.show()


# ## Plot the Gender

# In[9]:


li=[0,0]
for i in data['Gender']:
    li[i]+=1
print(li)
labels = ['Male','Female']
colors=['Turquoise','Orange']
plt.pie(li,labels = labels,colors=colors, autopct="%1.1f%%",shadow=True)
plt.show()


# ## Select the features

# In[10]:


x = data.iloc[:,0:2]


# ## Clustering

# In[11]:


kmeans = KMeans(4)
kmeans.fit(x)


# ## Clustering results

# In[12]:


clusters = x.copy()
clusters['cluster_pred']=kmeans.fit_predict(x)
clusters


# In[13]:


plt.scatter(clusters['Satisfaction'],clusters['Loyalty'],c=clusters['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()


# ## Standardize the variables

# Let's standardize and check the new result

# In[14]:


from sklearn import preprocessing
x_scaled = preprocessing.scale(x)
x_scaled


# ## Take advantage of the Elbow method

# In[15]:


wcss =[]
for i in range(1,10):
    # Clsuter solution with i clusters
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
    
wcss


# In[16]:


plt.plot(range(1,10),wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# ## Explore clustering solutions and select the number of clusters

# In[17]:


kmeans_new = KMeans(5)
kmeans_new.fit(x_scaled)
clusters_new = x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)


# In[18]:


clusters_new


# In[19]:


plt.scatter(clusters_new['Satisfaction'],clusters_new['Loyalty'],c=clusters_new['cluster_pred'],cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()

