import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# All Features - Coding and Exercise

import pandas as pd

X = np.load('data/MLindependent.npy',allow_pickle = True)
XNames = np.load('data/MLindependentNames.npy',allow_pickle = True)

#using elbow method to determine number of clusters
from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans =KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init =10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()#Applying K-Means to the mall dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter=300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s = 100, c = 'red', label = 'Lazy')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s = 100, c = 'blue', label = 'Average')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s = 100, c = 'green', label = 'Hard Worker')
#plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=3, c = 'yellow', label = 'Centroids')
plt.title('Clusters')
plt.xlabel(XNames[0])
plt.ylabel(XNames[1])
plt.legend()
plt.show()

