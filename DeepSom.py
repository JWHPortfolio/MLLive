import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

X = np.load('data/MLindependent.npy')
XNames = np.load('data/MLindependentNames.npy')

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = len(XNames), sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
'''
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
'''
show()
'
#get probablities'

print( som.distance_map())