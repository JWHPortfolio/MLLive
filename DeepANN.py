# Importing the libraries
import scipy
import numpy as np
import pandas as pd
import sklearn

X = np.load('data/MLindependent.npy',allow_pickle = True)
inputDim = X.shape[1]

y = np.load('data/MLdependent.npy',allow_pickle = True)

y = y.reshape(len(y),)
y = y.astype(int)

XNames = np.load('data/MLindependentNames.npy',allow_pickle = True)
yNames = np.load('data/MLdependentNames.npy',allow_pickle = True)
outputDim = len(yNames)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

output_dim = (inputDim+outputDim)
model = Sequential([
    Dense(output_dim, activation='relu'),
    Dense(6, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss=BinaryCrossentropy(),
              metrics=['accuracy'])
model.fit(X_train, y_train, batch_size= 5, epochs=20)


# Predicting the Test set results
y_pred = model.predict(X_train)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)
print (cm)
# % correct
print("TRAIN PREFORMANCE: ", (cm[0,0]+cm[1,1])/sum(sum(cm)))

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)
# % correct
print("TEST PERFORMANCE: ",(cm[0,0]+cm[1,1])/sum(sum(cm)))
