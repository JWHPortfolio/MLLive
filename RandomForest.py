import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

X = np.load('data/MLindependent.npy')

y = np.load('data/MLdependent.npy')

XNames = np.load('data/MLindependentNames.npy')
yNames = np.load('data/MLdependentNames.npy')

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10,criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
