# Importing the libraries
import scipy
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import sklearn

X = np.load('data/MLindependent.npy')

y = np.load('data/MLdependent.npy')
#y = y.tolist()

y = y.reshape(58,)
y = y.astype(int)

XNames = np.load('data/MLindependentNames.npy')
inputDim = len(XNames)
yNames = np.load('data/MLdependentNames.npy')
outputDim = len(yNames)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
#output_dim starts out as number of inputs + number of outputs which is 12/2 = 6
#init is a function that populates random numbers close to zero - uniform using uniform distribution
# input_dim is number of independent variables
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = inputDim))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
# the following is only for 1 output.  If more than 1 change output_dim accordinly and activation = "softmax"
classifier.add(Dense(output_dim = outputDim, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# batch size and number of epochs are "art" - try one and then rerun
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 20)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)