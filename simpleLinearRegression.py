import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


import pandas as pd

X = np.load('data/MLindependent.npy')

y = np.load('data/MLdependent.npy')

XNames = np.load('data/MLindependentNames.npy')
yNames = np.load('data/MLdependentNames.npy')

for name in XNames:
    print("I: ", name)
for name in yNames:
    print("D: ", name)

#Splitting the data set into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =1/3, random_state = 0)

#Fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#makes some predictions
X_test = X_test.reshape(-1,1)  #had to add this line JWH
regressor.predict(2)
ypred = regressor.predict(X_test)
#regressor.score()

plt.scatter(X_train.tolist(), y_train.tolist(), color = 'red')
pre = regressor.predict(X_train)
plt.scatter(X_train.tolist(), pre.tolist(), color = 'blue')
#plt.title("X vs y (Training Set)")
plt.title(XNames[0] + " vs " + yNames[0])
plt.xlabel(XNames[0])
plt.ylabel(yNames[0])
plt.show()


plt.scatter(X_test.tolist(), y_test.tolist(), color = 'red')

plt.scatter(X_train.tolist(), regressor.predict(X_train).tolist(), color = 'blue')
plt.title(XNames[0] + " vs " + yNames[0] + " (Training Set)")
plt.xlabel(XNames[0])
plt.ylabel(yNames[0])
plt.show()



#sc = StandardScaler()
#datasetUS = sc.inverse_transform(X)
#datasetUS