import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pickle
import copy

class Dataset:
    def __init__(self):
        self.dataset = pd.read_csv('data/survey.csv')
        
    def dataSelection(self, selectionString="2,1,3", extension="independent" ):     
        

        # get rid of leading columns
        dataset = self.dataset.iloc[1:,10:].values
        print("Selection: ", selectionString)

        #breakup set
        selArray = selectionString.split(",")
        
        # get names
        names = self.dataset.iloc[0:1,10:].values
        namesArray = []
        
        nameCount = 0
        for i in selArray:
            print(names[0,int(i)])
            namesArray.append(names[0, int(i)])
            nameCount += 1
        print("Selected Names: ", namesArray)   

        #change to set of integers (no repeats) in reverse order
        iSet = set()
        for i in selArray:
            iSet.add(int(i))
        length = len(dataset[0])  
        for i in reversed(range(0,length)):
           if i not in iSet:
            #Get rid of columns not wanted
            dataset = np.delete(dataset,i,1)
        print("LENGTH: ", len(dataset), " Shape: ", dataset.shape)
        
        #change categorical data to 1/0 columns
        labelencoder = LabelEncoder()
        catFeatures = {}
        for i in range(0, len(dataset[0])):
            try:
                #test to see if it is an integer
                int(dataset[:,i][0])
                # Do not run onehotencoder on this
                catFeatures.update({i:0})
            # Not an integer
            except ValueError:
                #determine length
                if( len(set(dataset[:,i])) > 2): 
                    print("MULTIPLE: ", len(set(dataset[:,i])))
                    dataset[:,i] = labelencoder.fit_transform(dataset[:,i])
                    catFeatures.update({i:len(set(dataset[:,i]))})
                else:
                    dataset[:,i] = labelencoder.fit_transform(dataset[:,i]) 
                    catFeatures.update({i:0})

        catFeaturesArray = []
        #print("Features: ", catFeatures.items())
        datasetFinal = copy.deepcopy(dataset)
        for key, value in catFeatures.items():
            if(value == 0):
                continue
            print("**Features - Key: ", key, " Value: ", value)
            print("**SHAPE: ", dataset.shape, " SHAPE2: ", dataset[:,key].shape)
            catFeaturesArray.append(key)
            onehotencoder = OneHotEncoder()
            #integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            integer_encoded = dataset[:,key].reshape(len(dataset[:,key]), 1)
            datasetencode = onehotencoder.fit_transform(integer_encoded, value).toarray()
            if( value > 2): #delete 1 column
                datasetencode = np.delete(datasetencode, 0, 1)
            print("Shape after: ", datasetencode.shape)
            print(datasetencode)
            # must delete the original column and add in this
            datasetFinal = np.delete(datasetFinal, key, axis=1)
            #dataset = np.concatenate([dataset, np.array([[1],[1]]).dot(datasetencode)], axis=1)
            datasetFinal = np.column_stack((datasetFinal, datasetencode))
            #np.concatenate(dataset, datasetencode, 1)
            #np.append(dataset, datasetencode, 1)
            
        print("COMPLETED DATASET SHAPE: ", datasetFinal.shape)
        #must delete one column for each categorical data set
        #delCol = 0
        #for key, value in catFeatures.items():
            #dataset = np.delete(dataset, delCol, 1)
            #adjusts for the disappearing column from above
            #print("Removed 1 Column due to onehotencoder")
            #delCol += value -1

        # Must scale the data
        # Feature Scaling
        
        filenameBase = 'data/ML' + extension
       
        if( nameCount > 1): #Onlyc need to scale when there is more than one feature
            sc = StandardScaler()
            dataset = sc.fit_transform(dataset)
        
            # save scaler to later reverse transform
         
            filename = filenameBase + '.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(sc, f, pickle.HIGHEST_PROTOCOL)

        #save data
        print("Data Shape: ", datasetFinal.shape)
        fileName = filenameBase +".npy"
        np.save(fileName, datasetFinal)
        
        #save names
        fileName = filenameBase + "Names.npy"
        #convert to numpy array
        namesNP = np.array(namesArray)
        
        np.save(fileName,namesNP)

        #datasetNew = np.load('data/MachineLine2.npy')

        #if( np.array_equal(dataset, datasetNew)):
            #message = "Exact Copy"
            #return dataset
        #else:
            #message = "Incorrect Data Copy"
        return datasetFinal


# d = Dataset()
# datas = d.dataSelection(selectionString = '1')
