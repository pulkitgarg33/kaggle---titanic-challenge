# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_test_set(dataset):
    
    dataset = dataset.drop('Name' , 1)
    dataset = dataset.drop('Cabin' , 1)
    dataset = dataset.drop('Ticket' , 1)
    dataset = dataset.drop('PassengerId' , 1)
    dataset = dataset.drop('Embarked' , 1)
    dataset = dataset.drop('Fare' , 1)

    
    
    #dealing with categorical data
    from sklearn.preprocessing import LabelEncoder
    label_encoder_a = LabelEncoder()
    dataset.iloc[: , 0]  = label_encoder_a.fit_transform( dataset.iloc[: , 0])
    dataset.iloc[: , 1]  = label_encoder_a.fit_transform( dataset.iloc[: , 1])
    
    
    #dealing with the misssing data
    dataset.iloc[:,2].fillna(dataset.iloc[: , 2].mean() , inplace = True)
    
    from sklearn.preprocessing import OneHotEncoder
    onehotencoder = OneHotEncoder(categorical_features=[0,1])
    dataset = onehotencoder.fit_transform(dataset).toarray()  
    
    
    return dataset


from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()

training_set = pd.read_csv('train.csv')
y = training_set.iloc[:,1].values 
training_set = training_set.drop('Survived' , 1)
training_set = preprocess_test_set(training_set)
training_set = standard_scaler.fit_transform(training_set)


# Fitting K-NN to the Training set
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state = 0)
#classifier.fit(training_set, y)


#preprocessing the test set
test_set = pd.read_csv('test.csv')
passenger = pd.DataFrame(test_set.iloc[: , 0])
test_set = test_set.drop('Name' , 1)
test_set = test_set.drop('Cabin' , 1)
test_set = test_set.drop('Ticket' , 1)
test_set = test_set.drop('PassengerId' , 1)
test_set = test_set.drop('Embarked' , 1)
test_set = test_set.drop('Fare' , 1)




#dealing with categorical data
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
test_set.iloc[: , 0]  = label_encoder.fit_transform( test_set.iloc[: , 0])
test_set.iloc[: , 1]  = label_encoder.fit_transform( test_set.iloc[: , 1])




#dealing with the misssing data
test_set.iloc[:,2].fillna(test_set.iloc[: , 2].mean() , inplace = True)
test_set.iloc[:,0].fillna(test_set.iloc[: , 0].mode()  , inplace = True)
test_set.iloc[:,1].fillna(test_set.iloc[: , 1].mode() , inplace = True)
test_set.iloc[:,3].fillna(test_set.iloc[: , 3].mode() , inplace = True)
test_set.iloc[:,4].fillna(test_set.iloc[: , 4].mode() , inplace = True)



from sklearn.preprocessing import OneHotEncoder
onehotencoder_a = OneHotEncoder(categorical_features=[0,1])
test_set = onehotencoder_a.fit_transform(test_set).toarray()   

test_set = standard_scaler.transform(test_set) 




from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
training_set = pca.fit_transform(training_set)
test_set = pca.transform(test_set)
explained_variance = pca.explained_variance_ratio_


#****************************************************************************************************
import keras
from keras.models import Sequential      # this is used to iinitialise pur neural network
from keras.layers import Dense        # this is used to make the different layers ofour nueral network

#initialising the ANN
classifier = Sequential()

#adding the input layer and the first hidden layer
classifier.add(Dense( units = 6 , input_shape = (5,) , kernel_initializer= 'uniform' , activation='relu' ))

# adding the second hidden layer
classifier.add(Dense( units = 4 , kernel_initializer= 'uniform' , activation='relu' ))
classifier.add(Dense( units = 6 , kernel_initializer= 'uniform' , activation='relu' ))
classifier.add(Dense( units = 2 , kernel_initializer= 'uniform' , activation='relu' ))

#adding the output layer
classifier.add(Dense( units = 1 , kernel_initializer= 'uniform' , activation='sigmoid' ))   # if the output has more than two categories than use the 'softmax, instead of sigmoid

#compiling the ANN
classifier.compile(optimizer='adam' , loss='binary_crossentropy' ,metrics=['accuracy'] ,)

#fitting the ANN to the training set
classifier.fit(training_set , y , batch_size=5 , epochs=100)




#****************************************************************************************************

y_pred = classifier.predict(test_set)

for k in range (0,418):
    if y_pred[k] >= 0.55:
        y_pred[k] = 1
    else:
        y_pred[k] = 0
        
passenger['Survived'] = y_pred

passenger.to_csv('final.csv' , index = False)





