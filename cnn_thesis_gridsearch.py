# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 13:17:36 2017

@author: Rithu
"""

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from PIL import Image
import os
from sklearn.model_selection import GridSearchCV

################ Data Preparation##################
nb_classes = 2

# input image dimensions
img_rows, img_cols = 64, 64

# load training data and do basic data normalization
path_train="dataset/training_set/"
path_test="dataset/test_set/"
newlist=[]
Xlist_train=[]
Ylist_train=[]

Xlist_test=[]
Ylist_test=[]

for directory in os.listdir(path_train):
    for file in os.listdir(path_train+directory):
        #print(path_train+directory+"/"+file)
        img=Image.open(path_train+directory+"/"+file)
           
        #img = img.resize((64,64), Image.NEAREST)
        featurevector=np.array(img)
        directoryvector=np.array(directory)
        featurevector=featurevector.flatten()[:4096].reshape(img_rows, img_cols, 1)
        directoryvector=directoryvector.flatten()[:4096]
        Xlist_train.append(featurevector)
        #print (len(Xlist_train))
        img.close()
        input_shape = (img_rows, img_cols, 1)
        Ylist_train.append(directoryvector)
        Ylist_train_binary=[]
        for i in Ylist_train:
            #print (i)
            if i == "cats":
                Ylist_train_binary.append(0)
                
            else:
                Ylist_train_binary.append(1)
                #print (newlist) 
                

for directory in os.listdir(path_test):
    for file in os.listdir(path_test+directory):
        img=Image.open(path_test+directory+"/"+file)
        featurevector=np.array(img)
        featurevector=featurevector.flatten()[:4096].reshape(img_rows, img_cols, 1)
        Xlist_test.append(featurevector)
        #print (len(Xlist_test))
        img.close()
        input_shape = (img_rows, img_cols, 1)
        Ylist_test.append(directory)
        Ylist_test_binary=[]
        for i in Ylist_test:
            #print (i)
            if i == "cats":
                Ylist_test_binary.append(0)
                #print (newlist)
            else:
                Ylist_test_binary.append(1)
                #print (newlist)
#
#Xlist_train = Xlist_train.astype('float32')
#for item in Xlist_train:
#    Xlist_train[item]=np.float(Xlist_train[item])
#    
#Xlist_train_new = [float(i) for i in Xlist_train][0]
#for index, item in enumerate(Xlist_train):
#    Xlist_train[index] = float(item)
Xlist_train_int = np.array(Xlist_train)
Xlist_train_float = np.array(Xlist_train_int) + 0
Xlist_train_float=Xlist_train_float / 255
#Xlist_train_float.shape
#print (Xlist_train_float)
Xlist_test_int = np.array(Xlist_test)
Xlist_test_float = np.array(Xlist_test_int) + 0
Xlist_test_float=Xlist_test_float / 255
#Xlist_test_float.shape

# convert class vectors to binary class matrices
Ylist_train_final = np_utils.to_categorical(Ylist_train_binary, nb_classes)
Ylist_test_final = np_utils.to_categorical(Ylist_test_binary, nb_classes)
#Ylist_train.shape
#Ylist_train_final.shape
#Ylist_test.shape
#Ylist_test_final.shape
def cnn_classifier(dense_layer_sizes, nb_filters, nb_conv, nb_pool):
    '''Creates model comprised of 2 convolutional layers followed by dense layers

    dense_layer_sizes: List of layer sizes. This list has one number for each layer
    nb_filters: Number of convolutional filters in each convolutional layer
    nb_conv: Convolutional kernel size
    nb_pool: Size of pooling area for max pooling
    '''
    model = Sequential()

    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
	                 padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    for layer_size in dense_layer_sizes:
        model.add(Dense(layer_size))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        model.compile(loss='sparse_categorical_crossentropy',
	                  optimizer='adadelta',
	                  metrics=['accuracy'])
        
    return model


dense_size_candidates = [[32], [64], [32, 32], [64, 64]]
cnn_model = KerasClassifier(cnn_classifier, batch_size=32)

###############  GridSearch HyperParameters ################

validator = GridSearchCV(cnn_model,
                         param_grid={'dense_layer_sizes': dense_size_candidates,
                                     # nb_epoch is avail for tuning even when not
                                     # an argument to model building function
                                     #'nb_epoch': [3,6,9],
                                     'epochs':[15],
                                     'nb_filters': [8],
                                     'nb_conv': [3],
                                     'nb_pool': [2]},
                         scoring='accuracy',
                         n_jobs=1)
validator.fit(Xlist_train_float, Ylist_train_binary)

print('The parameters of the best model are: ')
print(validator.best_params_)

# validator.best_estimator_ returns sklearn-wrapped version of best model.
# validator.best_estimator_.model returns the (unwrapped) keras model
best_model = validator.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(Xlist_test_float, Ylist_test_binary)
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)
