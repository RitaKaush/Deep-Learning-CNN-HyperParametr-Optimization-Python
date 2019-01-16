# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 13:17:36 2017

@author: Rithu
"""

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
from PIL import Image
import os
import re
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
        #print (len(featurevector))
        directoryvector=np.array(directory)
        featurevector=featurevector.flatten()[:4096].reshape(img_rows, img_cols, 1)
        directoryvector=directoryvector.flatten()[:4096]
        Xlist_train.append(featurevector)
        #print (len(featurevector))
        #print (len(Xlist_train))
        img.close()
        input_shape = (img_rows, img_cols, 1)
        #print(directoryvector)
        Ylist_train.append(directoryvector)
        Ylist_train_binary=[]
        for i in Ylist_train:
            #print (i)
            if i == "cats":
                Ylist_train_binary.append(0)
                
            else:
                Ylist_train_binary.append(1)
                #print (newlist) 

#print (len(Ylist_train))
#print (len(Ylist_train_binary))        


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
Xlist_train_float.shape
#print (Xlist_train_float)
Xlist_test_int = np.array(Xlist_test)
Xlist_test_float = np.array(Xlist_test_int) + 0
Xlist_test_float=Xlist_test_float / 255
Xlist_test_float.shape

# convert class vectors to binary class matrices
Ylist_train_final = np_utils.to_categorical(Ylist_train_binary, nb_classes)

Ylist_test_final = np_utils.to_categorical(Ylist_test_binary, nb_classes)
print (Ylist_test_final)
#Ylist_train.shape
#Ylist_train_final.shape
#Ylist_test.shape
#Ylist_test_final.shape
def cnn_classifier(dense_layer_sizes, nb_filters, nb_conv, 
                   nb_pool,optimizer,learn_rate,momentum,init_mode,
                   activation,dropout_rate,weight_constraint):
    
    model = Sequential()

    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
	                 padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(dense_layer_sizes))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
	                  optimizer=optimizer,
	                  metrics=['accuracy'])    
    
    return model


#dense_size_candidates = [[32], [64], [32, 32], [64, 64]]
# define the grid search parameters

#param_grid = [
#  {'batch_size': [10, 20, 40, 60], 'dense_layer_sizes': ['64']}
# ]

#batchsize=[[10],[20],[30]]
#dense_layer_sizes =[[64]]
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
activation = ['relu']
init_mode = ['uniform']
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
optimizer = ['SGD']
batch_size = [10]
epochs = [6]
dense_layer_sizes =[128]
nb_filters= [8]
nb_conv= [3]
nb_pool= [2]
param_grid = dict(batch_size=batch_size, epochs=epochs,
                  dense_layer_sizes=dense_layer_sizes,
                  nb_filters=nb_filters,nb_conv=nb_conv,nb_pool=nb_pool,
                  optimizer=optimizer,learn_rate=learn_rate,
                  momentum=momentum,
                  init_mode=init_mode,
                  activation=activation,
                  dropout_rate=dropout_rate, 
                  weight_constraint=weight_constraint)

cnn_model = KerasClassifier(cnn_classifier)

###############  GridSearch HyperParameters ################

validator = GridSearchCV(cnn_model,
                         param_grid=param_grid,
                         scoring='accuracy',
                         n_jobs=1)

validator.fit(Xlist_train_float, Ylist_train_binary, validation_data=None, shuffle=False, verbose=1)


print('The parameters of the best model are: ')
print(validator.best_params_)

# validator.best_estimator_ returns sklearn-wrapped version of best model.
# validator.best_estimator_.model returns the (unwrapped) keras model
best_model = validator.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(Xlist_test_float, Ylist_test_binary)
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)