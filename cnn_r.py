# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 21:02:14 2017

@author: Rithu
"""

#PART I- buiding CNN - not data preprocessing step here, since it has been done manually
#importing Keras packages for a CNN-each package referes to every step in the construction of CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
#dense is used to add a fully connected ANN
#initializing the CNN- similar to creating a vanilla CNN
cnn_classifier=Sequential()
#adding the different layers
#STEP 1- adding the convolutional layer - feature detector/filter plus ReLU to give the feature map
#many feature detectors with a convolution operation on the org image produces many feature maps-->convolution layer
#in practice, we start with 32 feature detectors of size 3x3 to generate 32 feature maps. then on the subsequent conv.layers
# we use 64 FDs, then 128, then 256 (all of 3x3 size-can be of any size)and so on. This type of huge computation works well in the case of
#computations on a GPU.
#input shape-shape of input img-all imgs are of diffr format/size- must force/convert all imgs into the same size-will do this before we fit all imgs into CNN
#colored imags shape-3d array 3 channels;BW imags-2d array 1 channel

#NOTE !!!-if you are using tensorflow backend specify input shape as (64,64,3). If using Theano backend use input shape as (3,64,64)
#this repr the input images to be of format 64x64 in 3 channels(RGB)

cnn_classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation='relu'))

#STEP 2- POOLING - reduces size of feature maps-max pooling-STRIDE 2-sliding subtable size=2x2 mostly

cnn_classifier.add(MaxPooling2D(pool_size=(2,2)))

# STEP to better accuracy and reduce overfitting, remove the input shape param as the input to this is no longer the actual image but it is the pooled feature maps from the 2 prev steps
cnn_classifier.add(Convolution2D(32,(3,3),activation='relu'))
cnn_classifier.add(MaxPooling2D(pool_size=(2,2)))

#STEP 3 - FLATTENING - taking all feature maps and convert it to a huge single vector. This is the input to a classic fully connected ANN
#keras understands that the prev layer is what that needs to be flattened. so dont need any explicit mention
cnn_classifier.add(Flatten())

#STEP 4- Create a classic fully connected ANN
#units=output dimension==>#nodes in hidden layer.best practice based on experimentation=128 (around 100, a power of 2)
cnn_classifier.add(Dense(128,activation='relu'))

#adding output layer-using sigmoid since v have a binary outcome. multiclass outcome==>softmax
cnn_classifier.add(Dense(1,activation='sigmoid'))

#Compiling CNN- SGD (called as adam),loss func-binary cross entropy(logarithmic),perf metric

cnn_classifier.compile('adam','binary_crossentropy',metrics=['accuracy'])

#PART-II Fitting the CNN to the images
#doing image augmentation - a keras process that preproceses the images to prevent overfitting
#without this process we will get great accuracy on training set and poor accuracy on test set(overfitting)
#type keras documentation on google.open the first link https://keras.io. search for code on image preprocessing on left menu.
#to prevent overfitting, we need lots of data.8000+2000 is not that great a number to train the cnn on. So we use a trick.
#the trick is data augmentation-to create batches of the images. on each batch it will apply random tranformations on random images-
#like rotating, flipping,shifting,shearing,skewing etc. so that now we have a vast variety of images to train on-augmented images.
#without creating new images, we augment the available images for a wider variety
#use the .flow_from_directory(directory) method since we have a directory to load the datasets from
#copy pasting the code snipper from the site.and make necessary changes
from keras.preprocessing.image import ImageDataGenerator

#rescale converts all pixel values between 0 and 1 using 1./255
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#creating training set:targetsize->size of img u expect in a cnn model

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')
#create test set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

#stepsperepoch-#of imgs in trainin set, validation_steps=#of imgs in test set                                         
cnn_classifier.fit_generator(training_set,
                            steps_per_epoch=8000,
                            epochs=25,
                            validation_data=test_set,
                            validation_steps=2000)

# accuracy will be improved : 1.when you increase the target_size. bigger it is the better because then the CNN will now have more pixels to relate to
#2.Adding more layers of convolution+pooling layers
#3.By adding more fully connected layers (ANN)
#4.by adding more feature detector in every subsq. conv.layer.


##################################code for single prediction################
# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
#creating a 3d array(RGB for color imgs) for the img loaded above as 64x64
test_image = image.img_to_array(test_image)
#adding a new dimension corresponding to a batch. this is wat the predict method expects. neural networks generally expect a batch and do not take a single value by itself. 
#even if there is a single data to be tested, it must be in a batch. axis=0==> the dimension being added has the zeroth index (THIS IS WHAT THE PREDICT METHOD EXPECTS)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn_classifier.predict(test_image)

#class_indices specify the 0 and 1 mapping to the labels.(ie) cats-0 & dogs-1
training_set.class_indices
#result is a 2d array where the value is stored in 1st row 1st col
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'