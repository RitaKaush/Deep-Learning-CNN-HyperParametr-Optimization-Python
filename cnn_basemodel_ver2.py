# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 19:12:19 2017

@author: Rithu
"""

#from tensorflow.contrib.keras.api.keras.layers import Dropout
#from tensorflow.contrib.keras.api.keras.models import Sequential
#from tensorflow.contrib.keras.api.keras.layers import Conv2D
#from tensorflow.contrib.keras.api.keras.layers import MaxPooling2D
#from tensorflow.contrib.keras.api.keras.layers import Flatten
#from tensorflow.contrib.keras.api.keras.layers import Dense
#from tensorflow.contrib.keras.api.keras.callbacks import Callback
#from tensorflow.contrib.keras.api.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.contrib.keras import backend
#import os
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
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras import backend 
 
class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_id = 0
        self.losses = ''
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses += "Epoch {}: accuracy -> {:.4f}, val_accuracy -> {:.4f}\n"\
            .format(str(self.epoch_id), logs.get('acc'), logs.get('val_acc'))
        self.epoch_id += 1
 
    def on_train_begin(self, logs={}):
        self.losses += 'Training begins...\n'
 
script_dir = "script_dir"
training_set_path = "dataset/training_set/"
test_set_path = "dataset/test_set/"
 
# Initialising the CNN
classifier = Sequential()
 
# Step 1 - Convolution

classifier.add(Conv2D(32, (3, 3), input_shape=(64,64, 3), activation='relu'))
 
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))  # 2x2 is optimal
 
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
 
# Step 3 - Flattening
classifier.add(Flatten())
 
# Step 4 - Full connection
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=1, activation='sigmoid'))
 
# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
 
# Part 2 - Fitting the CNN to the images
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
 
test_datagen = ImageDataGenerator(rescale=1. / 255)
 
training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size=(64,64),
                                                 batch_size=batch_size,
                                                 class_mode='binary')
 
test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size=(64,64),
                                            batch_size=batch_size,
                                            class_mode='binary')
 
# Create a loss history
history = LossHistory()
 
classifier.fit_generator(training_set,
                         steps_per_epoch=8000/batch_size,
                         epochs=30,
                         validation_data=test_set,
                         validation_steps=2000/batch_size,
                         workers=12,
                         max_q_size=100,
                        callbacks=[history] )
 
 
# Save model
model_backup_path = os.path.join(script_dir, 'cat_or_dogs_model_5.h5')
classifier.save(model_backup_path)
print("Model saved to", model_backup_path)
 
# Save loss history to file
loss_history_path = os.path.join(script_dir, 'loss_history_5.log')
myFile = open(loss_history_path, 'w+')
myFile.write(history.losses)
myFile.close()
 
backend.clear_session()
print("The model class indices are:", training_set.class_indices)
