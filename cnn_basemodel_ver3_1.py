# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 17:20:29 2017

@author: Rithu
"""

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
import os
from keras import backend
 
# Image dimensions
img_width, img_height = 150, 150 
 
"""
    Creates a CNN model
    p: Dropout rate
    input_shape: Shape of input 
"""
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

def create_model(p, input_shape=(32, 32, 3)):
    # Initialising the CNN
    model = Sequential()
    # Convolution + Pooling Layer 
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flattening
    model.add(Flatten())
    # Fully connection
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p/2))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compiling the CNN

    optimizer = RMSprop(lr=0.001,rho=0.9, epsilon=1e-08, decay=0.0)
    metrics=['accuracy']
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
    return model
"""
    Fitting the CNN to the images.
"""
def run_training(bs=32, epochs=10):
    history = LossHistory()
    
    train_datagen = ImageDataGenerator(rescale = 1./255, 
                                       shear_range = 0.2, 
                                       zoom_range = 0.2, 
                                       horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
 
    training_set = train_datagen.flow_from_directory(training_set_path,
                                                 target_size = (img_width, img_height),
                                                 batch_size = bs,
                                                 class_mode = 'binary')
                                                 
    test_set = test_datagen.flow_from_directory(test_set_path,
                                            target_size = (img_width, img_height),
                                            batch_size = bs,
                                            class_mode = 'binary')
                                            
    model = create_model(p=0.6, input_shape=(img_width, img_height, 3))                                  
    model.fit_generator(training_set,
                         steps_per_epoch=8000/bs,
                         epochs = epochs,
                         validation_data = test_set,
                         validation_steps = 2000/bs,
                         callbacks=[history])
    model_backup_path = os.path.join(script_dir, 'cat_or_dogs_model_42.h5')
    model.save(model_backup_path)
    print("Model saved to", model_backup_path)
 
    # Save loss history to file
    loss_history_path = os.path.join(script_dir, 'loss_history_42.log')
    myFile = open(loss_history_path, 'w+')
    myFile.write(history.losses)
    myFile.close()
 
    backend.clear_session()
    print("The model class indices are:", training_set.class_indices)
def main():
    run_training(bs=32, epochs=20)
 
""" Main """
if __name__ == "__main__":
    main()
