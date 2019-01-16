# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 13:33:14 2017

@author: Rithu
"""
######## PART I- buiding CNN #################


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from imutils import paths
import opencv
from PIL import Image
import glob

def build_classifier(optimizer):
# Initialising the CNN
    classifier = Sequential()
    
    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
    
    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Adding a second convolutional layer and pooling
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    
    # Step 3 - Flattening
    classifier.add(Flatten())
    
    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    
    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier


######## PART II- Fitting the CNN to the images #################

from keras.preprocessing.image import ImageDataGenerator

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary',
                                                 shuffle='false')

image_list = []
for filename in glob.glob('dataset/training_set/*/*.jpg'):
    im=Image.open(filename)
    image_list.append(im)
    length=len(image_list)
    print (length)
    x= image_list.reshape((length, -1))
    Image.close(filename)  
print(training_set.filenames)
print(training_set.samples)



label = []
for i in range(0, 8000):
    if i < 4000:
        label.append("cat")
    else:
        label.append("dog")

print(label)

labelf = []
for i in range(0, 20):
    labelf.append("cat")

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(image_list,label)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
