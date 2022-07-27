# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense , Dropout
from keras import initializers

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sz = 128
# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()

# First convolution layer and pooling
my_init = initializers.VarianceScaling(scale=1.0, mode='fan_avg', 
distribution='uniform', seed=None)

classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu' ,kernel_initializer=my_init))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
my_init2 = initializers.VarianceScaling(scale=1.0, mode='fan_avg', 
distribution='uniform', seed=None)
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer=my_init2))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# third convolution layer and pooling
my_init3 = initializers.VarianceScaling(scale=1.0, mode='fan_avg', 
distribution='uniform', seed=None)
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer=my_init3))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
my_init4 = initializers.VarianceScaling(scale=1.0, mode='fan_avg', 
distribution='uniform', seed=None)
classifier.add(Dense(units=128, activation='relu',kernel_initializer=my_init4))
classifier.add(Dropout(0.40))
my_init5 = initializers.VarianceScaling(scale=1.0, mode='fan_avg', 
distribution='uniform', seed=None)
classifier.add(Dense(units=96, activation='relu' ,kernel_initializer=my_init5))
classifier.add(Dropout(0.40))
my_init6 = initializers.VarianceScaling(scale=1.0, mode='fan_avg', 
distribution='uniform', seed=None)
classifier.add(Dense(units=64, activation='relu',kernel_initializer=my_init6 ))
my_init7 = initializers.VarianceScaling(scale=1.0, mode='fan_avg', 
distribution='uniform', seed=None)
classifier.add(Dense(units=3, activation='softmax',kernel_initializer=my_init7)) # softmax for more than 2

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2


# Step 2 - Preparing the train/test data and training the model
classifier.summary()
# Code copied from - https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataSet/trainingData_dru/',
                                                 target_size=(sz, sz),
                                                 batch_size=10,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataSet/testingData_dru/',
                                            target_size=(sz , sz),
                                            batch_size=10,
                                            color_mode='grayscale',
                                            class_mode='categorical') 
classifier.fit_generator(
        training_set,
        epochs=5,
        validation_data=test_set)# No of images in test set


# Saving the model
model_json = classifier.to_json()
with open("model_dru.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights('model_dru.h5')
print('Weights saved')