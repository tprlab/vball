from keras.models import Sequential, Model
from keras.layers import Conv2D, Convolution2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.optimizers import Adam



import numpy as np
from keras.preprocessing.image import ImageDataGenerator

size = 32

def createModel(input_shape, cls_n ):
    model = Sequential()

    activation = "relu"

    model = Sequential([
        Convolution2D(32,(3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(),
        Convolution2D(64,(3,3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(2, activation='softmax')
      ])

    opt = SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model



input_shape = (size, size, 3)

EPOCHS = 50
cls_n = 2

model = createModel(input_shape, cls_n)

train_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("train", color_mode="rgb", target_size = (size, size), batch_size = 32, class_mode = 'categorical')
model.fit_generator(training_set, steps_per_epoch = 20, epochs = EPOCHS, validation_steps = 10)

model_json = model.to_json()
with open("./model.json","w") as json_file:
  json_file.write(model_json)

model.save_weights("./model.h5")