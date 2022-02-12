import tensorflow as tf
import numpy as np
import time

size = 64

def createModel(input_shape, cls_n ):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(cls_n, activation='softmax')
      ])

    opt = tf.keras.optimizers.SGD(lr=0.01)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model



input_shape = (size, size, 1)

EPOCHS = 40
cls_n = 4

model = createModel(input_shape, cls_n)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("data", color_mode="grayscale", target_size = (size, size), batch_size = 32, class_mode = 'categorical')

t = time.time()
model.fit_generator(training_set, steps_per_epoch = 20, epochs = EPOCHS, validation_steps = 10)
t = time.time() - t
print("Training completed in {0:.2f} seconds".format(t))

model_json = model.to_json()
with open("./model.json","w") as json_file:
  json_file.write(model_json)

model.save_weights("./model.h5")