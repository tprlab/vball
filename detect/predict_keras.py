import tensorflow as tf
import numpy as np
import time



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")


def predict(data):
  kdata = np.vstack([np.reshape(d,[1,64,64,1]) for d in data])
  px = loaded_model.predict(kdata) 
  ret = [np.argmax(x) + 1 for x in px]
  return ret


