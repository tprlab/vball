import sys
import numpy as np
import cv2 as cv
import os

from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array 

size = 32
dim = 3

def handle_file(path):
  print ("Checking file", path)
  oimg = cv.imread(path)
  img = cv.resize(oimg, (size,size))
  img = np.reshape(img,[1,size,size,dim])
  return loaded_model.predict_classes(img), oimg



json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")

print("Loaded model from disk")


target = None
if len(sys.argv) > 2:
    target = int(sys.argv[2])

fails = 0
all = 0

root = sys.argv[1]

if os.path.isdir(root):
  for p in os.listdir(root):
    fp = root + "/" + p
    if os.path.isdir(fp):
      continue
    c, img = handle_file(fp)
    print (p, c[0], c)
    all += 1
    if target is not None:
      if c != target:
        fails += 1
        cv.imwrite("fails/" + p, img)

  if target is not None:    
    print ("Failures", fails, "of", all)
else:
    c,img = handle_file(root)
    print ("Prediction", c)




