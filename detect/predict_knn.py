import os
import numpy as np
import json
import joblib
import cv2 as cv


clf = joblib.load("knn-model.bin") 


def predict(data):
  kdata = np.vstack([np.ravel(mask) for mask in data])
  return clf.predict(kdata)
