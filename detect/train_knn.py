import os
import json
import numpy as np
import cv2 as cv

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from joblib import dump, load


ROOT_PATH = "data"
CHEER_PATH = os.path.join(ROOT_PATH, "cheer")
PLAY_PATH = os.path.join(ROOT_PATH, "play")
STAND_PATH = os.path.join(ROOT_PATH, "stand")
NP_PATH = os.path.join(ROOT_PATH, "noplay")


def read_folder(fpath):
  ret = []
  for f in os.listdir(fpath):
    fullpath = os.path.join(fpath, f)
    pic = cv.imread(fullpath, cv.IMREAD_GRAYSCALE)
    ret.append(np.ravel(pic))
  return ret



def read_data():
  cheer = read_folder(CHEER_PATH)
  play = read_folder(PLAY_PATH)
  stand = read_folder(STAND_PATH)
  noplay = read_folder(NP_PATH)
  return cheer, play, stand, noplay

def prepare_data():
  cheer, play, stand, noplay = read_data()
  data = []
  target = []

  data.extend(cheer)
  target.extend([1] * len(cheer))

  data.extend(noplay)
  target.extend([2] * len(noplay))

  data.extend(play)
  target.extend([3] * len(play))

  data.extend(stand)
  target.extend([4] * len(stand))

  return np.asarray(data), np.asarray(target)


def train():
  data, target = prepare_data()
  data, v_data, target, v_target = train_test_split(data, target, test_size=0.25, random_state=12)
  model = KNeighborsClassifier(n_neighbors=3)
  model.fit(data, target)
  dump(model,"knn-model.bin")
  tpred = model.predict(data)
  pred = model.predict(v_data)
  print(accuracy_score(target, tpred), accuracy_score(v_target, pred))
  

train()
  