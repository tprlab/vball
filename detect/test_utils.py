import os
import numpy as np
import json
import cv2 as cv


W = 1280
H = 720

def checkRect(r):
  left = r[0]
  top = r[1]
  right = left + r[2]
  bottom = top + r[3]
  if r[2] > 300 or r[3] > 300:
    return False
  if r[2] < 30 or r[3] < 30:
    return False
  return True

def checkRectCourt(r):
  left = r[0]
  top = r[1]
  right = left + r[2]
  bottom = top + r[3]
  if right < 150 or left > 930:
    return False
  return True



def filterRects(rects):
  good = [r for r in rects if checkRect(r) and checkRectCourt(r)]
  return np.asarray(good)


def get_mask(boxes):
  pic = np.zeros((H,W,1), np.uint8)
  clr = 255

  for r in boxes: 
    cv.rectangle(pic, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), clr, thickness=-1)

  pic = cv.resize(pic, (64, 64))
  return pic

def read_data(path):
  i = 1
  data = []
  while True:
    fname = "{0:05d}.jpg.json".format(i)
    fpath = os.path.join(path, fname)
    if not os.path.isfile(fpath):
      break

    with open(fpath) as f:
      raw = json.load(f)
      good = filterRects(raw)
      mask = get_mask(good)
      data.append(mask)

    i += 1
  return np.asarray(data)

MIN_RALLY_LEN = 3

def get_lead3(data, n):
  if data[n] == data[n + 1]:
    return data[n];
  if data[n] == data[n + 2]:
    return data[n];

  if data[n + 1] == data[n + 2]:
    return data[n + 1];
  return -1



def collect_rallies(data):
  ret = []
  start = -1
  for i in range(2, len(data)):
    idx = i - 2
    v = get_lead3(data, idx)
    if v == -1:
      continue
    if v == 3:
      if start == -1:
        start = idx
    else:
      if start != -1:
        if idx >= start + MIN_RALLY_LEN:
          ret.append((start, idx))
        start = -1

  if start != -1:
    ret.append((start, len(data) - 1))
  return ret

def cmp_rallies(src, comp):
  si = 0
  ci = 0
  E = 5
  eq = 0
  while si < len(src) and ci < len(comp):
    d = abs(src[si][0] - comp[ci][0]) + abs(src[si][1] - comp[ci][1])
    if d <= E:
      eq += 1
      ci += 1
      si += 1
      continue
    if src[si][0] < comp[ci][0]:
      si += 1
    elif src[si][0] > comp[ci][0]:
      ci += 1
    else:
      if src[si][1] < comp[ci][1]:
        si += 1
      else:
        ci += 1
  return eq
