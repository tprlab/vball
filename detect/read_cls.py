import os

CHEER = 1
NOPLAY = 2
PLAY = 3
STAND = 4

M = {}
M["cheer"] = CHEER
M["noplay"] = NOPLAY
M["play"] = PLAY
M["stand"] = STAND


def read_pic_classes(root):
  H = {}
  for subf in os.listdir(root):
    k = M.get(subf, 0)
    if k == 0:
      continue
    
    full_sub = os.path.join(root, subf)
    if not os.path.isdir(full_sub):
      continue

    for f in os.listdir(full_sub):
      n = int(f.split(".")[0])
      H[n] = k
  return H

def read_flat_vals(root):
  h = read_pic_classes(root)
  ret = []

  for i in range(len(h)):
    ret.append(h[i + 1])

  return ret

