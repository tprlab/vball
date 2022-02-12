import cv2 as cv
import os
import json
import test_utils

clr = (255, 0, 0)

INPATH = "json"
OUTPATH = "data"

test_utils.checkFolder(OUTPATH)

def processJsonDir(path):
  for f in os.listdir(path):
    fpath = os.path.join(path, f)
    if not os.path.isfile(fpath):
      continue
    outpic = os.path.join(OUTPATH, "{0}.jpg".format(f))
    
    with open(fpath) as jf:
      a = json.load(jf)
      rf = test_utils.filterRects(a)
      pic = test_utils.get_mask(rf)
      cv.imwrite(outpic, pic)


processJsonDir(INPATH)