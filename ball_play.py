import numpy as np
import cv2 as cv
import os
import ball_net as bn
import blobber

def draw_ball(mask, frame):
  cnts, _ = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

  k = 0
  for c in cnts:
    rx,ry,rw,rh  = cv.boundingRect(c)
    mn = min(rw, rh)
    mx = max(rw, rh)
    r = mx / mn
    if mn < 10 or mx > 40 or r > 1.5:
      continue

    cut_m = mask[ry : ry + rh, rx : rx + rw]
    cut_f = frame[ry : ry + rh, rx : rx + rw]

    cut_c = cv.bitwise_and(cut_f,cut_f,mask = cut_m)
    if bn.check_pic(cut_c) == 0:
      ((x, y), r) = cv.minEnclosingCircle(c)
      cv.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 3)


def test_clip(path):
  vs = cv.VideoCapture(path)
  backSub = cv.createBackgroundSubtractorMOG2()
  n = 0
  while(True):
    ret, frame = vs.read()
    if not ret or frame is None:
      break

    h = frame.shape[0]
    w = frame.shape[1]

    frame = cv.resize(frame, (int(w/2),int(h/2)))
    mask = backSub.apply(frame)

    mask = cv.dilate(mask, None)
    mask = cv.GaussianBlur(mask, (15, 15),0)
    ret,mask = cv.threshold(mask,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)
    blobber.handle_blobs(mask, frame)
    
    blobber.draw_ball_path(frame)
    blobber.draw_ball(frame)
    cv.imwrite("frames/frame-{:03d}.jpg".format(n), frame)
    cv.imshow('frame', frame)
    if cv.waitKey(10) == 27:
      break
    n += 1


test_clip("D:/Videos/aus4.avi")
