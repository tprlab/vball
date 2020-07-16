import sys
import cv2 as cv

def get_high(mask):
  cnts, _ = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

  chigh = None
  cy = 10000
  for c in cnts:
    x,y,w,h  = cv.boundingRect(c)
    s = min(w, h)
    if s < 15:
      continue

    if y < cy:
      r = max(w, h) / s
      if r > 1.5:
        continue

      cy = y
      chigh = c
  return chigh

def draw_high_cont(path):
  mask = cv.imread(path, cv.IMREAD_GRAYSCALE)
  chigh = get_high(mask)
  
  if chigh is not None:
    cmask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    cv.drawContours(cmask ,[chigh], 0,(255,0,0), 2)
    cv.imwrite("out.jpg", cmask)
    cv.imshow('frame', cmask)
    cv.waitKey()

def get_high_blobs(clip_path, out_path = None, clr_out_path = None):
  vs = cv.VideoCapture(clip_path)
  backSub = cv.createBackgroundSubtractorMOG2()

  n = 0
  while(True):
      ret, frame = vs.read()
      if not ret or frame is None:
        break

      mask = backSub.apply(frame)
      mask = cv.GaussianBlur(mask, (7, 7),0)
      ret,mask = cv.threshold(mask,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)

      cmask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

      chigh = get_high(mask)
      if chigh is not None:
        rx,ry,rw,rh  = cv.boundingRect(chigh)
        cut = mask[ry : ry + rh, rx : rx + rw]
        if not out_path is None:
          cv.imwrite("{0}/b-{1:03d}.jpg".format(out_path, n), cut)
        if not clr_out_path is None:
          cut_f = frame[ry : ry + rh, rx : rx + rw]
          cut_c = cv.bitwise_and(cut_f,cut_f,mask = cut)
          cv.imwrite("{0}/c-{1:03d}.jpg".format(clr_out_path, n), cut_c)

      print(n)
      n += 1




if __name__ == "__main__":
  draw_high_cont(sys.argv[1])
  #get_high_blobs(sys.argv[1], sys.argv[2], sys.argv[2])
  #get_high_blobs("D:/Videos/aus4.avi", "out", "clr")


