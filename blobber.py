import math
import cv2 as cv
import numpy as np
import ball_net as bn


cnt = 0

R = 50
EPS = 1e-6
EPS2 = 0.5

STATUS_INIT = 0
STATUS_STATIC = 1
STATUS_DIRECTED = 2


def pt_dist(x1, y1, x2, y2):
  dx = x1 - x2
  dy = y1 - y2
  return math.sqrt(dx * dx + dy * dy)

class Blob:
  def __init__(self, x, y, r, a):
    self.pts = [[x, y]]
    self.pp = [[r, a]]
    self.status = STATUS_INIT
    self.v = None
    self.age = a

  def fit(self, x, y, r):
    d = pt_dist(self.pts[-1][0], self.pts[-1][1], x, y)
    return d < R, d

  def add(self, x, y, r, a):
    self.pts.append([x, y])
    self.pp.append([r, a])
    self.age = a
    if len(self.pts) > 2:
      dx1 = self.pts[-2][0] - self.pts[-3][0]
      dy1 = self.pts[-2][1] - self.pts[-3][1]

      dx2 = x - self.pts[-2][0]
      dy2 = y - self.pts[-2][1]
      d = pt_dist(self.pts[-3][0], self.pts[-3][1], x, y)
      if dx1 * dx2 > 0 and dy1 * dy2 > 0 and d > 10:
        self.status = STATUS_DIRECTED
      elif self.status != STATUS_DIRECTED:
        self.status = STATUS_STATIC
        
B = []
bb = None

def get_ball_blob():
  return bb

def find_fblob(x, y, r):
  global B, cnt
  rbp = []
  sbp = []
  
  for b in B:
    ft, d = b.fit(x, y, r)
    if ft:
      if cnt - b.age < 4:
        rbp.append([b,d])
      elif b.status == STATUS_STATIC:
        sbp.append([b,d])

  if len(sbp) + len(rbp) == 0:
    return None
  rbp.sort(key = lambda e: e[1])
  if len(rbp) > 0:
    return rbp[0][0]

  sbp.sort(key = lambda e: e[1])
  return sbp[0][0]

def handle_blob(x, y, r):
  global B, cnt, bb
  b = find_fblob(x, y, r)
  if b is None:
    B.append(Blob(x, y, r, cnt))
    return
  b.add(x, y, r, cnt)
  if b.status == STATUS_DIRECTED:
    if bb is None:
      bb = b
    elif len(b.pts) > len(bb.pts):
      bb = b


def begin_gen():
  global bb
  bb = None

def end_gen():
  global cnt
  cnt += 1    


def handle_blobs(mask, frame):
  cnts, _ = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

  k = 0
  begin_gen()
  for c in cnts:
    rx,ry,rw,rh  = cv.boundingRect(c)
    mn = min(rw, rh)
    mx = max(rw, rh)
    r = mx / mn
    if mn < 10 or mx > 40 or r > 1.5:
      continue

    cut_m = mask[ry : ry + rh, rx : rx + rw]

    blob, nz = check_blob(cut_m, 0, 0, rw, rh)
    if not blob:
      continue
    pnz = nz / (rw * rh)
    if pnz < 0.5:
      continue

    cut_f = frame[ry : ry + rh, rx : rx + rw]
    cut_c = cv.bitwise_and(cut_f,cut_f,mask = cut_m)
    if bn.check_pic(cut_c) != 0:
      continue
    ((x, y), r) = cv.minEnclosingCircle(c)
    handle_blob(int(x), int(y), int(r))
    k += 1


  end_gen()

def check_blob(pic, x, y, w, h):
  dy = int(h / 5)
  y0 = y + 2 * dy
  cut_h = pic[y0 : y0 + dy, x : x + w]

  dx = int(w / 5)
  x0 = x + 2 * dx
  cut_v = pic[y : y + h, x0 : x0 + dx]

  hnz = cv.countNonZero(cut_h)
  vnz = cv.countNonZero(cut_v)
  nz = cv.countNonZero(pic)
  mn = min(hnz, vnz)
  r = max(hnz, vnz) / mn if mn > 0 else 1000
  return r < 1.5 and hnz / nz > 0.2 and vnz / nz > 0.2, nz



def draw_blobs(w, h):
  pic = np.zeros((h, w, 3), np.uint8)
  for b in B:
    clr = (200, 200, 200)
    if b.status == STATUS_STATIC:
      clr = (0, 200, 0)
    elif b.status == STATUS_DIRECTED:
      clr = (200, 0, 0)
      if not b.v is None:
        cv.line(pic,(b.pts[0][0], b.pts[0][1]),(b.pts[-1][0], b.pts[-1][1]),(255, 0, 0), 1)  
    for p in b.pts:
      cv.circle(pic, (p[0], p[1]), 3, clr, -1)

  bb = get_ball_blob()
  if not bb is None:
    cv.circle(pic, (bb.pts[-1][0], bb.pts[-1][1]), 10, (200, 0, 0), 3)

  return pic



def test_clip(path):
  vs = cv.VideoCapture(path)
  backSub = cv.createBackgroundSubtractorMOG2()
  n = 0
  while(True):
    ret, frame = vs.read()
    if not ret or frame is None:
      break

    h = int(frame.shape[0] / 2)
    w = int(frame.shape[1] / 2)

    frame = cv.resize(frame, (w, h))
    mask = backSub.apply(frame)

    mask = cv.dilate(mask, None)
    mask = cv.GaussianBlur(mask, (15, 15),0)
    ret,mask = cv.threshold(mask,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)

    handle_blobs(mask, frame)
    pic = draw_blobs(w, h)
    cv.imshow('frame', pic)
    #cv.imwrite("frames/frame-{:03d}.jpg".format(n), pic)    
    if cv.waitKey(10) == 27:
      break
    n += 1


if __name__ == "__main__":
  test_clip(sys.argv[1])