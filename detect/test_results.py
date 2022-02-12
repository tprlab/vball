import predict_knn
import predict_keras
import read_cls
import test_utils


DATA_PATH = "data"

def cmpF(a, b):
  return 1 if a == b else 0

data = test_utils.read_data("json")

V = read_cls.read_flat_vals(DATA_PATH)

predn = predict_knn.predict(data)

predt = predict_keras.predict(data)

L = len(data)
T = 0
N = 0


for i in range(len(data)):
  if predn[i] == V[i]:
    N +=1
  if predt[i] == V[i]:
    T +=1

print("KNN", round(N/L, 2), "TF", round(T/L, 2))

r0 = test_utils.collect_rallies(V)
rn = test_utils.collect_rallies(predn)
rt = test_utils.collect_rallies(predt)
print("--------Rallies -----------")
print(r0)
print("--------KNN Rallies--------")
print(rn)
print("--------TF Rallies---------")
print(rt)

print("KNN rallies success:", test_utils.cmp_rallies(r0, rn), "of", len(r0))
print("TF rallies success:", test_utils.cmp_rallies(r0, rt), "of", len(r0))




  