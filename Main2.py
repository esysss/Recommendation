#4.3 estefade az ravesh haye classic graph
from scipy import spatial
import Funcs as F
# from Recomms import *
import numpy as np
from sklearn.metrics import accuracy_score

T_train, T_test = F.read_data(have = True)
graph = F.bipartite(T_train)

emb = F.features(graph,T_test, have=True)

user,item = np.nonzero(T_test)

pred = []
real = []

for i,j in zip(user, item):
    pred.append(spatial.distance.cosine(emb['user'+str(i)],emb['item'+str(j)]))
    if T_test[i,j] > 2:
        real.append(1)
    else:
        real.append(0)

pred = np.array(pred)
real = np.array(real)
p = pred.copy()
p[pred<=pred.mean()] = 1
p[pred>pred.mean()] = 0

print(accuracy_score(real,p))