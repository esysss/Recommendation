# import Algorithms
import Funcs as F
import numpy as np
# from Merges import *
import pandas as pd
from Recomms import *
# from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import precision_score, ndcg_score, f1_score, accuracy_score
from tqdm import tqdm

n_clusters = 5 #number of clusters

T_train, T_test = F.read_data(have = True)
# T_train = F.T_bar(T_train)

# Pusers, Pitems = Algorithms.FUIS(T_train, n_clusters)
Pusers, Pitems = Algorithms.FCNMF(T_train, n_clusters)

#Decompose multiple clusters for recommender system methods
threshold = 1/n_clusters
partitions = {}
for c in range(n_clusters):
    users = np.where(Pusers[:,c] >= threshold)[0]
    items = np.where(Pitems[:,c] >= threshold)[0]

    partitions[c] = svd(users, items, T_train)
    # partitions[c] = svdplus(users, items, T_train)
    # partitions[c] = coclus(users, items, T_train)
    # partitions[c] = base(users, items, T_train)
    # partitions[c] = normal(users, items, T_train)
    # partitions[c] = knn(users, items, T_train)


#apply merge
user,item = np.nonzero(T_test)
T_pred = np.zeros(T_test.shape)
for i,j in zip(user, item):
    score = 0
    for k in partitions.keys():
        algo = partitions[k]
        # score += merge1(Pusers[i].copy(), Pitems[j].copy(), k) * algo.estimate(i,j)
        # score += merge2(Pusers[i], k) * algo.estimate(i,j)
        score += merge3(Pusers[i], Pitems[j], k) * algo.estimate(i,j)

    T_pred[i,j] = score

# T_test[np.nonzero(T_test)] = T_test[np.nonzero(T_test)]/max(T_test[np.nonzero(T_test)])
#
# T_pred[np.nonzero(T_pred)] = T_pred[np.nonzero(T_pred)]/max(T_pred[np.nonzero(T_pred)])

print('the mean absolute error : ', mae(T_test[np.nonzero(T_test)], T_pred[np.nonzero(T_test)]))