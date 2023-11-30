#4.2 estefade az ravesh haye gheyre graph


import Funcs as F
from Recomms import *
import numpy as np

T_train, T_test = F.read_data(have = True)
users,items = np.nonzero(T_train)
algo = svd(np.unique(users), np.unique(items), T_train)
# algo = coclus(np.unique(users), np.unique(items), T_train)

user,item = np.nonzero(T_test)
error = 0
for i,j in zip(user, item):
    est = algo.estimate(i,j)
    error+= abs(est - T_test[i,j])
print(error/len(user))