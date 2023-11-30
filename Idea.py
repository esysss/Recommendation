import Algorithms
import Funcs as F
import numpy as np
from Merges import *
import pandas as pd
from Recomms import *
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import precision_score, ndcg_score, f1_score, accuracy_score
from tqdm import tqdm

n_clusters = 5 #number of clusters

T_train, T_test = F.read_data()

g_num = np.sum(T_train, axis=-1)
q_num = np.sum(T_train, axis=-2)

g_denum = np.nonzero(T_train)[0]
q_denum = np.nonzero(T_train)[1]

g = g_num/np.bincount(g_denum)
q = q_num/np.bincount(q_denum)

