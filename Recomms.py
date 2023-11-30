import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import SVDpp
from surprise import KNNBasic
from surprise import SlopeOne, KNNBaseline
from surprise import CoClustering, BaselineOnly, NormalPredictor


def make_data(users, items, T):
    lst = []
    for user in users:
        address = np.nonzero(T[user,items])[0]
        lst += [(user,i,T[user,i]) for i in address if T[user,i] != 0]

    df = pd.DataFrame(lst, columns=['userID', 'itemID', 'rating'])
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

    return data.build_full_trainset()


def svd(users,items,T):
    data = make_data(users, items, T)
    algo = SVD()
    algo.fit(data)

    return algo

def svdplus(users, items, T):
    data = make_data(users, items, T)
    algo = SVDpp()
    algo.fit(data)

    return algo

def knn(users, items, T):
    data = make_data(users, items, T)
    algo = KNNBasic()
    algo.fit(data)

    return algo

def coclus(users, items, T):
    data = make_data(users, items, T)
    algo = CoClustering()
    algo.fit(data)

    return algo

def base(users, items, T):
    data = make_data(users, items, T)
    algo = BaselineOnly()
    algo.fit(data)

    return algo

def normal(users, items, T):
    data = make_data(users, items, T)
    algo = NormalPredictor()
    algo.fit(data)

    return algo

def save_plot(T_train, T_test, mark, name):
    T = T_train.copy()
    T[np.nonzero(T_test)] = T_test[np.nonzero(T_test)]
    plt.spy(T, markersize= mark)
    plt.title("Matrix sparsity")
    plt.xlabel("Users")
    plt.ylabel("Movies")
    plt.savefig(name+".png")
    print("the plot has been saved !!!")