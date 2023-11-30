import Funcs as f
import numpy as np
from scipy.sparse import csgraph
from sklearn.decomposition import NMF

def FCNMF(T_train, components = 5, information = False):
    model = NMF(n_components=components,tol=10e-6,max_iter = 300)
    V = model.fit_transform(T_train)
    H = model.components_

    # construct the matrix P
    P = np.concatenate((V, H.T), axis=0)
    print(P.shape)

    # normalize the rows of P to make U
    sum_of_rows = P.sum(axis=1)
    U = P / sum_of_rows[:, np.newaxis]
    U[np.isnan(U)] = 0

    Pusers = U[:T_train.shape[0], :]
    Pitems = U[T_train.shape[0]:, :]

    if information:
        print("The NMF is running on {} clusters".format(components))
        print("The NMF is done!\nV.shape = {} H.shape = {}".format(V.shape, H.shape))

    return Pusers, Pitems


def FUIS(T_train, n_clusters):
    #create M_UI
    D_row = np.sum(T_train, axis=-1)
    D_row = np.diag(D_row.astype(float) ** -1 / 2)
    D_row[D_row == np.inf] = 0

    D_col = np.sum(T_train, axis=-2)
    D_col = np.diag(D_col.astype(float) ** -1/2)
    D_col[D_col == np.inf] = 0

    S = D_row.dot(T_train).dot(D_col)

    M_UI = np.concatenate((np.eye(T_train.shape[0]), -S), axis=-1)
    temp = np.concatenate((-S.T, np.eye(T_train.shape[1])), axis=-1)
    M_UI = np.concatenate((M_UI, temp), axis=-2)

    #create M_UU & M_II
    W_user, W_item = f.dist(T_train,'cosine')

    L_Q = csgraph.laplacian(W_user, normed=False)
    L_R = csgraph.laplacian(W_item, normed=False)

    M_UU = M_II = np.zeros(M_UI.shape)

    M_UU[:L_Q.shape[0],:L_Q.shape[1]] = L_Q # [[L_Q,0],[0,0]]
    M_II[-L_R.shape[0]:,-L_R.shape[1]:] = L_R # [[0,0],[0,L_R]]

    M = M_UI + M_UU + M_II

    # find the r smallest eigenvectors of M
    w,v = np.linalg.eig(M)
    address = np.argsort(w)
    r = 3 #r smallest eigenvectors
    X_star = v[address[:r]]

    #create matrix P using fuzzy c-means
    U = f.cmeans(X_star.T, n_clusters)

    Pusers = U[:T_train.shape[0], :]
    Pitems = U[T_train.shape[0]:, :]

    return Pusers, Pitems

# Recommender Systems

# def pop(X_train, top):
#     #get the most popular items and recommend it to everyone
#
#     X_train[X_train > 0] = 1
#     p_score = np.sum(X_train,axis=-2)
#     items = np.argsort(p_score)[-top:]
#     p_score = p_score[items]
#     p_score -= min(p_score)
#     p_score = (p_score/max(p_score))*4+1 # scale between 1-5
#     # p_score = (p_score / max(p_score)) # scale between 0-1
#     return dict((i,s) for i,s in zip(items,p_score))



