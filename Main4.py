import Funcs as F
import numpy as np
from node2vec import Node2Vec as n2v
import pandas as pd
from scipy import spatial
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

T_train, T_test = F.read_data(have = True)
graph = F.bipartite(T_train)
g_emb = n2v(graph, dimensions=16, workers=8)

WINDOW = 1 # Node2Vec fit window
MIN_COUNT = 1 # Node2Vec min. count
BATCH_WORDS = 4 # Node2Vec batch words

mdl = g_emb.fit(
    window=WINDOW,
    min_count=MIN_COUNT,
    batch_words=BATCH_WORDS
)

# create 1st embeddings dataframe
df = (
    pd.DataFrame(
        [mdl.wv.get_vector(str(n)) for n in graph.nodes()],
        index = graph.nodes
    )
)

#create 2nd embedding
T_train, T_test = F.read_data(have = True)
graph = F.bipartite(T_train)

emb = F.features(graph,T_test, have=True)
df1 = pd.DataFrame.from_dict(emb,orient = 'index')

#create 3rd embedding
users,items = F.nmf(T_train,8)
columni = ['item{}'.format(i) for i in range(len(items))]
columnu = ['user{}'.format(i) for i in range(len(users))]
df21 = pd.DataFrame(users, index = columnu)
df22 = pd.DataFrame(items,index=columni)
df2 = df21.append(df22)

#concatinate the three
final_dict = {}
for i in df1.index:
    final_dict[i] = list(df.loc[i]) + list(df1.loc[i]) + list(df2.loc[i])

final_df = pd.DataFrame.from_dict(final_dict,orient = 'index')

#apply PCA
pca = PCA(n_components=8)
pca.fit(final_df.T)
pcaOutput = pca.components_

concat = list(final_df.index)

user,item = np.nonzero(T_test)
pred = []
real = []
for i,j in zip(user, item):
    pred.append(spatial.distance.cosine(pcaOutput[:,concat.index('user'+str(i))],
                                        pcaOutput[:,concat.index('item'+str(j))]))
    if T_test[i,j] > 2:
        real.append(1)
    else:
        real.append(0)

pred = np.array(pred)
real = np.array(real)
p = pred.copy()

threshold = 1
p[pred <= threshold] = 1
p[pred > threshold] = 0

print(accuracy_score(real,p))



