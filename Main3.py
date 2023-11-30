import Funcs as F
import numpy as np
from node2vec import Node2Vec as n2v
import pandas as pd
from scipy import spatial
from sklearn.metrics import accuracy_score

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

# create embeddings dataframe
df = (
    pd.DataFrame(
        [mdl.wv.get_vector(str(n)) for n in graph.nodes()],
        index = graph.nodes
    )
)

user,item = np.nonzero(T_test)
pred = []
real = []
for i,j in zip(user, item):
    pred.append(spatial.distance.cosine(df.loc['user'+str(i)],df.loc['item'+str(j)]))
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