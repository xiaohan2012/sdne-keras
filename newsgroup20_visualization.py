
# coding: utf-8
import matplotlib as mpl
mpl.use('Agg')

import networkx as nx

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt

from core import SDNE


categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
dataset = fetch_20newsgroups(categories=categories)


vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(dataset.data)


# build the graph
N = vectors.shape[0]

mat = kneighbors_graph(vectors, 128, metric='cosine', mode='distance', include_self=True)

mat.data = 1 - mat.data  # to similarity

g = nx.from_scipy_sparse_matrix(mat, create_using=nx.Graph())

model = SDNE(g, encode_dim=100, encoding_layer_dims=[1720, 200],
             l2_param=0.1)

model.fit(epochs=10, batch_size=32)

embeddings = model.get_node_embedding()


labels = dataset.target

pos = TSNE(n_components=2).fit_transform(embeddings)


plt.scatter(pos[:, 0], pos[:, 1], c=labels)
plt.savefig('figs/20newsgroup_viz.png')

