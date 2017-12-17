# coding: utf-8

import matplotlib as mpl
mpl.use('Agg')

import math
import networkx as nx
import pickle as pkl

from itertools import product
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import TSNE
from tqdm import tqdm

from matplotlib import pyplot as plt
from core import SDNE

batch_size = 32

categories = ['comp.graphics', 'rec.sport.baseball', 'talk.politics.guns']
dataset = fetch_20newsgroups(categories=categories)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(dataset.data)

# build the graph
N = vectors.shape[0]

mat = kneighbors_graph(vectors, N, metric='cosine', mode='distance', include_self=True)

mat.data = 1 - mat.data  # to similarity

g = nx.from_scipy_sparse_matrix(mat, create_using=nx.Graph())


parameter_grid = {'alpha': [2],
                  'l2_param': [1e-3],
                  'pretrain_epochs': [0],
                  'epochs': [5]}


parameter_values = list(product(*parameter_grid.values()))
parameter_keys = list(parameter_grid.keys())

parameter_dicts = [dict(list(zip(parameter_keys, values))) for values in parameter_values]


def one_run(params):
    plt.clf()
    alpha = params['alpha']
    l2_param = params['l2_param']
    pretrain_epochs = params['pretrain_epochs']
    epochs = params['epochs']

    model = SDNE(g, encode_dim=100, encoding_layer_dims=[1720, 200],
                 beta=2,
                 alpha=alpha,
                 l2_param=l2_param)
    model.pretrain(epochs=pretrain_epochs, batch_size=32)

    n_batches = math.ceil(g.number_of_edges() / batch_size)

    model.fit(epochs=epochs, log=True, batch_size=batch_size,
              steps_per_epoch=n_batches)

    embedding_path = 'embeddings/20newsgroup/alpha{}-l2_param{}-epochs{}-pre_epochs{}.pkl'.format(
        alpha, l2_param, epochs, pretrain_epochs
    )
    
    embeddings = model.get_node_embedding()
    labels = dataset.target
    pkl.dump((embeddings, labels), open(embedding_path, 'wb'))


for params in tqdm(parameter_dicts):
    one_run(params)
