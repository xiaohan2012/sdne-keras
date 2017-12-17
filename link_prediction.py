# coding: utf-8

import pickle as pkl
import numpy as np
import networkx as nx
import math

from core import SDNE
from sklearn.model_selection import train_test_split
from itertools import product
from tqdm import tqdm
from keras.callbacks import Callback


batch_size = 64
ks = [1, 5, 50, 100, 150, 250, 400, 500, 5000]

g = nx.read_edgelist('data/grqc.txt', create_using=nx.Graph())


g = nx.convert_node_labels_to_integers(g)


parameter_grid = {'beta': [2], 'alpha': [2], 'l2_param': [1e-4]}


parameter_values = list(product(*parameter_grid.values()))
parameter_keys = list(parameter_grid.keys())


parameter_dicts = [dict(list(zip(parameter_keys, values))) for values in parameter_values]


dev_ratio = 0.1
test_ratio = 0.15


train_set, test_edges = train_test_split(g.edges(), test_size=test_ratio)


_, dev_edges = train_test_split(train_set, test_size=dev_ratio / (1 - test_ratio))


print('#dev edges: {}, #test edges: {}'.format(len(dev_edges), len(test_edges)))
g.remove_edges_from(test_edges + dev_edges)


g.add_edges_from([(i, i) for i in np.arange(g.number_of_nodes())])


class PrecisionAtKEval(Callback):
    def __init__(self, g, true_edges, excluded_edges, decoder, ks):
        """
        ks = [2, 10, 100, 200, 300, 500, 800, 1000, 10000]
        """
        self.ks = ks
        self.maps = []
        self.decoder = decoder

        N = g.number_of_nodes()
        self.nodes = np.arange(N)[:, None]
        self.edges_to_eval = set([(i, j) for i in range(N) for j in range(i+1, N)])

        # we don't consider train edges and excluded_edges
        # only consider dev edges and other ones
        self.edges_to_eval -= set(g.edges()) | set(excluded_edges)
        true_edges = set(true_edges)

        # edges in true_edges are labeled 1
        # othe edges labeled 0
        # edge2label = {e: (e in true_edges)
        #               for e in self.edges_to_eval}
        self.true_y = np.array([(e in true_edges)
                                for e in self.edges_to_eval])

    def eval_map(self):
        # to enable calling this function outside of `keras.Model`
        reconstructed_adj = self.decoder.predict(self.nodes)

        pred_y = np.array([reconstructed_adj[u, v]
                           for u, v in self.edges_to_eval])

        sort_idx = np.argsort(pred_y)[::-1]  # index ranked by score from high to low

        return [precision_at_k(pred_y, self.true_y, k=k, sort_idx=sort_idx)
                for k in self.ks]

    def on_epoch_end(self, epoch, logs={}):
        row = self.eval_map()
        print('EPOCH {}'.format(epoch))
        print('[DEV] precision at k' + ' '.join(["{}:{}".format(k, r) for k, r in zip(self.ks, row)]))
        self.maps.append(row)


def precision_at_k(pred_y, true_y, k, sort_idx=None):
    if sort_idx is None:
        sort_idx = np.argsort(pred_y)[::-1]
    top_k_idx = sort_idx[:k]
    return np.sum(true_y[top_k_idx]) / k


def one_run(g, dev_edges, test_edges, params):
    # we divide the num. of edges by two because
    # becauses num. of edges in the paper about 2 times than the actual number of edges
    # the original:
    # ks = [2, 10, 100, 200, 300, 500, 800, 1000, 10000]
    model = SDNE(g, encode_dim=100, encoding_layer_dims=[5242, 500], **params)
    print('pre-training...')
    model.pretrain(epochs=1, batch_size=batch_size)
    print('training...')
    n_batches = math.ceil(g.number_of_edges() / batch_size)

    eval_callback = PrecisionAtKEval(g, dev_edges, test_edges,
                                     decoder=model.decoder,
                                     ks=ks)
    model.fit(epochs=200, batch_size=batch_size,
              steps_per_epoch=n_batches,
              callbacks=[eval_callback])

    test_evaluator = PrecisionAtKEval(
        g, test_edges, dev_edges,  # now we evaluate on test edges
        decoder=model.decoder,
        ks=ks)
    
    test_vals = test_evaluator.eval_map()
    test_result = dict(zip(ks, test_vals))
    return (eval_callback.maps, test_result)


result = [one_run(g, dev_edges, test_edges, params)
          for params in tqdm(parameter_dicts)]


pkl.dump(result, open('outputs/link_prediction_grqc_epochs.pkl', 'wb'))


# checking the best

# for params, scores in result[1].items():
#     scores['param'] = params


# df = pd.DataFrame.from_records(
#     [r for _, r in result],
#     columns=ks + ['param'])

# df.sort_values(by=[800, 1000], ascending=False)

################
# Hyper parameter tunning result
################

# - l2_param: 0.0001
# - alpha: 2
# - beta: 2
