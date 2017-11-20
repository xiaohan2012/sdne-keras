
# coding: utf-8

# In[70]:


import numpy as np
import networkx as nx

from core import SDNE
from sklearn.model_selection import train_test_split
from itertools import product
from tqdm import tqdm


# In[32]:


g = nx.read_edgelist('data/grqc.txt', create_using=nx.Graph())


# In[33]:


g = nx.convert_node_labels_to_integers(g)


# In[63]:


parameter_grid = {'beta': [2, 4, 8], 'alpha': [2, 4, 8], 'l2_param': [0.01, 0.1, 1, 10]}


# In[64]:


parameter_values = list(product(*parameter_grid.values()))
parameter_keys = list(parameter_grid.keys())


# In[65]:


parameter_dicts = [dict(list(zip(parameter_keys, values))) for values in parameter_values]


# In[66]:


len(parameter_dicts)


# In[35]:


dev_ratio = 0.1
test_ratio = 0.15


# In[36]:


train_set, test_edges = train_test_split(g.edges(), test_size=test_ratio)


# In[37]:


_, dev_edges = train_test_split(train_set, test_size=dev_ratio / (1 - test_ratio))


# In[38]:


g.remove_edges_from(dev_edges + test_edges)


# In[39]:


g.add_edges_from([(i, i) for i in np.arange(g.number_of_nodes())])


# In[ ]:


def precision_at_k(pred_y, true_y, k, sort_idx=None):
    if sort_idx is None:
        sort_idx = np.argsort(pred_y)[::-1]
    top_k_idx = sort_idx[:k]
    return np.sum(true_y[top_k_idx]) / k


# In[ ]:


def score(model, g, dev_edges, test_edges):
    # test_edges passed so to remove them from evauation
    N = g.number_of_nodes()

    nodes = np.arange(N)[:, None]
    reconstructed_adj = model.decoder.predict(nodes)

    all_edges = set([(i, j) for i in range(N) for j in range(i+1, N)])
    
    # we don't consider train edges and test_edges
    # only consider dev edges and other ones
    all_edges -= (set(g.edges()) | set(test_edges))
    
    edge2score = {(u, v): reconstructed_adj[u, v]
                  for u, v in all_edges}
    dev_edges = set(dev_edges)
    
    # edges in dev_edges are labeled 1
    # othe edges labeled 0
    edge2label = {e: (e in dev_edges)
                  for e in edge2score.keys()}

    edges_to_test = list(edge2label.keys())

    pred_y = np.array([edge2score[e] for e in edges_to_test])
    true_y = np.array([edge2label[e] for e in edges_to_test])

    sort_idx = np.argsort(pred_y)[::-1]

    scores = {}
    ks = [2, 10, 100, 200, 300, 500, 800, 1000, 10000]
    for k in ks:
        p = precision_at_k(pred_y, true_y, k=k, sort_idx=sort_idx)
        scores[k] = p
        print('precision@{}: {}'.format(k, p))
    return scores


# In[45]:


def one_run(g, dev_edges, test_edges, params):
    model = SDNE(g, encode_dim=100, encoding_layer_dims=[5242, 500], **params)
    print('pre-training...')
    model.pretrain(epochs=25, batch_size=64)
    print('training...')
    model.fit(epochs=25, batch_size=64)
    scores = score(model, g, dev_edges, test_edges)
    return (params, scores)


# In[71]:


result = [one_run(g, dev_edges, test_edges, params)for params in tqdm(parameter_dicts)]


# In[ ]:


import pickle as pkl
pkl.dump(result, open('outputs/link_prediction_grqc.pkl', 'wb'))

