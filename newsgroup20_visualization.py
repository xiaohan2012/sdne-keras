# coding: utf-8
import matplotlib as mpl
mpl.use('Agg')

import networkx as nx
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import kneighbors_graph

from matplotlib import pyplot as plt
from core import SDNE


# In[6]:


categories = ['comp.graphics', 'rec.sport.baseball', 'talk.politics.guns']
dataset = fetch_20newsgroups(categories=categories)


# In[9]:


vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(dataset.data)


# In[50]:


# build the graph
N = vectors.shape[0]

mat = kneighbors_graph(vectors, 256, metric='cosine', mode='distance', include_self=True)

mat.data = 1 - mat.data  # to similarity

g = nx.from_scipy_sparse_matrix(mat, create_using=nx.Graph())


# In[54]:


model = SDNE(g, encode_dim=100, encoding_layer_dims=[1720, 200],
             beta=10,
             alpha=10,
             l2_param=1)


# In[56]:


model.pretrain(epochs=5, batch_size=32)
model.fit(epochs=1, batch_size=32)


# In[57]:


embeddings = model.get_node_embedding()


# In[68]:


labels = dataset.target


# In[59]:


from sklearn.manifold import TSNE
pos = TSNE(n_components=2).fit_transform(embeddings)


# In[78]:


plt.scatter(pos[:, 0], pos[:, 1], c=labels)
# plt.legend(dataset.target_names)
plt.savefig('figs/20newsgroup_viz.png')

