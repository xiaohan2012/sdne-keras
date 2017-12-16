
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import networkx as nx
import numpy as np
from core import SDNE


# In[3]:


# the graph
g = nx.karate_club_graph()


# In[13]:


model = SDNE(g, encode_dim=8, alpha=2)
model.fit(batch_size=10, epochs=100, verbose=1)


# In[14]:


node_embeddings = model.get_node_embedding()


# In[15]:


community_1 = [1, 2, 3, 4, 5, 6, 7, 8,  11, 12, 13, 14, 17, 18, 20, 22]


# In[16]:


node_color = np.zeros(g.number_of_nodes())
node_color[community_1] = 1


# In[17]:


# visualization using default layout
nx.draw_networkx(g, node_color=node_color, font_color='white') 


# In[18]:


from sklearn.manifold import TSNE
pos = TSNE(n_components=2).fit_transform(node_embeddings)


# In[19]:


# visualization using the learned embedding
nx.draw_networkx(g, pos, node_color=node_color, font_color='white')

