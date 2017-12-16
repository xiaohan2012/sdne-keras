
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pickle as pkl
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from itertools import product
from tqdm import tqdm


# In[9]:


embedding_path = 'embeddings/20newsgroup/alpha2-l2_param0.001-epochs1-pre_epochs0.pkl'
embeddings, labels = pkl.load(open(embedding_path, 'rb'))


# In[10]:


parameter_names = ['perplexity', 'n_iter']
# parameter_values = [[10, 30, 50], [500, 750, 1000, 1250]]
parameter_values = [[10, 20], [500, 1000]]

value_combinations = list(product(*parameter_values))

parameter_dicts = [dict(list(zip(parameter_names, values))) for values in value_combinations]


# In[11]:


nrow, ncol = len(parameter_values[0]), len(parameter_values[1])
width = 5
fig, axes = plt.subplots(nrow, ncol, figsize=(width * ncol, width * nrow))

for i, perp in tqdm(enumerate(parameter_values[0])):
    for j, n_iter in enumerate(parameter_values[1]):        
        ax = axes[i, j]
            
        pos = TSNE(n_components=2, perplexity=perp, n_iter=n_iter).fit_transform(embeddings)
        ax.scatter(pos[:, 0], pos[:, 1], c=labels)
# plt.legend(dataset.target_names)
# path = 'figs/20newsgroup_viz-alpha{}-l2_param{}-epochs{}-pre_epochs{}.png'.format(
#     alpha, l2_param, epochs, pretrain_epochs
# )
# plt.savefig(path)


# In[40]:


import seaborn as sns
import pandas as pd

pos = TSNE(n_components=2, perplexity=20, n_iter=500).fit_transform(embeddings)



# In[43]:


# prettier plot
df = pd.DataFrame()

df['x'] = pos[:, 0]
df['y'] = pos[:, 1]
legends = ['comp.graphics', 'rec.sport.baseball', 'talk.politics.guns']
df['class'] = [legends[l] for l in labels]

sns.set_context("notebook", font_scale=1.5)
sns.set_style("ticks")


# Create scatterplot of dataframe
sns.lmplot('x', # Horizontal axis
           'y', # Vertical axis
           data=df, # Data source
           fit_reg=False, # Don't fix a regression line
           hue="class", # Set color,
           legend=True,
           scatter_kws={"s": 25, 'alpha': 0.5}) # S marker size
sns.despine(top=True, left=True, right=True, bottom=True)
plt.xticks([]); plt.yticks([])
plt.xlabel(''); plt.ylabel('')
plt.savefig('figs/20newsgroup_viz.png')
#plt.legend(,
#           loc='best')

