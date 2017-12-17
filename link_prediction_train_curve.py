# coding: utf-8

import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt


result = pkl.load(open('outputs/link_prediction_grqc_epochs.pkl', 'rb'))


ks = np.array([2, 10, 100, 200, 300, 500, 800, 1000, 10000]) / 2

train_scores = np.array(result[0][0])

size = 3
fig, axes = plt.subplots(3, 3, figsize=(size * 3, size * 3 / 1.5), sharex=True)
xs = np.arange(train_scores.shape[0])
for c, k in enumerate(ks):
    i, j = int(c / 3), int(c % 3)
    ys = train_scores[:, c]
    ax = axes[i, j]
    ax.plot(xs[0:-1:10], ys[0:-1:10], '-o')
    # sns.pointplot(x=xs[0:-1:10], y=ys[0:-1:10], ax=ax)
    ax.set_ylabel('p@{:}'.format(int(k)), fontsize=18)

    ax.patch.set_visible(False)
    if i == 2:
        ax.set_xlabel('epoch', fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
fig.tight_layout()
fig.savefig('figs/link_prediction_train.png')


for k in ks:
    v = result[0][1][k]
    print('- **p@{}**: *{}*'.format(int(k), v))
