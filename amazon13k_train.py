# coding: utf-8

import networkx as nx
import os
import tensorflow as tf
import math
import numpy as np
import pandas as pd

from core import SDNE
from tensorflow.contrib.tensorboard.plugins import projector

batch_size = 128
epochs = 50

g = nx.read_gpickle('data/amazon13k.pkl')


model = SDNE(g, encode_dim=100, encoding_layer_dims=[3200, 800],
             beta=2,
             alpha=2,
             l2_param=1e-3)


n_batches = math.ceil(g.number_of_edges() / batch_size)
model.fit(epochs=epochs, log=True,
          log_dir="log/amazon13k",
          batch_size=batch_size,
          steps_per_epoch=n_batches)


embedding_values = model.encoder.predict(np.arange(g.number_of_nodes())[:, None])
embedding_var = tf.Variable(embedding_values, name='node_embeddings')


emb_df = pd.DataFrame(embedding_values)
emb_df.to_csv('embeddings/amazon13k/embedding.tsv', sep='\t', header=False, index=False)

LOG_DIR = 'log/amazon13k'

with tf.Session() as sess:
    saver = tf.train.Saver([embedding_var])

    sess.run(embedding_var.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'embeddings.ckpt'))
    
    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = 'node_labels.tsv'
    # Saves a config file that TensorBoard will read during startup.
    
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
