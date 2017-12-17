# coding: utf-8

import tensorflow as tf
import pickle as pkl
import pandas as pd
import os

from tensorflow.contrib.tensorboard.plugins import projector


embedding_path = 'embeddings/20newsgroup/alpha2-l2_param0.001-epochs1-pre_epochs0.pkl'
embeddings, labels = pkl.load(open(embedding_path, 'rb'))


emb_df = pd.DataFrame(embeddings)
emb_df.to_csv('embeddings/20newsgroup/embedding.tsv', sep='\t', header=False, index=False)

embedding_var = tf.Variable(embeddings, name='node_embeddings')


LOG_DIR = 'log/20newsgroup'


df = pd.Series(labels, name='label')
df.to_frame().to_csv(LOG_DIR + '/node_labels.tsv', index=False, header=False)

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
