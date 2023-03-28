import numpy as np
import scipy.sparse as sp
import networkx as nx
import tensorflow as tf

# Load data
G = nx.read_edgelist('flickr.edges')
features = np.loadtxt('flickr.features', delimiter=',') 
labels = np.loadtxt('flickr.labels', dtype=int)

# Convert NetworkX graph object to adjacency matrix
adj_matrix = nx.adjacency_matrix(G).toarray().astype(np.float32)

# Preprocess data
labels = tf.keras.utils.to_categorical(labels)

# Save data to npz archive
np.savez('flickr.npz', adj_data=adj_matrix.data, adj_indices=adj_matrix.indices, adj_indptr=adj_matrix.indptr, adj_shape=adj_matrix.shape, attr_data=features, labels=labels)
