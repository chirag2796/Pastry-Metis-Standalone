import time
import numpy as np
import networkx as nx
import requests

# tf part
import tensorflow as tf
from tensorflow.keras import layers

# dgl part
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
# from dgl.data import CiteseerGraphDataset

class available_datasets_already_available_with_dgl:
    def __init__(self, dataset_name):
        self.dataset = dataset_name

data = load_data(available_datasets_already_available_with_dgl('citeseer'))



# dataset = CiteseerGraphDataset()
# g = dataset[0]
# num_class = dataset.num_classes
# # get node feature
# features = g.ndata['feat']
# # get data split
# train_mask = g.ndata['train_mask']
# val_mask = g.ndata['val_mask']
# test_mask = g.ndata['test_mask']
# # get labels
# labels = g.ndata['label']




# features = tf.convert_to_tensor(features, dtype=tf.float32)
# labels = tf.convert_to_tensor(labels, dtype=tf.int64)
# train_mask = tf.convert_to_tensor(train_mask, dtype=tf.bool)
# val_mask = tf.convert_to_tensor(val_mask, dtype=tf.bool)
# test_mask = tf.convert_to_tensor(test_mask, dtype=tf.bool)

# in_feats = features.shape[1]
# n_classes = num_class
# # n_edges = data.graph.number_of_edges()
# n_edges = 9228


features = tf.convert_to_tensor(data.features, dtype=tf.float32)
labels = tf.convert_to_tensor(data.labels, dtype=tf.int64)
train_mask = tf.convert_to_tensor(data.train_mask, dtype=tf.bool)
val_mask = tf.convert_to_tensor(data.val_mask, dtype=tf.bool)
test_mask = tf.convert_to_tensor(data.test_mask, dtype=tf.bool)

in_feats = features.shape[1]
n_classes = data.num_labels
n_edges = data.graph.number_of_edges()


print("""----Data statistics------'
#Edges %d
#Classes %d
#Train samples %d
#Val samples %d
#Test samples %d""" %
      (n_edges, n_classes,
       train_mask.numpy().sum(),
       val_mask.numpy().sum(),
       test_mask.numpy().sum()))

# graph preprocess and calculate normalization factor
g = data.graph
g.remove_edges_from(nx.selfloop_edges(g))
g.add_edges_from(zip(g.nodes(), g.nodes()))

g = DGLGraph(g)


# nx.draw(g.to_networkx())  # you can draw it better than me. I am useless


n_edges = g.number_of_edges()

degs = tf.cast(tf.identity(g.in_degrees()), dtype=tf.float32)
norm = tf.math.pow(degs, -0.5)
norm = tf.where(tf.math.is_inf(norm), tf.zeros_like(norm), norm)
g.ndata['norm'] = tf.expand_dims(norm, -1)

# class GraphConv(layers.Layer):
#     def __init__(self,
#                 out_feats):
#         super(GraphConv, self).__init__()
        
#         # a very simple dense layer takes node features as input and outputs a lower dimension representation
#         self.denselayer = layers.Dense(out_feats, use_bias=False)

#     def call(self, graph, feat):
#         # make a local copy of the graph -- something related to not changing the global variable graph, I dont really care...
#         graph = graph.local_var()

#         feat = self.denselayer(feat)
#         graph.ndata['h'] = feat
#         graph.update_all(dgl.function.copy_src(src='h', out='m'),
#                          dgl.function.sum(msg='m', out='h'))
#         rst = graph.ndata['h']

#         return rst


class GraphConv(layers.Layer):
    def __init__(self, out_feats):
        super(GraphConv, self).__init__()
        self.denselayer = layers.Dense(out_feats, use_bias=False)

    def call(self, g, feat):
        g = g.local_var()
        feat = self.denselayer(feat)
        g.ndata['h'] = feat
        g.update_all(dgl.function.copy_src(src='h', out='m'),
                     dgl.function.sum(msg='m', out='h'))
        rst = g.ndata['h']
        return rst



class GCN(tf.keras.Model):
    def __init__(self, g,in_feats,n_hidden,n_classes):
        super(GCN, self).__init__()
        self.g = g
        
        self.input_layer = GraphConv(n_hidden)
        self.hidden_layer = GraphConv(n_hidden)  # create more if you like
        self.output_layer = GraphConv(n_classes)
    
    def call(self, features):
        h = features
        h = self.input_layer(self.g, h)
        h = self.hidden_layer(self.g, h)
        h = self.output_layer(self.g, h)
        return h


model = GCN(g,
           in_feats = in_feats,
           n_hidden=16,
           n_classes=n_classes)

loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, epsilon=1e-8)

for x in range (100):
    with tf.GradientTape() as tape:
        logits = model(features)
        loss_value = loss_func(labels[train_mask], logits[train_mask])

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))



    # evaluate
    logits_test = model(features, training = False)

    logits_val = logits[val_mask]
    labels_val = labels[val_mask]
    indices = tf.math.argmax(logits_val, axis = 1)
    acc = tf.reduce_mean(tf.cast(indices == labels_val, dtype = tf.float32))

    print(acc.numpy().round(2))