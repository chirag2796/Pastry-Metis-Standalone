#!/usr/bin/python
import os
from model_structure import get_model_structure_by_path

from model_validation import evaluate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
import sys
# import texthero as hero
# from tensorflow import keras
# from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")

import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pickle

hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 4
batch_size = 256


def run_experiment(model, x_train, y_train):
    # Compile the model.
    # model.compile(
    #     optimizer=keras.optimizers.Adam(learning_rate),
    #     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    # )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping],
    )

    return history, model

# vectorize sentences embeddings
def vectorize(sentence, model, dimension=50):
    ans = [0]*dimension
    for word in sentence:
        if word in model:
            ans += model[word]
    return ans

# train model
def train_model(model, x_train, y_train):
    history, model = run_experiment(model, x_train, y_train)
    return model

def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)


# class GraphConvLayer(layers.Layer):
#     def __init__(
#         self,
#         hidden_units,
#         dropout_rate=0.2,
#         aggregation_type="mean",
#         combination_type="concat",
#         normalize=False,
#         *args,
#         **kwargs,
#     ):
#         super(GraphConvLayer, self).__init__(*args, **kwargs)

#         self.aggregation_type = aggregation_type
#         self.combination_type = combination_type
#         self.normalize = normalize

#         self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
#         if self.combination_type == "gated":
#             self.update_fn = layers.GRU(
#                 units=hidden_units,
#                 activation="tanh",
#                 recurrent_activation="sigmoid",
#                 dropout=dropout_rate,
#                 return_state=True,
#                 recurrent_dropout=dropout_rate,
#             )
#         else:
#             self.update_fn = create_ffn(hidden_units, dropout_rate)

#     def prepare(self, node_repesentations, weights=None):
#         # node_repesentations shape is [num_edges, embedding_dim].
#         messages = self.ffn_prepare(node_repesentations)
#         if weights is not None:
#             messages = messages * tf.expand_dims(weights, -1)
#         return messages

#     def aggregate(self, node_indices, neighbour_messages):
#         # node_indices shape is [num_edges].
#         # neighbour_messages shape: [num_edges, representation_dim].
#         num_nodes = tf.math.reduce_max(node_indices) + 1
#         if self.aggregation_type == "sum":
#             aggregated_message = tf.math.unsorted_segment_sum(
#                 neighbour_messages, node_indices, num_segments=num_nodes
#             )
#         elif self.aggregation_type == "mean":
#             aggregated_message = tf.math.unsorted_segment_mean(
#                 neighbour_messages, node_indices, num_segments=num_nodes
#             )
#         elif self.aggregation_type == "max":
#             aggregated_message = tf.math.unsorted_segment_max(
#                 neighbour_messages, node_indices, num_segments=num_nodes
#             )
#         else:
#             raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

#         return aggregated_message

#     def update(self, node_repesentations, aggregated_messages):
#         # node_repesentations shape is [num_nodes, representation_dim].
#         # aggregated_messages shape is [num_nodes, representation_dim].
#         if self.combination_type == "gru":
#             # Create a sequence of two elements for the GRU layer.
#             h = tf.stack([node_repesentations, aggregated_messages], axis=1)
#         elif self.combination_type == "concat":
#             # Concatenate the node_repesentations and aggregated_messages.
#             h = tf.concat([node_repesentations, aggregated_messages], axis=1)
#         elif self.combination_type == "add":
#             # Add node_repesentations and aggregated_messages.
#             h = node_repesentations + aggregated_messages
#         else:
#             raise ValueError(f"Invalid combination type: {self.combination_type}.")

#         # Apply the processing function.
#         node_embeddings = self.update_fn(h)
#         if self.combination_type == "gru":
#             node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

#         if self.normalize:
#             node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
#         return node_embeddings

#     def call(self, inputs):
#         """Process the inputs to produce the node_embeddings.
#         inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
#         Returns: node_embeddings of shape [num_nodes, representation_dim].
#         """

#         node_repesentations, edges, edge_weights = inputs
#         # Get node_indices (source) and neighbour_indices (target) from edges.
#         node_indices, neighbour_indices = edges[0], edges[1]
#         # neighbour_repesentations shape is [num_edges, representation_dim].
#         neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)

#         # Prepare the messages of the neighbours.
#         neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)
#         # Aggregate the neighbour messages.
#         aggregated_messages = self.aggregate(node_indices, neighbour_messages)
#         # Update the node embedding with the neighbour messages.
#         return self.update(node_repesentations, aggregated_messages)




# class GNNNodeClassifier(tf.keras.Model):
#     def __init__(
#         self,
#         graph_info,
#         num_classes,
#         hidden_units,
#         aggregation_type="sum",
#         combination_type="concat",
#         dropout_rate=0.2,
#         normalize=True,
#         *args,
#         **kwargs,
#     ):
#         super(GNNNodeClassifier, self).__init__(*args, **kwargs)

#         # Unpack graph_info to three elements: node_features, edges, and edge_weight.
#         node_features, edges, edge_weights = graph_info
#         self.node_features = node_features
#         self.edges = edges
#         self.edge_weights = edge_weights
#         # Set edge_weights to ones if not provided.
#         if self.edge_weights is None:
#             self.edge_weights = tf.ones(shape=edges.shape[1])
#         # Scale edge_weights to sum to 1.
#         self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

#         # Create a process layer.
#         self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")
#         # Create the first GraphConv layer.
#         self.conv1 = GraphConvLayer(
#             hidden_units,
#             dropout_rate,
#             aggregation_type,
#             combination_type,
#             normalize,
#             name="graph_conv1",
#         )
#         # Create the second GraphConv layer.
#         self.conv2 = GraphConvLayer(
#             hidden_units,
#             dropout_rate,
#             aggregation_type,
#             combination_type,
#             normalize,
#             name="graph_conv2",
#         )
#         # Create a postprocess layer.
#         self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
#         # Create a compute logits layer.
#         self.compute_logits = layers.Dense(units=num_classes, name="logits")

#     def call(self, input_node_indices):
#         # Preprocess the node_features to produce node representations.
#         x = self.preprocess(self.node_features)
#         # Apply the first graph conv layer.
#         x1 = self.conv1((x, self.edges, self.edge_weights))
#         # Skip connection.
#         x = x1 + x
#         # Apply the second graph conv layer.
#         x2 = self.conv2((x, self.edges, self.edge_weights))
#         # Skip connection.
#         x = x2 + x
#         # Postprocess node embedding.
#         x = self.postprocess(x)
#         # Fetch node embeddings for the input node_indices.
#         node_embeddings = tf.gather(x, input_node_indices)
#         # Compute logits
#         return self.compute_logits(node_embeddings)


if __name__ == "__main__":
    print("Training for "+str(sys.argv[1])+ " begins...")
    # path = "/Users/taehwan/Desktop/Research/pastry_amazon/src/rice/tutorial/FederatedML_Amazon"

    device_id = "device_" + str(sys.argv[1])
    
    path = "/home/chirag/fl/keras-gnn/cora_partition/FederatedML"
    
    gnn_model = get_model_structure_by_path(path + "/saved_structure")

    # load the model
    # gnn_model = keras.models.load_model(path+'/init_model_'+str(sys.argv[1])+'_'+str(sys.argv[2]))
    gnn_model.load_weights(path+'/init_model_'+str(sys.argv[1])+'_'+str(sys.argv[2]) + ".h5")

    print("1. Init Model Loaded")    

    # device_id = "device_1"
    # device_id = "device_full"
    print("Loading: ", "data_partitions/{}_data_partition.pickle".format(device_id))
    with open(path+"/data_partitions/{}_data_partition.pickle".format(device_id), 'rb') as handle:
        graph_partition = pickle.load(handle)

    x_train = graph_partition["x_train"]
    x_test = graph_partition["x_test"]
    y_train = graph_partition["y_train"]
    y_test = graph_partition["y_test"]
    
    # x_train = train_data.paper_id.to_numpy()

    # print("x_train: ", x_train.shape)
    # print("x_test: ", x_test.shape)
    # print("y_train: ", y_train.shape)
    # print("y_test: ", y_test.shape)
    # print(type(x_train))
    # print(type(y_train))
    # # exit()


    new_model = train_model(model=gnn_model, x_train=x_train, y_train=y_train)
    
    print("6. Model Trained")

    evaluate(new_model, data_origin="SAVED")
    
    # new_model.save(path+'/new_model_'+str(sys.argv[1])+'_'+str(sys.argv[2])+'.h5')
    new_model.save_weights(path+'/new_model_'+str(sys.argv[1])+'_'+str(sys.argv[2]) + ".h5")

    print("7. Model Saved as" + path+'/new_model_'+str(sys.argv[1])+'_'+str(sys.argv[2]) + ".h5")


    print("Loading again 1")
    new_model_loaded = get_model_structure_by_path(path + "/saved_structure")
    new_model_loaded.load_weights(path+'/new_model_'+str(sys.argv[1])+'_'+str(sys.argv[2]) + ".h5")
    evaluate(new_model_loaded, data_origin="SAVED")
    
    print("Training for "+str(sys.argv[1])+ " ends...")