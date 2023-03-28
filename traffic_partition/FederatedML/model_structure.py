#!/usr/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import numpy as np
import typing
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

import warnings
warnings.filterwarnings("ignore")

import networkx as nx
import matplotlib.pyplot as plt

import pickle

batch_size = 64
input_sequence_length = 12
forecast_horizon = 3
multi_horizon = False


def compute_adjacency_matrix(
    route_distances: np.ndarray, sigma2: float, epsilon: float
):
    """Computes the adjacency matrix from distances matrix.

    It uses the formula in https://github.com/VeritasYin/STGCN_IJCAI-18#data-preprocessing to
    compute an adjacency matrix from the distance matrix.
    The implementation follows that paper.

    Args:
        route_distances: np.ndarray of shape `(num_routes, num_routes)`. Entry `i,j` of this array is the
            distance between roads `i,j`.
        sigma2: Determines the width of the Gaussian kernel applied to the square distances matrix.
        epsilon: A threshold specifying if there is an edge between two nodes. Specifically, `A[i,j]=1`
            if `np.exp(-w2[i,j] / sigma2) >= epsilon` and `A[i,j]=0` otherwise, where `A` is the adjacency
            matrix and `w2=route_distances * route_distances`

    Returns:
        A boolean graph adjacency matrix.
    """
    num_routes = route_distances.shape[0]
    route_distances = route_distances / 10000.0
    w2, w_mask = (
        route_distances * route_distances,
        np.ones([num_routes, num_routes]) - np.identity(num_routes),
    )
    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask


class GraphInfo:
    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes


class GraphConv(layers.Layer):
    def __init__(
        self,
        in_feat,
        out_feat,
        graph_info: GraphInfo,
        aggregation_type="mean",
        combination_type="concat",
        activation: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.weight = tf.Variable(
            initial_value=keras.initializers.glorot_uniform()(
                shape=(in_feat, out_feat), dtype="float32"
            ),
            trainable=True,
        )
        self.activation = layers.Activation(activation)

    def aggregate(self, neighbour_representations: tf.Tensor):
        aggregation_func = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)

        if aggregation_func:
            return aggregation_func(
                neighbour_representations,
                self.graph_info.edges[0],
                num_segments=self.graph_info.num_nodes,
            )

        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

    def compute_nodes_representation(self, features: tf.Tensor):
        """Computes each node's representation.

        The nodes' representations are obtained by multiplying the features tensor with
        `self.weight`. Note that
        `self.weight` has shape `(in_feat, out_feat)`.

        Args:
            features: Tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

        Returns:
            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
        """
        return tf.matmul(features, self.weight)

    def compute_aggregated_messages(self, features: tf.Tensor):
        neighbour_representations = tf.gather(features, self.graph_info.edges[1])
        aggregated_messages = self.aggregate(neighbour_representations)
        return tf.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation: tf.Tensor, aggregated_messages: tf.Tensor):
        if self.combination_type == "concat":
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        return self.activation(h)

    def call(self, features: tf.Tensor):
        """Forward pass.

        Args:
            features: tensor of shape `(num_nodes, batch_size, input_seq_len, in_feat)`

        Returns:
            A tensor of shape `(num_nodes, batch_size, input_seq_len, out_feat)`
        """
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)


class LSTMGC(layers.Layer):
    """Layer comprising a convolution layer followed by LSTM and dense layers."""

    def __init__(
        self,
        in_feat,
        out_feat,
        lstm_units: int,
        input_seq_len: int,
        output_seq_len: int,
        graph_info: GraphInfo,
        graph_conv_params: typing.Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # graph conv layer
        if graph_conv_params is None:
            graph_conv_params = {
                "aggregation_type": "mean",
                "combination_type": "concat",
                "activation": None,
            }
        self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)

        self.lstm = layers.LSTM(lstm_units, activation="relu")
        self.dense = layers.Dense(output_seq_len)

        self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len

    def call(self, inputs):
        """Forward pass.

        Args:
            inputs: tf.Tensor of shape `(batch_size, input_seq_len, num_nodes, in_feat)`

        Returns:
            A tensor of shape `(batch_size, output_seq_len, num_nodes)`.
        """

        # convert shape to  (num_nodes, batch_size, input_seq_len, in_feat)
        inputs = tf.transpose(inputs, [2, 0, 1, 3])

        gcn_out = self.graph_conv(
            inputs
        )  # gcn_out has shape: (num_nodes, batch_size, input_seq_len, out_feat)
        shape = tf.shape(gcn_out)
        num_nodes, batch_size, input_seq_len, out_feat = (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )

        # LSTM takes only 3D tensors as input
        gcn_out = tf.reshape(gcn_out, (batch_size * num_nodes, input_seq_len, out_feat))
        lstm_out = self.lstm(
            gcn_out
        )  # lstm_out has shape: (batch_size * num_nodes, lstm_units)

        dense_output = self.dense(
            lstm_out
        )  # dense_output has shape: (batch_size * num_nodes, output_seq_len)
        output = tf.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len))
        return tf.transpose(
            output, [1, 2, 0]
        )  # returns Tensor of shape (batch_size, output_seq_len, num_nodes)





def get_model_structure(path):
    
    route_distances = pd.read_csv(
        os.path.join(path, "PeMSD7_Full", "PeMSD7_W_228.csv"), header=None
    ).to_numpy()
    

    sigma2 = 0.1
    epsilon = 0.5
    adjacency_matrix = compute_adjacency_matrix(route_distances, sigma2, epsilon)
    node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
    graph = GraphInfo(
        edges=(node_indices.tolist(), neighbor_indices.tolist()),
        num_nodes=adjacency_matrix.shape[0],
    )


    in_feat = 1
    batch_size = 64
    epochs = 20
    input_sequence_length = 12
    forecast_horizon = 3
    multi_horizon = False
    out_feat = 10
    lstm_units = 64
    graph_conv_params = {
        "aggregation_type": "mean",
        "combination_type": "concat",
        "activation": None,
    }

    st_gcn = LSTMGC(
        in_feat,
        out_feat,
        lstm_units,
        input_sequence_length,
        forecast_horizon,
        graph,
        graph_conv_params,
    )
    inputs = layers.Input((input_sequence_length, graph.num_nodes, in_feat))
    outputs = st_gcn(inputs)

    gnn_model = keras.models.Model(inputs, outputs)
    gnn_model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.0002),
        loss=keras.losses.MeanSquaredError(),
        metrics = ['accuracy']
    )
    return gnn_model


def save_model_structure(path):

    route_distances = pd.read_csv(
        os.path.join(path, "PeMSD7_Full", "PeMSD7_W_228.csv"), header=None
    ).to_numpy()
    

    sigma2 = 0.1
    epsilon = 0.5
    adjacency_matrix = compute_adjacency_matrix(route_distances, sigma2, epsilon)
    node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
    graph = GraphInfo(
        edges=(node_indices.tolist(), neighbor_indices.tolist()),
        num_nodes=adjacency_matrix.shape[0],
    )


    in_feat = 1
    batch_size = 64
    epochs = 20
    input_sequence_length = 12
    forecast_horizon = 3
    multi_horizon = False
    out_feat = 10
    lstm_units = 64
    graph_conv_params = {
        "aggregation_type": "mean",
        "combination_type": "concat",
        "activation": None,
    }

    st_gcn = LSTMGC(
        in_feat,
        out_feat,
        lstm_units,
        input_sequence_length,
        forecast_horizon,
        graph,
        graph_conv_params,
    )
    inputs = layers.Input((input_sequence_length, graph.num_nodes, in_feat))
    outputs = st_gcn(inputs)

    gnn_model = keras.models.Model(inputs, outputs)
    gnn_model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.0002),
        loss=keras.losses.MeanSquaredError(),
        metrics = ['accuracy']
    )
    return gnn_model



# def get_model_structure_by_path(path):
#     with open(path + "/edges.pickle", 'rb') as handle:
#         edges = pickle.load(handle)
    
#     with open(path + "/edge_weights.pickle", 'rb') as handle:
#         edge_weights = pickle.load(handle)

#     with open(path + "/node_features.pickle", 'rb') as handle:
#         node_features = pickle.load(handle)

#     with open(path + "/num_classes.pickle", 'rb') as handle:
#         num_classes = pickle.load(handle)


#     graph_info = (node_features, edges, edge_weights)

#     print("Edges shape:", edges.shape)
#     print("Nodes shape:", node_features.shape)
    
#     gnn_model = GNNNodeClassifier(
#         graph_info=graph_info,
#         num_classes=num_classes,
#         hidden_units=hidden_units,
#         dropout_rate=dropout_rate,
#         name="gnn_model",
#     )

#     # print("GNN output shape:", gnn_model([1, 10, 100]))
#     gnn_model([1, 10, 100])


#     gnn_model.compile(
#             optimizer=keras.optimizers.Adam(learning_rate),
#             loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#             metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
#     )
#     return gnn_model

# gnn_model = get_model_structure()


# def save_new_model_structure(path):
#     model = get_model_structure()
#     model.save(path, save_format='tf')


if __name__=="__main__":
    path = "/home/chirag/fl/keras-gnn/traffic_partition/FederatedML/saved_structure/model_structure"
    save_new_model_structure(path)

    new_model = get_model_structure_by_path(path)
    new_model.summary()
    