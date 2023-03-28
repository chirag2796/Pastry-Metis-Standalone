# Import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import deepchem as dc

# Load and process Tox21 dataset with GraphConv featurizer
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = tox21_datasets

# Define GCN layer class
class GraphConvLayer(layers.Layer):

  def __init__(self, units):
    super(GraphConvLayer, self).__init__()
    self.units = units

  def build(self, input_shape):
    self.weight = self.add_weight(
        shape=(input_shape[-1], self.units),
        initializer="random_normal",
        trainable=True,
    )
    self.bias = self.add_weight(
        shape=(self.units,), initializer="random_normal", trainable=True
    )

  def call(self, inputs):
    node_features = inputs[0]
    adjacency_matrix = inputs[1]
    node_features = tf.matmul(node_features, self.weight) + self.bias
    node_features = tf.nn.relu(node_features)
    node_features = tf.matmul(adjacency_matrix, node_features)
    return node_features

# Define GCN model class
class GCNModel(keras.Model):

  def __init__(self):
    super(GCNModel, self).__init__()
    self.gcn1 = GraphConvLayer(64)
    self.gcn2 = GraphConvLayer(64)
    self.flatten = layers.Flatten()
    self.dense1 = layers.Dense(128)
    self.dropout1 = layers.Dropout(0.5)
    self.dense2 = layers.Dense(12)

  def call(self, inputs):
    node_features = inputs[0]
    adjacency_matrix = inputs[1]
    node_features = self.gcn1([node_features, adjacency_matrix])
    node_features = self.gcn2([node_features, adjacency_matrix])
    node_features = self.flatten(node_features)
    node_features = self.dense1(node_features)
    node_features = self.dropout1(node_features)
    output = self.dense2(node_features)
    return output

# Create GCN model instance
model = GCNModel()

# Compile model with loss and optimizer
model.compile(
  loss=keras.losses.BinaryCrossentropy(from_logits=True),
  optimizer=keras.optimizers.Adam(learning_rate=0.001),
  metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)],
)

# Define custom generator class to feed data to model
class Tox21Generator(keras.utils.Sequence):

  def __init__(self, dataset, batch_size=32):
    self.dataset = dataset
    self.batch_size = batch_size

  def __len__(self):
    return int(np.ceil(len(self.dataset)/float(self.batch_size)))

  def __getitem__(self, idx):
    # Get a batch of data starting from index idx*self.batch_size
    batch_data = self.dataset.select(np.arange(idx*self.batch_size, (idx+1)*self.batch_size))
    # Extract node features and adjacency matrix from data
    node_features = batch_data.X
    adjacency_matrix = []
    for i in range(len(batch_data)):
      adj_mat = batch_data[i].w
      adjacency_matrix.append(adj_mat)
    # Pad node features and adjacency matrix to have same dimensions
    pad_length = 75 - len(node_features)
    if pad_length > 0:
      node_features = np.pad(node_features, ((0,pad_length),(0,0)), mode='constant')
      adjacency_matrix = [np.pad(adj_mat, ((0,pad_length-len(adj_mat)),(0,pad_length-len(adj_mat))), mode='constant') for adj_mat in adjacency_matrix]
    # Extract labels from data
    labels = batch_data.y
    return [node_features, np.array(adjacency_matrix)], labels




# Create generators for train, valid and test datasets
train_gen = Tox21Generator(train_dataset)
valid_gen = Tox21Generator(valid_dataset)
test_gen = Tox21Generator(test_dataset)

# Train model for 10 epochs with batch size of 32
model.fit(train_gen, epochs=10, batch_size=32, validation_data=valid_gen)

# Evaluate model on test dataset
model.evaluate(test_gen)