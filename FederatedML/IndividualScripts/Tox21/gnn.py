import tensorflow as tf
import numpy as np
import deepchem as dc
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Layer

# Load the Tox21 dataset
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = tox21_datasets

# Extract features from the molecules
featurizer = dc.feat.graph_features.ConvMolFeaturizer()
train_features = featurizer.featurize(train_dataset.X)
valid_features = featurizer.featurize(valid_dataset.X)
test_features = featurizer.featurize(test_dataset.X)

# Define the GCN layer
class GraphConvolution(Layer):
    def __init__(self, units, activation=None, use_bias=True, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[1], self.units),
                                      initializer='glorot_uniform',
                                      name='kernel')
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer='zeros',
                                        name='bias')
        super(GraphConvolution, self).build(input_shape)

    def call(self, X, adj):
        X = tf.matmul(X, self.kernel)
        X = tf.matmul(adj, X)
        if self.use_bias:
            X = X + self.bias
        if self.activation is not None:
            X = self.activation(X)
        return X

# Create the input layer
inputs = Input(shape=(train_features.shape[1],))

# Add the GCN layer with 64 units and ReLU activation
gcn = GraphConvolution(64, activation='relu', use_bias=True)(inputs)

# Add a dropout layer with a rate of 0.5
gcn = Dropout(0.5)(gcn)

# Add another GCN layer with 64 units and ReLU activation
gcn = GraphConvolution(64, activation='relu', use_bias=True)(gcn)

# Add a dropout layer with a rate of 0.5
gcn = Dropout(0.5)(gcn)

# Add the output layer with a sigmoid activation
outputs = Dense(1, activation='sigmoid')(gcn)

# Create the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Convert the target variable to a one-hot encoded format
train_targets = np.eye(12)[train_dataset.y]
valid_targets = np.eye(12)[valid_dataset.y]

# Train the model for 50 epochs
model.fit(train_features, train_targets, epochs=2, batch_size=32, validation_data=(valid_features, valid_targets))

# Evaluate the model on the test set
# test_loss, test_acc = model.evaluate(test_dataset.X, test_dataset.y)
# print('Test Loss: {:.4f}'.format(test_loss))
# print('Test Accuracy: {:.4f}'.format(test_acc))

