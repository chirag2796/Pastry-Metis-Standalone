import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# from spektral.datasets import citeseer
import spektral
from spektral.layers import GraphConv

from scipy.sparse import coo_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

# Load dataset
# adjacency, features, labels, _, _, _ = citeseer.load_data()



from tensorflow.keras import activations, initializers, constraints
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers

# class GraphConv(Layer):
#     def __init__(self, units, activation=None, use_bias=True,
#                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
#                  kernel_regularizer=None, bias_regularizer=None,
#                  activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
#                  **kwargs):
#         super(GraphConv, self).__init__(**kwargs)
#         self.units = units
#         self.activation = activations.get(activation)
#         self.use_bias = use_bias
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.activity_regularizer = regularizers.get(activity_regularizer)
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)

#     def build(self, input_shape):
#         assert len(input_shape) == 2
#         self.input_dim = input_shape[-1]
#         self.kernel = self.add_weight(shape=(self.input_dim, self.units),
#                                       initializer=self.kernel_initializer,
#                                       name='kernel',
#                                       regularizer=self.kernel_regularizer,
#                                       constraint=self.kernel_constraint)
#         if self.use_bias:
#             self.bias = self.add_weight(shape=(self.units,),
#                                         initializer=self.bias_initializer,
#                                         name='bias',
#                                         regularizer=self.bias_regularizer,
#                                         constraint=self.bias_constraint)
#         else:
#             self.bias = None
#         self.built = True

#     def call(self, inputs):
#         features, adjacency = inputs
#         output = tf.sparse.sparse_dense_matmul(adjacency, features)
#         output = tf.matmul(output, self.kernel)
#         if self.use_bias:
#             output = tf.nn.bias_add(output, self.bias)
#         if self.activation is not None:
#             output = self.activation(output)
#         return output

#     def compute_output_shape(self, input_shape):
#         assert input_shape and len(input_shape) == 2
#         return input_shape[0], self.units





cora_dataset = spektral.datasets.citation.Citation(name='cora')
test_mask = cora_dataset.mask_te
train_mask = cora_dataset.mask_tr
val_mask = cora_dataset.mask_va
graph = cora_dataset.graphs[0]
features = graph.x
adjacency = graph.a
labels = graph.y


# adjacency, features, labels, _, _, _ = spektral.datasets.citation.Citation('citeseer')


# Parameters
N = adjacency.shape[0]       # Number of nodes in the graph
F = features.shape[1]        # Number of node features
n_classes = labels.shape[1]  # Number of classes
l2_reg = 5e-4                # Regularization rate for l2
learning_rate = 1e-2         # Learning rate for optimizer
epochs = 200                 # Number of training epochs
batch_size = N               # Batch size
es_patience = 10             # Patience for early stopping

# Convert the data to a list of sparse matrices
A_list = [coo_matrix(adjacency).astype(np.float32)]
X_list = [features.astype(np.float32)]

# Convert the labels to integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(np.argmax(labels, axis=1))

# Shuffle the data
X, A, y = shuffle(features, adjacency, labels, random_state=0)

# Model definition
X_in = Input(shape=(F, ))
A_in = Input((N, ), sparse=True)

# Graph Convolutional Layers
H = Dropout(0.5)(X_in)
H = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=l2(l2_reg))(H)
H = tf.keras.layers.Dropout(0.5)(H)
H = tf.keras.layers.Dense(n_classes, activation='softmax')(H)
output = GraphConv(n_classes, activation='softmax')([H,A_in])

# Build model
model = Model(inputs=[X_in, A_in], outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training
validation_data = ([X]+A_list, y, val_mask)
model.fit([X]+A_list, y, sample_weight=train_mask, batch_size=batch_size, epochs=epochs,
          validation_data=validation_data, shuffle=False, callbacks=[
        EarlyStopping(patience=es_patience, restore_best_weights=True)
    ])

# Evaluation
test_results = model.evaluate([X]+A_list, y, sample_weight=test_mask, batch_size=N)
print('Test loss: {:.3f}'.format(test_results[0]))
print('Test accuracy: {:.3f}'.format(test_results[1]))
