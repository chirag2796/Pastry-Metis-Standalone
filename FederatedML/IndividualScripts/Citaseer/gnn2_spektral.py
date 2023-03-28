import numpy as np
from keras.layers import Input, Dropout, Dense
from keras.models import Model
# from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from keras.layers import concatenate, Reshape, Flatten
from keras.layers import Lambda, Layer
from keras import backend as K
from keras.utils.vis_utils import plot_model
from spektral.layers import GraphConv

from spektral.datasets import citeseer
from spektral.utils import normalized_laplacian

# Load dataset
A, X, y, train_mask, val_mask, test_mask = citeseer.load_data()

# Parameters
N = X.shape[0]              # Number of nodes in the graph
F = X.shape[1]              # Number of node features
n_classes = y.shape[1]      # Number of classes
l2_reg = 5e-4               # Regularization rate for l2
learning_rate = 1e-2        # Learning rate for optimizer
epochs = 200                # Number of training epochs
batch_size = N              # Batch size
es_patience = 10            # Patience for early stopping

# Convert the data to a list of sparse matrices
A_list = [normalized_laplacian(adjacency).astype(np.float32) for adjacency in A]
X_list = [X.astype(np.float32)]

# Model definition
X_in = Input(shape=(F, ))
A_in = [Input((None, ), sparse=True) for _ in range(len(A_list))]

# Graph Convolutional Layers
H = Dropout(0.5)(X_in)
H = GraphConv(16, activation='relu', kernel_regularizer=l2(l2_reg))([H]+A_in)
H = Dropout(0.5)(H)
H = GraphConv(n_classes, activation='softmax')([H]+A_in)
output = Flatten()(H)

# Build model
model = Model(inputs=[X_in] + A_in, outputs=output)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', weighted_metrics=['acc'])

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
