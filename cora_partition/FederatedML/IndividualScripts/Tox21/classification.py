import tensorflow as tf
import numpy as np
import deepchem as dc
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Layer

# Load the Tox21 dataset
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = tox21_datasets

# Create the input layer
inputs = Input(shape=(train_dataset.X.shape[1],))


# # Create the input layer
# inputs = Input(shape=(train_dataset.X.shape[0], 1))

# # Reshape the input data to a 2D array
# train_X = train_dataset.X.reshape((-1, 1))




# Add a dense layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(inputs)

# Add a dropout layer with a rate of 0.5
x = Dropout(0.5)(x)

# Add another dense layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(x)

# Add a dropout layer with a rate of 0.5
x = Dropout(0.5)(x)

# Add the output layer with a sigmoid activation
outputs = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model for 50 epochs
model.fit(train_dataset.X, train_dataset.y, epochs=3, batch_size=32, validation_data=(valid_dataset.X, valid_dataset.y))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_dataset.X, test_dataset.y)
print('Test Loss: {:.4f}'.format(test_loss))
print('Test Accuracy: {:.4f}'.format(test_acc))
