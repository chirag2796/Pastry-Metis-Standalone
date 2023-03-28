import tensorflow as tf
import numpy as np
import deepchem as dc
from deepchem.feat import RDKitDescriptors
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Layer

# Load the Tox21 dataset
tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = tox21_datasets

# Filter out non-drug-like molecules
featurizer = RDKitDescriptors()
train_dataset = dc.data.DiskDataset.from_numpy(train_dataset.X, train_dataset.y, train_dataset.w, train_dataset.ids)
train_dataset = train_dataset.subset(lambda x: featurizer.featurize(x[0])[0][0] < 1000)

valid_dataset = dc.data.DiskDataset.from_numpy(valid_dataset.X, valid_dataset.y, valid_dataset.w, valid_dataset.ids)
valid_dataset = valid_dataset.subset(lambda x: featurizer.featurize(x[0])[0][0] < 1000)

test_dataset = dc.data.DiskDataset.from_numpy(test_dataset.X, test_dataset.y, test_dataset.w, test_dataset.ids)
test_dataset = test_dataset.subset(lambda x: featurizer.featurize(x[0])[0][0] < 1000)

train_features = featurizer.featurize(train_dataset.X)
valid_features = featurizer.featurize(valid_dataset.X)
test_features = featurizer.featurize(test_dataset.X)

# Create the input layer
inputs = Input(shape=(train_features.shape[1],))

# Add a dense layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(inputs)

# Add a dropout layer with a rate of 0.5
x = Dropout(0.5)(x)

# Add another dense layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(x)

# Add a dropout layer with a rate of 0.5
x = Dropout(0.5)(x)

# Add the output layer with a softmax activation
outputs = Dense(12, activation='softmax')(x)

# Create the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model with categorical crossentropy loss and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Convert the target variable to a one-hot encoded format
train_targets = np.eye(12)[train_dataset.y]

# Train the model for 50 epochs
model.fit(train_features, train_targets, epochs=50, batch_size=32, validation_data=(valid_features, np.eye(12)[valid_dataset.y]))

# Evaluate the model on the test set
test_targets = np.eye(12)[test_dataset.y]
test_loss, test_acc = model.evaluate(test_features, test_targets)
print('Test Loss: {:.4f}'.format(test_loss))
print('Test Accuracy: {:.4f}'.format(test_acc))
