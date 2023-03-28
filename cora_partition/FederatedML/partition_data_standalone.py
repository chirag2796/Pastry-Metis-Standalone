import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pickle

path = "/home/chirag/fl/keras-gnn/cora_partition"



"""
This function compiles and trains an input model using the given training data.
"""


def run_experiment(model, x_train, y_train):
    # Compile the model.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
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

    return history


"""
This function displays the loss and accuracy curves of the model during training.
"""


def display_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "test"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()


"""
## Implement Feedforward Network (FFN) Module
We will use this module in the baseline and the GNN models.
"""


def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)


"""
## Build a Baseline Neural Network Model
### Prepare the data for the baseline model
"""

zip_file = keras.utils.get_file(
    fname="cora.tgz",
    origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
    extract=True,
)
data_dir = os.path.join(os.path.dirname(zip_file), "cora")
"""
### Process and visualize the dataset
Then we load the citations data into a Pandas DataFrame.
"""

citations = pd.read_csv(
    os.path.join(data_dir, "cora.cites"),
    sep="\t",
    header=None,
    names=["target", "source"],
)
print("Citations shape:", citations.shape)

"""
Now we display a sample of the `citations` DataFrame.
The `target` column includes the paper ids cited by the paper ids in the `source` column.
"""

citations.sample(frac=1).head()

"""
Now let's load the papers data into a Pandas DataFrame.
"""

column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
papers = pd.read_csv(
    os.path.join(data_dir, "cora.content"), sep="\t", header=None, names=column_names,
)
print("Papers shape:", papers.shape)

"""
Now we display a sample of the `papers` DataFrame. The DataFrame includes the `paper_id`
and the `subject` columns, as well as 1,433 binary column representing whether a term exists
in the paper or not.
"""

print(papers.sample(5).T)

"""
Let's display the count of the papers in each subject.
"""

print(papers.subject.value_counts())

"""
We convert the paper ids and the subjects into zero-based indices.
"""

class_values = sorted(papers["subject"].unique())
class_idx = {name: id for id, name in enumerate(class_values)}
paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])




"""
### Split the dataset into stratified train and test sets
"""

train_data, test_data = [], []

for _, group_data in papers.groupby("subject"):
    # Select around 50% of the dataset for training.
    random_selection = np.random.rand(len(group_data.index)) <= 0.5
    train_data.append(group_data[random_selection])
    test_data.append(group_data[~random_selection])

train_data = pd.concat(train_data).sample(frac=1)
test_data = pd.concat(test_data).sample(frac=1)

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

"""
## Implement Train and Evaluate Experiment
"""

hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 30
batch_size = 256

feature_names = set(papers.columns) - {"paper_id", "subject"}
num_features = len(feature_names)
num_classes = len(class_idx)

# Create train and test features as a numpy array.
x_train = train_data[feature_names].to_numpy()
x_test = test_data[feature_names].to_numpy()
# Create train and test targets as a numpy array.
y_train = train_data["subject"]
y_test = test_data["subject"]

x_train = train_data.paper_id.to_numpy()

print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)


full_graph_partition = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
    }
with open("data_partitions/{}_data_partition.pickle".format("device_full"), 'wb') as handle:
        pickle.dump(full_graph_partition, handle, protocol=pickle.HIGHEST_PROTOCOL)

devices_relative_resources = {
    "device_1": 0.42,
    "device_2": 0.35,
    "device_3": 0.23
}

def partition_graph(devices_relative_resources, arr):
    partioned_data = {}
    partition_indices = {}
    cur_index = 0
    for device_id, relative_resource in devices_relative_resources.items():
        partition_index = int(len(arr)*relative_resource)
        partition_index += cur_index
        cur_index = partition_index
        partition_indices[device_id] = partition_index
    
    # print(partition_indices)
    cur_index = 0
    for device_id in devices_relative_resources:
        # print(device_id, cur_index, partition_indices[device_id])
        partioned_data[device_id] = arr[cur_index: partition_indices[device_id]]
        cur_index = partition_indices[device_id]
    
    # partitions = np.split(arr, len(arr)*relative_resource)
    # partitoned_data[device_id] = partition
    return partioned_data

partitioned_x_train = partition_graph(devices_relative_resources, x_train)
partitioned_x_test = partition_graph(devices_relative_resources, x_test)
partitioned_y_train = partition_graph(devices_relative_resources, y_train)
partitioned_y_test = partition_graph(devices_relative_resources, y_test)

# print(partitioned_x_train)
# exit()

for device_id in devices_relative_resources:
    print(device_id, len(partitioned_x_train[device_id]))
    graph_partition = {
        "x_train": partitioned_x_train[device_id],
        "x_test": partitioned_x_test[device_id],
        "y_train": partitioned_y_train[device_id],
        "y_test": partitioned_y_test[device_id],
    }
    print("\n", device_id)
    print("x_train: ", graph_partition['x_train'].shape)
    # print("x_test: ", x_test.shape)
    print("y_train: ", graph_partition['y_train'].shape)
    # print("y_test: ", y_test.shape)


    # print(graph_partition)
    with open("data_partitions/{}_data_partition.pickle".format(device_id), 'wb') as handle:
        pickle.dump(graph_partition, handle, protocol=pickle.HIGHEST_PROTOCOL)