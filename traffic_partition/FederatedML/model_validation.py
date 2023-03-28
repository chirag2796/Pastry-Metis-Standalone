from model_structure import get_model_structure

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
import pickle

path = "/home/chirag/fl/keras-gnn/traffic_partition/FederatedML"
#
def create_tf_dataset(
    data_array: np.ndarray,
    input_sequence_length: int,
    forecast_horizon: int,
    batch_size: int = 128,
    shuffle=True,
    multi_horizon=True,
):

    inputs = timeseries_dataset_from_array(
        np.expand_dims(data_array[:-forecast_horizon], axis=-1),
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )

    target_offset = (
        input_sequence_length
        if multi_horizon
        else input_sequence_length + forecast_horizon - 1
    )
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = timeseries_dataset_from_array(
        data_array[target_offset:],
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset.prefetch(16).cache()




def initialize():
    device_id = "device_full"
    with open(path+"/data_partitions/{}_data_partition.pickle".format(device_id), 'rb') as handle:
        graph_partition = pickle.load(handle)

    test_array = graph_partition["test_array"]
    # print(f"test set size: {test_array.shape}")

    batch_size = 64
    input_sequence_length = 12
    forecast_horizon = 3
    multi_horizon = False


    test_dataset = create_tf_dataset(
        test_array,
        input_sequence_length,
        forecast_horizon,
        batch_size=test_array.shape[0],
        shuffle=False,
        multi_horizon=multi_horizon,
    )

    return test_dataset



def evaluate(model, test_dataset=None):
    print("Now evaluating")
    
    if test_dataset is None:
        test_dataset = initialize()

    test_loss, test_accuracy = model.evaluate(test_dataset)

    # Print the test accuracy
    print(f"Test loss: {test_loss}")
    print(f"Test accuracy: {test_accuracy}")
    print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
    return test_loss, test_accuracy


if __name__=="__main__":
    path = "/home/chirag/fl/keras-gnn/traffic_partition/FederatedML"


    new_model_loaded = get_model_structure(path)
    new_model_loaded.load_weights("/home/chirag/fl/keras-gnn/traffic_partition/FederatedML/new_model_a_0.h5")
    evaluate(new_model_loaded)