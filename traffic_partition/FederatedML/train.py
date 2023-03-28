#!/usr/bin/python
import os
from model_structure import get_model_structure

from model_validation import evaluate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
import sys
import warnings
warnings.filterwarnings("ignore")

import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

import pickle

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




if __name__ == "__main__":
    print("Training for "+str(sys.argv[1])+ " begins...")
    # path = "/Users/taehwan/Desktop/Research/pastry_amazon/src/rice/tutorial/FederatedML_Amazon"

    device_id = "device_" + str(sys.argv[1])
    
    path = "/home/chirag/fl/keras-gnn/traffic_partition/FederatedML"
    
    

    print("1. Init Model Loaded")    

    # device_id = "device_1"
    # device_id = "device_full"
    print("Loading: ", "data_partitions/{}_data_partition.pickle".format(device_id))
    with open(path+"/data_partitions/{}_data_partition.pickle".format(device_id), 'rb') as handle:
        graph_partition = pickle.load(handle)

    train_array = graph_partition["train_array"]
    val_array = graph_partition["val_array"]
    test_array = graph_partition["test_array"]


    print(f"train set size: {train_array.shape}")
    print(f"val set size: {val_array.shape}")
    print(f"test set size: {test_array.shape}")

    batch_size = 128
    input_sequence_length = 12
    forecast_horizon = 3
    multi_horizon = False

    train_dataset, val_dataset = (
        create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size)
        for data_array in [train_array, val_array]
    )

    test_dataset = create_tf_dataset(
        test_array,
        input_sequence_length,
        forecast_horizon,
        batch_size=test_array.shape[0],
        shuffle=False,
        multi_horizon=multi_horizon,
    )

    gnn_model = get_model_structure(path)

    # load the model
    # gnn_model = keras.models.load_model(path+'/init_model_'+str(sys.argv[1])+'_'+str(sys.argv[2]))
    gnn_model.load_weights(path+'/init_model_'+str(sys.argv[1])+'_'+str(sys.argv[2]) + ".h5")


    in_feat = 1
    batch_size = 128
    epochs = 20
    input_sequence_length = 12
    forecast_horizon = 3
    multi_horizon = False
    out_feat = 10
    lstm_units = 32
    graph_conv_params = {
        "aggregation_type": "mean",
        "combination_type": "concat",
        "activation": None,
    }


    gnn_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=1,
        batch_size = batch_size
        # callbacks=[keras.callbacks.EarlyStopping(patience=10)],
    )

    print("6. Model Trained")

    # evaluate(gnn_model, test_dataset=test_dataset)

    # Evaluate the model on the test set
    # test_loss, test_accuracy = gnn_model.evaluate(test_dataset)

    # # Print the test accuracy
    # print(f"Test loss: {test_loss}")
    # print(f"Test accuracy: {test_accuracy}")

    
    
    
    # new_model.save(path+'/new_model_'+str(sys.argv[1])+'_'+str(sys.argv[2])+'.h5')
    gnn_model.save_weights(path+'/new_model_'+str(sys.argv[1])+'_'+str(sys.argv[2]) + ".h5")

    print("7. Model Saved as" + path+'/new_model_'+str(sys.argv[1])+'_'+str(sys.argv[2]) + ".h5")


    print("Training for "+str(sys.argv[1])+ " ends...")