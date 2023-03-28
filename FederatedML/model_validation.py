from model_structure import get_model_structure_by_path

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle

path = "/home/chirag/fl/keras-gnn/cora_partition_metis/FederatedML"
# data_origin = "NEW"
# data_origin = "SAVED"


def initialize(data_origin="NEW"):
    if data_origin == "SAVED":
        device_id = "device_full"
        print("Loading: ", "data_partitions/{}_data_partition.pickle".format(device_id))
        with open(path+"/data_partitions/{}_data_partition.pickle".format(device_id), 'rb') as handle:
            graph_partition = pickle.load(handle)

        x_train = graph_partition["x_train"]
        x_test = graph_partition["x_test"]
        y_train = graph_partition["y_train"]
        y_test = graph_partition["y_test"]
    return x_test, y_test



def evaluate(model, data_origin="NEW"):
    print("Now evaluating")
    
    x_test, y_test = initialize(data_origin)

    test_loss, test_accuracy = model.evaluate(x=x_test, y=y_test, verbose=1)
    print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
    return test_loss, test_accuracy

if __name__=="__main__":
    # initialize(data_origin="SAVED")

    # gnn_model = get_model_structure()
    # # gnn_model.load_weights("/home/chirag/fl/keras-gnn/cora_partition_metis/FederatedML/Received_Models/new_model_<0x9F8BDA..>_0.h5")
    # gnn_model.load_weights("/home/chirag/fl/keras-gnn/cora_partition_metis/FederatedML/new_model_<0x39607A..>_0.h5")
    # evaluate(gnn_model, data_origin="SAVED")
    path = "/home/chirag/fl/keras-gnn/cora_partition_metis/FederatedML"


    new_model_loaded = get_model_structure_by_path(path + "/saved_structure")
    new_model_loaded.load_weights("/home/chirag/fl/keras-gnn/cora_partition_metis/FederatedML/new_model_<0xFCE0E5..>_1.h5")
    evaluate(new_model_loaded, data_origin="SAVED")