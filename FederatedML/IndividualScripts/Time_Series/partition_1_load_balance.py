import pickle
import sys
import pandas as pd
import numpy as np
import os
import typing
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras


data_dir = "PeMSD7_Full"

route_distances = pd.read_csv(
    os.path.join(data_dir, "PeMSD7_W_228.csv"), header=None
).to_numpy()
speeds_array = pd.read_csv(os.path.join(data_dir, "PeMSD7_V_228.csv"), header=None).to_numpy()


print(f"route_distances shape={route_distances.shape}")
print(f"speeds_array shape={speeds_array.shape}")

train_size, val_size = 0.7, 0.1

def preprocess(data_array: np.ndarray, train_size: float, val_size: float):
    num_time_steps = data_array.shape[0]
    num_train, num_val = (
        int(num_time_steps * train_size),
        int(num_time_steps * val_size),
    )
    train_array = data_array[:num_train]
    mean, std = train_array.mean(axis=0), train_array.std(axis=0)

    train_array = (train_array - mean) / std
    val_array = (data_array[num_train : (num_train + num_val)] - mean) / std
    test_array = (data_array[(num_train + num_val) :] - mean) / std

    return train_array, val_array, test_array

train_array, val_array, test_array = preprocess(speeds_array, train_size, val_size)

print(f"train set size: {train_array.shape}")
print(f"validation set size: {val_array.shape}")
print(f"test set size: {test_array.shape}")

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

if __name__ == "__main__":
    print("Data Partitioning begins...")
    
    path = "/home/chirag/fl/keras-gnn/cora_partition_metis/FederatedML/IndividualScripts/Time_Series"

    print(sys.argv[0])
    print(sys.argv[1])
    

    # devices_relative_resources = {
    #     "device_" + str(sys.argv[1]): float(sys.argv[2]),
    #     "device_" + str(sys.argv[3]): float(sys.argv[4]),
    # }

    devices_relative_resources = {}

    arguments = sys.argv[1].split()
    arguments = list(map(lambda s:s.replace('"',''), arguments))


    for i in range(0, len(arguments)-1, 2):
        devices_relative_resources["device_{}".format(arguments[i])] = float(arguments[i+1])

    print(devices_relative_resources)

    partitioned_train_array = partition_graph(devices_relative_resources, train_array)
    # partitioned_x_test = partition_graph(devices_relative_resources, x_test)
    # partitioned_y_train = partition_graph(devices_relative_resources, y_train)
    # partitioned_y_test = partition_graph(devices_relative_resources, y_test)

    # print(partitioned_x_train)
    # exit()

    for device_id in devices_relative_resources:
        print(device_id, len(partitioned_train_array[device_id]))
        graph_partition = {
            "train_array": partitioned_train_array[device_id],
            # "x_test": partitioned_x_test[device_id],
            # "y_train": partitioned_y_train[device_id],
            # "y_test": partitioned_y_test[device_id],
        }
        print("\n", device_id)
        print("train_array: ", graph_partition['train_array'].shape)
        # print("x_test: ", x_test.shape)
        # print("y_train: ", graph_partition['y_train'].shape)
        # print("y_test: ", y_test.shape)


        # print(graph_partition)
        with open(path + "/data_partitions/{}_data_partition.pickle".format(device_id), 'wb') as handle:
            pickle.dump(graph_partition, handle, protocol=pickle.HIGHEST_PROTOCOL)