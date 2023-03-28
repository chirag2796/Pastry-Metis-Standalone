import os
import sys
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import metis
import pickle



def create_graph(citations):
    G = nx.DiGraph()
    G.add_edges_from(citations[["source", "target"]].values)
    return G

def metis_partition(G, num_partitions):
    edge_cuts, parts = metis.part_graph(G, num_partitions)
    partitioned_data = [[] for _ in range(num_partitions)]
    for idx, part in enumerate(parts):
        partitioned_data[part].append(idx)
    return partitioned_data

def get_partitioned_data(partitioned_data, papers):
    result = []
    for partition in partitioned_data:
        partition_papers = papers[papers['paper_id'].isin(partition)]
        result.append(partition_papers)
    return result




if __name__ == "__main__":
    print("Data Partitioning begins...")
    
    path = "/home/chirag/fl/keras-gnn/cora_partition_metis/FederatedML"

    print(sys.argv[0])
    print(sys.argv[1])
    

    # devices_relative_resources = {
    #     "device_" + str(sys.argv[1]): float(sys.argv[2]),
    #     "device_" + str(sys.argv[3]): float(sys.argv[4]),
    # }

    

    arguments = sys.argv[1].split()
    arguments = list(map(lambda s:s.replace('"',''), arguments))

    device_names = [arguments[i] for i in range(0, len(arguments), 2)]
    

    print("Device Names: ", device_names)


    zip_file = keras.utils.get_file(
        fname="cora.tgz",
        origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
        extract=True,
    )
    data_dir = os.path.join(os.path.dirname(zip_file), "cora")


    citations = pd.read_csv(
        os.path.join(data_dir, "cora.cites"),
        sep="\t",
        header=None,
        names=["target", "source"],
    )

    citations.sample(frac=1).head()



    column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
    papers = pd.read_csv(
        os.path.join(data_dir, "cora.content"), sep="\t", header=None, names=column_names,
    )
    print(papers.sample(5).T)


    class_values = sorted(papers["subject"].unique())
    class_idx = {name: id for id, name in enumerate(class_values)}
    paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

    papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
    citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
    citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
    papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])





    # Create a graph from the citation dataset
    G = create_graph(citations)

    # Number of partitions you want to create
    num_partitions = 5

    # Perform METIS partitioning
    partitioned_data = metis_partition(G, num_partitions)

    # Get the partitioned datasets
    partitioned_datasets = get_partitioned_data(partitioned_data, papers)

    # Example usage of partitioned datasets for training
    for i, partitioned_data in enumerate(partitioned_datasets):
        print(f"Training model on partition {i+1}/{num_partitions}")
        train_partition_data, test_partition_data = [], []
        for _, group_data in partitioned_data.groupby("subject"):
            random_selection = np.random.rand(len(group_data.index)) <= 0.5
            train_partition_data.append(group_data[random_selection])
            test_partition_data.append(group_data[~random_selection])

        train_partition_data = pd.concat(train_partition_data).sample(frac=1)
        test_partition_data = pd.concat(test_partition_data).sample(frac=1)

        print(train_partition_data.shape)
        print(type(train_partition_data))


        hidden_units = [32, 32]
        learning_rate = 0.01
        dropout_rate = 0.5
        num_epochs = 20
        batch_size = 256

        feature_names = set(papers.columns) - {"paper_id", "subject"}
        num_features = len(feature_names)
        num_classes = len(class_idx)

        # Create train and test features as a numpy array.
        x_train = train_partition_data[feature_names].to_numpy()
        x_test = test_partition_data[feature_names].to_numpy()
        # Create train and test targets as a numpy array.
        y_train = train_partition_data["subject"]
        y_test = test_partition_data["subject"]
        # Data building end

        x_train = train_partition_data.paper_id.to_numpy()


        graph_partition = {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        print("x_train: ", graph_partition['x_train'].shape)
        # print("x_test: ", x_test.shape)
        print("y_train: ", graph_partition['y_train'].shape)
        # print("y_test: ", y_test.shape)

        with open("{}/data_partitions/device_{}_data_partition.pickle".format(path, device_names[i]), 'wb') as handle:
            pickle.dump(graph_partition, handle, protocol=pickle.HIGHEST_PROTOCOL)


