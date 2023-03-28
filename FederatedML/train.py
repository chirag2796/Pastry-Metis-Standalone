#!/usr/bin/python
import os
from model_structure import get_model_structure_by_path

from model_validation import evaluate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
import sys
# import texthero as hero
# from tensorflow import keras
# from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")

import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pickle

hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 1
batch_size = 128


def run_experiment(model, x_train, y_train):
    # Compile the model.
    # model.compile(
    #     optimizer=keras.optimizers.Adam(learning_rate),
    #     loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #     metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    # )
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

    return history, model

# vectorize sentences embeddings
def vectorize(sentence, model, dimension=50):
    ans = [0]*dimension
    for word in sentence:
        if word in model:
            ans += model[word]
    return ans

# train model
def train_model(model, x_train, y_train):
    history, model = run_experiment(model, x_train, y_train)
    return model

def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)



if __name__ == "__main__":
    print("Training for "+str(sys.argv[1])+ " begins...")
    # path = "/Users/taehwan/Desktop/Research/pastry_amazon/src/rice/tutorial/FederatedML_Amazon"

    device_id = "device_" + str(sys.argv[1])
    
    path = "/home/chirag/fl/keras-gnn/cora_partition_metis/FederatedML"
    
    gnn_model = get_model_structure_by_path(path + "/saved_structure")

    # load the model
    # gnn_model = keras.models.load_model(path+'/init_model_'+str(sys.argv[1])+'_'+str(sys.argv[2]))
    gnn_model.load_weights(path+'/init_model_'+str(sys.argv[1])+'_'+str(sys.argv[2]) + ".h5")

    print("1. Init Model Loaded")    

    # device_id = "device_1"
    # device_id = "device_full"
    print("Loading: ", "data_partitions/{}_data_partition.pickle".format(device_id))
    with open(path+"/data_partitions/{}_data_partition.pickle".format(device_id), 'rb') as handle:
        graph_partition = pickle.load(handle)

    x_train = graph_partition["x_train"]
    x_test = graph_partition["x_test"]
    y_train = graph_partition["y_train"]
    y_test = graph_partition["y_test"]
    
    # x_train = train_data.paper_id.to_numpy()

    # print("x_train: ", x_train.shape)
    # print("x_test: ", x_test.shape)
    # print("y_train: ", y_train.shape)
    # print("y_test: ", y_test.shape)
    # print(type(x_train))
    # print(type(y_train))
    # # exit()


    new_model = train_model(model=gnn_model, x_train=x_train, y_train=y_train)
    
    print("6. Model Trained")

    evaluate(new_model, data_origin="SAVED")
    
    # new_model.save(path+'/new_model_'+str(sys.argv[1])+'_'+str(sys.argv[2])+'.h5')
    new_model.save_weights(path+'/new_model_'+str(sys.argv[1])+'_'+str(sys.argv[2]) + ".h5")

    print("7. Model Saved as" + path+'/new_model_'+str(sys.argv[1])+'_'+str(sys.argv[2]) + ".h5")


    print("Loading again 1")
    new_model_loaded = get_model_structure_by_path(path + "/saved_structure")
    new_model_loaded.load_weights(path+'/new_model_'+str(sys.argv[1])+'_'+str(sys.argv[2]) + ".h5")
    evaluate(new_model_loaded, data_origin="SAVED")
    
    print("Training for "+str(sys.argv[1])+ " ends...")