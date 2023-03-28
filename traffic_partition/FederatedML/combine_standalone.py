#!/usr/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from os import walk
from tensorflow import keras
from numpy import array, average
from keras.models import clone_model
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from model_structure import get_model_structure
# from model_validation import evaluate
import numpy as np
import tensorflow as tf
import warnings
import sys
import time, pickle
warnings.filterwarnings("ignore")

hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256

import copy

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



# load all models from folder
def load_all_models(path):
	mypath = path + "/Received_Models"
	filenames = next(walk(mypath), (None, None, []))[2]  # [] if no file
	all_models = []
	for file in filenames:
		print("Here: ", mypath+"/"+file)
		# print("Sleeping")
		# time.sleep(500)
		if file[0] != ".":
			model = get_model_structure(path)
			# model.built = True
			model.load_weights(mypath+"/"+file)
			# model = keras.models.load_model(mypath+"/"+file)
			
			# model_to_add = copy.copy(model)
			# all_models.append(model_to_add)
			all_models.append(model)
	return all_models

def combine_models(members, weights):
    # how many layers need to be averaged
    n_layers = len(members[0].get_weights())
    # create a set of average model weights
    avg_model_weights = list()
    for layer in range(n_layers):
        # collect this layer from each model
        layer_weights = array([model.get_weights()[layer] for model in members])

        # print(layer_weights)

        # weighted average of weights for this layer
        avg_layer_weights = average(layer_weights, axis=0, weights=weights)
        # store average layer weights
        avg_model_weights.append(avg_layer_weights)
	
	
    # create a new model with the same structure
    # model = clone_model(members[0])
    # model.set_weights(avg_model_weights)
    # # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate),
    #         loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #         metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])
	
    model = get_model_structure(path)
    model.set_weights(avg_model_weights)

    return model


if __name__ == '__main__':
	print("Model Combining begins...")
	# path = "/Users/taehwan/Desktop/Research/pastry_amazon/src/rice/tutorial/FederatedML_Amazon"
	# path = "/home/ec2-user/FederatedML"
	path = "/home/chirag/fl/keras-gnn/traffic_partition/FederatedML"
	all_models = load_all_models(path)
	# prepare an array of equal weights
	n_models = len(all_models)
	weights = [1/n_models for i in range(1, n_models+1)]
	# combine models
	combined_model = combine_models(all_models, weights)


	device_id = "device_" + str(sys.argv[1])
	with open(path+"/data_partitions/{}_data_partition.pickle".format(device_id), 'rb') as handle:
		graph_partition = pickle.load(handle)

	
	test_array = graph_partition["test_array"]

	input_sequence_length = 12
	forecast_horizon = 3
	multi_horizon = False

	# evaluate(combined_model)
	test_dataset = create_tf_dataset(
        test_array,
        input_sequence_length,
        forecast_horizon,
        batch_size=test_array.shape[0],
        shuffle=False,
        multi_horizon=multi_horizon,
    )
	test_loss, test_accuracy = combined_model.evaluate(test_dataset)

    # Print the test accuracy
	print(f"Test loss: {test_loss}")
	print(f"Test accuracy: {test_accuracy}")

	# save model
	# combined_model.save(path + '/init_model_'+str(sys.argv[1])+'.h5')
	# combined_model.save_weights(path + '/init_model_'+str(sys.argv[1])+'.h5')
	print("Model Combining successfully completes...")


	