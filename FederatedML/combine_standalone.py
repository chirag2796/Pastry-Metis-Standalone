#!/usr/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from os import walk
from tensorflow import keras
from numpy import array, average
from keras.models import clone_model
from model_structure import get_model_structure
from model_validation import evaluate
import warnings
import sys
import time
warnings.filterwarnings("ignore")

hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256

import copy

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
			model = get_model_structure()
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

        print(layer_weights)

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
	
    model = get_model_structure()
    model.set_weights(avg_model_weights)

    return model


if __name__ == '__main__':
	print("Model Combining begins...")
	# path = "/Users/taehwan/Desktop/Research/pastry_amazon/src/rice/tutorial/FederatedML_Amazon"
	# path = "/home/ec2-user/FederatedML"
	path = "/home/chirag/fl/keras-gnn/cora_partition_metis/FederatedML"
	all_models = load_all_models(path)
	# prepare an array of equal weights
	n_models = len(all_models)
	weights = [1/n_models for i in range(1, n_models+1)]
	# combine models
	combined_model = combine_models(all_models, weights)

	evaluate(combined_model)

	# save model
	# combined_model.save(path + '/init_model_'+str(sys.argv[1])+'.h5')
	# combined_model.save_weights(path + '/init_model_'+str(sys.argv[1])+'.h5')
	print("Model Combining successfully completes...")


	