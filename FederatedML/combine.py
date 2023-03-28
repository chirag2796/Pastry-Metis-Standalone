#!/usr/bin/python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from os import walk
from tensorflow import keras
from numpy import array, average
from keras.models import clone_model
from model_structure import get_model_structure_by_path
from model_validation import evaluate
import warnings
import sys
import time
import re
import csv
warnings.filterwarnings("ignore")

hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 300
batch_size = 256

# load all models from folder
def load_all_models(path, devices_relative_resources):
	mypath = path + "/Received_Models"
	filenames = next(walk(mypath), (None, None, []))[2]  # [] if no file
	all_models = []
	relative_weighting = []
	for file in filenames:
		print("Here: ", mypath+"/"+file)

		node_id = re.match(r"model\_(.*)\_", file).group(1)

		# print("Sleeping")
		# time.sleep(500)
		if file[0] != ".":
			# model = get_model_structure()
			model = get_model_structure_by_path(path + "/saved_structure")
			# model.built = True


			model.load_weights(mypath+"/"+file)
			# model = keras.models.load_model(mypath+"/"+file)
			all_models.append(model)
			relative_weighting.append(devices_relative_resources[node_id])
			print(file, devices_relative_resources[node_id])
	return all_models, relative_weighting

def combine_models(members, weights):
    # how many layers need to be averaged
    n_layers = len(members[0].get_weights())
    # create a set of average model weights
    avg_model_weights = list()
    for layer in range(n_layers):
        # collect this layer from each model
        layer_weights = array([model.get_weights()[layer] for model in members])
        # weighted average of weights for this layer
        # avg_layer_weights = average(layer_weights, axis=0, weights=weights)
        avg_layer_weights = average(layer_weights, axis=0)
        # store average layer weights
        avg_model_weights.append(avg_layer_weights)
	
	
    # create a new model with the same structure
    # model = clone_model(members[0])
    # model.set_weights(avg_model_weights)
    # # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.compile(optimizer=keras.optimizers.Adam(learning_rate),
    #         loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #         metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])
	
    # model = get_model_structure()

    model = get_model_structure_by_path(path + "/saved_structure")
    model.set_weights(avg_model_weights)

    return model


if __name__ == '__main__':
	print("Model Combining begins...")
	print(sys.argv)

	iteration = sys.argv[1]

	devices_relative_resources = {}

	arguments = sys.argv[2:]

	for i in range(0, len(arguments)-1, 2):
		devices_relative_resources["{}".format(arguments[i])] = float(arguments[i+1])

	print(devices_relative_resources)

	# path = "/Users/taehwan/Desktop/Research/pastry_amazon/src/rice/tutorial/FederatedML_Amazon"
	# path = "/home/ec2-user/FederatedML"
	path = "/home/chirag/fl/keras-gnn/cora_partition_metis/FederatedML"
	all_models, relative_weighting = load_all_models(path, devices_relative_resources)
	# prepare an array of equal weights
	n_models = len(all_models)
	# weights = [1/n_models for i in range(1, n_models+1)]
	# combine models
	print("Relative weights: ", relative_weighting)

	combined_model = combine_models(all_models, relative_weighting)

	test_loss, test_accuracy = evaluate(combined_model, data_origin="SAVED")

	# save model
	# combined_model.save(path + '/init_model_'+str(sys.argv[1])+'.h5')
	combined_model.save_weights(path + '/init_model_'+str(sys.argv[1])+'.h5')

	eval_filename = "metrics.csv"
	if int(iteration)==0:
		with open(eval_filename, "w", newline="") as csvfile:
			csvwriter = csv.writer(csvfile)
			csvwriter.writerow([iteration, str(test_loss), str(test_accuracy)])
	else:
		with open(eval_filename, "a", newline="") as csvfile:
			csvwriter = csv.writer(csvfile)
			csvwriter.writerow([iteration, str(test_loss), str(test_accuracy)])

	print("Model Combining successfully completes...")


	