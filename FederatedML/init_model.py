import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model_structure import save_model_structure, get_model_structure_by_path

path = "/home/chirag/fl/keras-gnn/cora_partition_metis/FederatedML"


gnn_model = save_model_structure(path + "/saved_structure")

gnn_model.save_weights(path+'/init_model_0.h5')

# gnn_model.save(path+'/init_model_0.h5', save_format='tf')
