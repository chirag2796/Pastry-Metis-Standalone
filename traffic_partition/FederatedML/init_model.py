from model_structure import save_model_structure

path = "/home/chirag/fl/keras-gnn/traffic_partition/FederatedML"


gnn_model = save_model_structure(path)

gnn_model.save_weights(path+'/init_model_0.h5')

# gnn_model.save(path+'/init_model_0.h5', save_format='tf')
