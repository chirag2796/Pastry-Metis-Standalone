#!/bin/bash
# LOG_LOCATION="/Users/taehwan/Desktop/Research/pastry_amazon/src/rice/tutorial/FederatedML_Amazon/LOGS"
# LOG_LOCATION="/home/ec2-user/FederatedML/LOGS"
LOG_LOCATION="/home/chirag/fl/keras-gnn/cora_partition_metis/FederatedML/LOGS"
exec >> $LOG_LOCATION/train_model.txt 2>&1
# source "/Users/taehwan/Desktop/Research/pastry_amazon/src/rice/tutorial/FederatedML_Amazon/headnode/bin/activate"
# python /Users/taehwan/Desktop/Research/pastry_amazon/src/rice/tutorial/FederatedML_Amazon/train.py $1 $2
# source "/home/ec2-user/FederatedML/headnode/bin/activate"
# python /home/ec2-user/FederatedML/train.py $1 $2

source "/home/chirag/fl/keras-gnn/keras-gnn-env/bin/activate"
python /home/chirag/fl/keras-gnn/cora_partition_metis/FederatedML/train.py $1 $2