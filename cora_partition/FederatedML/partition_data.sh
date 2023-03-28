#!/bin/bash
LOG_LOCATION="/home/chirag/fl/keras-gnn/cora_partition/FederatedML/LOGS"
exec >> $LOG_LOCATION/partition_data.txt 2>&1

source "/home/chirag/fl/keras-gnn/keras-gnn-env/bin/activate"
# python /home/chirag/fl/keras-gnn/cora_partition/FederatedML/partition_data.py $1 $2 $3 $4
python /home/chirag/fl/keras-gnn/cora_partition/FederatedML/partition_data.py "$*"