#!/bin/bash
LOG_LOCATION="/home/chirag/fl/keras-gnn/traffic_partition/FederatedML/LOGS"
exec >> $LOG_LOCATION/partition_data.txt 2>&1

source "/home/chirag/fl/keras-gnn/keras-gnn-env/bin/activate"
python /home/chirag/fl/keras-gnn/traffic_partition/FederatedML/partition_data.py "$*"