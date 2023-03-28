#!/bin/bash
# LOG_LOCATION="/Users/taehwan/Desktop/Research/pastry_amazon/src/rice/tutorial/FederatedML_Amazon/LOGS"
# LOG_LOCATION="/home/ec2-user/FederatedML/LOGS"
LOG_LOCATION="/home/chirag/fl/keras-gnn/traffic_partition/FederatedML/LOGS"
exec >> $LOG_LOCATION/combine_model.txt 2>&1
# source "/Users/taehwan/Desktop/Research/pastry_amazon/src/rice/tutorial/FederatedML_Amazon/headnode/bin/activate"
# python /Users/taehwan/Desktop/Research/pastry_amazon/src/rice/tutorial/FederatedML_Amazon/combine.py $1
# rm -v /Users/taehwan/Desktop/Research/pastry_amazon/src/rice/tutorial/FederatedML_Amazon/Received_Models/*

# source "/home/ec2-user/FederatedML/headnode/bin/activate"
# python /home/ec2-user/FederatedML/combine.py $1
# rm -v /home/ec2-user/FederatedML/Received_Models/*

source "/home/chirag/fl/keras-gnn/keras-gnn-env/bin/activate"
python /home/chirag/fl/keras-gnn/traffic_partition/FederatedML/combine.py $@
rm -v /home/chirag/fl/keras-gnn/traffic_partition/FederatedML/Received_Models/*