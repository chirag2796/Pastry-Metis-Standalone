#!/bin/bash
python3 -m venv headnode
source headnode/bin/activate
pip3 install --no-cache-dir tensorflow
pip install numpy
pip install pandas
pip install texthero
