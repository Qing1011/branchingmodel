#!/bin/bash

source activate gnn
nohup python3 GNN_Regression.py 0 6 > output.log 2>&1 &