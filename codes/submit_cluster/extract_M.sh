#!/bin/bash

source activate gnn
nohup python3 mobility_matrix.py > output.log 2>&1 &
