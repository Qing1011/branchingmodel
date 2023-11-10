#!/bin/bash

source activate gnn
nohup python3 SEIR_M_stocha_run.py > output.log 2>&1 &