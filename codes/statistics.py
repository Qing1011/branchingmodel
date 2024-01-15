import torch
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch_geometric.utils.convert import from_scipy_sparse_matrix, from_networkx
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import torch.nn as nn


import random
import numpy as np
from scipy.io import loadmat, savemat
import pandas as pd
import scipy.special as SS
import scipy.stats as SSA
import copy
import math
from sklearn.model_selection import ParameterGrid
import os
import numpy.linalg as LA
import gzip
from scipy import sparse
from torch.utils.data import random_split
import time

# load pickle module
import pickle
import networkx as nx
# from tqdm import tqdm
import sys
import h5py
import scipy.stats as stats


def compare_data_objects(data1, data2, tolerance=1e-7):
    # Compare each attribute of the Data objects
    for key in data1.keys:
        if key in data2:
            if not torch.allclose(data1[key], data2[key], atol=tolerance):
                return False
        else:
            return False
    return True


def compare_datasets(dataset1, dataset2, tolerance=1e-7):
    if len(dataset1) != len(dataset2):
        return False

    for item1, item2 in zip(dataset1, dataset2):
        if not compare_data_objects(item1, item2, tolerance):
            return False

    return True


def bootstrap_median_confidence_interval(data, ci=95, n_bootstraps=10000):
    bootstrapped_medians = []
    n = len(data)

    for _ in range(n_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrapped_medians.append(np.median(sample))

    lower_bound = np.percentile(bootstrapped_medians, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrapped_medians, 100 - (100 - ci) / 2)
    medians = np.percentile(bootstrapped_medians, 50)

    return medians, lower_bound, upper_bound


def find_best_learning_rate_and_test_mse(file_dir, x_number, hlayer, slipt, lr_list, g=True):
    if g:
        sub_dir = 'regression_{}_layer_{}' .format(x_number, hlayer)
    else:
        sub_dir = 'regression_ng_{}_layer_{}' .format(x_number, hlayer)
#     print(sub_dir)
    RES = np.zeros((10, 7))
    for lr_idx in range(70):
        with h5py.File(file_dir+sub_dir+'/res_{}_{}.hdf5'.format(slipt, lr_idx), 'r') as f:
            lr_pos = lr_idx % 7
            run = lr_idx // 7
            val_loss = f['val_mse'][()]
            RES[run, lr_pos] = val_loss[-1]
#     print(RES)
    min_positions = np.argmin(RES, axis=1)
    index_counts = np.bincount(min_positions)
    most_frequent_index = np.argmax(index_counts)

    test_mse_best = np.infty
    test_mse_position = 0
    for lr_idx in range(most_frequent_index, 70, 7):
        with h5py.File(file_dir+sub_dir+'/res_{}_{}.hdf5'.format(slipt, lr_idx), 'r') as f:
            test_mse = f['test_mse'][()]
            if test_mse < test_mse_best:
                test_mse_best = test_mse
                test_mse_position = lr_idx

    return test_mse_best, lr_list[most_frequent_index], test_mse_position
