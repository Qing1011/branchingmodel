import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
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
from GCN import *


def main():

    # datafolder = '/Users/qingyao/Documents/branching_data/gnn_regression/'

    num_x = int(sys.argv[2])
    layer = int(sys.argv[3])
    R0 = sys.argv[4]
    with_g = int(sys.argv[5])
    datafolder = '/rds/general/user/qy1815/ephemeral/branching_simulation_data_{}/'.format(
        R0)

    save_dir = '/rds/general/user/qy1815/home/branching_superspreading/regression_{}_layer_{}_{}_g-{}/'.format(
        num_x, layer, R0, with_g)

    job_idx = int(sys.argv[1])-1  # ensemble/parameter index
    s = job_idx//70
    es_idx = job_idx % 70  # ensemble/parameter index
    s = int(s)  # seed index

    # load data
    # WN = np.loadtxt('W_avg.csv')
    dataset = torch.load(datafolder+'dataset_{}.pt'.format(num_x))
    lr_list = [np.power(0.5, i) for i in range(2, 16, 2)]*10
    my_lr = lr_list[es_idx]

    torch.manual_seed(s)
    all_data_len = len(dataset)
    train_size = int(all_data_len * 0.6)
    val_size = int(all_data_len * 0.2)
    test_size = all_data_len - train_size - val_size

    train_data, val_data, test_data = random_split(
        dataset, [train_size, val_size, test_size])

    # Now we can create a DataLoader
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
    # Create a model and an optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = GCN_ng(num_node_features=num_x, hidden_channels=[
    #                128, 64, 16, 8], num_hlayers=layer).to(device)
    model = MLP(num_node_features=num_x, hidden_channels=[
        128, 64, 16, 8], num_hlayers=layer).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=my_lr, weight_decay=5e-4)

    # training and validation
    counter = 0
    count_epochs = 0
    best = float("inf")
    epochs = 200
    patience = 15
    loss_ep = []

    seed = int(time.time()) + es_idx
    # random.seed(seed)
    torch.manual_seed(seed)

    for epoch in range(epochs):
        loss, myres, reals = train(model, train_loader, optimizer, device)
        # loss_ep.append(loss)
        val_loss = validate(model, val_loader, device)
        loss_ep.append(val_loss)
        if val_loss < best:
            best = val_loss
            counter = 0
            # Save the best model
            torch.save(model.state_dict(
            ), save_dir+'best_model_{}_{}.pth'.format(s, es_idx))
        else:
            counter += 1
            count_epochs += 1

        if counter > patience:
            break

    # testing
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True)
    predictions, y_true, test_mse = test(model, test_loader, device)

    predictions_np = predictions.cpu().detach().numpy()
    y_true_np = y_true.cpu().detach().numpy()

    with h5py.File(save_dir+'res_{}_{}.hdf5'.format(s, es_idx), 'w') as f:
        # f['val_mse'] = val_loss
        f.create_dataset('val_mse', data=np.array(loss_ep))
        f.create_dataset('predictions', data=predictions_np)
        f.create_dataset('y_true', data=y_true_np)
        f['test_mse'] = test_mse


if __name__ == "__main__":
    main()
