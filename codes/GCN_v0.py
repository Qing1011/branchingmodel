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

# load pickle module
import pickle
import networkx as nx
# from tqdm import tqdm
import sys
import h5py


class GCN(torch.nn.Module):  # the simpliest model that GNN and it is classical, used as baseline
    def __init__(self, num_node_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 16)
        self.conv4 = GCNConv(16, 8)
        self.fc = torch.nn.Linear(8, 1)

    """
        hyperparameters:
        - number of hidden layers
        - number of hidden channels
        - dropout rate (now it's zero)
        - learning rate <- most important to tune
        - weight decay
        - etc etc.
    """

    def forward(self, data):
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_weight)
        x = F.elu(x)

        x = self.conv2(x, edge_index, edge_weight)
        x = F.elu(x)

        x = self.conv3(x, edge_index, edge_weight)
        x = F.elu(x)

        x = self.conv4(x, edge_index, edge_weight)
        x = F.elu(x)

        x = global_mean_pool(x, batch)
        x = self.fc(x)

        return x


class GCN_ng(torch.nn.Module):  # the simpliest model that GNN and it is classical, used as baseline
    def __init__(self, num_node_features):
        super(GCN_ng, self).__init__()
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 16)
        # self.conv4 = GCNConv(16, 8)
        self.fc = torch.nn.Linear(16, 1)

    """
        hyperparameters:
        - number of hidden layers
        - number of hidden channels
        - dropout rate (now it's zero)
        - learning rate <- most important to tune
        - weight decay
        - etc etc.
    """

    def forward(self, data):
        # x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x, batch = data.x, data.batch
        edge_index = torch.Tensor([[], []]).to(x.device).long()

        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = self.conv3(x, edge_index)
        x = F.elu(x)

        # x = self.conv4(x, edge_index)
        # x = F.elu(x)

        x = global_mean_pool(x, batch)
        x = self.fc(x)

        return x


class OLP(torch.nn.Module):
    """
       one_layer_perceptron
    """

    def __init__(self, num_node_features):
        super(GCN_ng, self).__init__()
        self.fc = torch.nn.Linear(num_node_features, 1)

    def forward(self, data):
        # x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x, batch = data.x, data.batch
        edge_index = torch.Tensor([[], []]).to(x.device).long()

        x = global_mean_pool(x, batch)
        x = self.fc(x)

        return x


def train(model, myloader, optimizer, device):
    model.train()
    loss_all = 0
    correct = 0
    y_true = []
    total = 0
    results = []
    for data in myloader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        y_true.append(label)
        loss = F.mse_loss(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()

        optimizer.step()
        results.append(output)

    return loss_all / len(myloader.dataset), results, y_true


def validate(model, loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # No need to track gradients during validation
        for data in loader:
            data = data.to(device)
            out = model(data)
            # Assume you have a loss function defined, e.g., MSE for regression
            loss = F.mse_loss(out, data.y)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def test(model, testloader, device):
    model.eval()
    predictions = []
    total_loss = 0
    true_values = []
    with torch.no_grad():
        for data in testloader:
            data = data.to(device)
            output = model(data)
            loss = F.mse_loss(output, data.y)
            total_loss += loss.item() * data.num_graphs
            predictions.append(output.cpu())
            true_values.append(data.y.cpu())
    return torch.cat(predictions, dim=0), torch.cat(true_values, dim=0), total_loss / len(testloader.dataset)
