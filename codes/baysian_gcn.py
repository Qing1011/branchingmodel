import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GCNConv, global_mean_pool
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

@variational_estimator
class GCNBayesian(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCNBayesian, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 64)
        self.conv3 = GCNConv(64, 32)
        self.post_pool_norm = nn.LayerNorm(32)
        self.dropout = Dropout(p=0.5)
        self.fc = Linear(32, 8)
        # self.fc2 = Linear(hidden_channels // 2, hidden_channels // 4)
        self.blinear = BayesianLinear(in_features=8, out_features=1,prior_sigma_1=3.0,     # ↑ broader wide component (e.g., 2.0–5.0)
        prior_sigma_2=0.1,     # keep narrow component modest
        prior_pi=0.8 )          # ↑ weight on the wide component)     # Predicting r

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = getattr(data, "edge_attr", None) 
        x = self.conv1(x, edge_index,edge_weight=edge_weight)
        x = F.elu(x)
        x = self.conv2(x, edge_index,edge_weight=edge_weight)
        x = F.elu(x)
        x = self.conv3(x, edge_index,edge_weight=edge_weight)
        x = F.elu(x)
        x = self.dropout(x)
        x = global_mean_pool(x, batch)  # Pool over graph
        x = self.post_pool_norm(x) 
        x = F.elu(self.fc(x))
        # x = F.relu(self.fc2(x))
        mu_logr = self.blinear(x)  # Output a distribution over r
        return mu_logr

    def sample_elbo_loss(self, data, criterion, sample_nbr=3, complexity_cost_weight=3e-3):
        return self.sample_elbo(
            inputs=data,
            labels=data.y,
            criterion=criterion,
            sample_nbr=sample_nbr,
            complexity_cost_weight=complexity_cost_weight
        )
