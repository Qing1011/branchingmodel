import torch
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import matplotlib.pyplot as plt
import scipy.special as SS
import scipy.stats as SSA

def torch_negative_binomial(n, p, size):
    # Generate gamma distribution
    gamma = torch.distributions.Gamma(n, (1 - p)/p).sample(sample_shape=torch.Size([size]))
    # Generate Poisson distribution
    return torch.distributions.Poisson(gamma).sample()

class EpidemicSimulator(MessagePassing):
    def __init__(self, r, p, max_time_step):
        super(EpidemicSimulator, self).__init__(aggr='add')
        self.r = r
        self.p = p  
        self.max_time_step = max_time_step
        self.Z = 3 # latent period
        self.Zb = 1 # scale parameter for Z
        self.D = 5 # infectious period
        self.Db = 1 # scale parameter for beta
    
    def forward(self, x, edge_index, edge_attr, step):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr, step=step)

    def message(self, x_j, edge_index, edge_attr):
        # x_j has shape [E, num_features]
        # edge_attr has shape [E, num_edge_features]
        # Get the new infections from x_j.
        new_infections = x_j[:, 0:1]  # Shape: [E, 1]
        # Compute the messages.
        messages = new_infections * edge_attr.view(-1, 1)
        return messages

    def update(self, aggr_out, x, step):
        # x has shape [N, num_features], it is the original node features
        # The new infections are the aggregated messages.
        new_infections = aggr_out # aggr_out has shape [N, 1], it contains the updated infections
        #### Add the effective infections to the column corresponding to the current step.####
        ## diffuse the new_infections to different times 
        new_infections_int  = new_infections.round().int()
        inf_sizes = new_infections_int.squeeze().tolist()
        for i, inf_size_i in enumerate(inf_sizes):
            gamma_dist1 = torch.distributions.Gamma(self.Z, 1/self.Zb)
            gamma_dist2 = torch.distributions.Gamma(self.D, 1/self.Db)
            latency_p = gamma_dist1.sample(sample_shape=torch.Size([inf_size_i]))
            infectious_p = gamma_dist2.sample(sample_shape=torch.Size([inf_size_i]))
            v = torch.rand(inf_size_i)
            delay_days = latency_p + v * infectious_p
            for j,delay_t in enumerate(delay_days):
                t_j = (1+step+delay_t).ceil().int()
                if t_j > self.max_time_step:
                    pass
                else:
                    x[i,t_j] = x[i,t_j] + 1
        ##generate new infections based on the current time infectors
        population = x[:, 1:2]
        new_generation = x[:, 2+step:3+step] ## the infectors at time ti
        total_infection = torch.sum(x[:, 2:3+step], dim=1,keepdim=True) 
        rate = (population - total_infection) / population # Compute the rate.
        rate[rate<0] = 0
        temp = new_generation.round().int()
        sizes = temp.squeeze().tolist()
        # Initialize an empty tensor to store the results
        results = torch.empty_like(new_generation)
        # Generate negative binomial for each size
        for i, size in enumerate(sizes):
            result = torch_negative_binomial(self.r, self.p, size)
            temp_sum = result.sum()
            effective_infections = (rate[i] * temp_sum)
            results[i] = effective_infections
        
        new_infections = results
        ######^^^^^^#######
        # The rest of the features remain the same.
        other_features = x[:, 2:]
        # Concatenate the new infections, the population, and the other features to get the new node features.
        x_new = torch.cat([new_infections, population, other_features], dim=1)
        return x_new

def simulate_dynamics(data, R0, r, num_steps,edge_index):
    p = r/(R0+r)   
    simulator = EpidemicSimulator(r,p,max_time_step=61)
    x = data.x
    for ti in range(num_steps):
        x = simulator(x, edge_index, data.edge_attr,ti)
    return x