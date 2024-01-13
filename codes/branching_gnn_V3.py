import torch
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import matplotlib.pyplot as plt
import scipy.special as SS
import scipy.stats as SSA


class EpidemicSimulator(MessagePassing):
    def __init__(self, r, p, weight, max_time_step, prior_scale=1.):
        super(EpidemicSimulator, self).__init__(aggr='add')
        self.r = r
        # self.r = PyroModule[torch.nn.Parameter(torch.Tensor([r]))]
        # self.r = PyroSample(dist.Normal(0.1, prior_scale).to_event(2))
#         self.p = p
        self.p_prime = 1-p
        self.max_time_step = max_time_step
        self.Z = 3  # latent period
        self.Zb = 1  # scale parameter for Z
        self.D = 5  # infectious period
        self.Db = 1  # scale parameter for beta
        self.weight = torch.Tensor(weight)
#         print(self.weight)
        self.offspring = []

    def forward(self, x, edge_index, edge_attr, step):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr, step=step)

    def message(self, x, edge_index, edge_attr, step):
        new_infectors = x[:, 2+step:3+step]  # the infectors at time ti
        population = x[:, 1]
        total_infection = torch.sum(x[:, 2:3+step], dim=1, keepdim=True)
        rate = (population - total_infection) / population  # Compute the rate.
        rate[rate < 0] = 0

        new_effective_infectors = new_infectors*rate
        new_infectors_int = new_effective_infectors.round().int()
#         temp = new_infectors.round().int()
        cases = new_infectors_int.squeeze().tolist()
        # Initialize an empty tensor to store the results
        results = torch.zeros_like(new_infectors)
        # Generate negative binomial for each size
        for i, size in enumerate(cases):
            #             print(size)
            if size > 0:
                offspring_per_case = torch.distributions.negative_binomial.NegativeBinomial(
                    self.r, self.p_prime).sample(sample_shape=torch.Size([size]))
#                 offspring_per_case = torch.tensor([2]*size)
                # torch.distributions.Categorical(
                #     self.weight).sample(sample_shape=torch.Size([size])) ##cutoff version
#                 self.offspring.extend(offspring_per_case.tolist())
                temp_sum = offspring_per_case.sum()
            else:
                temp_sum = 0
#             print(temp_sum)
            results[i] = temp_sum
        ###### ^^^^^^#######
        # Compute the messages.
        results_aligned = results[edge_index[0]]
        messages = results_aligned * edge_attr.view(-1, 1)

        return messages

    def update(self, aggr_out, x, step):
        # The new infections are the aggregated messages.
        # aggr_out has shape [N, 1], it contains the updated infections
        new_infections = aggr_out
        #### Add the effective infections to the column corresponding to the current step.####
        # immu first
        new_infections_int = new_infections.round().int()
        # diffuse the new_infections to different times
        inf_sizes = new_infections_int.squeeze().tolist()
        for i, inf_size_i in enumerate(inf_sizes):
            gamma_dist1 = torch.distributions.Gamma(self.Z, 1/self.Zb)
            gamma_dist2 = torch.distributions.Gamma(self.D, 1/self.Db)
            latency_p = gamma_dist1.rsample(
                sample_shape=torch.Size([inf_size_i]))
            infectious_p = gamma_dist2.rsample(
                sample_shape=torch.Size([inf_size_i]))
            v = torch.rand(inf_size_i)
#             delay_days = torch.tensor([2]*inf_size_i)
            delay_days = latency_p + v * infectious_p
#             print(step, delay_days)
            for j, delay_t in enumerate(delay_days):
                t_j = (2+step+delay_t).ceil().int()
                if t_j > self.max_time_step:
                    pass
                else:
                    x[i, t_j] = x[i, t_j] + 1

        return x


def simulate_dynamics(data, R0, r, num_steps):
    p = r/(R0+r)
    xx = np.arange(0, 100, 1)  # define the range of x values the cutoff is 200
    # pmf = SSA.nbinom.pmf(xx, r, p)  # calculate the probability mass function
    pmf = SSA.nbinom.pmf(xx, r.detach().numpy(), p.detach().numpy())
    weights_n = pmf/np.sum(pmf)
#     print(weights_n)
    x = data.x
    T_len = x.shape[1]
    simulator = EpidemicSimulator(r, p, weights_n, max_time_step=(T_len-1))
    for ti in range(num_steps):
        x = simulator(x, data.edge_index, data.edge_attr, ti)
    return x
