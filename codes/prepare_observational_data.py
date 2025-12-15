import gzip
import numpy as np
import scipy.stats as SSA
import torch
from torch_geometric.data import Data,DataLoader

def load_gzipped_numpy(filename):
    try:
        with gzip.open(filename, 'rb') as f:
            return np.load(f, allow_pickle=True)
    except FileNotFoundError:
        return [0]
    

def observational_para():
    # aver_delay=13.9 aver_delay = alpha*scale_theta
    x = np.arange(0.0, 21, 1) 
    alpha = 4.27
    scale_theta = 3.25 ## scale = 1/beta
    y1 = SSA.gamma.pdf(x, a=alpha, scale=scale_theta)
    raw=np.array([y1[i+1] for i in range(20)])
    prob_gamma=(raw/np.sum(raw)).reshape(20,1)
    sim_daily_rate = 0.15
    return prob_gamma, sim_daily_rate


def add_observation_noise(data, prob_gamma, sim_daily_rate):
    """
    Add observation noise to the data
    """
    T = data.shape[1]
    num_fips = data.shape[0]
    new_case = np.zeros((T,num_fips))
    max_delay = prob_gamma.shape[0]
    for t in range(T):
        deltacase = data[:,t]
        deltacase = deltacase.reshape(1,num_fips)
        prob_case = prob_gamma.dot(deltacase)
        new_case[t:min(t+max_delay,T)] += prob_case[:min(T-t,max_delay)]
    case_daily_obs = sim_daily_rate*new_case
    case_daily_obs = case_daily_obs.T
    return case_daily_obs


def prepare_dataset(export_dir, rs, t_len, edge_index, edge_weights):
    dataset = []
    t = int(10+t_len)
    print(t, type(t))
    for r_idx in range(len(rs)):
        r = rs[r_idx]
        sub_export_dir = export_dir+'/branching_r-{}/'.format(np.round(r,3))
#         r_c = r_class[r]
        print(r)
        # print(sub_export_dir)

        for g_idx in range(300):
            export_names = sub_export_dir+'NewInf_r-{}_{}.npy.gz'.format(np.round(r,3),(g_idx)) ## this is the output of the cluster job index g_idx+1
            g_i = load_gzipped_numpy(export_names)
            print(export_names)
            g_i_new = g_i[:,10:t]
            
            matrix = torch.from_numpy(g_i_new)
            y = torch.log(torch.tensor([[r]], dtype=torch.float))
            data = Data(x=matrix, edge_index=edge_index, edge_attr=edge_weights, y=y)
            data.x = data.x.float()
            dataset.append(data)
    print('finish reading ^______^')
    return dataset


def prepare_dataset_obs(export_dir, rs, t_len, edge_index, edge_weights,prob_gamma, sim_daily_rate):
    dataset = []
    t = int(10+t_len)
    print(t, type(t))
    for r_idx in range(len(rs)):
        r = rs[r_idx]
        sub_export_dir = export_dir+'branching_r-{}/'.format(np.round(r,3))
#         r_c = r_class[r]
        print(r)

        for g_idx in range(300):
            export_names = sub_export_dir+'NewInf_r-{}_{}.npy.gz'.format(np.round(r,3),(g_idx+1))
            g_i = load_gzipped_numpy(export_names)
            g_i_obs = add_observation_noise(g_i, prob_gamma, sim_daily_rate)
            g_i_new = g_i_obs[:,10:t]
            
            matrix = torch.from_numpy(g_i_new)
            y = torch.log(torch.tensor([[r]], dtype=torch.float))
            data = Data(x=matrix, edge_index=edge_index, edge_attr=edge_weights, y=y)
            data.x = data.x.float()
            dataset.append(data)
    print('finish reading ^______^')
    return dataset