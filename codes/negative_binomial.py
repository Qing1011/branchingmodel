import numpy as np
from branching import branching
from scipy.io import loadmat, savemat

def negative_binomial(n, p, size=None):
    # Sample from a negative binomial distribution.
    # n: number of successes
    # p: probability of success
    # size: number of samples
    # return: samples   
    return np.random.negative_binomial(n, p, size)

def branchingmodel(r):
    # load data, prepare
    data = loadmat('../data/commutedata_100.mat')
    nl = data['nl'].astype(np.int32)
    part = data['part'].astype(np.int32)
    Cave = data['Cave'].astype(np.float64)
    population = loadmat('../data/population.mat')['population'].astype(np.float64)
    
    num_loc = len(population)
    T = 60
    num_ens = 300
    
    # compute wji
    # mobility factor
    theta = 1
    w = np.zeros_like(Cave)
    for i in range(num_loc):
        for j in range(part[i], part[i+1]-1):
            w[j] = theta * Cave[j] / population[i]
        w[part[i]] = max(1 - np.sum(w[part[i]+1:part[i+1]-1]), 0.2)
        w[part[i]:part[i+1]-1] = w[part[i]:part[i+1]-1] / np.sum(w[part[i]:part[i+1]-1])
    
    # pathogen characteristics
    # initialize parameters
    R0 = 2.5
    Z = 3  # latent period
    Zb = 1  # scale parameter for Z
    D = 5  # infectious period
    Db = 1  # scale parameter for b
    alpha = 0.1  # reporting rate 10%
    para = np.array([R0, r, Z, Zb, D, Db, alpha]).reshape((7, 1)) @ np.ones((1, num_ens))
    
    # initialize variables
    NewInf = np.zeros((num_loc, T, num_ens))
    Obs = np.zeros((num_loc, T, num_ens))
    # seeding
    l0 = 1859  # New York County NY
    NewInf[l0, 0, :] = 100
    
    for t in range(T):
        print(t+1)
        for k in range(num_ens):
            # branching process
            NewInf[:, t, k] = branching(nl, part, w, t, NewInf[:, t, k], para[:, k], population)
    
    Obs = alpha * NewInf
    DailyInf = np.sum(NewInf, axis=0)
    
    # plot results here if needed
    
    # save results
    if r == 0.1:
        savemat('NewInf_01.mat', {'NewInf': NewInf, 'Obs': Obs})
    elif r == 0.3:
        savemat('NewInf_03.mat', {'NewInf': NewInf, 'Obs': Obs})
    elif r == 5:
        savemat('NewInf_5.mat', {'NewInf': NewInf, 'Obs': Obs})



