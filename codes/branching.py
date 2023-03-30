import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import pandas as pd
import scipy.special as SS
import scipy.stats as SSA
import copy
import random
import math
from numba import njit

## the model input neighbour list, idx, probability, time, the previous infections, parameters, population size 
## the model output for kth ensemble, the county everytime
## it iterates every time
@numba.jit(nopython=True)
def superspreading_T_Loc(T,num_fips,initials,weights_n,pop,paras,WN):
    Z, Zb, D, Db = paras
    ## initialise 
    l0, i0 = initials
    
    NewInf = np.zeros((num_fips,T*10))
    TotInf = np.zeros((num_fips,T*10))
    
    NewInf[l0,0] = i0
    TotInf[:,0] = NewInf[:,0]
    ### for each time step and each location
    for ti in range(T):
        print(ti)
        for l in range(num_fips):
            infectors = int(NewInf[l,ti])
            ### the number of possible new infections the size = 100 can infect individually from the NB 
            z_all = np.random.choice(len(weights_n), size = infectors, p=weights_n) ## this is a vector
            ### however, the populations have immunited people, the number will decrease i did not agree with this part 
            ## as the infection happens inside the location of the infectious people, for immunity is the pop of the 
            z_immunity = np.round(z_all*(1-TotInf[l,ti]/pop[l]))
            
            z_num = np.int64(np.sum(z_immunity))
            NF_l = np.zeros((2,z_num),dtype=np.int64)
            ## for the time distribution
            latency_p = SSA.gamma.rvs(a = Z,scale=Zb,size = z_num)
            infectious_p = SSA.gamma.rvs(a = D,scale=Db,size = z_num)
            v = np.random.random_sample(z_num)
            delay_days = latency_p+v*infectious_p

            NF_l[0,:] = np.ceil(delay_days+ti) ## make it idx int
            ## for the location distribution
            loc_idx = np.random.choice(np.arange(num_fips), size = z_num, p=WN[:,l])
            NF_l[1,:] = loc_idx

            ## infections merge into the matrix
            NewInf_l = np.zeros_like(NewInf)
            for i in range(z_num):
                t_i, loc_i = NF_l[:,i]
                NewInf_l[loc_i,t_i] = NewInf_l[loc_i,t_i]+1
            NewInf = NewInf + NewInf_l
            TotInf = np.cumsum(NewInf,axis=1)
    return NewInf, TotInf