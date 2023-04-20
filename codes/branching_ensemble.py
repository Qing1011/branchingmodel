import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import pandas as pd
import copy
from branchingv1 import *

# WN = np.loadtxt('W_avg.csv')
WN = np.loadtxt('W_rand.csv')
Cave = np.loadtxt('Cave.csv')
pop = np.loadtxt('pop_new.csv')

num_fips = len(pop)
T = 40
num_ens = 1000 ##300

# pathogen characteristics
# initialize parameters
R0 = 2.5  ## the model input

Z = 3 # latent period
Zb = 1 # scale parameter for Z
D = 5 # infectious period
Db = 1 # scale parameter for b
alpha = 0.1 # reporting rate 10%
# R0,r,Z;Zb;D;Db,alpha


#initialize variables
# seeding
l0 = 1859-1 # start with New York County NY in python -1, in matlab is 1859
i0 = 100 ## the starting t=0, in matlab it is 1
# initials = (l0,i0)

x_cutoff = 100
# r = 0.05 ## the parameters (in the paper of Lloyd smith is the k)

for r in [0.05,0.1]:
    p = r/(R0+r)

    weights = np.zeros(x_cutoff)
    for i in range(x_cutoff):
        temp1=SS.gamma(r+i)/SS.gamma(r)/SS.gamma((i+1))*np.power(p,r)*np.power((1-p),i)
        weights[i] = temp1
    weights_n = weights/np.sum(weights)
    print(r)
    E_NewInf = np.zeros((num_ens, num_fips,T*10))
    E_TotInf = np.zeros((num_ens, num_fips,T*10))

    for en_i in range(num_ens):
        print(en_i)
        E_NewInf[en_i], E_TotInf[en_i] = superspreading_T_Loc(40,num_fips,(l0,i0),weights_n,pop,(3,1,5,1),WN)

    np.save('NewInf_r_{}_randm'.format(r),E_NewInf)