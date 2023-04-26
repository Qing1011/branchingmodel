import numpy as np
# import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import pandas as pd
import scipy.special as SS
import scipy.stats as SSA
import copy
import random
import math
from branchingv1 import *
import sys
from sklearn.model_selection import ParameterGrid
import gzip

def main():
    s = sys.argv[1]
    s = int(s) - 1
    rs = np.arange(0.01, 1, 0.02)
    R0s = np.arange(1.5, 6.5, 0.1)
    param_grid = {'R0': R0s, 'r' : rs}
    grid = ParameterGrid(param_grid)
    para_dict = list(grid)
    para_i = para_dict[s]
    R0 = para_i['R0']
    r = para_i['r']
    ## load data
    WN = np.loadtxt('W_avg.csv')
    pop = np.loadtxt('pop_new.csv')

    num_fips = len(pop)
    T = 60
    num_ens = 500

    # pathogen characteristics
    # initialize parameters

    Z = 3 # latent period
    Zb = 1 # scale parameter for Z
    D = 5 # infectious period
    Db = 1 # scale parameter for b
    alpha = 0.1 # reporting rate 10%

    #initialize variables
    # seeding
    l0 = 1859-1 # start with New York County NY in python -1, in matlab is 1859
    i0 = 100 ## the starting t=0, in matlab it is 1
    # initials = (l0,i0)

    x_cutoff = 100
    p = r/(R0+r)
    weights = np.zeros(x_cutoff)
    for i in range(x_cutoff):
        temp1=SS.gamma(r+i)/SS.gamma(r)/SS.gamma((i+1))*np.power(p,r)*np.power((1-p),i)
        weights[i] = temp1
    weights_n = weights/np.sum(weights)
    print(r)

    # E_NewInf = np.zeros((num_ens, num_fips,T*10))
    # E_TotInf = np.zeros((num_ens, num_fips,T*10))

    for en_i in range(num_ens):
        # print(en_i)
        E_NewInf, E_TotInf = superspreading_T_Loc(T,num_fips,(l0,i0),weights_n,pop,(Z,Zb,D,Db),WN)

        # np.save('NewInf_r_{}_randm'.format(r),E_NewInf)
        save_dir = '/ifs/scratch/msph/ehs/qy2290/branching_results/'
        f = gzip.GzipFile(save_dir+"NewInf_R0-{}_r-{}_{}.npy.gz" .format(np.round(R0,2),np.round(r,2),en_i), "w")
        np.save(file=f, arr=E_NewInf)
        f.close()
        # f = save_dir+"NewInf_R0-{}_r-{}.txt" .format(np.round(R0,2),np.round(r,2))
        # np.savetxt(f, E_NewInf)
        

if __name__== "__main__":
    main()

