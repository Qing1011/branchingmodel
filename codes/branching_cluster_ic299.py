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

def check_file_exists(filename):
    """Check if the file exists.
    if exist return False, do not run the program else return True, run the program
    """
    try:
        f = open(filename, 'r')
        f.close()
        return False
    except FileNotFoundError:
        return True

def main():
    s = sys.argv[1]
    s = int(s) - 1
    e = int(sys.argv[2])
    rs = np.arange(0.01, 1, 0.02)
    R0s = np.arange(1.5, 6.5, 0.1)
    param_grid = {'R0': R0s, 'r' : rs}
    grid = ParameterGrid(param_grid)
    para_dict = list(grid)
    para_i = para_dict[s]
    R0 = para_i['R0']
    r = para_i['r']
    ## check if the file exists
    check_dir = '/rds/general/user/qy1815/ephemeral/branching_results300/'
    file_name = check_dir+"NewInf_R0-{}_r-{}.npy.gz" .format(np.round(R0,2),np.round(r,2))
    if check_file_exists(file_name):
        ## load data
        WN = np.loadtxt('W_avg.csv')
        pop = np.loadtxt('pop_new.csv')

        num_fips = len(pop)
        T = 60
        num_ens = 100 ##300 ###500 intially

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
        # print(r)

        E_NewInf = np.zeros((num_ens, num_fips,T))
        E_TotInf = np.zeros((num_ens, num_fips,T))
        save_dir = '/rds/general/user/qy1815/ephemeral/branching299/'
        

        for en_i in range(num_ens):
            # print(en_i)
            E_NewInf_i, E_TotInf_i = superspreading_T_Loc(T,num_fips,(l0,i0),weights_n,pop,(Z,Zb,D,Db),WN)
            E_NewInf[en_i,:,:] = E_NewInf_i[:,:T]
            E_TotInf[en_i,:,:] = E_TotInf_i[:,:T]
            # np.save('NewInf_r_{}_randm'.format(r),E_NewInf)
            # save_dir = '/ifs/scratch/msph/ehs/qy2290/branching_results/'
            # f = gzip.GzipFile(save_dir+"NewInf_R0-{}_r-{}_{}.npy.gz" .format(np.round(R0,2),np.round(r,2),en_i), "w")
            # np.save(file=f, arr=E_NewInf)
            # f.close()
        f = gzip.GzipFile(save_dir+"NewInf_R0-{}_r-{}_{}.npy.gz" .format(np.round(R0,2),np.round(r,2),e), "w")
        np.save(file=f, arr=E_NewInf)
        f.close()
    else:
        print('file already exists')
        

if __name__== "__main__":
    main()

