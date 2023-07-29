import numpy as np
# import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import pandas as pd
import scipy.special as SS
import scipy.stats as SSA
import copy
import random
import math
from branchingv2 import *
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
    s = int(s) ## parameter index
    es_idx = int(sys.argv[2]) ## ensemble index
        ## load data
    WN = np.loadtxt('W_avg.csv')
    pop = np.loadtxt('pop_new.csv')
    para_dict = np.load('para_dict.npy',allow_pickle=True)
    ## set parameters
    para_i = para_dict[s]
    R0 = para_i['R0']
    r = para_i['r']
    p = r/(R0+r)

    num_fips = len(pop)
    T = 60
    # num_ens = 100 ##300 ###500 intially when R0 gets larger, we need fewer ensemble members, std is smaller

    # pathogen characteristics
    Z = 3 # latent period
    Zb = 1 # scale parameter for Z
    D = 5 # infectious period
    Db = 1 # scale parameter for b
    alpha = 0.1 # reporting rate 10%

    #initialize variables
    # seeding
    l0 = 1859-1 # start with New York County NY in python -1, in matlab is 1859
    i0 = 100 ## the starting t=0, in matlab it is 1

    E_NewInf = np.zeros((num_fips,T))
    E_TotInf = np.zeros((num_fips,T))
    save_dir = '/rds/general/user/qy1815/ephemeral/branching_R0-{}_r-{}/' .format(np.round(R0,2),np.round(r,3))
    
    ss = np.random.SeedSequence(es_idx)
    E_NewInf_i, E_TotInf_i = superspreading_T_Loc(T,num_fips,(l0,i0),(r,p),pop,(Z,Zb,D,Db),WN,ss)
    E_NewInf[:,:] = E_NewInf_i[:,:T]
    E_TotInf[:,:] = E_TotInf_i[:,:T]

    f = gzip.GzipFile(save_dir+"NewInf_R0-{}_r-{}_{}.npy.gz" .format(np.round(R0,2),np.round(r,3),es_idx), "w")
    np.save(file=f, arr=E_NewInf)
    f.close()


if __name__== "__main__":
    main()

