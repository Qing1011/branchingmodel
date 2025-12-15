import numpy as np
# import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import pandas as pd
import scipy.special as SS
import scipy.stats as SSA
import copy
import random
import math
from branching_v_tx import *
import sys
# from sklearn.model_selection import ParameterGrid
import gzip
import os


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

def load_gzipped_numpy(filename):
    try:
        with gzip.open(filename, 'rb') as f:
            return np.load(f, allow_pickle=True)
    except FileNotFoundError:
        return [0]

def main():
    s = sys.argv[1]
    s = int(s)  # r parameter index
    es_idx = int(sys.argv[2])  # ensemble index
    
    r_observed = float(sys.argv[3])  # observed r
    real_idx = int(sys.argv[4])  # index for realisation idx
    alpha = float(sys.argv[5])  # reporting rate at the beginning
    
    Rtx_folder = sys.argv[6]
    observed_folder = sys.argv[7]
    cluster_sim_folder = sys.argv[8]

    # load data
    WN = np.loadtxt('W_avg.csv')
    pop = np.loadtxt('pop_new.csv')
    # para_dict = np.load('para_dict.npy', allow_pickle=True)
    # set parameters
    r_list = np.array([20, 10, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.025, 5., 2.5  , 13.333,  3.333,  1.333,  0.667,  0.286,  0.133, 0.067,  0.033, 0.37, 7.4])
    # para_i = para_dict
    # Rtx = np.loadtxt('Rtx_synthetic.csv')
    # Rtx = np.loadtxt('Rt_array_r-{}.csv'.format(r_observed)) # no reporting rate version
    Rtx = np.loadtxt(Rtx_folder+'Rt_array_r-{}_{}_rr.csv'.format(r_observed,real_idx)) # with reporting rate version
    #### check the saving file part wether is is consistent with the r_observed

    r = r_list[s]
    Ptx = r/(Rtx+r)

    num_fips = len(pop)
    T = 60
    # num_ens = 100 ##300 ###500 intially when R0 gets larger, we need fewer ensemble members, std is smaller

    # pathogen characteristics
    Z = 3  # latent period
    Zb = 1  # scale parameter for Z
    D = 5  # infectious period
    Db = 1  # scale parameter for b
    # alpha = 0.1  # reporting rate 10%

    # initialize variables
    # seeding
    export_names = observed_folder + 'NewInf_r-{}_{}_rr.npy.gz' .format(np.round(r_observed, 3), real_idx) ### with the reporting rate
    # export_names = '../test_data/nrr_den/NewInf_r-{}_{}.npy.gz' .format(np.round(r_observed, 3), real_idx) ### without the reporting and the density version
    g_observed = load_gzipped_numpy(export_names)
    # l0 = 1859-1  # start with New York County NY in python -1, in matlab is 1859
    # i0 = 50  # the starting t=0, in matlab it is 1
    # l1 = 228  # Santa Clara County in the state of California
    # i1 = 50  
    # l2 = 1229 # Suffolk County in the state of Massachusetts 
    # i2 = 20  
    # l3 = 2969  # King County in the state of Washington
    # i3 = 20 
    # location_ls = [l0, l1, l2, l3]
    # seed_ls = [i0, i1, i2, i3] 
    location_ls = list(np.where(g_observed[:,0]>0)[0])
    seed_ls = list(g_observed[location_ls,0]/alpha)

    E_NewInf = np.zeros((num_fips, T))
    E_TotInf = np.zeros((num_fips, T))

    ss = np.random.SeedSequence(es_idx)
    E_NewInf_i, E_TotInf_i = superspreading_T_Loc_varrying(
        T, num_fips, (location_ls, seed_ls), (r, Ptx), pop, (Z, Zb, D, Db), WN, ss)
    E_NewInf[:, :] = E_NewInf_i[:, :T]
    E_TotInf[:, :] = E_TotInf_i[:, :T]

    cluster_folder = cluster_sim_folder +'r-{}_{}_simulation_res/'.format(np.round(r_observed, 3), real_idx)    
    save_dir = cluster_folder + 'branching_r-{}/' .format(np.round(r, 3))
    os.makedirs(save_dir, exist_ok=True)

    f = gzip.GzipFile(
        save_dir+"NewInf_r-{}_{}.npy.gz" .format(np.round(r, 3), es_idx), "w")
    np.save(file=f, arr=E_NewInf*alpha)
    # np.save(file=f, arr=E_NewInf)
    f.close()


if __name__ == "__main__":
    main()
