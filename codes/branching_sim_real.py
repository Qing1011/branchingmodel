import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import scipy.special as SS
import scipy.stats as SSA
import copy
import random
import math
import sys
import gzip
import h5py
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
    

def superspreading_T_Loc_mobility_varrying(T,Initials_inf, nbi_para, pop, paras, WN_t, rand_seed):
    # initialise
    Z, Zb, D, Db = paras
    r, Ptx = nbi_para
    child_seeds = rand_seed.spawn(T)
    num_fips = len(pop)
    NewInf = np.zeros((num_fips, T))
    TotInf = np.zeros((num_fips, T))
    # initial_days = Initials_inf.shape[1]
    # NewInf[:, :initial_days] = Initials_inf
    # TotInf[:, :initial_days] = Initials_inf
    NewInf[:, 0] = Initials_inf
    TotInf[:, 0] = Initials_inf

    for ti in range(T):
        # print(ti)
        WN = WN_t[ti]
        infectors = np.int64(NewInf[:, ti])
        pop_immu = 1-TotInf[:, ti]/pop[:]
        pop_immu[pop_immu < 0] = 0
        # create list of immu_prob * number of infectors
        #immu_all = np.repeat(pop_immu, infectors)
        rng = np.random.default_rng(child_seeds[ti])
        
        # total_num_infectors = np.sum(infectors)
        P_t = Ptx[ti,:] 
        params = list(zip(r*np.ones(num_fips), P_t, infectors, range(num_fips)))
        
        # samples = []
        totoal_new_infection_loc = []
        total_new = []
        # Generate samples
        for r_x, p_x, size,loc_idx in params:
            if size > 0:
                sam_i = rng.negative_binomial(n=r_x, p=p_x, size=size)
                # sam_i = np.array([2]*size)
                # samples.append(sam_i)
                real_new_infection = np.int64(np.round(sam_i*pop_immu[loc_idx]))
                total_new.extend(real_new_infection)
                new_inf_loc = [loc_idx]*np.sum(real_new_infection)
                totoal_new_infection_loc.extend(new_inf_loc)
            # else:
            #     samples.append(np.array([]))

        z_num = np.int64(np.sum(total_new))
        NF = np.zeros((2, z_num), dtype=np.int64)
        # for the time distribution
        latency_p = SSA.gamma.rvs(a=Z, scale=Zb, size=z_num, random_state=rng)
        infectious_p = SSA.gamma.rvs(
            a=D, scale=Db, size=z_num, random_state=rng)
        v = rng.random(z_num)
        delay_days = latency_p+v*infectious_p  # 3+5*0.5

        NF[0, :] = np.ceil(delay_days+ti)  # make it idx int
        # for the location distribution
        NF[1, :] = totoal_new_infection_loc

        time_column = NF[0, :]
        o_l_column = NF[1, :]
        result_list = []
        for i in range(z_num):
            if time_column[i] <= (T - 1):
                d_l_value = rng.choice(np.arange(num_fips), size=1, p=WN[:, o_l_column[i]])[0]
                result_list.append([time_column[i], o_l_column[i], d_l_value])
        if len(result_list) == 0:
            continue
        # NF_ii = np.array(result_list)
        # for (t, o, d) in NF_ii:
        #     NewInf[d, t] = NewInf[d, t]+1
        NF_ii = np.asarray(result_list, dtype=np.int64).reshape(-1, 3)
        np.add.at(NewInf, (NF_ii[:,2], NF_ii[:,0]), 1)
        # TotInf = np.cumsum(NewInf, axis=1)
        TotInf[:, ti] = (TotInf[:, ti-1] if ti > 0 else 0) + NewInf[:, ti]

    return NewInf, TotInf


def main():
    s = sys.argv[1]
    s = int(s)  # parameter index
    es_idx = int(sys.argv[2])  # ensemble index
    # load data
    # WN = np.loadtxt('W_avg.csv')
    with h5py.File('NMM_t_3142.hdf5', 'r') as f:
        WM_t = f['WM'][...] # Load precomputed WM_t starting from 2020-02-23
    pop = np.loadtxt('pop_new.csv')
    # para_dict = np.load('para_dict.npy', allow_pickle=True)
    # set parameters
    r_list = np.array([20, 10, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.025, 5., 2.5  , 13.333,  3.333,  1.333,  0.667,  0.286,  0.133, 0.067,  0.033, 0.37, 7.4])
    # para_i = para_dict

    Rtx = np.loadtxt('Rt_real.csv')
    Ini_seed = np.loadtxt('seed_real.csv', delimiter=',')

    r = r_list[s]
    Ptx = r/(Rtx+r)

    num_fips = len(pop)
    T = 21
    # num_ens = 100 ##300 ###500 intially when R0 gets larger, we need fewer ensemble members, std is smaller

    # pathogen characteristics
    Z = 3  # latent period
    Zb = 1  # scale parameter for Z
    D = 5  # infectious period
    Db = 1  # scale parameter for b
    alpha = 0.1  # reporting rate 10%

    E_NewInf = np.zeros((num_fips, T))
    E_TotInf = np.zeros((num_fips, T))

    ss = np.random.SeedSequence(es_idx)
    E_NewInf_i, E_TotInf_i = superspreading_T_Loc_mobility_varrying(
        T, Ini_seed, (r, Ptx), pop, (Z, Zb, D, Db), WM_t, ss)

    E_NewInf[:, :] = E_NewInf_i[:, :T]
    E_TotInf[:, :] = E_TotInf_i[:, :T]
    # save_dir = '/rds/general/user/qy1815/ephemeral/Rtx/branching_r-{}/' .format(np.round(r, 3))
    
    save_dir = 'branching_r-{}/' .format(np.round(r, 3))
    os.makedirs(save_dir, exist_ok=True)
    f = gzip.GzipFile(
        save_dir+"NewInf_r-{}_{}.npy.gz" .format(np.round(r, 3), es_idx), "w")
    np.save(file=f, arr=E_NewInf)
    f.close()


if __name__ == "__main__":
    main()