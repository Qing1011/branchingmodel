import numpy as np
# import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import pandas as pd
import scipy.special as SS
import scipy.stats as SSA
import copy
import random
import math


def get_new_infections_position(infectors, total_new, num_fips):
    cum_inft = np.cumsum(infectors)
    cum_new_infections = np.cumsum(total_new)

    totoal_new_infection_l = np.zeros(int(np.sum(total_new)))
    num_inft_s_idx = 0
    inft_s_idx = 0
    for l_indx in range(num_fips):
        num_inft = infectors[l_indx]
        if num_inft > 0:
            num_inft_e_idx = num_inft_s_idx+num_inft
            infection_list_l = total_new[num_inft_s_idx:num_inft_e_idx]
            # the list of the number of infections caused by every infector
            num_ift_l = np.sum(infection_list_l)
            inft_e_idx = inft_s_idx + num_ift_l
            totoal_new_infection_l[int(inft_s_idx):int(inft_e_idx)] = l_indx
            # the position of infection in the list of all infections
            num_inft_s_idx = num_inft_e_idx
            inft_s_idx = inft_e_idx
    return totoal_new_infection_l


def superspreading_T_Loc(T, num_fips, initials, weights_n, pop, paras, WN):
    Z, Zb, D, Db = paras
    # initialise
    l0, i0 = initials

    NewInf = np.zeros((num_fips, T*10))
    TotInf = np.zeros((num_fips, T*10))

    NewInf[l0, 0] = i0
    TotInf[:, 0] = NewInf[:, 0]
    for ti in range(T):
        # print(ti)
        infectors = np.int64(NewInf[:, ti])
        total_num_infectors = np.sum(infectors)
        pop_immu = 1-TotInf[:, ti]/pop[:]
        pop_immu[pop_immu < 0] = 0
        immu_all = []  # create list of immu_prob * number of infectors
        for idx in range(len(pop_immu)):
            pop_immu_i = [pop_immu[idx]]*infectors[idx]
            immu_all.extend(pop_immu_i)
        tt = np.random.choice(
            len(weights_n), size=total_num_infectors, p=weights_n)  # this is a vector
        # tt = SSA.nbinom.rvs(0.025, 0.025/(2.5+0.025), size=total_num_infectors)
        # to be assigned, every new infections for the infector
        total_new = np.round(tt*immu_all)
        totoal_new_infection_loc = get_new_infections_position(
            infectors, total_new, num_fips)

        z_num = np.int64(np.sum(total_new))
        NF = np.zeros((2, z_num), dtype=np.int64)
        # for the time distribution
        latency_p = SSA.gamma.rvs(a=Z, scale=Zb, size=z_num)
        infectious_p = SSA.gamma.rvs(a=D, scale=Db, size=z_num)
        v = np.random.random_sample(z_num)
        delay_days = latency_p+v*infectious_p  # 3+5*0.5

        NF[0, :] = np.ceil(delay_days+ti)  # make it idx int
        # for the location distribution
        NF[1, :] = totoal_new_infection_loc
        df = pd.DataFrame(NF.T, columns=['time', 'o_l'])
        l_list = np.arange(num_fips)
        df['d_l'] = df['o_l'].apply(
            lambda x: np.random.choice(l_list, size=1, p=WN[:, x])[0])
        NF_ii = np.array(df)
        for (t, o, d) in NF_ii:
            NewInf[d, t] = NewInf[d, t]+1
        TotInf = np.cumsum(NewInf, axis=1)
    return NewInf, TotInf
