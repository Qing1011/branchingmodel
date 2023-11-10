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


def superspreading_T_Loc(T, num_fips, initials, nbi_para, pop, paras, WN, rand_seed):
    # initialise
    Z, Zb, D, Db = paras
    l0, i0 = initials
    r, p = nbi_para
    child_seeds = rand_seed.spawn(T)

    NewInf = np.zeros((num_fips, T))
    TotInf = np.zeros((num_fips, T))
    NewInf[l0, 0] = i0
    TotInf[:, 0] = NewInf[:, 0]

    for ti in range(T):
        infectors = np.int64(NewInf[:, ti])
        total_num_infectors = np.sum(infectors)
        pop_immu = 1-TotInf[:, ti]/pop[:]
        pop_immu[pop_immu < 0] = 0
        # create list of immu_prob * number of infectors
        immu_all = np.repeat(pop_immu, infectors)
        rng = np.random.default_rng(child_seeds[ti])
        tt = rng.negative_binomial(r, p, total_num_infectors)
        # xx = np.arange(0, 100, 1)  # define the range of x values the
        # calculate the probability mass function
        # pmf = SSA.nbinom.pmf(xx, r, p)
        # weights_n = pmf/np.sum(pmf)
        # tt = rng.choice(
        #     len(weights_n), size=total_num_infectors, p=weights_n)
        # to be assigned, every new infections for the infector
        total_new = np.round(tt*immu_all)
        totoal_new_infection_loc = get_new_infections_position(
            infectors, total_new, num_fips)

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
        df = pd.DataFrame(NF.T, columns=['time', 'o_l'])
        l_list = np.arange(num_fips)
        df['d_l'] = df['o_l'].apply(
            lambda x: np.random.choice(l_list, size=1, p=WN[:, x])[0])
        df = df[df['time'] <= (T-1)]
        NF_ii = np.array(df)
        for (t, o, d) in NF_ii:
            NewInf[d, t] = NewInf[d, t]+1
        TotInf = np.cumsum(NewInf, axis=1)

    return NewInf, TotInf
