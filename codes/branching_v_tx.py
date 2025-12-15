import numpy as np
# import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import pandas as pd
import scipy.special as SS
import scipy.stats as SSA
import copy
import random
import math



def superspreading_T_Loc_varrying(T, num_fips, initials, nbi_para, pop, paras, WN, rand_seed):
    # initialise
    Z, Zb, D, Db = paras
    l_ls, i_ls = initials
    r, Ptx = nbi_para
    child_seeds = rand_seed.spawn(T)

    NewInf = np.zeros((num_fips, T))
    TotInf = np.zeros((num_fips, T))
    for idx in range(len(l_ls)):
        NewInf[l_ls[idx], 0] = i_ls[idx]
        # NewInf[l0, 0] = i0
    TotInf[:, 0] = NewInf[:, 0]

    for ti in range(T):
        infectors = np.int64(NewInf[:, ti])
        pop_immu = 1-TotInf[:, ti]/pop[:]
        pop_immu[pop_immu < 0] = 0
        # create list of immu_prob * number of infectors
        #immu_all = np.repeat(pop_immu, infectors)
        rng = np.random.default_rng(child_seeds[ti])
        
        # total_num_infectors = np.sum(infectors)
        P_t = Ptx[ti,:] 
        params = list(zip(r*np.ones(num_fips), P_t, infectors, range(num_fips)))
        
        samples = []
        totoal_new_infection_loc = []
        total_new = []
        # Generate samples
        for r_x, p_x, size,loc_idx in params:
            if size > 0:
                sam_i = np.random.negative_binomial(n=r_x, p=p_x, size=size)
                # sam_i = np.array([2]*size)
                samples.append(sam_i)
                real_new_infection = np.int64(np.round(sam_i*pop_immu[loc_idx]))
                total_new.extend(real_new_infection)
                new_inf_loc = [loc_idx]*np.sum(real_new_infection)
                totoal_new_infection_loc.extend(new_inf_loc)
            else:
                samples.append(np.array([]))

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



def superspreading_T_Loc_mobility_varrying(T,initials_inf, nbi_para, pop, paras, WN_t, rand_seed):
    # initialise
    Z, Zb, D, Db = paras
    r, Ptx = nbi_para
    child_seeds = rand_seed.spawn(T)
    num_fips = len(pop)
    NewInf = np.zeros((num_fips, T))
    TotInf = np.zeros((num_fips, T))
    NewInf[:, 0] = initials_inf
    TotInf[:, 0] = initials_inf

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
                sam_i = np.random.negative_binomial(n=r_x, p=p_x, size=size)
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
                d_l_value = np.random.choice(np.arange(num_fips), size=1, p=WN[:, o_l_column[i]])[0]
                result_list.append([time_column[i], o_l_column[i], d_l_value])
        NF_ii = np.array(result_list)
        for (t, o, d) in NF_ii:
            NewInf[d, t] = NewInf[d, t]+1
        TotInf = np.cumsum(NewInf, axis=1)

    return NewInf, TotInf
