import torch
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import matplotlib.pyplot as plt
import scipy.special as SS
import pandas as pd
import scipy.stats as SSA


def superspreading_T_Loc_real(T, num_fips, initials, nbi_para, pop, paras, file_path, rand_seed):
    # initialise
    # initials is a list of location and seed
    Z, Zb, D, Db = paras
#     l0, i0 = initials
    r, P_ts = nbi_para
    child_seeds = rand_seed.spawn(T)

    NewInf = np.zeros((num_fips, T))
    TotInf = np.zeros((num_fips, T))

    for l0, i0 in initials:
        NewInf[l0, 0] = i0
        TotInf[:, 0] = NewInf[:, 0]
    for ti in range(T):
        #         print('the current simulating time', ti)
        p_ti = P_ts.iloc[ti].values[1:]
        date_parts = P_ts.iloc[ti]['Date'].split('-')
        y, m, d = date_parts
#         WN = np.load(file_path+'M_{}{}_{}.csv')
        ################# Mobility#######################
        M_test = np.loadtxt(
            file_path+'M_{}{}_{}.csv'.format(y, m, d), delimiter=',')
        M_pop = M_test/pop[None, :]
        # if there is no one go out from one site
        temp = np.sum(M_pop, axis=0)
        if np.sum(temp == 0) == 0:
            M_pop_norm = M_pop/temp[None, :]
        else:
            #             print('there is no one go out from one site')
            mask = temp == 0
            temp[mask] = 1
            M_pop_norm = M_pop/temp[None, :]
            M_pop_norm[:, mask] = 0
            M_pop_norm[mask, mask] = 1
#         WN_small = M_pop_norm[:100,:100]
        WN = M_pop_norm  # WN_small/np.sum(WN_small,axis=0)[None,:]
        ########################################
        infectors = np.int64(NewInf[:, ti])
        pop_immu = 1-TotInf[:, ti]/pop[:]
        pop_immu[pop_immu < 0] = 0

        rng = np.random.default_rng(child_seeds[ti])
        totoal_new_infection_loc = []
        total_new = 0
        for i in range(num_fips):
            infectors_loc_i = infectors[i]
            if infectors_loc_i > 0:
                #                 print(i,infectors_loc_i)
                xx = np.arange(0, 100, 1)  # define the range of x values the
                p = p_ti[i]
                pmf = SSA.nbinom.pmf(xx, r, p)
                weights_n = pmf/np.sum(pmf)
                # np.array([2]*infectors_loc_i) # #rng
                ttt = np.random.choice(
                    len(weights_n), size=infectors_loc_i, p=weights_n)
                new_array = np.round(ttt*pop_immu[i])
                if len(new_array) > 0:
                    #                     print(ttt)
                    new_s = np.int64(np.sum(new_array))

                    totoal_new_infection_loc.extend([i]*new_s)
                    total_new = total_new + new_s
        # to be assigned, every new infections for the infector
#         total_new = np.round(tt*immu_all)
#         totoal_new_infection_loc = get_new_infections_position(
#             infectors, total_new, num_fips)
        z_num = np.int64(total_new)

        NF = np.zeros((2, z_num), dtype=np.int64)
        # for the time distribution
        latency_p = SSA.gamma.rvs(a=Z, scale=Zb, size=z_num, random_state=rng)
        infectious_p = SSA.gamma.rvs(
            a=D, scale=Db, size=z_num, random_state=rng)
        v = rng.random(z_num)
        delay_days = latency_p+v*infectious_p  # latency_p+v*infectious_p  # 3+5*0.5

        NF[0, :] = np.ceil(delay_days+ti)  # make it idx int
        # for the location distribution
        NF[1, :] = totoal_new_infection_loc
        df = pd.DataFrame(NF.T, columns=['time', 'o_l'])
        l_list = np.arange(num_fips)
        df['d_l'] = df['o_l'].apply(lambda x: rng.choice(
            l_list, size=1, p=WN[:, x])[0])  # np.random
        df = df[df['time'] <= (T-1)]
        NF_ii = np.array(df)
        for (t, o, d) in NF_ii:
            NewInf[d, t] = NewInf[d, t]+1
        TotInf = np.cumsum(NewInf, axis=1)

    return NewInf, TotInf
