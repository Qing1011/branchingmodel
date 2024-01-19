import numpy as np
# import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import pandas as pd
import scipy.special as SS
import scipy.stats as SSA
import copy
import random
import math
from multiprocessing import Pool
import h5py
import sys

# not symmetric!! travel out is added up tp zero, travel in is not add to zero


# M_stocha_nbi.h5 is for negative binomial distribution
# M_stocha_poi.h5 is for poisson distribution


def state_M(M, states):
    N = M.shape[0]
    # sum up the prob and then sum the value
    state_outflow = (np.sum(M, axis=0)-np.diag(M))*states
    state_in_w = np.zeros_like(M)
    for j in range(N):
        state_in_w[:, j] = M[:, j]*(states[j])
    np.fill_diagonal(state_in_w, 0)
    state_inflow = np.sum(state_in_w, axis=1)
    return state_outflow, state_inflow


def SEIR_M_St(params, pop, initials, M, T, rand_seed, dt=1, nbi_r=False):
    """
    SEIR model with migration
    :param params: list of parameters [R0, Z, D]
    :param pop: population size of each location
    :param initials: list of initial values [start_pos, E0]
    :param M: migration matrix
    :param T: number of time steps
    :param dt: time step size
    """
    # Parameters
    R0 = params[0]
    Z = params[1]
    D = params[2]

    start_pos, E0 = initials
    N = len(pop)  # number of locations
    # initialize
    NewInf = np.zeros((N, T))
    NewInf[start_pos, 0] = E0
    x = np.zeros((N, 4))  # S, E, I for three populations
    x[:, 0] = pop
    x[start_pos, 0] = pop[start_pos] - E0  # Susceptibles
    x[start_pos, 1] = E0  # Exposed

    # Define distributions
    if nbi_r:
        def infection_distribution(mean_newinf, my_seed):
            # it has the size of number of counties
            p = nbi_r / (mean_newinf + nbi_r)
            rng = np.random.default_rng(my_seed)
            newinf = rng.negative_binomial(nbi_r, p)
            return newinf
    else:
        def infection_distribution(mean_newinf, my_seed):
            rng = np.random.default_rng(my_seed)
            return rng.poisson(mean_newinf)

    child_seeds = rand_seed.spawn(T*(1 / dt))
    for t in range(1, T):  # Starting from 1 because we already initialized NewInf at time 0
        #         print(t)
        for tt in range(int(1 / dt)):
            S = x[:, 0]
            E = x[:, 1]
            I = x[:, 2]
            R = x[:, 3]
            ds_expected = - R0 / D * S * I / pop
            mean_newinf = - ds_expected
            ti = t*10 + tt
            my_seed = child_seeds[ti]
            # ds is negative
            ds = - infection_distribution(mean_newinf, my_seed)
            NewInf[:, t] = - ds

            de = - ds - E/Z
            di = E/Z - I/D
            dr = I/D
            S += ds*dt
            E += de*dt
            I += di*dt
            R += dr*dt
            # then travel
            s_moveout, s_movein = state_M(M, S)
            e_moveout, e_movein = state_M(M, E)
            i_moveout, i_movein = state_M(M, I)
            r_moveout, r_movein = state_M(M, R)
            S += (- s_moveout + s_movein)*dt
            E += (- e_moveout + e_movein)*dt
            I += (- i_moveout + i_movein)*dt
            R += (- r_moveout + r_movein)*dt
    return NewInf


def parallel_run(args):
    return SEIR_M_St(*args)


def main():
    # Important guard for cross-platform use of multiprocessing

    s = sys.argv[1]
    s = int(s)  # parameter index
    es_idx = int(sys.argv[2])  # ensemble index
    ss = np.random.SeedSequence(es_idx)
    # file_dir = '../notebook/'
    file_dir = './'
    WN = np.loadtxt(file_dir + 'W_avg.csv')
    # Cave = np.loadtxt(file_dir + 'Cave.csv')
    pop = np.loadtxt(file_dir + 'pop_new.csv')
    rs = np.array([20, 1, 0.1, 0.025])
    r = rs[s]
    num_fips = len(pop)
    T = 60
    num_ens = 300
    Z = 3  # latent period
    Zb = 1  # scale parameter for Z
    D = 5  # infectious period
    Db = 1  # scale parameter for b
    alpha = 0.1  # reporting rate 10%
    # seeding
    l0 = 1859-1  # start with New York County NY in python -1, in matlab is 1859
    i0 = 100  # the starting t=0, in matlab it is 1
    initials = (l0, i0)
    R0 = 2.5

    New_Inf_s = SEIR_M_St([2.5, 3, 5], pop, initials, WN, T, ss, 0.1, r)
    np.savetxt(file_dir + 'New_Inf_nbi_{}_{}.csv'.format(r, es_idx),
               New_Inf_s, delimiter=',')


if __name__ == "__main__":
    main()

    # # for r in [0.025]:  # 20, 1, 0.1, 0.025
    #
    # args_list = [([2.5, 3, 5], pop, initials, WN, T, 0.1, r)
    #                 for _ in range(num_ens)]
    # pool = Pool()
    # results = pool.map(parallel_run, args_list)
    # pool.close()
    # pool.join()

    # with h5py.File(file_dir + 'M_stocha_nbi_{}.h5'.format(r), 'w') as hf:
    #     hf.create_dataset("data", data=results)

    # print('r = {} done'.format(r))

    # args_list = [([2.5, 3, 5], pop, initials, WN, T, 0.1, False)
    #              for _ in range(num_ens)]
    # pool = Pool()
    # results = pool.map(parallel_run, args_list)
    # pool.close()
    # pool.join()

    # with h5py.File(file_dir + 'M_stocha_poi.h5', 'w') as hf:
    #     hf.create_dataset("data", data=results)
