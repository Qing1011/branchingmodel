import numpy as np
import random
import math

def mexFunction(nlmxhs, plhs, nrhs, prhs):

    # declare variables
    mxnl, mxpart, mxw, mxT0, mxNewInf, mxpara, mxpop = prhs
    # input variables
    nl = mxnl.copy()
    part = mxpart.copy()
    w = mxw.copy()
    T0 = mxT0.copy()
    NewInf = mxNewInf.copy()
    para = mxpara.copy()
    pop = mxpop.copy()
    # figure out dimensions
    num_mp, num_loc = nl.shape[0], part.shape[0]-1
    T = NewInf.shape[1]
    # associate outputs
    mxNewInf1 = plhs[0] = np.zeros((num_loc, T), dtype=np.double)
    # output variables
    NewInf1 = mxNewInf1
    # do something
    generator = random.Random()
    generator.seed() # uses system time as seed
    # initialize auxiliary variables
    totalinfection = np.zeros(num_loc, dtype=np.double)
    # change index in nl and part (0-based index)
    nl -= 1
    part -= 1
    Tcnt = T0[0]-1 # the current time (note index from 0)
    # total infection
    for l in range(num_loc):
        totalinfection[l] = NewInf[l:Tcnt*num_loc+1:num_loc].sum()
    # initialize NewInf1
    NewInf1[:,:] = NewInf[:,:]
    # prepare generators
    # para: R0,r,Z;Zb;D;Db,alpha
    R0, r = para[0], para[1]
    p = r / (R0 + r)
    # use a discrete distribution to generate NB
    # prepare NB pdf
    weights = []
    for k in range(500):
        weight = math.exp(-para[5]*(k**para[6]))
        weights.append(weight)
    # normalize pdf
    sum_weights = sum(weights)
    weights = [w/sum_weights for w in weights]
    # generate NB
    nb_rvs = np.random.choice(np.arange(500), size=100000, replace=True, p=weights)
    # branching process simulation
    for i in range(num_mp):
        k = 0
        while True:
            # infectors
            inf_inds = np.random.choice(np.where(pop[nl[i],:] > 0)[0], size=pop[nl[i],:].sum(), replace=True)
            newinfection = len(inf_inds)
            if newinfection == 0:
                break
            k += 1
            # locations of infectors
            locs = np.searchsorted(part, inf_inds, side='right') - 1
            # infected individuals
            infected = np.zeros(num_loc, dtype=np.double)
            for j in range(newinfection):
                loc = locs[j]
                v = generator.uniform(0, 1)
                if v < w[loc]:
                    infected[loc] += 1
            # update NewInf1
            NewInf1[:, Tcnt] += infected
            # stop simulation if no new infections
            if sum(infected) == 0:
                break
            # stopping rule: exceed threshold of total infections
            if k >= nb_rvs[newinfection-1]:
                break
        # move to the next subpopulation
        Tcnt += 1
       
