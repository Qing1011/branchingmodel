import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

def SEIR_M(params, pop, initials,M, T, dt=0.01):
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

    start_pos = initials[0]
    E0 = initials[1]

    N = len(pop)  # number of locations
    map_arr = [] # create a idx list to exclude the locations of flowin
    for i in range(N):
        indices = list(range(N))
        indices.remove(i)
        map_arr.append(indices)
    map_arr = np.array(map_arr)
    
    # initialize
    NewInf = np.zeros((N, T))
    NewInf[start_pos, 0] = E0
    x = np.zeros((N, 3))  # S, E, I for three populations
    x[:,0] = pop
    x[start_pos, 0] = pop[start_pos] - E0  # Susceptibles
    x[start_pos, 1] = E0  # Exposed

    for t in range(1, T):  # Starting from 1 because we already initialized NewInf at time 0
        print(t)
        for _ in range(int(1 / dt)):
            xnew = x.copy()
            infection = (R0 / D * x[:, 0] * x[:, 2] / pop) * dt  
            NewInf[:, t] += infection
            # print(infection)
            migration_outflow = (np.sum(M,axis=0) - np.diag(M))[:,None] * x ### sum up the prob and then sum the value
    #         migration_inflow = np.einsum('ij,ijk->ik', M[map_arr, np.arange(3)[:, None]], x[map_arr])
            migration_inflow = np.einsum('ij,ijk->ik', M[np.arange(N)[:, None], map_arr], x[map_arr])
            
            xnew[:, 0] = xnew[:, 0] - infection + migration_inflow[:, 0] - migration_outflow[:, 0]
            xnew[:, 1] = xnew[:, 1] + infection - (x[:, 1] / Z * dt) + migration_inflow[:, 1] - migration_outflow[:, 1]
            xnew[:, 2] = xnew[:, 2] + (x[:, 1] / Z * dt) - (x[:, 2] / D * dt) + migration_inflow[:, 2] - migration_outflow[:, 2]
            x = xnew

    return NewInf