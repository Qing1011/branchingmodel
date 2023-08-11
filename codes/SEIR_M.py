import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat


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


def SEIR_M(params, pop, initials, M, T, dt=1):
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

    for t in range(1, T):  # Starting from 1 because we already initialized NewInf at time 0
        #         print(t)
        for _ in range(int(1 / dt)):
            S = x[:, 0]
            E = x[:, 1]
            I = x[:, 2]
            R = x[:, 3]
    #             x_new = x.copy()
            ds = - R0 / D * S * I / pop
    #             NewInf[:, t] += infection
            NewInf[:, t] = - ds  # x[:, 1]/Z #ds
            de = R0 / D * S * I / pop - E/Z
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
    return x, NewInf


def SEIR_M_old(params, pop, initials, M, T, dt=0.01):
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
    map_arr = []  # create a idx list to exclude the locations of flowin
    for i in range(N):
        indices = list(range(N))
        indices.remove(i)
        map_arr.append(indices)
    map_arr = np.array(map_arr)

    # initialize
    NewInf = np.zeros((N, T))
    NewInf[start_pos, 0] = E0
    x = np.zeros((N, 3))  # S, E, I for three populations
    x[:, 0] = pop
    x[start_pos, 0] = pop[start_pos] - E0  # Susceptibles
    x[start_pos, 1] = E0  # Exposed

    for t in range(1, T):  # Starting from 1 because we already initialized NewInf at time 0
        print(t)
        for _ in range(int(1 / dt)):
            xnew = x.copy()
            infection = (R0 / D * x[:, 0] * x[:, 2] / pop) * dt
            NewInf[:, t] += infection
            # print(infection)
            # sum up the prob and then sum the value
            migration_outflow = (np.sum(M, axis=0) - np.diag(M))[:, None] * x
    #         migration_inflow = np.einsum('ij,ijk->ik', M[map_arr, np.arange(3)[:, None]], x[map_arr])
            migration_inflow = np.einsum(
                'ij,ijk->ik', M[np.arange(N)[:, None], map_arr], x[map_arr])

            xnew[:, 0] = xnew[:, 0] - infection + \
                migration_inflow[:, 0] - migration_outflow[:, 0]
            xnew[:, 1] = xnew[:, 1] + infection - \
                (x[:, 1] / Z * dt) + migration_inflow[:, 1] - \
                migration_outflow[:, 1]
            xnew[:, 2] = xnew[:, 2] + (x[:, 1] / Z * dt) - (x[:, 2] / D * dt) + \
                migration_inflow[:, 2] - migration_outflow[:, 2]
            x = xnew

    return NewInf


def SEIR():
    N = 1628706  # 1000  # Population size
    E0, I0, R0 = 100, 0, 0  # Initial conditions
    beta = 0.5  # Infection rate
    gamma = 1/5  # Recovery rate
    sigma = 1/3  # Rate at which an exposed person becomes infective
    T = 60  # Number of time steps

    S = N - E0 - I0 - R0
    E, I, R = E0, I0, R0

    S_list, E_list, I_list, R_list = [S], [E], [I], [R]
    new_inf_list = [0]
    dt = 0.01
    for _ in range(5900):
        new_inf = beta * S * I / N

        dS = -beta * S * I / N
        dE = beta * S * I / N - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I

        S += dS * dt
        E += dE * dt
        I += dI * dt
        R += dR * dt

        S_list.append(S)
        E_list.append(E)
        I_list.append(I)
        R_list.append(R)
        new_inf_list.append(new_inf*dt)
        re = np.stack([S_list, E_list, I_list, R_list, new_inf_list])
    return re
