import numpy as np

def negative_binomial(n, p, size=None):
    # Sample from a negative binomial distribution.
    # n: number of successes
    # p: probability of success
    # size: number of samples
    # return: samples   
    return np.random.negative_binomial(n, p, size)

def ising_model(n, beta, size=None):
    # Sample from an Ising model.
    # n: number of spins
    # beta: inverse temperature
    # size: number of samples
    # return: samples
    return np.random.binomial(1, 1 / (1 + np.exp(-2 * beta)), size=(size, n))



