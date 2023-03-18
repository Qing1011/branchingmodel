import numpy as np

def negative_binomial(n, p, size=None):
    # Sample from a negative binomial distribution.
    # n: number of successes
    # p: probability of success
    # size: number of samples
    # return: samples   
    return np.random.negative_binomial(n, p, size)



