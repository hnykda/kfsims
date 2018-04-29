import numpy as np
from scipy.stats import multivariate_normal as mvn


def static_noise(N, mod=1, d=2):
    return mvn.rvs(cov=np.eye(d) * mod, size=N)


def hill_noise(N, low=1, mid=10, top=15):
    """  ____
        /
    ___/
    """
    lower = mvn.rvs(cov=np.eye(2) * low, size=50)
    middle = np.array([mvn.rvs(cov=np.eye(2) * i, size=1) for i in range(mid)])
    upper = mvn.rvs(cov=np.eye(2) * top, size=N - mid - 20)
    return np.concatenate([lower, middle, upper])


def sin_noise(N, sin_halves=2, shift=0):
    a = np.sin([np.pi * (sin_halves * i / N) + shift/np.pi for i in range(N)])
    return np.array([a, a]).T
