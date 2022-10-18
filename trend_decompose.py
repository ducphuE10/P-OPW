from itertools import chain
from cvxopt import blas, lapack, solvers
from cvxopt import matrix, spmatrix, sin, mul, div, normal, spdiag
solvers.options['show_progress'] = 0

# Code get from : https://github.com/elsonidoq/py-l1tf

def get_second_derivative_matrix(n):
    """
    :param n: The size of the time series
    :return: A matrix D such that if x.size == (n,1), D * x is the second derivate of x
    """
    m = n - 2
    D = spmatrix(list(chain(*[[1, -2, 1]] * m)),
                 list(chain(*[[i] * 3 for i in range(m)])),
                 list(chain(*[[i, i + 1, i + 2] for i in range(m)])))
    return D


def _l1tf(corr, delta):
    """
        minimize    (1/2) * ||x-corr||_2^2 + delta * sum(y)
        subject to  -y <= D*x <= y
    Variables x (n), y (n-2).
    :param x:
    :return:
    """

    n = corr.size[0]
    m = n - 2

    D = get_second_derivative_matrix(n)

    P = D * D.T
    q = -D * corr

    G = spmatrix([], [], [], (2*m, m))
    G[:m, :m] = spmatrix(1.0, range(m), range(m))
    G[m:, :m] = -spmatrix(1.0, range(m), range(m))

    h = matrix(delta, (2*m, 1), tc='d')

    res = solvers.qp(P, q, G, h)

    return corr - D.T * res['x']


from cvxopt import matrix, spmatrix, sin, mul, div, normal, spdiag
import pandas as pd
# from .impl import _l1tf
import numpy as np

def l1filter(corr, delta):
    """
    :param corr: Corrupted signal, should be a numpy array / pandas Series
    :param delta: Strength of regularization
    :return: The filtered series
    """

    m = float(corr.min())
    M = float(corr.max())
    denom = M - m
    # if denom == 0, corr is constant
    t = (corr-m) / (1 if denom == 0 else denom)

    if isinstance(corr, np.ndarray):
        values = matrix(t)
    elif isinstance(corr, pd.Series):
        values = matrix(t.values[:])
    else:
        raise ValueError("Wrong type for corr")

    values = _l1tf(values, delta)
    values = values * (M - m) + m

    if isinstance(corr, np.ndarray):
        values = np.asarray(values).squeeze()
    elif isinstance(corr, pd.Series):
        values = pd.Series(values, index=corr.index, name=corr.name)

    return values


def remove_outliers(t, delta, mad_factor=3):
    """
    :param t: an instance of pd.Series
    :param delta: parameter for l1tf function
    """
    filtered_t = l1tf(t, delta)

    diff = t.values - np.asarray(filtered_t).squeeze()
    t = t.copy()
    t[np.abs(diff - np.median(diff)) > mad_factor * mad(diff)] = np.nan

    t = t.fillna(method='ffill').fillna(method='bfill')
    return t


def strip_na(s):
    """
    :param s: an instance of pd.Series
    Removes the NaN from the extremes
    """
    m = s.min()
    lmask = s.fillna(method='ffill').fillna(m-1) == m-1
    rmask = s.fillna(method='bfill').fillna(m-1) == m-1
    mask = np.logical_or(lmask, rmask)
    return s[np.logical_not(mask)]

def df_l1tf(df, delta=3, remove_outliers=False, mad_factor=3):
    """
    Applies the l1tf function to the whole dataframe optionally removing outliers
    :param df: A pandas Dataframe
    :param delta: The delta parameter of the l1tf function
    :param remove_outliers: Whether outliers should be removed
    :param mad_factor: Strength of the outlier detection technique
    """
    l1tf_d = {}
    if remove_outliers: wo_outliers_d = {}
    ks = df.keys()

    for i, k in enumerate(ks):
        if i % 50 == 0: print(i, 'of', len(ks))
        t = strip_na(df[k])

        if remove_outliers:
            t = remove_outliers(t, delta, mad_factor)
            wo_outliers_d[k] = t
        s = l1tf(t, delta)
        l1tf_d[k] = s

    if remove_outliers:
        return pd.DataFrame(l1tf_d), pd.DataFrame(wo_outliers_d)
    else:
        return pd.DataFrame(l1tf_d)


#--------------------------------------------------------------------------------------------------------------------





def soft_thresholding(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def huber_loss(x, threshold):
    return np.where(np.abs(x) < threshold, 0.5 * x ** 2, threshold * (np.abs(x) - 0.5 * threshold))


def huber_loss_derivative(x, threshold):
    return np.where(np.abs(x) < threshold, x, threshold * np.sign(x))

def diagonal_huber(x, threshold):
    # return np.diag(np.divide(huber_loss_derivative(x, threshold),huber_loss(x, threshold)))
    return np.diag(np.divide(huber_loss_derivative(x, threshold),x))

def update_trend(D, y, trend, u, z, threshold):
    A = diagonal_huber(y - trend, threshold)
    trend = y - threshold * np.linalg.inv(( A + threshold * D.T @ D)) @ D.T @ (u - z + D @ y)
    return trend

def robustTrend(y, lambda1=0.4, lambda2=0.4, penalty_parameter=0.9, max_iter=200):
    n = len(y)
    t = np.zeros((n,))
    z = np.zeros((2 * n,))
    u = np.zeros((2 * n,))

    #first order matrix
    D1 = np.diag(np.ones(n-1),1) - np.diag(np.ones(n-1),-1)
    #second order matrix
    D2 = np.diag(np.ones(n-2),2) - 2*np.diag(np.ones(n-1),1) + np.diag(np.ones(n-2),-2)

    D = np.concatenate((lambda1*D1, lambda2*D2), axis=0)

    for iteration in range(max_iter):
        x = y - t
        A = diagonal_huber(x, penalty_parameter)
        t = update_trend(D, y, t, u, z, penalty_parameter)
        z = soft_thresholding(D @ t + u, penalty_parameter)
        u = u + D @ t - z
    return t