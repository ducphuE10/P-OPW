import numpy as np 
from trend_decompose import l1filter
import ot


def res2prob(residual):
    # residual = (residual - np.mean(residual))/np.std(residual)
    # residual = (residual - np.min(residual))/(np.max(residual) - np.min(residual))
    residual = np.abs(residual)
    # residual = (residual - np.mean(residual))/np.std(residual)
    # residual = residual/np.median(residual)
    residual = np.exp(-residual/2)
    probs = residual/np.sum(residual)
    return probs

def get_prob(y1, y2, lam1=0.5, lam2=0.5):
    y1_trend = l1filter(y1,lam1)
    y2_trend = l1filter(y2,lam2)
    y1_res = y1 - y1_trend
    y2_res = y2 - y2_trend

    y1_prob = res2prob(y1_res).reshape(-1,1)
    y2_prob = res2prob(y2_res).reshape(-1,1)

    join_prob = y1_prob.dot(y2_prob.T)

    return y1_prob, y2_prob


def robustOPW(D_hat ,a,b, lambda1=50, lambda2=0.1, delta=1, metric='euclidean'):
    '''
    Input
    D: distance matrix
    a: probabilities (normalization)
    b: probabilities (normalization)
    Return
    distance: distance between two sequences
    T: 
    '''



    

    T = ot.sinkhorn(a, b, D_hat)

    return 0