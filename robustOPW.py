import numpy as np
from trend_decompose import l1filter, robustTrend
import ot



class L1TrendFilter:
    def __init__(self, lambda1=0.5, lambda2=0.5):
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def __repr__(self):
        return 'L1TrendFilter(lambda1={}, lambda2={})'.format(self.lambda1, self.lambda2)

    def res2prob(self, residual):
        residual = np.abs(residual)
        residual = np.exp(-residual/3)
        probs = residual/np.sum(residual)
        return probs

    def fit(self, x1, x2):
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)
        res1 = x1 - l1filter(x1, self.lambda1)
        res2 = x2 - l1filter(x2, self.lambda2)
        x1_prob = self.res2prob(res1)
        x2_prob = self.res2prob(res2)

        self.x1_prob = x1_prob
        self.x2_prob = x2_prob          
        self.x1_trend = x1 - res1
        self.x2_trend = x2 - res2
        return x1_prob, x2_prob

    def get_prob(self, x1, x2):
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)
        res1 = l1filter(x1, self.lambda1)
        res2 = l1filter(x2, self.lambda2)
        x1_prob = self.res2prob(res1)
        x2_prob = self.res2prob(res2)
        return x1_prob, x2_prob

    def get_trend(self):
        return self.x1_trend, self.x2_trend




class RobustTrendFilter(L1TrendFilter):
    def __init__(self, lambda1=0.5, lambda2=0.5, penalty=0.9, max_iter=20):
        super().__init__(lambda1, lambda2)
        self.penalty = penalty
        self.max_iter = max_iter

    def __repr__(self):
        return 'RobustTrendFilter(lambda1={}, lambda2={}, penalty={}, max_iter={})'.format(self.lambda1, self.lambda2, self.penalty, self.max_iter)

    def fit(self, x1, x2):
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)
        trend1 = robustTrend(x1, lambda1=self.lambda1, lambda2=self.lambda2, penalty_parameter=self.penalty, max_iter=self.max_iter)
        trend2 = robustTrend(x2, lambda1=self.lambda1, lambda2=self.lambda2, penalty_parameter=self.penalty, max_iter=self.max_iter)
        x1_prob = self.res2prob(x1 - trend1)
        x2_prob = self.res2prob(x2 - trend2)
        self.x1_trend, self.x2_trend = trend1, trend2
        return x1_prob, x2_prob

    def get_prob(self, x1, x2):
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)
        trend1 = robustTrend(x1, lambda1=self.lambda1, lambda2=self.lambda2, penalty_parameter=self.penalty, max_iter=self.max_iter)
        trend2 = robustTrend(x2, lambda1=self.lambda1, lambda2=self.lambda2, penalty_parameter=self.penalty, max_iter=self.max_iter)
        x1_prob = self.res2prob(x1 - trend1)
        x2_prob = self.res2prob(x2 - trend2)
        return x1_prob, x2_prob





def res2prob(residual):
    residual = np.abs(residual)
    residual = np.exp(-residual/3)
    probs = residual/np.sum(residual)
    return probs

def get_prob(y1, y2, lam1=0.5, lam2=0.5):
    y1_trend = l1filter(y1,lam1)
    y2_trend = l1filter(y2,lam2)
    y1_res = y1 - y1_trend
    y2_res = y2 - y2_trend

    y1_prob = res2prob(y1_res).reshape(-1,1)
    y2_prob = res2prob(y2_res).reshape(-1,1)
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