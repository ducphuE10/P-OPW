import ot
import numpy as np
from abc import ABC, abstractmethod

def relative_element_trans(x, metric='euclidean'):
    if metric == 'euclidean':
        x_tru =  np.concatenate((np.zeros((1,x.shape[1])), x[:-1]), axis=0)
        distance = np.linalg.norm(x - x_tru, axis=1)
        fx = np.cumsum(distance)
        sum_distance = fx[-1]
        fx = fx / sum_distance
        return fx


class BaseOrderPreserve:
    def __init__(self,lambda1, lambda2, delta):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.delta = delta

    @abstractmethod
    def get_d_matrix(self, x1, x2):
        pass

    def fit(self, x1, x2, x1_trend=None, x2_trend=None, a=None, b=None, metric='euclidean'):
        """
        Parameters
        ---------
        x1 : array-like, shape (n1, d)
            matrix with `n1` samples of dimension `d`

        x2 : array-like, shape (n2, d), optional
            matrix with `n2` samples of dimension `d` (if None then :math:`\mathbf{x_2} = \mathbf{x_1}`)

        a : array-like, shape (n1, )
        b : array-like, shape (n2, )

        metric : str, optional

        Returns
        -------
        dist : float distance between :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`
        M : array-like, shape (`n1`, `n2`)
            transportation matrix between samples in :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`
        """
        tolerance = .5e-2
        maxIter = 200


        N = x1.shape[0]
        M = x2.shape[0]


        if a is None:
            a = np.ones((N, 1)) / N
        if b is None:
            b = np.ones((M, 1)) / M


        if x1.ndim == 1:
            x1 = x1.reshape(-1, 1)
            x2 = x2.reshape(-1, 1)

        if x1_trend is None:
            x1_trend = x1
        if x2_trend is None:
            x2_trend = x2

        d_matrix = self.get_d_matrix(x1, x2)
        # d_matrix = self.get_d_matrix(x1_trend, x2_trend)
        P = np.exp(-d_matrix ** 2 / (2 * self.delta ** 2)) / (self.delta * np.sqrt(2 * np.pi))
        P = a@b.T * P



        row_col_matrix = np.mgrid[1:N + 1, 1:M + 1]
        row = row_col_matrix[0] / N  # row = (i+1)/N
        col = row_col_matrix[1] / M  # col = (j+1)/M
        S = self.lambda1 / ((row - col) ** 2 + 1)

        D = ot.dist(x1, x2, metric=metric)
        # D = D_goc

        # max_distance = 200 * self.lambda2
        # D = np.clip(D, 0, max_distance)

        K = np.exp((S - D) / self.lambda2) * P

        compt = 0
        u = np.ones((N, 1)) / N

        while compt < maxIter:

            u = a / (K @ (b / (K.T @ u)))
            # assert not np.isnan(u).any(), "nan in u"
            if np.isnan(u).any():
                self.dis = np.inf
                return
            compt += 1

            if compt % 20 == 0 or compt == maxIter:
                v = b / (K.T @ u)
                u = a / (K @ v)

                criterion = np.linalg.norm(
                    np.sum(np.abs(v * (K.T @ u) - b), axis=0), ord=np.inf)
                if criterion < tolerance:
                    break


        U = K * D
        self.dis = np.sum(u * (U @ v))
        self.T = np.diag(u[:, 0]) @ K @ np.diag(v[:, 0])

        # return self.dis, self.T



    def get_distance(self):
        return self.dis

    def get_transportation_matrix(self):
        return self.T


class TOPWBase(BaseOrderPreserve):
    def _cal_d_matrix_from_mn(self, m, n, M, N):
        pass

    def get_d_matrix(self, x1, x2):
        N = x1.shape[0]
        M = x2.shape[0]

        fx = relative_element_trans(x1)
        fy = relative_element_trans(x2)
        m,n = np.meshgrid(fy, fx)
        # d_matrix = np.maximum(m,n) / np.minimum(m,n)
        d_matrix = self._cal_d_matrix_from_mn(m,n, M, N)
        return d_matrix


class TOPW2(TOPWBase):
    def __init__(self, lambda1, lambda2, delta):
        super().__init__(lambda1, lambda2, delta)

    def _cal_d_matrix_from_mn(self, m, n, M, N):
        return np.abs(m - n) / np.sqrt(1 / N **2 + 1 / M ** 2)


class TOPW1(TOPWBase):
    r"""Solve ot with :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}` with topw1 prior.


    """
    def __init__(self, lambda1, lambda2, delta):
        super().__init__(lambda1, lambda2, delta)

    def _cal_d_matrix_from_mn(self, m, n, M, N):
        return np.maximum(m,n) / np.minimum(m,n)


class OPW(BaseOrderPreserve):
    def __init__(self, lambda1, lambda2, delta):
        super().__init__(lambda1, lambda2, delta)

    def get_d_matrix(self, x1, x2):

        N = x1.shape[0]
        M = x2.shape[0]

        mid_para = np.sqrt((1/(N**2) + 1/(M**2)))

        row_col_matrix = np.mgrid[1:N+1, 1:M+1]
        row = row_col_matrix[0] / N   # row = (i+1)/N
        col = row_col_matrix[1] / M   # col = (j+1)/M

        d_matrix = np.abs(row - col) / mid_para
        return d_matrix


