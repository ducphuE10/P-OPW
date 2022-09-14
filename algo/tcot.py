import ot
import numpy as np


def tcot(X, Y ,a=None,b=None, metric='euclidean', ld=50):

    tolerance = .5e-2
    maxIter = 20

    N = X.shape[0]
    M = Y.shape[0]
    dim = X.shape[1]
    if dim != Y.shape[1]:
        print("X and Y must have the same number of columns")
        return

    if a is None:
        a = np.ones((N, 1)) / N
    if b is None:
        b = np.ones((M, 1)) / M

    distance_matrix = ot.dist(X, Y, metric=metric)
    row_col_matrix = np.mgrid[1:N+1, 1:M+1]
    row = row_col_matrix[0] / N   # row = (i+1)/N
    col = row_col_matrix[1] / M   # col = (j+1)/M
    distance_time_matrix = np.abs(row - col) + 1

    D = distance_matrix * distance_time_matrix
    # print("Distance after:\n",distance_time_matrix)
    K = np.exp(-D / ld)
    # a = np.ones((N, 1)) / N
    # b = np.ones((M, 1)) / M

    compt = 0
    u = np.ones((N, 1)) / N

    while compt < maxIter:
        u = a / (K @ (b / (K.T @ u)))
        assert not np.isnan(u).any(), "nan in u"
        compt += 1

        if compt % 20 == 0 or compt == maxIter:
            v = b / (K.T @ u)
            u = a / (K @ v)

            criterion = np.linalg.norm(
                np.sum(np.abs(v * (K.T @ u) - b), axis=0), ord=np.inf)
            if criterion < tolerance:
                break

    U = K * D
    dis = np.sum(u * (U @ v))
    T = np.diag(u[:, 0]) @ K @ np.diag(v[:, 0])
    # dis = np.sum(T * D)
    # T = ot.sinkhorn(X, Y, D, reg=ld, numItermax=10000, log=True)
    # dis = ot.sinkhorn2(X, Y, D, reg=ld, numItermax=10000)
    return dis, T