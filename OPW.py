import numpy as np
import ot


def POT_feature(M,m,nb_dummies=1):
    a = np.ones(M.shape[0])
    b = np.ones(M.shape[1])

    if m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater"
                         " than 0.")
    elif m > np.min((np.sum(a), np.sum(b))):
        raise ValueError("Problem infeasible. Parameter m should lower or"
                         " equal than min(|a|_1, |b|_1).")

    b_extended = np.append(b, [(np.sum(a) - m) / nb_dummies] * nb_dummies)
    a_extended = np.append(a, [(np.sum(b) - m) / nb_dummies] * nb_dummies)
    M_extended = np.zeros((len(a_extended), len(b_extended)))
    M_extended[-nb_dummies:, -nb_dummies:] = np.max(M) * 2
    M_extended[:len(a), :len(b)] = M


def opw(X, Y, lambda1=1, lambda2=0.1, delta=1, metric='euclidean'):
    """preserved OT
    Args:
        X (ndarray): view1 (len_seq, feature)
        Y (ndarray): view2 (len_seq, feature)
        lambda1 (int, optional): weight of first term. Defaults to 50.
        lambda2 (float, optional): weight of second term. Defaults to 0.1.
        delta (int, optional): _description_. Defaults to 1.
    Returns:
        distance, ot_plan: distance is the distance between views, ot_plan is the transport plan
    """
    tolerance = .5e-2
    maxIter = 20

    N = X.shape[0]
    M = Y.shape[0]
    dim = X.shape[1]
    if dim != Y.shape[1]:
        print("X and Y must have the same number of columns")
        return

    # P = np.zeros((N, M))
    mid_para = np.sqrt((1/(N**2) + 1/(M**2)))

    # for i in range(N):
    #     for j in range(M):
    #         d = np.abs((i+1)/N - (j+1)/M) / mid_para
    #         P[i, j] = np.exp(-d**2/(2*delta**2)) / (delta*np.sqrt(2*np.pi))
    row_col_matrix = np.mgrid[1:N+1, 1:M+1]
    row = row_col_matrix[0] / N   # row = (i+1)/N
    col = row_col_matrix[1] / M   # col = (j+1)/M

    d_matrix = np.abs(row - col) / mid_para


    P = np.exp(-d_matrix**2/(2*delta**2)) / (delta*np.sqrt(2*np.pi))

    # S = np.zeros((N, M))
    # for i in range(N):
    #     for j in range(M):
    #         S[i, j] = lambda1/(((i+1)/N - (j+1)/M) ** 2 + 1)
    # S = lambda1 / ((row - col) ** 2 + 1)
    S = lambda1 * (row - col) ** 2

    D = ot.dist(X, Y, metric=metric)

    # Clip the distance matrix to prevent numerical errors
    max_distance = 200 * lambda2
    D = np.clip(D, 0, max_distance)

    K = np.exp((S - D) / lambda2) * P


    a = np.ones((N, 1)) / N
    b = np.ones((M, 1)) / M

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

    # return dis,T
    return dis



