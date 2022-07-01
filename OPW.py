import numpy as np
import ot
import cv2

def opw(X, Y, lambda1=1, lambda2=0.1, delta=1, metric='euclidean'):
    """preserved OT
    Args:
        X (ndarray): view1
        Y (ndarray): view2
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
    S = lambda1 / ((row - col) ** 2 + 1)

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

    return dis


# import os
# import seaborn as sns
# import matplotlib.pyplot as plt
# from os import listdir
# from os.path import isfile, join
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# N = 10
# d = 3

# data1 = np.arange(N*d).reshape(N, d)
# # perm = np.array([1, 2, 0, 5, 4, 3, 8, 9, 6, 7])
# perm = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
# data2 = data1.copy()[perm]

# data1 = []
# path_data1 = ['img/img1/'+f for f in listdir('img/img1')]
# for p1 in path_data1:
#     img = cv2.imread(p1)
#     # img = resize(p1,1)
#     flat_img = img.reshape((-1,))
#     data1.append(flat_img)


# data1 = np.array(data1)
# print(data1.shape)
#
# # perm = np.array([11,10,9,8,7, 6, 5, 4, 3, 2, 1, 0])
# # data2 = data1.copy()[perm]
# #
# data2 = []
# path_data2 = ['img/img2/'+f for f in listdir('img/img2')]
# for p2 in path_data2:
#     img = cv2.imread(p2)
#     # img = resize(p2, 10)
#     flat_img = img.reshape((-1,))
#     data2.append(flat_img)
#
# data2 = np.array(data2)
# print(data2.shape)
#
#
#
#
# dis, T = opw(data1, data2, lambda1=0.01, lambda2=0.001, delta=30,metric='euclidean')
# # print(perm)
# sns.heatmap(T)
# plt.show()