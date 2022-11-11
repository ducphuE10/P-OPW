from ast import parse
from time import time
from utils import get_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from OPW import opw
import numpy as np
# from DTW import dtw_distance
from algo.l1_ot_dis import TrendOTDis
import ray
import argparse



@ray.remote
def opw_(X, Y):
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    return opw(X, Y)



@ray.remote
def robustOPW_(X, Y):
    return trend_ot_dis.dist(X, Y)


def dtw_(X, Y):
    X = np.array(X, dtype=np.double)
    Y = np.array(Y, dtype=np.double)
    return dtw_distance(X, Y)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, default=1)
    parser.add_argument('-l1', '--lambda1', type=float, default=1)
    parser.add_argument('-l2', '--lambda2', type=float, default=0.1)
    parser.add_argument('-k','--n_neighbors', type=int, default=1, help='number of neighbors')
    parser.add_argument('-t','--test_size', type=int, default=-1, help='test size')
    parser.add_argument('-m', '--method', type = str, default='opw',help = 'opw or topw1 or topw2')
    args = parser.parse_args()

    delta, lambda1, lambda2 = args.delta, args.lambda1, args.lambda2
    k = args.n_neighbors
    method = args.method

    trend_ot_dis = TrendOTDis(lambda1, lambda2, delta, method)

    X_train, y_train = get_data('FacesUCR/FacesUCR_TRAIN.txt')
    X_test, y_test = get_data('FacesUCR/FacesUCR_TEST.txt')


    

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train, y_test = np.array(y_train,dtype=np.int8), np.array(y_test,dtype=np.int8)

    test_size = args.test_size
    if test_size > 0:
        X_test = X_test[:test_size]
        y_test = y_test[:test_size]
    else:
        test_size = len(X_test)

    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('Train size: ', train_size)
    print('Test size: ', test_size)


    ray.init()
    start = time()

    results = []

    for i in range(train_size):
        for j in range(test_size):
            results.append(robustOPW_.remote(X_train[i], X_test[j]))

    results_values = ray.get(results)
    ray.shutdown()

    result = np.array(results_values).reshape(train_size, test_size)
    y_pred = y_train[np.argmin(result, axis=0)]

    print("Accuracy: ",accuracy_score(y_test, y_pred))  



    end = time()

    print(f"cost time: {end - start}")


