import logging
from time import time
from utils import get_data
from sklearn.metrics import accuracy_score
from OPW import opw
import numpy as np
from DTW import dtw_distance
from algo.l1_ot_dis import TrendOTDis
import ray
import argparse
import os
from algo.t_opw1 import t_opw1
from algo.drop_dtw import drop_dtw_cost
from progress_bar import ProgressBar
import pandas as pd
import sys
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean


DATA_FOLDER = 'UCR_data/UCRArchive_2018'

@ray.remote
def opw_(X, Y, pba=None):
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    dist = opw(X, Y)
    if pba is not None:
        pba.update.remote(1)
    return dist

@ray.remote
def opw1(X,Y, pba=None):
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    dist =  t_opw1(X, Y)[0]
    if pba is not None:
        pba.update.remote(1)
    return dist

@ray.remote
def dtw_(X, Y, pba=None):
    X = np.array(X, dtype=np.double)
    Y = np.array(Y, dtype=np.double)
    dist = dtw_distance(X, Y)
    if pba is not None:
        pba.update.remote(1)
    return dist

@ray.remote
def sdtw_(X, Y, pba=None):
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    D = SquaredEuclidean(X, Y)
    sdtw_instance = SoftDTW(D, gamma=1.0)
    dist = sdtw_instance.compute()
    if pba is not None:
        pba.update.remote(1)
    return dist

@ray.remote
def drop_dtw(X, Y, pba=None):
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    dist = drop_dtw_cost(X, Y)
    if pba is not None:
        pba.update.remote(1)
    return dist







if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FacesUCR')
    parser.add_argument('--delta', type=float, default=1)
    parser.add_argument('-l1', '--lambda1', type=float, default=1)
    parser.add_argument('-l2', '--lambda2', type=float, default=0.1)
    parser.add_argument('-k','--n_neighbors', type=int, default=1, help='number of neighbors')
    parser.add_argument('-t','--test_size', type=int, default=-1, help='test size')
    parser.add_argument('-m', '--method', type=str, default='opw', help='method in [opw, dtw, softdtw]')
    parser.add_argument('-n', '--n_jobs', type=int, default=8, help='number of jobs')


    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=f'./logs/{args.dataset}_{args.method}.log', filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(args)

    delta, lambda1, lambda2 = args.delta, args.lambda1, args.lambda2
    k = args.n_neighbors
    method = args.method
    dist_func = None
    if method == 'opw':
        dist_func = opw_
    elif method == 'dtw':
        dist_func = dtw_
    elif method == 'softdtw':
        dist_func = sdtw_
    elif method == 'drop_dtw':
        dist_func = drop_dtw

    train_path = os.path.join(DATA_FOLDER, args.dataset, args.dataset + '_TRAIN.tsv')
    test_path = os.path.join(DATA_FOLDER, args.dataset, args.dataset + '_TEST.tsv')

    X_train, y_train = get_data(train_path)
    X_test, y_test = get_data(test_path, dataset_size=args.test_size)


    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train, y_test = np.array(y_train,dtype=np.int8), np.array(y_test,dtype=np.int8)

    logging.info('X_train shape: {}'.format(X_train.shape))
    logging.info('X_test shape: {}'.format(X_test.shape))


    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    print('Train size: ', train_size)
    print('Test size: ', test_size)


    ray.init(num_cpus=args.n_jobs)
    pb = ProgressBar(total=train_size * test_size)
    actor = pb.actor
    start = time()

    results = []

    # import ipdb; ipdb.set_trace()
    for i in range(train_size):
        for j in range(test_size):
            results.append(dist_func.remote(X_train[i], X_test[j], actor))
            # a = dist_func(X_train[i], X_test[j])
    pb.print_until_done()
    results_values = ray.get(results)
    ray.shutdown()


    result = np.array(results_values).reshape(train_size, test_size)
    y_pred = y_train[np.argmin(result, axis=0)]

    accuracy = accuracy_score(y_test, y_pred)
    logging.info("Accuracy: {}".format(accuracy))

    end = time()
    cost_time = end - start
    # print(f"cost time: ", cost_time)
    logging.info("cost time: {}".format(cost_time))
    experiment_file_path = './exp/experiments_results_newopw.csv'

    df = pd.read_csv(experiment_file_path, index_col=0)
    #add new row if not exist
    df.loc[args.dataset, args.method] = accuracy
    df.to_csv(experiment_file_path)


