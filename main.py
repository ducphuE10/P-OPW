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
from progress_bar import ProgressBar
import pandas as pd
import sys
from datetime import datetime
from utils import get_distance
from popw import entropic_opw_2
DATA_FOLDER = 'UCR_data/UCRArchive_2018'

@ray.remote
def popw_(X, Y, lambda1, lambda2, delta = 1, m=0.8,pba=None):
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    D = get_distance(X,Y)
    a = np.ones(X.shape[0])/X.shape[0]
    b = np.ones(Y.shape[0])/Y.shape[0]
    dist = entropic_opw_2(a, b, D, lambda1, lambda2, delta, m ,dropBothSides=True)
    if pba is not None:
        pba.update.remote(1)
    return dist


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
def robustOPW_(X, Y, pba=None, mean=None, std=None):
    dist =  trend_ot_dis.dist(X, Y, mean=mean, std=std)
    if pba is not None:
        pba.update.remote(1)
    return dist


def dtw_(X, Y):
    X = np.array(X, dtype=np.double)
    Y = np.array(Y, dtype=np.double)
    return dtw_distance(X, Y)





if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FaceUCR')
    parser.add_argument('--delta', type=float, default=1)
    parser.add_argument('-l1', '--lambda1', type=float, default=50)
    parser.add_argument('-l2', '--lambda2', type=float, default=0.1)
    parser.add_argument('-k','--n_neighbors', type=int, default=1, help='number of neighbors')
    parser.add_argument('-t','--test_size', type=int, default=-1, help='test size')
    parser.add_argument('-m', '--method', type=str, default='opw', help='method in [popw ,opw, topw1, topw2]')
    parser.add_argument('-n', '--n_jobs', type=int, default=8, help='number of jobs')
    parser.add_argument('-e', '--exp_file', type=str, default='exp', help='exp file')

    trend_parser = parser.add_argument_group('Trend filter args')
    trend_parser.add_argument('-tmethod','--trend_method', type=str, default='l1', help='trend filter method in [l1, robust]')
    trend_parser.add_argument('-tl1','--trend_lambda1', type=float, default=0.4)
    trend_parser.add_argument('-tl2','--trend_lambda2', type=float, default=0.4)
    trend_parser.add_argument('-tp','--trend_penalty', type=float, default=0.9, help='penalty for robust trend filter')
    trend_parser.add_argument('-tmax_iter','--trend_max_iter', type=int, default=20)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename=f'./logs/{args.dataset}_{args.method}_{args.trend_method}.log', filemode='w')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info(args)
    trend_args = args

    delta, lambda1, lambda2 = args.delta, args.lambda1, args.lambda2
    k = args.n_neighbors
    method = args.method

    trend_ot_dis = TrendOTDis(lambda1=lambda1, lambda2=lambda2, delta=delta, algo=method, trend_method=args.trend_method, trend_args=trend_args)

    train_path = os.path.join(DATA_FOLDER, args.dataset, args.dataset + '_TRAIN.tsv')
    test_path = os.path.join(DATA_FOLDER, args.dataset, args.dataset + '_TEST.tsv')

    X_train, y_train = get_data(train_path)
    X_test, y_test = get_data(test_path, dataset_size=args.test_size)




    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)

    # X_train = (X_train - X_train_mean) / X_train_std
    # X_test = (X_test - X_train_mean) / X_train_std

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

    for i in range(train_size):
        for j in range(test_size):
            results.append(robustOPW_.remote(X_train[i], X_test[j], actor, mean=X_train_mean, std=X_train_std))
            # results.append(opw_.remote(X_train[i], X_test[j]))
            # results.append(opw1.remote(X_train[i], X_test[j], actor))
            # opw1(X_train[i], X_test[j])
            # robustOPW_(X_train[i], X_test[j])
    pb.print_until_done()
    results_values = ray.get(results)
    ray.shutdown()

    result = np.array(results_values).reshape(train_size, test_size)
    y_pred = y_train[np.argmin(result, axis=0)]

    accuracy = accuracy_score(y_test, y_pred)
    logging.info("Accuracy: {}".format(accuracy))

    end = time()
    cost_time = end - start
    logging.info("cost time: {}".format(cost_time))
    EXP_FOLDER = 'exp'
    date_time = datetime.now().strftime("%Y%m%d")
    experiment_file_path = os.path.join(EXP_FOLDER, args.exp_file + '.csv')
    df = pd.read_csv(experiment_file_path, index_col=0)
    df.loc[args.dataset, args.method+"_norm"] = accuracy
    df.to_csv(experiment_file_path)

    # df = pd.read_csv(experiment_file_path, index_col=0)
    # #add new row if not exist
    # df.loc[args.dataset, args.method] = accuracy
    # df.to_csv(experiment_file_path)



