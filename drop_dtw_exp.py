#import wandb
import numpy as np
import wandb
import argparse
import ray
from algo.drop_dtw import drop_dtw_cost
from sklearn.metrics import accuracy_score
from time import time
from utils import get_data
import os
from progress_bar import ProgressBar
import logging
import matplotlib.pyplot as plt

DATA_FOLDER = 'UCR_data/UCRArchive_2018'

logging.basicConfig(level=logging.INFO)



@ray.remote
def drop_dtw(X, Y, pba=None):
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    dist = drop_dtw_cost(X, Y, 0.2)
    if pba is not None:
        pba.update.remote(1)
    return dist


def get_train_test_data():
    train_path = os.path.join(DATA_FOLDER, args.dataset, args.dataset + '_TRAIN.tsv')
    test_path = os.path.join(DATA_FOLDER, args.dataset, args.dataset + '_TEST.tsv')

    X_train, y_train = get_data(train_path)
    X_test, y_test = get_data(test_path, dataset_size=-1)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train, y_test = np.array(y_train,dtype=np.int8), np.array(y_test,dtype=np.int8)
    return X_train, y_train, X_test, y_test

def random_add_noise_with_seed(X, noise_ratio, seed):
    np.random.seed(seed)
    outlier = np.max(X) * np.random.choice([-1, 1], X.shape)
    # choice noise ratio from each row in X without replacement
    outlier_choice = [np.random.choice(X.shape[1], int(X.shape[1] * noise_ratio), replace=False) for _ in range(X.shape[0])]
    outlier_mask = np.zeros(X.shape)
    for i, choice in enumerate(outlier_choice):
        outlier_mask[i, choice] = 1
    # noise_mask = np.random.choice([0, 1], X.shape, p=[1-noise_ratio, noise_ratio])
    X_noise = X + outlier * outlier_mask
    return X_noise



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FacesUCR')
    parser.add_argument('--noise_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.test:
        import matplotlib.pyplot as plt
        X_train, y_train, X_test, y_test = get_train_test_data()
        X_test_noise = random_add_noise_with_seed(X_test, args.noise_ratio, args.seed)
        plt.plot(X_test[0], color='red', label='before')
        plt.plot(X_test_noise[0], color='blue', label='after')
        plt.legend()
        plt.show()
        exit()

    #check if config args is exist in wandb
    # wandb.init(project='sequence-alignment',entity='quydo', config=args)
    wandb.init(project="sequence-classify", entity="sequence-learning", config=args)

    wandb.run.summary["method"] = 'drop-dtw'
    wandb.run.name = wandb.run.summary["method"] + '_' + args.dataset + '_noise_ratio_' + str(args.noise_ratio) + '_seed_' + str(args.seed)


    X_train, y_train, X_test, y_test = get_train_test_data()
    train_size, test_size = X_train.shape[0], X_test.shape[0]
    logging.info("train size: {}, test size: {}".format(train_size, test_size))
    wandb.run.summary["train_size"] = train_size
    wandb.run.summary["test_size"] = test_size
    wandb.run.summary["time_series_length"] = X_train.shape[1]


    X_test_noise = random_add_noise_with_seed(X_test, args.noise_ratio, args.seed)
    logging.info("X_test_noise shape: {}".format(X_test_noise.shape))

    #save image sameple before and after noise to wandb
    plt.plot(X_test[0], color='red', label='before')
    plt.plot(X_test_noise[0], color='blue', label='after')
    plt.legend()
    wandb.log({"sample": wandb.Image(plt)})

    ray.init(num_cpus=4)
    pb = ProgressBar(total=train_size * test_size)
    actor = pb.actor
    start = time()

    results = []

    # import ipdb; ipdb.set_trace()
    for i in range(train_size):
        for j in range(test_size):
            results.append(drop_dtw.remote(X_train[i], X_test_noise[j], actor))
            # a = dist_func(X_train[i], X_test[j])
    pb.print_until_done()
    results_values = ray.get(results)
    ray.shutdown()


    result = np.array(results_values).reshape(train_size, test_size)
    y_pred = y_train[np.argmin(result, axis=0)]

    accuracy = accuracy_score(y_test, y_pred)
    logging.info("Accuracy: {}".format(accuracy))
    wandb.run.summary["accuracy"] = accuracy

    end = time()
    cost_time = end - start
    logging.info("cost time: {}".format(cost_time))
    wandb.run.summary["cost_time"] = cost_time


    #done wandb
    wandb.finish()

