from utils import get_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from OPW import opw
import numpy as np
from DTW import dtw_distance
# import argparse
#
# parser = argparse.ArgumentParser(description='use KNN to classification time series data')
# parser.add_argument('distance', type=str, help='L2 or DTW or OPW')
# parser.add_argument('K', type=int, help='number of neighbors')
# args = parser.parse_args()


def opw_(X, Y):
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    return opw(X, Y)


def dtw_(X, Y):
    X = np.array(X, dtype=np.double)
    Y = np.array(Y, dtype=np.double)
    return dtw_distance(X, Y)


def KNN(X_train, y_train, X_test, y_test, distance='OPW', K=1):
    return 0


if __name__ == '__main__':
    delta, lambda1, lambda2 = 1, 1, 0.1  # on UCR dataset

    X_train, y_train = get_data('FacesUCR/FacesUCR_TRAIN.txt')
    X_test, y_test = get_data('FacesUCR/FacesUCR_TEST.txt')

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # neigh = KNeighborsClassifier(n_neighbors=2, metric=dtw_)
    # neigh = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    neigh = KNeighborsClassifier(n_neighbors=2, metric=opw_)
    neigh.fit(X_train, y_train)

    y_pred = neigh.predict(X_test)

    print("Accuracy of 1NN: %.2f %%" % (100 * accuracy_score(y_test, y_pred)))
