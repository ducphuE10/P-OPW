from utils import get_data
import os 
import ot
import argparse
import numpy as np
from utils import get_distance
from popw import entropic_opw_2,entropic_opw_1
from tqdm import tqdm
DATA_FOLDER = 'UCR_data/UCRArchive_2018'


def popw_(X, Y, lambda1, lambda2, delta = 1, m=0.8,pba=None):
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    # D = get_distance(X,Y,distance= 'norm2')
    D = ot.dist(X,Y)

    a = np.ones(X.shape[0])/X.shape[0]
    b = np.ones(Y.shape[0])/Y.shape[0]
    dist = entropic_opw_2(a, b, D, lambda1, lambda2, delta, m,dropBothSides=True)
    if pba is not None:
        pba.update.remote(1)
    return dist


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Chinatown')


args = parser.parse_args()
    
train_path = os.path.join(DATA_FOLDER, args.dataset, args.dataset + '_TRAIN.tsv')
test_path = os.path.join(DATA_FOLDER, args.dataset, args.dataset + '_TEST.tsv')

X_train, y_train = get_data(train_path)
X_test, y_test = get_data(test_path, dataset_size=-1)


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train, y_test = np.array(y_train,dtype=np.int8), np.array(y_test,dtype=np.int8)

train_size = X_train.shape[0]
test_size = X_test.shape[0]
print('Train size: ', train_size)
print('Test size: ', test_size)

results_values = []


for i in tqdm(range(train_size)):
        for j in range(test_size):
            results_values.append(popw_(X_train[i],X_test[j],lambda1=0,lambda2=0.1,delta=1, m=0.9))

result = np.array(results_values).reshape(train_size, test_size)
y_pred = y_train[np.argmin(result, axis=0)]


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
