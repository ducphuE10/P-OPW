import matplotlib.pyplot as plt
import numpy as np

def visualize(data,trend,threshold = None, fig_size = (30,16)):
    '''
    data: input data : numpy array
    trend: result of apply trend filter : numpy array
    threshold: for residual : scalar
    fig_size: figure size
    '''
    fig = plt.figure(figsize=fig_size)
    #TREND & DATA
    plt.subplot(2,1,1)
    residual = data - trend
    dataline = plt.plot(np.arange(data.shape[0]),data,label = 'data')
    trendline = plt.plot(np.arange(data.shape[0]),trend, label = 'trend')
    plt.legend(loc = 'upper left')
    if threshold:
        idx = np.where(np.abs(residual) > threshold)[0]
        plt.scatter(idx, data[idx],c = 'red')

    #RESIDUAL
    plt.subplot(2,1,2)
    plt.plot(np.arange(data.shape[0]),residual,label = 'residual')
    if threshold:
        idx = np.where(np.abs(residual) > threshold)[0]
        plt.scatter(idx, residual[idx],c = 'red')

    plt.legend(loc = 'upper left')
    plt.show()


def get_data(path, dataset_size=-1):
    with open(path, 'r') as f:
        data = []
        labels = []
        for line in f.readlines():
            label = int(float(line.split()[0]))
            record = [float(j) for j in  line.split()[1:]]
            data.append(record)
            labels.append(label)
    if dataset_size > 0:
        data = data[:dataset_size]
        labels = labels[:dataset_size]
    return data, labels

train_data, train_labels = get_data('FacesUCR/FacesUCR_TRAIN.txt')

#--------------------------------------------------------------------------------
import numpy as np
from numpy.linalg import norm

def cal_cosin(i,j):
  return np.dot(i,j)/(norm(i)*norm(j))


def norm1(i,j):
  return norm(i-j,1)

def norm2(i,j):
  return norm(i-j,2)

def get_distance(x,y,distance = 'cosin'):
    '''TODO: implement faster version'''
    D = []
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            assert x[i].shape == y[j].shape
            if distance == 'norm1':
                D.append(norm1(x[i],y[j])/x[i].shape[0])
            elif distance == 'norm2':
                D.append(norm2(x[i],y[j]))
            elif distance == 'mix':
                D.append(1-cal_cosin(x[i],y[j]) + norm2(x[i],y[j]))
            elif distance == 'cosin':
                D.append(1-cal_cosin(x[i],y[j]))
            else:
                raise NotImplementedError
            
    D = np.array(D).reshape(x.shape[0],-1)
    return 